import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import softmax
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import leidenalg as la

LR_Actor = 0.0001
LR_Critic = 0.0001
LR_AttnSAGE = 0.0001
GAMMA = 0.99
TAU = 0.01
MEMORY_CAPACITY = 500
BATCH_SIZE = 32
POLICY_DELAY = 2
NOISE_CLIP = 0.5
POLICY_NOISE = 0.2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LCR(affinity_matrices, lambda_penalty=0.3,
        lambda1=0.5, lambda2=0.3, lambda3=0.2):

    sorted_applications = {}

    for app, affinity_matrix in affinity_matrices.items():
        n_nodes = len(affinity_matrix)

        if n_nodes == 1:
            sorted_applications[app] = [0]
            continue
        G = nx.from_numpy_array(np.array(affinity_matrix), create_using=nx.DiGraph)
        edges = []
        weights = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if affinity_matrix[i][j] > 0:
                    edges.append((i, j))
                    weights.append(affinity_matrix[i][j])
        if len(edges) == 0:
            sorted_applications[app] = list(range(n_nodes))
            continue
        ig_graph = ig.Graph(n=n_nodes, edges=edges, directed=True)
        ig_graph.es['weight'] = weights

        try:
            partition = la.find_partition(ig_graph, la.ModularityVertexPartition,
                                          weights='weight')
        except:
            sorted_applications[app] = list(range(n_nodes))
            continue

        current_membership = list(partition.membership)

        def compute_cross_community_penalty(membership, aff_matrix):
            penalty = 0
            for i in range(len(membership)):
                for j in range(len(membership)):
                    if aff_matrix[i][j] > 0 and membership[i] != membership[j]:
                        penalty += aff_matrix[i][j]
            return penalty

        def compute_modified_modularity(membership, base_modularity, aff_matrix, lam):
            cross_penalty = compute_cross_community_penalty(membership, aff_matrix)
            return base_modularity - lam * cross_penalty

        initial_base_Q = partition.modularity
        current_Q_prime = compute_modified_modularity(
            current_membership, initial_base_Q, affinity_matrix, lambda_penalty
        )

        max_iterations = 10
        for iteration in range(max_iterations):
            improved = False

            for node in range(n_nodes):
                current_com = current_membership[node]

                neighbor_comm_weights = {}
                for j in range(n_nodes):
                    if node != j:
                        weight = affinity_matrix[node][j] + affinity_matrix[j][node]
                        if weight > 0:
                            j_com = current_membership[j]
                            neighbor_comm_weights[j_com] = neighbor_comm_weights.get(j_com, 0) + weight

                if not neighbor_comm_weights:
                    continue

                best_com = max(neighbor_comm_weights, key=neighbor_comm_weights.get)

                if best_com != current_com:
                    old_node_penalty = sum(w for c, w in neighbor_comm_weights.items() if c != current_com)
                    new_node_penalty = sum(w for c, w in neighbor_comm_weights.items() if c != best_com)
                    delta_Q_prime = lambda_penalty * (old_node_penalty - new_node_penalty)

                    if delta_Q_prime > 0:
                        current_membership[node] = best_com
                        current_Q_prime += delta_Q_prime
                        improved = True

            if not improved:
                break

        communities = {}
        for node, com_id in enumerate(current_membership):
            if com_id not in communities:
                communities[com_id] = []
            communities[com_id].append(node)

        sorted_microservices_list = []

        for com_id in sorted(communities.keys()):
            nodes = communities[com_id]

            if len(nodes) == 1:
                sorted_microservices_list.extend(nodes)
                continue

            subgraph = G.subgraph(nodes)
            importance_scores = {}

            for node in nodes:
                try:
                    pagerank = nx.pagerank(subgraph, weight='weight')
                    pr_score = pagerank.get(node, 0)
                except:
                    pr_score = 1.0 / len(nodes)

                try:
                    if len(nodes) > 2:
                        betweenness = nx.betweenness_centrality(subgraph, weight='weight')
                        bc_score = betweenness.get(node, 0)
                    else:
                        bc_score = 0
                except:
                    bc_score = 0

                predecessors = list(subgraph.predecessors(node))
                n_dependencies = len(predecessors) if len(predecessors) > 0 else 1
                dependency_score = 1.0 / n_dependencies

                importance = lambda1 * pr_score + lambda2 * bc_score + lambda3 * dependency_score
                importance_scores[node] = importance

            sorted_community = sorted(importance_scores.keys(),
                                      key=lambda x: importance_scores[x],
                                      reverse=True)
            sorted_microservices_list.extend(sorted_community)

        sorted_applications[app] = sorted_microservices_list

    return sorted_applications

class WeightedGATLayer(nn.Module):

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.5, negative_slope=0.2):
        super(WeightedGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        N = x.size(0)
        x = self.lin(x).view(N, self.heads, self.out_channels)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        alpha = alpha_src[src_idx] + alpha_dst[dst_idx]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        alpha = softmax(alpha, dst_idx, num_nodes=N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros(N, self.heads, self.out_channels, device=x.device)
        src_features = x[src_idx]
        weighted_features = src_features * alpha.unsqueeze(-1)
        out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(weighted_features), weighted_features)

        out = out.view(N, self.heads * self.out_channels)
        out = out + self.bias

        return out


class AttnSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dim=16, output_dim=8, heads=4, dropout=0.5):
        super(AttnSAGE, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.gat = WeightedGATLayer(hidden_dim, output_dim, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(heads * output_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.output_dim = heads * output_dim

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.embedding(x))
        x = self.sage(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.gat(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

    def get_graph_embedding(self, x, edge_index, edge_weight=None):
        node_embeddings = self.forward(x, edge_index, edge_weight)
        graph_embedding = torch.mean(node_embeddings, dim=0)
        return graph_embedding, node_embeddings


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class PLMD_GRL:

    def __init__(self, raw_state_dim, action_dim, n_services, n_edges,
                 service_features, edge_features, edge_index,
                 affinity_matrix=None,
                 attn_hidden_dim=16, attn_output_dim=8, attn_heads=4,
                 grad_clip=1.0):

        self.raw_state_dim = raw_state_dim
        self.action_dim = action_dim
        self.n_services = n_services
        self.n_edges = n_edges
        self.grad_clip = grad_clip

        attn_final_dim = attn_output_dim * attn_heads
        self.attn_final_dim = attn_final_dim
        self.state_dim = attn_final_dim + raw_state_dim

        self.service_features = torch.tensor(service_features, dtype=torch.float32).to(device)
        self.edge_features = torch.tensor(edge_features, dtype=torch.float32).to(device)
        self.node_features = torch.cat([self.service_features, self.edge_features], dim=0)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        if affinity_matrix is not None:
            self.affinity_matrix = np.array(affinity_matrix)
        else:
            self.affinity_matrix = np.ones((n_services, n_services))
            np.fill_diagonal(self.affinity_matrix, 0)

        self.edge_weights = self._compute_initial_edge_weights()

        input_feature_dim = self.node_features.shape[1]
        self.attn_sage = AttnSAGE(
            input_dim=input_feature_dim,
            hidden_dim=attn_hidden_dim,
            output_dim=attn_output_dim,
            heads=attn_heads
        ).to(device)

        self.memory = np.zeros((24, MEMORY_CAPACITY, self.raw_state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = np.zeros(24, dtype=np.int32)
        self.actor_eval = Actor(self.state_dim, action_dim).to(device)
        self.actor_target = Actor(self.state_dim, action_dim).to(device)
        self.critic_eval_1 = Critic(self.state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(self.state_dim, action_dim).to(device)
        self.critic_eval_2 = Critic(self.state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(self.state_dim, action_dim).to(device)

        self.optimizer = optim.Adam([
            {'params': self.attn_sage.parameters(), 'lr': LR_AttnSAGE},
            {'params': self.actor_eval.parameters(), 'lr': LR_Actor},
            {'params': self.critic_eval_1.parameters(), 'lr': LR_Critic},
            {'params': self.critic_eval_2.parameters(), 'lr': LR_Critic},
        ])

        self.loss_critic = nn.MSELoss()
        self.cost_critic = []
        self.cost_actor = []
        self.cost_attn_sage = []
        self.hard_update(self.actor_target, self.actor_eval)
        self.hard_update(self.critic_target_1, self.critic_eval_1)
        self.hard_update(self.critic_target_2, self.critic_eval_2)
        self.update_counter = 0

    def _compute_initial_edge_weights(self):
        edge_index_np = self.edge_index.cpu().numpy()
        num_edges = edge_index_np.shape[1]
        edge_weights = []

        for idx in range(num_edges):
            src, dst = edge_index_np[0, idx], edge_index_np[1, idx]
            if src < self.n_services and dst >= self.n_services:
                edge_weights.append(1.0)
            elif src >= self.n_services and dst < self.n_services:
                edge_weights.append(1.0)
            elif src < self.n_services and dst < self.n_services:
                edge_weights.append(float(self.affinity_matrix[src, dst]))
            else:
                edge_weights.append(1.0)
        return torch.tensor(edge_weights, dtype=torch.float32).to(device)

    def compute_edge_weights(self, deployment_state):
        edge_index_np = self.edge_index.cpu().numpy()
        num_edges = edge_index_np.shape[1]
        edge_weights = []

        for idx in range(num_edges):
            src, dst = edge_index_np[0, idx], edge_index_np[1, idx]
            if src < self.n_services and dst >= self.n_services:
                edge_idx = dst - self.n_services
                edge_weights.append(1.0 if deployment_state[src, edge_idx] == 1 else 0.0)
            elif src >= self.n_services and dst < self.n_services:
                edge_idx = src - self.n_services
                edge_weights.append(1.0 if deployment_state[dst, edge_idx] == 1 else 0.0)
            elif src < self.n_services and dst < self.n_services:
                edge_weights.append(float(self.affinity_matrix[src, dst]))
            else:
                edge_weights.append(1.0)
        return torch.tensor(edge_weights, dtype=torch.float32).to(device)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def _get_graph_embedding_cached(self, deployment_state, training=False):
        edge_weights = self.compute_edge_weights(deployment_state)

        if training:
            self.attn_sage.train()
            graph_embedding, _ = self.attn_sage.get_graph_embedding(
                self.node_features, self.edge_index, edge_weights
            )
        else:
            self.attn_sage.eval()
            with torch.no_grad():
                graph_embedding, _ = self.attn_sage.get_graph_embedding(
                    self.node_features, self.edge_index, edge_weights
                )
        return graph_embedding

    def extract_state_features(self, raw_state, training=False):
        raw_state_np = np.array(raw_state).flatten()
        deployment_state = raw_state_np[:self.n_services * self.n_edges].reshape(
            self.n_services, self.n_edges
        )

        graph_embedding = self._get_graph_embedding_cached(deployment_state, training)
        raw_state_tensor = torch.FloatTensor(raw_state).to(device)
        if raw_state_tensor.dim() == 1:
            raw_state_tensor = raw_state_tensor.unsqueeze(0)
        graph_embedding = graph_embedding.unsqueeze(0)
        enhanced_state = torch.cat([graph_embedding, raw_state_tensor], dim=1)
        return enhanced_state

    def extract_batch_state_features(self, raw_states_batch, training=True):

        self.attn_sage.train() if training else self.attn_sage.eval()
        batch_size = raw_states_batch.shape[0]
        enhanced_states = []
        for i in range(batch_size):
            raw_state = raw_states_batch[i].cpu().numpy()
            deployment_state = raw_state[:self.n_services * self.n_edges].reshape(
                self.n_services, self.n_edges
            )

            edge_weights = self.compute_edge_weights(deployment_state)

            if training:
                graph_embedding, _ = self.attn_sage.get_graph_embedding(
                    self.node_features, self.edge_index, edge_weights
                )
            else:
                with torch.no_grad():
                    graph_embedding, _ = self.attn_sage.get_graph_embedding(
                        self.node_features, self.edge_index, edge_weights
                    )

            raw_state_tensor = raw_states_batch[i].unsqueeze(0)
            graph_embedding = graph_embedding.unsqueeze(0)
            enhanced_state = torch.cat([graph_embedding, raw_state_tensor], dim=1)
            enhanced_states.append(enhanced_state)

        return torch.cat(enhanced_states, dim=0)

    def choose_action(self, raw_state):
        enhanced_state = self.extract_state_features(raw_state, training=False)
        self.actor_eval.eval()
        with torch.no_grad():
            action = self.actor_eval(enhanced_state)
        return action.cpu().numpy()

    def store_transition(self, state, action, reward, next_state, t):
        raw_state = np.array(state).flatten()
        raw_next_state = np.array(next_state).flatten()
        action_flat = np.array(action).flatten()
        reward_flat = np.array([reward]).flatten()

        transition = np.hstack((raw_state, action_flat, reward_flat, raw_next_state))
        index = int(self.pointer[t] % MEMORY_CAPACITY)
        self.memory[t, index, :] = transition
        self.pointer[t] += 1

    def learn(self, t, update_policy=True):

        self.update_counter += 1
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        buffer = self.memory[t, indices, :]
        state_end = self.raw_state_dim
        action_end = state_end + self.action_dim
        reward_end = action_end + 1
        buffer_raw_state = torch.FloatTensor(buffer[:, :state_end]).to(device)
        buffer_action = torch.FloatTensor(buffer[:, state_end:action_end]).to(device)
        buffer_reward = torch.FloatTensor(buffer[:, action_end:reward_end]).to(device)
        buffer_raw_next_state = torch.FloatTensor(buffer[:, reward_end:]).to(device)
        buffer_state = self.extract_batch_state_features(buffer_raw_state, training=True)

        with torch.no_grad():
            buffer_next_state = self.extract_batch_state_features(buffer_raw_next_state, training=False)

        with torch.no_grad():
            noise = (torch.randn_like(buffer_action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(buffer_next_state) + noise).clamp(-1.0, 1.0)
            Q_target1 = self.critic_target_1(buffer_next_state, next_action)
            Q_target2 = self.critic_target_2(buffer_next_state, next_action)
            Q_target = buffer_reward + GAMMA * torch.min(Q_target1, Q_target2)

        Q1 = self.critic_eval_1(buffer_state, buffer_action)
        Q2 = self.critic_eval_2(buffer_state, buffer_action)
        loss_critic_1 = self.loss_critic(Q1, Q_target)
        loss_critic_2 = self.loss_critic(Q2, Q_target)
        loss_critic = loss_critic_1 + loss_critic_2

        loss_actor = torch.tensor(0.0)
        if update_policy and self.update_counter % POLICY_DELAY == 0:
            actor_action = self.actor_eval(buffer_state)
            loss_actor = -self.critic_eval_1(buffer_state, actor_action).mean()

        self.optimizer.zero_grad()

        total_loss = loss_critic
        if update_policy and self.update_counter % POLICY_DELAY == 0:
            total_loss = total_loss + loss_actor

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.attn_sage.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_eval_1.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_eval_2.parameters(), self.grad_clip)

        self.optimizer.step()

        if update_policy and self.update_counter % POLICY_DELAY == 0:
            self.soft_update(self.actor_target, self.actor_eval)
            self.soft_update(self.critic_target_1, self.critic_eval_1)
            self.soft_update(self.critic_target_2, self.critic_eval_2)
            self.cost_actor.append(loss_actor.item())

        self.cost_critic.append((loss_critic_1.item(), loss_critic_2.item()))

        attn_sage_grad_norm = 0
        for p in self.attn_sage.parameters():
            if p.grad is not None:
                attn_sage_grad_norm += p.grad.norm().item()
        self.cost_attn_sage.append(attn_sage_grad_norm)

    def compute_reward(self, avg_time_prev, avg_time_curr, load_variance_prev, load_variance_curr,
                       min_avg_time, iteration, k1=10, k2=1, k3=10, k4=100):
        if iteration == 0:
            RT = (avg_time_prev - avg_time_curr) / k1
        elif iteration >= 1 and avg_time_curr <= min_avg_time + 10:
            RT = k2 + (min_avg_time - avg_time_curr) / k3
        else:
            RT = (min_avg_time - avg_time_curr) / k4

        RL = -load_variance_curr
        total_reward = RT + RL
        return total_reward, RT, RL

    def plot_cost(self):
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        if len(self.cost_critic) > 0:
            critic_loss_1, critic_loss_2 = zip(*self.cost_critic)
            plt.plot(critic_loss_1, label="Critic1 Loss", alpha=0.7)
            plt.plot(critic_loss_2, label="Critic2 Loss", alpha=0.7)
        plt.ylabel('Q Loss')
        plt.xlabel('Training Steps')
        plt.legend()
        plt.title('Critic Networks Loss')

        plt.subplot(1, 3, 2)
        if len(self.cost_actor) > 0:
            plt.plot(self.cost_actor, label="Actor Loss", color='green', alpha=0.7)
        plt.ylabel('Actor Loss')
        plt.xlabel('Training Steps')
        plt.legend()
        plt.title('Actor Network Loss')

        plt.subplot(1, 3, 3)
        if len(self.cost_attn_sage) > 0:
            plt.plot(self.cost_attn_sage, label="AttnSAGE Grad Norm", color='orange', alpha=0.7)
        plt.ylabel('Gradient Norm')
        plt.xlabel('Training Steps')
        plt.legend()
        plt.title('AttnSAGE Gradient Norm')

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.show()

    def save_model(self, path):
        torch.save({
            'attn_sage': self.attn_sage.state_dict(),
            'actor_eval': self.actor_eval.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_eval_1': self.critic_eval_1.state_dict(),
            'critic_target_1': self.critic_target_1.state_dict(),
            'critic_eval_2': self.critic_eval_2.state_dict(),
            'critic_target_2': self.critic_target_2.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.attn_sage.load_state_dict(checkpoint['attn_sage'])
        self.actor_eval.load_state_dict(checkpoint['actor_eval'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_eval_1.load_state_dict(checkpoint['critic_eval_1'])
        self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
        self.critic_eval_2.load_state_dict(checkpoint['critic_eval_2'])
        self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

