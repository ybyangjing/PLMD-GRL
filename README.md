# PLMD-GRL: Proximity-Aware and Load-Balanced Microservice Deployment for Graph Reinforcement Learning

This is the official PyTorch code for the paper:

**PLMD-GRL: Proximity-Aware and Load-Balanced Microservice Deployment for Graph Reinforcement Learning in Edge-Cloud Environments**

**Keli Liu, Jing Yang\*, Yuling Chen, Xiaoli Ruan, Han Zhao, Shaobo Li and Minyi Guo**

> **Note: The complete code will be uploaded after the paper is accepted.** Currently, this repository contains the core algorithmic implementation.

<div align="center">
  <img src="framework.png" width="100%" alt="PLMD-GRL Framework"/>
</div>

## Requirements

Install all required dependencies into a new virtual environment via conda.

```bash
conda create -n plmd_grl python=3.8
conda activate plmd_grl

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install torch-geometric numpy networkx python-igraph leidenalg matplotlib
