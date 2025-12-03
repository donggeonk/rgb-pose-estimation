import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A  # Adjacency matrix (fixed or learnable)
        
        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        A = self.A.to(x.device)
        
        x = self.conv(x)
        x = x.view(N, self.out_channels, A.size(0), T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(ST_GCN_Block, self).__init__()
        
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1, inplace=True)
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class PoseRefinerModel(nn.Module):
    def __init__(self, num_joints=33, in_channels=3, hidden_dim=64):
        super(PoseRefinerModel, self).__init__()
        
        # 1. Define Graph (Adjacency Matrix)
        # We use a simplified adjacency strategy: 1 for self, 1 for connected neighbor
        self.A = self.get_adjacency_matrix(num_joints)
        
        # 2. Network Architecture
        # Input is (N, 3, T, V)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, hidden_dim, self.A, residual=False),
            ST_GCN_Block(hidden_dim, hidden_dim, self.A),
            ST_GCN_Block(hidden_dim, hidden_dim, self.A),
            ST_GCN_Block(hidden_dim, hidden_dim, self.A),
            ST_GCN_Block(hidden_dim, in_channels, self.A, residual=False) # Project back to 3D
        ])

    def get_adjacency_matrix(self, num_joints):
        # Define MediaPipe connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
        ]
        A = torch.zeros(num_joints, num_joints)
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1
        # Add self-loops
        A = A + torch.eye(num_joints)
        # Normalize
        D = torch.sum(A, dim=1)
        D_inv = torch.pow(D, -1).diag()
        A_norm = torch.mm(D_inv, A)
        return A_norm.unsqueeze(0) # (1, V, V)

    def forward(self, x):
        # x shape: (N, T, V, C) -> needs to be (N, C, T, V) for Conv2d
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Store input for residual connection (We learn the correction delta)
        x_input = x 
        
        # Normalization
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)
        
        # Forward pass
        for gcn in self.st_gcn_networks:
            x = gcn(x)
            
        # Add the learned correction to the original input
        # Output = Input + Correction
        out = x_input + x 
        
        # Permute back to (N, T, V, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out