"""Triangle-update axial transformer for pairwise (bus x bus) targets.

DQ(i,j) depends on paths through intermediate buses k. Row/column attention only
mixes cells that share a row or column; a triangle multiplicative update
(AlphaFold-style) mixes cell (i,j) with the pair of cells (i,k) and (j,k)
(outgoing edges) or (k,i) and (k,j) (incoming edges) for every bus k. Both
directions are used because DQ is not symmetric.

Each block:  outgoing triangle update -> incoming triangle update
             -> axial row/column attention + feed-forward (reused unchanged).

Input (B, C, N, N) where C = in_channels (the ablation wrapper masks excluded
channels to zero before calling). The transposed matrix and per-bus row/column
means of every channel are added as extra input features, so the network can
see Ddc(i,j) and Ddc(j,i) and closeness/betweenness-like summaries directly.
"""
import torch
from torch import nn

from transformer_model import (AxialSelfAttentionBlock, OutputProjection,
                               ScalarTokenEmbedding, VmagHead)


class TriangleMultiplication(nn.Module):
    """Gated triangle multiplicative update on a (B, N, N, D) pair tensor."""

    def __init__(self, d_model: int, hidden: int, outgoing: bool):
        super().__init__()
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 2 * hidden)
        self.gate = nn.Linear(d_model, 2 * hidden)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.out_gate = nn.Linear(d_model, d_model)

    def mix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sum over the intermediate bus k, in float32 (sum of N products)."""
        with torch.autocast(device_type=a.device.type, enabled=False):
            a, b = a.float(), b.float()
            if self.outgoing:   # out[i,j] = sum_k a[i,k] * b[j,k]
                return torch.einsum('bikc,bjkc->bijc', a, b)
            return torch.einsum('bkic,bkjc->bijc', a, b)   # sum_k a[k,i] * b[k,j]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zn = self.norm(z)
        a, b = (self.proj(zn) * torch.sigmoid(self.gate(zn))).chunk(2, dim=-1)
        mixed = self.out_proj(self.out_norm(self.mix(a, b)))
        return mixed * torch.sigmoid(self.out_gate(zn))


class TriangleBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, dropout, hidden):
        super().__init__()
        self.outgoing = TriangleMultiplication(d_model, hidden, outgoing=True)
        self.incoming = TriangleMultiplication(d_model, hidden, outgoing=False)
        self.drop = nn.Dropout(dropout)
        self.axial = AxialSelfAttentionBlock(d_model, num_heads, mlp_ratio, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.drop(self.outgoing(z))
        z = z + self.drop(self.incoming(z))
        return self.axial(z)


class TriangleDQNet(nn.Module):
    def __init__(self, n_buses: int = 30, d_model: int = 96, depth: int = 4,
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 in_channels: int = 1, tri_dim: int = None):
        super().__init__()
        self.n_buses = n_buses
        self.in_channels = in_channels
        # pair features: [x, x^T]; node features: [row means, column means]
        self.embed = ScalarTokenEmbedding(d_model, n_buses, 2 * in_channels)
        self.node_row = nn.Linear(2 * in_channels, d_model)
        self.node_col = nn.Linear(2 * in_channels, d_model)
        hidden = tri_dim or max(8, d_model // 2)
        self.blocks = nn.ModuleList([
            TriangleBlock(d_model, num_heads, mlp_ratio, dropout, hidden)
            for _ in range(depth)])
        self.out = OutputProjection(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N, N) -> pair tokens (B, N, N, d_model)."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'Expected (batch,{self.in_channels},N,N), got {tuple(x.shape)}')
        pair = torch.cat([x, x.transpose(2, 3)], dim=1)                  # (B,2C,N,N)
        node = torch.cat([x.mean(dim=3), x.mean(dim=2)], dim=1)          # (B,2C,N)
        node = node.transpose(1, 2)                                      # (B,N,2C)
        z = self.embed(pair)
        z = z + self.node_row(node)[:, :, None, :] + self.node_col(node)[:, None, :, :]
        for block in self.blocks:
            z = block(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.encode(x))                                  # (B,1,N,N)


class TriangleVmagNet(nn.Module):
    """Same backbone, per-bus vector head (row pooling) for Vmag. Output (B, N)."""

    def __init__(self, n_buses: int = 30, d_model: int = 96, depth: int = 4,
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 in_channels: int = 1):
        super().__init__()
        self.n_buses = n_buses
        self.backbone = TriangleDQNet(n_buses, d_model, depth, num_heads, mlp_ratio,
                                      dropout, in_channels)
        self.vmag_head = VmagHead(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vmag_head(self.backbone.encode(x))
