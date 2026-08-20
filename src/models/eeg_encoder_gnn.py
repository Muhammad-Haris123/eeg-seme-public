"""
GNN-based EEG encoder that respects 10-20 electrode topology.

Works without torch_geometric via ManualGraphConv.
If torch_geometric is available, a PyG variant can be used.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv  # type: ignore

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]

ADJACENCY: Dict[str, List[str]] = {
    "Fp1": ["Fp2", "F7", "F3"],
    "Fp2": ["Fp1", "F8", "F4"],
    "F7": ["Fp1", "F3", "T3"],
    "F3": ["Fp1", "F7", "Fz", "C3"],
    "Fz": ["F3", "F4", "Cz"],
    "F4": ["Fp2", "Fz", "F8", "C4"],
    "F8": ["Fp2", "F4", "T4"],
    "T3": ["F7", "C3", "T5"],
    "C3": ["F3", "T3", "Cz", "P3"],
    "Cz": ["Fz", "C3", "C4", "Pz"],
    "C4": ["F4", "Cz", "T4", "P4"],
    "T4": ["F8", "C4", "T6"],
    "T5": ["T3", "P3", "O1"],
    "P3": ["C3", "T5", "Pz", "O1"],
    "Pz": ["Cz", "P3", "P4"],
    "P4": ["C4", "Pz", "T6", "O2"],
    "T6": ["T4", "P4", "O2"],
    "O1": ["T5", "P3", "Pz"],
    "O2": ["T6", "P4", "Pz"],
}


def build_adjacency_matrix() -> torch.Tensor:
    """Binary (19, 19) adjacency with self-loops."""
    n = len(CHANNEL_NAMES)
    idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
    adj = torch.zeros((n, n), dtype=torch.float32)
    for ch, nbrs in ADJACENCY.items():
        i = idx[ch]
        adj[i, i] = 1.0
        for nb in nbrs:
            j = idx[nb]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def build_edge_index() -> torch.Tensor:
    """COO edge index (2, n_edges) from adjacency."""
    adj = build_adjacency_matrix()
    src, dst = torch.where(adj > 0)
    return torch.stack([src, dst], dim=0).long()


def _triu_pairs(n_channels: int = 19) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pairs.append((i, j))
    return pairs


PAIR_INDEX = _triu_pairs(19)  # 171 pairs in the same order as torch.triu_indices


def _node_pair_mask() -> torch.Tensor:
    """
    Map node -> involved pair indices.
    Returns mask shape (19, 171), float in {0,1}.
    """
    n_channels = 19
    n_pairs = len(PAIR_INDEX)
    mask = torch.zeros((n_channels, n_pairs), dtype=torch.float32)
    for p_idx, (i, j) in enumerate(PAIR_INDEX):
        mask[i, p_idx] = 1.0
        mask[j, p_idx] = 1.0
    return mask


NODE_PAIR_MASK = _node_pair_mask()
NODE_PAIR_COUNT = NODE_PAIR_MASK.sum(dim=1, keepdim=True).clamp(min=1.0)  # (19, 1)


def reshape_features_to_nodes(x_flat: torch.Tensor) -> torch.Tensor:
    """
    Map flat EEG features (batch, 2185) -> node features (batch, 19, 35).

    Mapping used:
    - Per-channel direct features (25):
        PSD[0:380] reshaped to (19, 20)
        Band powers[380:475] reshaped to (19, 5)
    - Pairwise summaries (10):
        coherence[475:1330] -> (5, 171), summarize per node by mean over incident pairs
        plv[1330:2185] -> (5, 171), summarize per node similarly
      => 5 coherence + 5 PLV summary features per node
    """
    if x_flat.ndim != 2 or x_flat.shape[1] != 2185:
        raise ValueError(f"Expected x_flat shape (batch, 2185), got {tuple(x_flat.shape)}")

    bsz = x_flat.shape[0]
    device = x_flat.device

    psd = x_flat[:, 0:380].view(bsz, 19, 20)
    bands = x_flat[:, 380:475].view(bsz, 19, 5)

    coh = x_flat[:, 475:1330].view(bsz, 5, 171)
    plv = x_flat[:, 1330:2185].view(bsz, 5, 171)

    mask = NODE_PAIR_MASK.to(device).unsqueeze(0).unsqueeze(1)  # (1,1,19,171)
    count = NODE_PAIR_COUNT.to(device).unsqueeze(0).unsqueeze(1)  # (1,1,19,1)

    coh_node = (coh.unsqueeze(2) * mask).sum(dim=-1) / count.squeeze(-1)  # (batch, 5, 19)
    plv_node = (plv.unsqueeze(2) * mask).sum(dim=-1) / count.squeeze(-1)  # (batch, 5, 19)

    coh_node = coh_node.transpose(1, 2)  # (batch, 19, 5)
    plv_node = plv_node.transpose(1, 2)  # (batch, 19, 5)

    node_feats = torch.cat([psd, bands, coh_node, plv_node], dim=-1)  # (batch, 19, 35)
    return node_feats


class ManualGraphConv(nn.Module):
    """
    GraphSAGE-style graph convolution without torch_geometric.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features)
        self.W_neighbor = nn.Linear(in_features, out_features)

        adj = build_adjacency_matrix()
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg
        self.register_buffer("adj_norm", adj_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 19, in_features) -> (batch, 19, out_features)
        """
        self_out = self.W_self(x)
        adj = self.adj_norm.to(x.device)
        neighbor_feats = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        neighbor_out = self.W_neighbor(neighbor_feats)
        return self_out + neighbor_out


class EEGGraphEncoder(nn.Module):
    """
    Manual GNN EEG encoder.
    Output shape: (batch, latent_dim) where latent_dim defaults to 128.
    """

    def __init__(self, input_dim: int = 2185, latent_dim: int = 128, n_channels: int = 19):
        super().__init__()
        self.n_channels = n_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.features_per_node = 35

        self.conv1 = ManualGraphConv(self.features_per_node, 128)
        self.conv2 = ManualGraphConv(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_proj = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}")

        x_nodes = reshape_features_to_nodes(x)  # (batch, 19, 35)
        h = self.relu(self.conv1(x_nodes))  # (batch, 19, 128)
        h = self.dropout(h)
        h = self.relu(self.conv2(h))  # (batch, 19, 64)

        mean_pool = h.mean(dim=1)  # (batch, 64)
        max_pool = h.max(dim=1).values  # (batch, 64)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (batch, 128)
        return self.output_proj(pooled)


class EEGGraphEncoderPyG(nn.Module):
    """
    Optional PyG implementation (used only if torch_geometric is available).
    """

    def __init__(self, input_dim: int = 2185, latent_dim: int = 128, n_channels: int = 19):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric not available")
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.features_per_node = 35

        self.conv1 = SAGEConv(self.features_per_node, 128)
        self.conv2 = SAGEConv(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output_proj = nn.Linear(128, latent_dim)
        self.register_buffer("edge_index", build_edge_index())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}")
        bsz = x.shape[0]
        x_nodes = reshape_features_to_nodes(x)  # (B, 19, 35)
        outs = []
        for b in range(bsz):
            h = x_nodes[b]  # (19, 35)
            h = self.relu(self.conv1(h, self.edge_index))
            h = self.dropout(h)
            h = self.relu(self.conv2(h, self.edge_index))  # (19, 64)
            mean_pool = h.mean(dim=0, keepdim=True)  # (1,64)
            max_pool = h.max(dim=0, keepdim=True).values  # (1,64)
            pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (1,128)
            outs.append(self.output_proj(pooled))
        return torch.cat(outs, dim=0)


def get_gnn_encoder(input_dim: int = 2185, latent_dim: int = 128, **kwargs) -> nn.Module:
    """Factory: PyG version if available, else manual implementation."""
    if HAS_PYG:
        return EEGGraphEncoderPyG(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
    return EEGGraphEncoder(input_dim=input_dim, latent_dim=latent_dim, **kwargs)

