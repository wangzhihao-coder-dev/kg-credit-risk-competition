"""Non-backtracking GraphSAGE for unlisted business clients (the report's main graph model).

Two GraphSAGE layers (Hamilton et al., 2017; mean aggregator with a separate weight for the node itself) over the union
of all relation types. As in RelationalRiskGNN, the client is encoded from its own information only and every
counterparty from its full public information. Non-backtracking: in layer 2 a neighbour j sends client i the layer-1
state j would have *without* i, so i's own full-information view (its market price and rating, which an unlisted
client does not have) can never return to i through i -> j -> i. With two layers this is the only path back to i.

Pure PyTorch; parameter shapes and initialisation order follow PyTorch Geometric's SAGEConv, so a seed gives the same
model as a SAGEConv-based implementation without adding that dependency.
"""
import torch
from torch import nn


def union_edges(edges):
    """Unique directed pairs (source, target) of a [E, 2] edge list, as a [2, U] tensor."""
    return torch.unique(edges.t(), dim=1) if edges.numel() else edges.t()


def reverse_multiplicity(edges_2xE, n):
    """For each edge j -> i: how many times i -> j occurs in the same list (i is then one of j's own neighbours)."""
    if edges_2xE.shape[1] == 0:
        return torch.zeros(0, dtype=torch.long)
    keys, counts = torch.unique(edges_2xE[0] * n + edges_2xE[1], return_counts=True)
    q = edges_2xE[1] * n + edges_2xE[0]
    pos = torch.searchsorted(keys, q).clamp_max(len(keys) - 1)
    return torch.where(keys[pos] == q, counts[pos], torch.zeros_like(q))


class NonBacktrackingSAGE(nn.Module):
    def __init__(self, input_dim, hidden=32, neighbour_dim=None, dropout=0.2, non_backtracking=True):
        super().__init__()
        self.enc_client = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.enc_firm = nn.Sequential(nn.Linear(neighbour_dim or input_dim, hidden), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + input_dim, 1)
        self.lin_l, self.lin_r = nn.ModuleList(), nn.ModuleList()
        for _ in range(2):  # SAGEConv: lin_l (neighbours, bias) and lin_r (self, no bias), then reset_parameters()
            l, r = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
            l.reset_parameters(); r.reset_parameters()
            self.lin_l.append(l); self.lin_r.append(r)
        self.non_backtracking = non_backtracking
        self._cache = {}

    def _structure(self, edges, n):
        key = (edges.data_ptr(), tuple(edges.shape), n)
        if key not in self._cache:
            self._cache = {key: (lambda u: (u, reverse_multiplicity(u, n)))(union_edges(edges))}
        return self._cache[key]

    def forward(self, x, edges, relations=None, x_neighbour=None):
        """Same call as RelationalRiskGNN; relation types are merged (GraphSAGE has one weight for all neighbours)."""
        n = len(x); ei, mult = self._structure(edges, n); src, dst = ei[0], ei[1]
        h0 = self.enc_client(x); s = self.enc_firm(x if x_neighbour is None else x_neighbour)
        cnt = torch.bincount(dst, minlength=n).to(s.dtype)[:, None]
        total = torch.zeros_like(s).index_add_(0, dst, s[src]); mean = total / cnt.clamp_min(1)
        u = self.lin_l[0](mean) + self.lin_r[0](h0)
        mask = self.dropout(torch.ones_like(u))  # node-level dropout mask shared by every message a node sends
        h1 = torch.relu(u) * mask
        u_e = u[src]  # layer-1 state of j = src as sent to i = dst
        if self.non_backtracking:
            k = mult.to(s.dtype)[:, None]; left = cnt[src] - k
            mean_ex = torch.where(left > 0, (total[src] - k * s[dst]) / left.clamp_min(1), torch.zeros_like(s[dst]))
            u_e = u_e + nn.functional.linear(mean_ex - mean[src], self.lin_l[0].weight)
        h1e = torch.relu(u_e) * mask[src]
        agg = torch.zeros_like(h1).index_add_(0, dst, h1e) / cnt.clamp_min(1)
        h = self.dropout(torch.relu(self.lin_l[1](agg) + self.lin_r[1](h1)))
        return self.head(torch.cat((h, x), dim=1)).squeeze(-1)
