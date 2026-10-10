"""Relational GNN for risk contagion: per-relation mean AND max aggregation with learned relation gates.

Mean aggregation dilutes a single distressed counterparty among many healthy ones; the max channel keeps the
worst related party visible, and the gate lets the model switch a relation off when it carries no signal.
"""
import torch
from torch import nn

from graph_models import mean_messages


def max_messages(x, edges):
    """Element-wise maximum over incoming neighbours; nodes without neighbours receive zeros."""
    out = torch.zeros_like(x)
    if edges.numel() == 0:
        return out
    source, target = edges[:, 0], edges[:, 1]
    reduced = torch.full_like(x, float('-inf')).scatter_reduce(0, target[:, None].expand(-1, x.shape[1]), x[source],
                                                               reduce='amax', include_self=True)
    has = torch.zeros(len(x), dtype=torch.bool, device=x.device); has[target] = True
    out[has] = reduced[has]
    return out


class RelationalRiskGNN(nn.Module):
    def __init__(self, input_dim, hidden=32, n_relations=4, aggregation='maxmean', dropout=0.2, neighbour_dim=None):
        super().__init__()
        if aggregation not in ('maxmean', 'mean', 'none'):
            raise ValueError('Unknown aggregation')
        self.aggregation, self.n_relations = aggregation, n_relations
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        # Optional separate view of each node *as a neighbour* (e.g. a listed counterparty's market signals that the
        # unlisted client itself does not have). Used for first-hop messages only.
        self.neighbour_encoder = nn.Sequential(nn.Linear(neighbour_dim, hidden), nn.ReLU()) if neighbour_dim else None
        self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.mean_layers = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_relations)]) for _ in range(2)])
        self.max_layers = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_relations)]) for _ in range(2)])
        self.gates = nn.Parameter(torch.zeros(2, n_relations))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + input_dim, 1)

    def forward(self, x, edges, relations, x_neighbour=None):
        h = self.encoder(x)
        for layer in range(2):
            update = self.self_layers[layer](h)
            source = self.neighbour_encoder(x_neighbour) if layer == 0 and self.neighbour_encoder is not None else h
            if self.aggregation != 'none':
                for r in range(self.n_relations):
                    selected = edges[relations == r]
                    if selected.numel() == 0:
                        continue
                    message = self.mean_layers[layer][r](mean_messages(source, selected))
                    if self.aggregation == 'maxmean':
                        message = message + self.max_layers[layer][r](max_messages(source, selected))
                    update = update + torch.sigmoid(self.gates[layer, r]) * message
            h = self.dropout(torch.relu(update))
        return self.head(torch.cat((h, x), dim=1)).squeeze(-1)
