"""Relational GNN for risk contagion: per-relation mean AND max aggregation with learned relation gates.

Mean aggregation dilutes a single distressed counterparty among many healthy ones; the max channel keeps the
worst related party visible, and the gate lets the model switch a relation off when it carries no signal.

non_backtracking=True (off by default): in layer 2 a neighbour j sends node i the layer-1 state j would have without i
(mean: i removed, max: next largest value where i is the maximum). In the unlisted-client view this keeps a client's
own full-information view from returning to it through i -> j -> i, the only such path in a two-layer model.
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
    def __init__(self, input_dim, hidden=32, n_relations=4, aggregation='maxmean', dropout=0.2, neighbour_dim=None, non_backtracking=False):
        super().__init__()
        self.non_backtracking, self._cache = non_backtracking, {}
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
        if self.non_backtracking and self.aggregation == 'maxmean':
            return self._forward_non_backtracking(x, edges, relations, x_neighbour)
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

    def _structure(self, edges, relations, n):
        key = (edges.data_ptr(), tuple(edges.shape), relations.data_ptr(), n)
        if key not in self._cache:
            by_rel = [edges[relations == r].t() for r in range(self.n_relations)]
            j, i = edges[:, 0], edges[:, 1]
            mult = []
            for e in by_rel:  # how often i is j's neighbour in relation r (edge i -> j present)
                if e.shape[1] == 0:
                    mult.append(torch.zeros(len(edges), dtype=torch.long)); continue
                keys, counts = torch.unique(e[0] * n + e[1], return_counts=True); q = i * n + j
                pos = torch.searchsorted(keys, q).clamp_max(len(keys) - 1)
                mult.append(torch.where(keys[pos] == q, counts[pos], torch.zeros_like(q)))
            self._cache = {key: (by_rel, mult, [(relations == r).nonzero().squeeze(1) for r in range(self.n_relations)])}
        return self._cache[key]

    def _forward_non_backtracking(self, x, edges, relations, x_neighbour):
        n = len(x); by_rel, mult, slices = self._structure(edges, relations, n)
        j_of, i_of = edges[:, 0], edges[:, 1]
        h0 = self.encoder(x); s = self.neighbour_encoder(x_neighbour) if self.neighbour_encoder is not None else h0
        u = self.self_layers[0](h0); stats = []
        for r in range(self.n_relations):
            sel = by_rel[r].t()
            if sel.numel() == 0:
                stats.append(None); continue
            mean_r, max_r = mean_messages(s, sel), max_messages(s, sel)
            u = u + torch.sigmoid(self.gates[0, r]) * (self.mean_layers[0][r](mean_r) + self.max_layers[0][r](max_r))
            stats.append((sel, mean_r, max_r))
        mask = self.dropout(torch.ones_like(u))  # same Bernoulli draws as dropout(relu(u))
        h1 = torch.relu(u) * mask
        corr = torch.zeros(len(edges), u.shape[1], dtype=u.dtype)
        for r, st in enumerate(stats):
            idx = (mult[r] > 0).nonzero().squeeze(1)
            if st is None or len(idx) == 0:
                continue
            sel, mean_r, max_r = st; src, dst = sel[:, 0], sel[:, 1]
            cnt = torch.bincount(dst, minlength=n).to(s.dtype)[:, None]
            total = torch.zeros_like(s).index_add_(0, dst, s[src])
            ties = torch.zeros_like(s).index_add_(0, dst, (s[src] == max_r[dst]).to(s.dtype)).detach()
            below = torch.where(s[src] < max_r[dst], s[src], torch.full_like(s[src], float('-inf')))
            second = torch.full_like(s, float('-inf')).scatter_reduce(0, dst[:, None].expand(-1, s.shape[1]), below, reduce='amax', include_self=True)
            second = torch.where(torch.isfinite(second), second, torch.zeros_like(second))
            jj, ii, k = j_of[idx], i_of[idx], mult[r][idx].to(s.dtype)[:, None]
            left = cnt[jj] - k
            mean_ex = torch.where(left > 0, (total[jj] - k * s[ii]) / left.clamp_min(1), torch.zeros_like(s[ii]))
            use_second = (s[ii] >= max_r[jj]) & (ties[jj] <= k)
            max_ex = torch.where(left > 0, torch.where(use_second, second[jj], max_r[jj]), torch.zeros_like(s[ii]))
            delta = self.mean_layers[0][r](mean_ex - mean_r[jj]) + self.max_layers[0][r](max_ex - max_r[jj])
            corr = corr.index_add(0, idx, torch.sigmoid(self.gates[0, r]) * delta)
        h1e = torch.relu(u[j_of] + corr) * mask[j_of]
        update = self.self_layers[1](h1)
        for r in range(self.n_relations):
            e = slices[r]
            if len(e) == 0:
                continue
            dst, vals = i_of[e], h1e[e]
            cnt = torch.bincount(dst, minlength=n).to(h1.dtype)[:, None]
            mean_r = torch.zeros_like(h1).index_add_(0, dst, vals) / cnt.clamp_min(1)
            red = torch.full_like(h1, float('-inf')).scatter_reduce(0, dst[:, None].expand(-1, h1.shape[1]), vals, reduce='amax', include_self=True)
            max_r = torch.zeros_like(h1); has = cnt.squeeze(1) > 0; max_r[has] = red[has]
            update = update + torch.sigmoid(self.gates[1, r]) * (self.mean_layers[1][r](mean_r) + self.max_layers[1][r](max_r))
        h = self.dropout(torch.relu(update))
        return self.head(torch.cat((h, x), dim=1)).squeeze(-1)
