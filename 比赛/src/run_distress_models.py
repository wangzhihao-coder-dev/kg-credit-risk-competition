"""Competition model: two-stage intra-risk + contagion-risk scoring of all North-American listed firms.

Stage 1 (encoder): a gradient-boosted own-risk model scores every firm-quarter from its own information
(out-of-fold by firm inside the training period). Stage 2 (propagation + decoder): neighbour risk is summarised per
relation (max / mean / count of high-risk neighbours / realised recent distress) and diffused over the union graph
for three hops; a second model combines own and contagion signals. A relational max+mean GNN is trained end to end
as a competing architecture. Controls: same features with neighbours randomly re-drawn inside each quarter, and
no-edge / rewired GNNs. Selection uses validation AP only.
"""
import argparse, copy, json, math, time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import torch
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn

from audit_rating_history import ROOT
from build_distress_panel import EVENT_NAMES, FEATURE_NAMES, FIN_NAMES, MARKET_NAMES, RATING_NAMES
from distress_gnn import RelationalRiskGNN
from experiment_metrics import evaluate, paired_cluster_bootstrap
from run_global_contagion import ks_statistic

RELATIONS = ['supplier', 'customer', 'people', 'investor']
CONTAGION_NAMES = [f'{r}_{k}' for r in RELATIONS for k in ('n', 'max', 'mean', 'n_high', 'share_high', 'recent_distress', 'late_filing', 'crash',
                                                         'max_deterioration', 'mean_deterioration')] + \
                  ['customer_dependence_max', 'customer_dependence_weighted_risk', 'customer_dependence_max_risk', 'customer_revenue_at_high_risk'] + \
                  ['diffused_1', 'diffused_2', 'diffused_3', 'diffused_distress_1', 'diffused_distress_2']


def split_of(pointdate, config):
    t = date.fromisoformat(pointdate) + timedelta(days=1); h = timedelta(days=config['horizon_days'] + config['maturity_days'])
    if t + h < date.fromisoformat(config['validation_start']):
        return 'train'
    if date.fromisoformat(config['validation_start']) <= t and t + h < date.fromisoformat(config['test_start']):
        return 'validation'
    if t >= date.fromisoformat(config['test_start']):
        return 'test'
    return 'purged'


def preprocess(X, fit_mask, quantiles):
    low, high = np.nanquantile(X[fit_mask], quantiles, axis=0); med = np.nanmedian(X[fit_mask], axis=0)
    low, high, med = (np.nan_to_num(a) for a in (low, high, med))
    missing = ~np.isfinite(X); Z = np.where(missing, med, np.clip(X, low, high))
    mu, sd = Z[fit_mask].mean(0), Z[fit_mask].std(0) + 1e-6
    keep = missing[fit_mask].mean(0) > 0.001
    return np.concatenate(((Z - mu) / sd, missing[:, keep].astype(np.float32)), axis=1).astype(np.float32)


def lgbm(config, seed):
    return LGBMClassifier(**config['lightgbm'], random_state=seed, n_jobs=1, verbosity=-1)


def neighbours(panel, randomized_seed=None):
    """Directed (target <- source) neighbour lists per relation, optionally re-drawn at random inside each quarter."""
    n = len(panel['pointdate']); rng = np.random.default_rng(randomized_seed) if randomized_seed is not None else None
    if rng is not None:
        perm = np.arange(n)
        for p in np.unique(panel['pointdate']):
            idx = np.flatnonzero(panel['pointdate'] == p); perm[idx] = rng.permutation(idx)
        remap = lambda e: np.stack((perm[e[:, 0]], e[:, 1]), axis=1) if len(e) else e
    else:
        remap = lambda e: e
    sup = panel['edges_supply']
    rel = {'supplier': remap(sup), 'customer': remap(sup[:, ::-1].copy()),
           'people': remap(np.concatenate((panel['edges_people'], panel['edges_people'][:, ::-1]))),
           'investor': remap(np.concatenate((panel['edges_investor'], panel['edges_investor'][:, ::-1])))}
    weight = panel['supply_weight'].astype(float) if 'supply_weight' in panel else np.full(len(sup), np.nan)
    keep = {k: v[:, 0] != v[:, 1] for k, v in rel.items()}
    out = {k: v[keep[k]] for k, v in rel.items()}
    out['customer_weight'] = weight[keep['customer']]  # customer dependence aligned with the filtered customer edges
    return out


def previous_quarter_delta(panel, score):
    """Own-score change since the firm's previous quarter (NaN when the firm was absent last quarter)."""
    points = sorted(set(panel['pointdate'])); prev = dict(zip(points[1:], points[:-1]))
    where = {(g, p): i for i, (g, p) in enumerate(zip(panel['gvkey'], panel['pointdate']))}
    delta = np.full(len(score), np.nan)
    for i, (g, p) in enumerate(zip(panel['gvkey'], panel['pointdate'])):
        j = where.get((g, prev.get(p)))
        if j is not None:
            delta[i] = score[i] - score[j]
    return delta


def contagion_features(rel, score, high, signals, delta):
    n = len(score); out = []
    for r in RELATIONS:
        e = rel[r]; src, dst = e[:, 0], e[:, 1]
        count = np.bincount(dst, minlength=n).astype(float)
        mx = np.full(n, np.nan); np.fmax.at(mx, dst, score[src])
        mean = np.bincount(dst, weights=score[src], minlength=n) / np.maximum(count, 1); mean[count == 0] = np.nan
        nh = np.bincount(dst, weights=(score[src] > high).astype(float), minlength=n)
        cols = [np.log1p(count), mx, mean, np.log1p(nh), np.where(count > 0, nh / np.maximum(count, 1), np.nan)]
        for s in signals:
            cols.append(np.where(count > 0, np.bincount(dst, weights=s[src], minlength=n) / np.maximum(count, 1), np.nan))
        d = delta[src]; ok = np.isfinite(d)
        worst = np.full(n, np.nan); np.fmax.at(worst, dst[ok], d[ok])
        k = np.bincount(dst[ok], minlength=n)
        cols += [worst, np.where(k > 0, np.bincount(dst[ok], weights=d[ok], minlength=n) / np.maximum(k, 1), np.nan)]
        out += cols
    # Revenue dependence on customers: target node = supplier, neighbours = its customers, weight = share of its sales.
    e = rel['customer']; src, dst = e[:, 0], e[:, 1]; w = np.nan_to_num(rel['customer_weight'], nan=0.0)
    risk = 1 / (1 + np.exp(-score[src]))
    conc = np.zeros(n); np.maximum.at(conc, dst, w)
    worst = np.zeros(n); np.maximum.at(worst, dst, w * risk)
    has = np.bincount(dst, minlength=n) > 0
    out += [np.where(has, conc, np.nan), np.where(has, np.bincount(dst, weights=w * risk, minlength=n), np.nan),
            np.where(has, worst, np.nan), np.where(has, np.bincount(dst, weights=w * (score[src] > high), minlength=n), np.nan)]
    union = np.concatenate([rel[r] for r in RELATIONS])
    A = sparse.coo_matrix((np.ones(len(union)), (union[:, 1], union[:, 0])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel(); P = sparse.diags(1 / np.maximum(deg, 1)) @ A
    centred = score - np.nanmean(score); d1 = P @ centred; d2 = P @ d1; d3 = P @ d2
    e1 = P @ signals[0].astype(float); e2 = P @ e1
    has = deg > 0
    out += [np.where(has, d1, np.nan), np.where(has, d2, np.nan), np.where(has, d3, np.nan), np.where(has, e1, np.nan), np.where(has, e2, np.nan)]
    return np.column_stack(out)


def metrics(y, p, k):
    m = evaluate(y, p, k); m['ks'] = ks_statistic(y, p)
    order = np.argsort(-p); m['capture_top5pct'] = float(y[order[:max(1, int(.05 * len(y)))]].sum() / y.sum())
    return m


def train_gnn(Z, panel, rel_edges, split, y, sup, kind, seed, config, Z_neighbour=None, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    edges = np.concatenate([rel_edges[r] for r in RELATIONS]); relations = np.concatenate([np.full(len(rel_edges[r]), i) for i, r in enumerate(RELATIONS)])
    if kind.endswith('_rewired'):
        rng = np.random.default_rng(seed); perm = np.arange(len(Z))
        for p in np.unique(panel['pointdate']):
            idx = np.flatnonzero(panel['pointdate'] == p); perm[idx] = rng.permutation(idx)
        edges = np.stack((perm[edges[:, 0]], edges[:, 1]), axis=1)
    blocks = {}
    for s in ('train', 'validation', 'test'):
        nodes = np.flatnonzero(split == s); local = -np.ones(len(Z), dtype=np.int64); local[nodes] = np.arange(len(nodes))
        e = edges[(local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)]; r = relations[(local[edges[:, 0]] >= 0) & (local[edges[:, 1]] >= 0)]
        if kind == 'mlp':
            e, r = e[:0], r[:0]
        target = sup[nodes]
        blocks[s] = (torch.from_numpy(Z[nodes]), torch.from_numpy(local[e]), torch.from_numpy(r), torch.from_numpy(np.flatnonzero(target)),
                     torch.tensor(y[nodes][target], dtype=torch.float32), torch.from_numpy(Z_neighbour[nodes]) if Z_neighbour is not None else None)
    aggregation = 'none' if kind == 'mlp' else ('mean' if kind.startswith('gnn_mean') else 'maxmean')
    model = RelationalRiskGNN(Z.shape[1], config['hidden'], len(RELATIONS), aggregation, neighbour_dim=Z_neighbour.shape[1] if Z_neighbour is not None else None)
    x, e, r, idx, target, xn = blocks['train']
    pos_weight = torch.tensor(math.sqrt((len(target) - target.sum().item()) / max(target.sum().item(), 1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, state, best_epoch = -1, None, 0
    for epoch in range(config['epochs']):
        model.train(); optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, e, r, xn)[idx], target, pos_weight=pos_weight)
        loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad():
            vx, ve, vr, vidx, vt, vn = blocks['validation']; score = average_precision_score(vt.numpy(), model(vx, ve, vr, vn)[vidx].numpy())
        if score > best + 1e-6:
            best, state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
        if epoch + 1 - best_epoch >= config['patience']:
            break
    model.load_state_dict(state); model.eval(); pred = np.full(len(Z), np.nan)
    with torch.no_grad():
        for s, (bx, be, br, _, _, bn) in blocks.items():
            pred[np.flatnonzero(split == s)] = torch.sigmoid(model(bx, be, br, bn)).numpy()
    gates = torch.sigmoid(model.gates).detach().numpy().round(3).tolist()
    details = {'best_epoch': best_epoch, 'validation_ap': best, 'gates': gates}
    return (pred, details, model, blocks) if return_model else (pred, details)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='distress_model_distress_unlisted_config.json')
    config = json.loads((ROOT / 'experiments' / parser.parse_args().config).read_text())
    torch.set_num_threads(1)  # one OpenMP runtime at a time; see graph_models.configure_cpu_runtime
    raw = np.load(ROOT / 'data/processed/distress_panel/panel.npz', allow_pickle=False)
    panel = {k: raw[k] for k in raw.files}
    target_col = {'bankruptcy': 0, 'distress': 1}[config['target']]
    y = panel['labels'][:, target_col].astype(int); sup = panel['supervised'] & (y >= 0)
    split = np.array([split_of(p, config) for p in panel['pointdate']]); groups = panel['gvkey']
    X = panel['X'].astype(float)
    cols = lambda names: [FEATURE_NAMES.index(n) for n in names]
    feature_sets = {'own_financial': cols(FIN_NAMES), 'own_financial_market': cols(FIN_NAMES + MARKET_NAMES),
                    'own_full': cols(FIN_NAMES + MARKET_NAMES + EVENT_NAMES + RATING_NAMES)}
    feature_sets['own_unlisted'] = cols(FIN_NAMES + EVENT_NAMES)  # what an unlisted business client discloses to a bank
    own_key = 'own_unlisted' if config.get('own_info') == 'unlisted' else 'own_full'
    train_sup = sup & (split == 'train')
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'config': config, 'runs': {}, 'comparisons': {},
              'counts': {s: {'rows': int((sup & (split == s)).sum()), 'positive_rows': int(y[sup & (split == s)].sum()),
                             'positive_firms': len(set(groups[sup & (split == s) & (y == 1)])), 'firms': len(set(groups[sup & (split == s)]))}
                         for s in ('train', 'validation', 'test')}}
    print(json.dumps(report['counts']))
    preds = {}

    def stage_one(feature_idx, seed):
        """Out-of-fold scores inside training rows; full-train model for every other node."""
        Xs = X[:, feature_idx]; p = np.full(len(X), np.nan)
        rows = np.flatnonzero(train_sup)
        for fit, hold in GroupKFold(config['folds']).split(rows, groups=groups[rows]):
            m = lgbm(config, seed).fit(Xs[rows[fit]], y[rows[fit]]); p[rows[hold]] = m.predict_proba(Xs[rows[hold]])[:, 1]
        m = lgbm(config, seed).fit(Xs[rows], y[rows]); other = np.setdiff1d(np.arange(len(X)), rows)
        p[other] = m.predict_proba(Xs[other])[:, 1]
        return p, m

    def record(name, p, extra=None):
        preds[name] = p
        report['runs'][name] = {s: metrics(y[sup & (split == s)], p[sup & (split == s)], config['review_k']) for s in ('validation', 'test')}
        if extra:
            report['runs'][name].update(extra)
        v, t = report['runs'][name]['validation'], report['runs'][name]['test']
        print(f"{name:42s} val AP {v['ap']:.4f} test AP {t['ap']:.4f} AUC {t['roc_auc']:.4f} KS {t['ks']:.4f} top5% {t['capture_top5pct']:.3f}", flush=True)

    # Stage 1 ladder (own information only).
    own_scores = {}
    for name, idx in feature_sets.items():
        if name == 'own_unlisted' and own_key != 'own_unlisted':
            continue
        ps = [stage_one(idx, s) for s in config['seeds']]
        own_scores[name] = np.mean([p for p, _ in ps], axis=0)
        record(f'gbdt_{name}', own_scores[name])
    score = np.log(np.clip(own_scores['own_full'], 1e-6, 1 - 1e-6) / (1 - np.clip(own_scores['own_full'], 1e-6, 1 - 1e-6)))
    high = np.quantile(score[train_sup], 0.95)
    late = (X[:, FEATURE_NAMES.index('late_filing_365')] > 0).astype(float)
    crash = (np.nan_to_num(X[:, FEATURE_NAMES.index('ret_12m')], nan=0) < -0.5).astype(float)
    signals = [panel['recent_distress'].astype(float), late, crash]
    delta = previous_quarter_delta(panel, score)
    real = contagion_features(neighbours(panel), score, high, signals, delta)
    control = contagion_features(neighbours(panel, randomized_seed=config['control_seed']), score, high, signals, delta)
    report['relation_coverage_test'] = {r: float((real[sup & (split == 'test'), CONTAGION_NAMES.index(f'{r}_n')] > 0).mean()) for r in RELATIONS}

    # Stage 2: own features + own score + contagion block (and ablations).
    # Neighbours always contribute the full-information score (listed counterparties are observable to a bank);
    # the client itself contributes only its own information set.
    own = X[:, feature_sets[own_key]]
    own_logit = np.log(np.clip(own_scores[own_key], 1e-6, 1 - 1e-6) / (1 - np.clip(own_scores[own_key], 1e-6, 1 - 1e-6)))
    relation_cols = lambda rs: [CONTAGION_NAMES.index(c) for c in CONTAGION_NAMES if c.split('_')[0] in rs]
    dependence_cols = [CONTAGION_NAMES.index(c) for c in CONTAGION_NAMES if '_dependence_' in c or 'revenue_at' in c]
    diffusion_cols = [CONTAGION_NAMES.index(c) for c in CONTAGION_NAMES if c.startswith('diffused')]
    variants = {'two_stage_no_contagion': (real, []),
                'two_stage_contagion': (real, list(range(len(CONTAGION_NAMES)))),
                'two_stage_random_neighbours': (control, list(range(len(CONTAGION_NAMES)))),
                'two_stage_supply_only': (real, relation_cols(['supplier', 'customer'])),
                'two_stage_people_only': (real, relation_cols(['people'])),
                'two_stage_investor_only': (real, relation_cols(['investor'])),
                'two_stage_diffusion_only': (real, diffusion_cols),
                'two_stage_customer_dependence_only': (real, dependence_cols)}
    importance = None
    for name, (block, keep) in variants.items():
        Xs = np.concatenate((own, own_logit[:, None], block[:, keep]), axis=1); ps = []
        for seed in config['seeds']:
            m = lgbm(config, seed).fit(Xs[train_sup], y[train_sup]); ps.append(m.predict_proba(Xs)[:, 1])
            if name == 'two_stage_contagion' and importance is None:
                contrib = m.predict(Xs[sup & (split == 'test')], pred_contrib=True)[:, :-1]
                names = [FEATURE_NAMES[i] for i in feature_sets[own_key]] + ['stage1_own_score'] + CONTAGION_NAMES
                share = np.abs(contrib).mean(0); share = share / share.sum()
                groups_of = {'own_financial': FIN_NAMES, 'own_market': MARKET_NAMES, 'own_events': EVENT_NAMES, 'own_rating': RATING_NAMES,
                             'stage1_own_score': ['stage1_own_score']}
                groups_of = {g: [c for c in cs if c in names] for g, cs in groups_of.items()}
                groups_of.update({f'contagion_{r}': [c for c in CONTAGION_NAMES if c.startswith(r)] for r in RELATIONS})
                groups_of['contagion_diffusion'] = [c for c in CONTAGION_NAMES if c.startswith('diffused')]
                groups_of['contagion_customer'] = [c for c in groups_of['contagion_customer'] if c.split('_')[1] != 'dependence' and 'revenue_at' not in c]
                groups_of['contagion_customer_dependence'] = [c for c in CONTAGION_NAMES if '_dependence_' in c or 'revenue_at' in c]
                importance = {'group_share': {g: float(sum(share[names.index(c)] for c in cs)) for g, cs in groups_of.items()},
                              'top_features': sorted(({'feature': n, 'share': float(s)} for n, s in zip(names, share)), key=lambda d: -d['share'])[:15]}
        record(name, np.mean(ps, axis=0))
    report['shap_two_stage'] = importance

    # Relational GNNs (end-to-end) and controls.
    Z = preprocess(X[:, feature_sets[own_key]], train_sup, config['clip_quantiles']); rel = neighbours(panel)
    Z_neighbour = preprocess(X[:, feature_sets['own_full']], train_sup, config['clip_quantiles']) if own_key != 'own_full' else None
    for kind in config['gnn_models']:
        runs = [train_gnn(Z, panel, rel, split, y, sup, kind, s, config, Z_neighbour) for s in config['seeds']]
        record(kind, np.mean([p for p, _ in runs], axis=0), {'seed_runs': [d for _, d in runs]})

    test = sup & (split == 'test'); g = groups[test]
    pairs = [('gbdt_own_financial_market', 'gbdt_own_financial'), ('gbdt_own_full', 'gbdt_own_financial_market'),
             ('two_stage_no_contagion', f'gbdt_{own_key}'), ('two_stage_contagion', 'two_stage_no_contagion'),
             ('two_stage_contagion', 'two_stage_random_neighbours'), ('two_stage_random_neighbours', 'two_stage_no_contagion'),
             ('two_stage_supply_only', 'two_stage_no_contagion'), ('two_stage_people_only', 'two_stage_no_contagion'),
             ('two_stage_investor_only', 'two_stage_no_contagion'), ('two_stage_diffusion_only', 'two_stage_no_contagion'),
             ('two_stage_customer_dependence_only', 'two_stage_no_contagion')]
    pairs += [(k, 'mlp') for k in config['gnn_models'] if k != 'mlp'] + [('gnn_maxmean', 'gnn_mean'), ('gnn_maxmean', 'gnn_maxmean_rewired')]
    for a, b in pairs:
        if a in preds and b in preds:
            report['comparisons'][f'{a}_vs_{b}'] = paired_cluster_bootstrap(y[test], preds[a][test], preds[b][test], g, config['bootstrap_draws'])
            c = report['comparisons'][f'{a}_vs_{b}']
            print(f"  {a} vs {b}: {c['ap_difference']:+.4f} [{c['ap_difference_ci95'][0]:+.4f}, {c['ap_difference_ci95'][1]:+.4f}]")
    controls = ('random', 'rewired', 'mlp')
    candidates = [k for k in preds if not any(c in k for c in controls)]
    if own_key == 'own_unlisted':
        # An unlisted client has no own market price or rating: models built on those inputs are reference rows only.
        candidates = [k for k in candidates if k not in ('gbdt_own_financial_market', 'gbdt_own_full')]
    report['selection_candidates'] = candidates
    selected = max(candidates, key=lambda k: report['runs'][k]['validation']['ap'])
    report['selected_by_validation'] = selected

    # Scorecard for the selected model: grades on training-score quantiles.
    p = preds[selected]; sc = 600 - 50 / math.log(2) * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cuts = np.quantile(sc[train_sup], [.01, .05, .20, .50]); grade = np.digitize(sc, cuts)
    report['scorecard'] = {'score_cuts': cuts.tolist(), 'grades': [{'grade': 'EDCBA'[k], 'test_rows': int((test & (grade == k)).sum()),
                           'test_events': int(y[test & (grade == k)].sum()), 'test_event_rate': float(y[test & (grade == k)].mean()) if (test & (grade == k)).any() else None,
                           'test_event_firms': len(set(groups[test & (grade == k) & (y == 1)]))} for k in range(5)]}
    # Path cases: test firms in the selected model's top 2% whose strongest related party is high-risk.
    if importance is not None:
        tidx = np.flatnonzero(test); top = tidx[np.argsort(-p[tidx])[:max(1, int(.02 * len(tidx)))]]
        cases = []
        for i in top:
            best = None
            for r in RELATIONS:
                e = rel[r]; srcs = e[e[:, 1] == i, 0]
                if len(srcs):
                    j = srcs[np.argmax(score[srcs])]
                    if best is None or score[j] > best[2]:
                        best = (r, j, score[j])
            if best and best[2] > high:
                cases.append({'risk_rank': int(np.where(top == i)[0][0]) + 1, 'p': float(p[i]), 'own_stage1_p': float(own_scores[own_key][i]),
                              'relation': best[0], 'neighbour_stage1_p': float(1 / (1 + math.exp(-best[2]))),
                              'neighbour_recent_distress': int(panel['recent_distress'][best[1]]), 'label': int(y[i])})
        report['path_cases'] = {'top2pct': len(top), 'with_high_risk_neighbour': len(cases),
                                'event_rate_with': float(np.mean([c['label'] for c in cases])) if cases else None,
                                'event_rate_top2pct': float(y[top].mean()), 'examples': cases[:8]}
    out = ROOT / 'experiments' / config['output']
    out.write_text(json.dumps(report, indent=2, default=float) + '\n')
    np.savez_compressed(ROOT / 'data/processed/distress_panel' / f"predictions_{config['target']}_{own_key}.npz", y=y, split=split, supervised=sup, **preds)
    print('selected', selected); print(json.dumps(report.get('shap_two_stage', {}).get('group_share', {}), indent=1))


if __name__ == '__main__':
    main()
