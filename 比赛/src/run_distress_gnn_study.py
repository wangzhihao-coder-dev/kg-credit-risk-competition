"""Study of the relational max+mean GNN for unlisted business clients.

Main model: client sees only its own financial + event information; neighbours (listed counterparties) are encoded
from their full information. Ablations (same seeds, budget and early stopping): one relation type at a time, and a
neighbour view restricted to the same unlisted information (tests whether counterparties' market signals carry the
contagion). For the main model: score card on training-score quantiles, top-k capture, and relation attribution by
removing one relation type at inference for the highest-risk test clients.
"""
import argparse, json, math
from datetime import datetime, timezone

import numpy as np
import torch

from audit_rating_history import ROOT
from build_distress_panel import EVENT_NAMES, FEATURE_NAMES, FIN_NAMES, MARKET_NAMES, RATING_NAMES
from experiment_metrics import paired_cluster_bootstrap
from run_distress_models import RELATIONS, metrics, neighbours, preprocess, split_of, train_gnn


def subset(rel, keep):
    return {r: (v if r in keep else v[:0]) for r, v in rel.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='distress_model_distress_unlisted_config.json')
    config = json.loads((ROOT / 'experiments' / parser.parse_args().config).read_text())
    torch.set_num_threads(1)
    raw = np.load(ROOT / 'data/processed/distress_panel/panel.npz'); panel = {k: raw[k] for k in raw.files}
    y = panel['labels'][:, {'bankruptcy': 0, 'distress': 1}[config['target']]].astype(int); sup = panel['supervised'] & (y >= 0)
    split = np.array([split_of(p, config) for p in panel['pointdate']]); groups = panel['gvkey']
    X = panel['X'].astype(float); cols = lambda names: [FEATURE_NAMES.index(n) for n in names]
    train_sup = sup & (split == 'train'); test = sup & (split == 'test')
    Z = preprocess(X[:, cols(FIN_NAMES + EVENT_NAMES)], train_sup, config['clip_quantiles'])
    Z_full = preprocess(X[:, cols(FIN_NAMES + MARKET_NAMES + EVENT_NAMES + RATING_NAMES)], train_sup, config['clip_quantiles'])
    rel = neighbours(panel)
    variants = {'main_all_relations_full_neighbour_view': (rel, Z_full),
                'supply_only': (subset(rel, {'supplier', 'customer'}), Z_full),
                'people_only': (subset(rel, {'people'}), Z_full),
                'investor_only': (subset(rel, {'investor'}), Z_full),
                'neighbour_view_unlisted_only': (rel, None)}
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'config': config, 'runs': {}, 'comparisons': {}}
    preds, models = {}, []
    for name, (edges, view) in variants.items():
        outs = [train_gnn(Z, panel, edges, split, y, sup, 'gnn_maxmean', s, config, view, return_model=(name.startswith('main'))) for s in config['seeds']]
        preds[name] = np.mean([o[0] for o in outs], axis=0)
        if name.startswith('main'):
            models = [(o[2], o[3]) for o in outs]
        report['runs'][name] = {s: metrics(y[sup & (split == s)], preds[name][sup & (split == s)], config['review_k']) for s in ('validation', 'test')}
        t = report['runs'][name]['test']
        print(f"{name:40s} val AP {report['runs'][name]['validation']['ap']:.4f} test AP {t['ap']:.4f} AUC {t['roc_auc']:.4f} top5% {t['capture_top5pct']:.3f}", flush=True)
    main_name = 'main_all_relations_full_neighbour_view'
    for other in [k for k in preds if k != main_name]:
        c = paired_cluster_bootstrap(y[test], preds[main_name][test], preds[other][test], groups[test], config['bootstrap_draws'])
        report['comparisons'][f'{main_name}_vs_{other}'] = c
        print(f"  main vs {other}: {c['ap_difference']:+.4f} [{c['ap_difference_ci95'][0]:+.4f}, {c['ap_difference_ci95'][1]:+.4f}]")

    # Score card and capture for the main model.
    p = preds[main_name]; score = 600 - 50 / math.log(2) * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cuts = np.quantile(score[train_sup], [.01, .05, .20, .50]); grade = np.digitize(score, cuts)
    report['scorecard'] = {'score_cuts': cuts.tolist(), 'grades': [
        {'grade': 'EDCBA'[k], 'test_rows': int((test & (grade == k)).sum()), 'test_events': int(y[test & (grade == k)].sum()),
         'test_event_rate': float(y[test & (grade == k)].mean()) if (test & (grade == k)).any() else None,
         'test_event_firms': len(set(groups[test & (grade == k) & (y == 1)]))} for k in range(5)]}
    order = np.argsort(-p[test]); yt = y[test][order]
    report['capture'] = {f'top_{q}pct': float(yt[:max(1, int(len(yt) * q / 100))].sum() / yt.sum()) for q in (1, 5, 10, 20)}
    report['test_event_rate'] = float(y[test].mean())

    # Relation attribution: drop one relation type at inference for the top 5% test clients (seed-averaged).
    tidx = np.flatnonzero(split == 'test'); local = {g: i for i, g in enumerate(tidx)}
    top = [local[i] for i in np.flatnonzero(test)[np.argsort(-p[test])[:max(1, int(.05 * test.sum()))]]]
    full, dropped = [], {r: [] for r in RELATIONS}
    with torch.no_grad():
        for model, blocks in models:
            x, e, r, _, _, xn = blocks['test']; full.append(torch.sigmoid(model(x, e, r, xn)).numpy()[top])
            for k, name in enumerate(RELATIONS):
                keep = r != k; dropped[name].append(torch.sigmoid(model(x, e[keep], r[keep], xn)).numpy()[top])
    full = np.mean(full, axis=0); attribution = {name: full - np.mean(v, axis=0) for name, v in dropped.items()}
    matrix = np.column_stack([attribution[r] for r in RELATIONS]); dominant = np.argmax(matrix, axis=1)
    y_top = y[tidx[top]]
    report['relation_attribution_top5pct'] = {
        'clients': len(top), 'event_rate': float(y_top.mean()),
        'mean_score_drop_when_removed': {r: float(attribution[r].mean()) for r in RELATIONS},
        'dominant_relation_share': {r: float((dominant == k).mean()) for k, r in enumerate(RELATIONS)},
        'event_rate_by_dominant_relation': {r: float(y_top[dominant == k].mean()) if (dominant == k).any() else None for k, r in enumerate(RELATIONS)},
        'share_where_relations_explain_over_half': float((matrix.clip(min=0).sum(1) > 0.5 * full).mean())}
    print(json.dumps({k: report[k] for k in ('scorecard', 'capture', 'relation_attribution_top5pct')}, indent=1))
    out = ROOT / 'experiments' / config['output'].replace('_results.json', '_gnn_study.json')
    out.write_text(json.dumps(report, indent=2, default=float) + '\n')
    np.savez_compressed(ROOT / 'data/processed/distress_panel' / f"gnn_study_{config['target']}.npz", y=y, split=split, supervised=sup, **preds)


if __name__ == '__main__':
    main()
