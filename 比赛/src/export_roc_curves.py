"""Test-period ROC curves of the report's model comparison (unlisted-client view, distress target).

Reads the per-row predictions that run_distress_models.py saves under data/processed/ and writes only aggregate curve
points to experiments/roc_curves.json, so the report builder never touches row-level data.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from audit_rating_history import ROOT

PREDICTIONS = ROOT / 'data/processed/distress_panel/predictions_distress_own_unlisted.npz'
RESULTS = ROOT / 'experiments/distress_distress_unlisted_results.json'
OUT = ROOT / 'experiments/roc_curves.json'
MODELS = ['gbdt_own_unlisted', 'mlp', 'two_stage_contagion', 'gnn_maxmean_nb', 'sage_nb', 'sage_nb_rewired']
# Denser at low false-positive rates, where a review list actually operates.
GRID = np.unique(np.concatenate((np.linspace(0, .1, 101), np.linspace(.1, 1, 181))))


def roc_points(y, p, grid=GRID):
    """True-positive rate at each false-positive rate on the grid (the ROC step function, highest TPR at that FPR)."""
    fpr, tpr, _ = roc_curve(y, p)
    return tpr[np.searchsorted(fpr, grid, side='right') - 1]


def main():
    raw = np.load(PREDICTIONS, allow_pickle=False)
    test = raw['supervised'] & (raw['split'] == 'test'); y = raw['y'][test]
    reported = json.loads(RESULTS.read_text())['runs']
    out = {'source': PREDICTIONS.name, 'split': 'test', 'n': int(test.sum()), 'positives': int(y.sum()),
           'fpr': GRID.round(4).tolist(), 'models': {}}
    for name in MODELS:
        p = raw[name][test]; auc = float(roc_auc_score(y, p))
        if abs(auc - reported[name]['test']['roc_auc']) > 1e-9:
            raise SystemExit(f'{name}: AUC {auc} differs from {RESULTS.name}')
        out['models'][name] = {'auc': auc, 'tpr': roc_points(y, p).round(4).tolist()}
        print(f'{name:22s} AUC {auc:.4f}')
    OUT.write_text(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
