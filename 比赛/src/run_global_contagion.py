"""Related-party rating-event contagion on the global recorded-rating task (supply chain + industry peers).

Neighbor signals are point-in-time rating states/actions dated on or before the observation
timestamp; neighbor future events and any outcome label are never read. Same panel, splits,
clipping and model settings as run_global_baselines.py (rating_enhancement configuration).
"""
import argparse, csv, hashlib, json, math, time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_rating_history import ROOT, read_verified
from experiment_metrics import evaluate, paired_cluster_bootstrap
from global_panel import financial_features, rating_features
from rating_state import DEFAULT_SYMBOLS, LIVE_SYMBOLS, rating_stamp

ORDER = 'AAA AA+ AA AA- A+ A A- BBB+ BBB BBB- BB+ BB BB- B+ B B- CCC+ CCC CCC- CC C'.split()
NEIGHBOR_NAMES = ['rated', 'mean_ordinal', 'worst_ordinal', 'downgraded_365', 'default_730', 'negative_outlook']
OWN_EVENT_NAMES = ['own_negative_outlook', 'own_watch_negative', 'own_downgraded_365', 'own_notches_change_365']
CLIMATE_NAMES = ['industry_downgrade_share_365', 'industry_negative_outlook_share', 'industry_rated_peers']


def ks_statistic(y, p):
    y = np.asarray(y)[np.argsort(-np.asarray(p), kind='stable')]
    return float(np.max(np.abs(np.cumsum(y) / y.sum() - np.cumsum(1 - y) / (1 - y).sum())))


def entity_rows():
    """gvkey -> issuer rating rows under the same static one-to-one mapping as build_global_panel."""
    raw = ROOT / 'data/raw/wrds'
    links, _ = read_verified(raw / 'ciq_gvkey_history', 'links.csv')
    history, _ = read_verified(raw / 'ratings_history_2015_2023', 'history.csv')
    companies = defaultdict(set); reverse = defaultdict(set)
    for r in links:
        g = r['symbolvalue'].zfill(6); companies[r['companyid']].add(g); reverse[g].add(r['companyid'])
    groups = defaultdict(list); ids = defaultdict(set)
    for r in history:
        if (r['ratingtypecode'], r['orgdebttypecode'], r['globalornationalscaleind'], r['longtermflag']) == ('FCLONG', 'ICR', 'G', '1'):
            groups[r['entity_id']].append(r)
            if r['ciqcompanyid']:
                ids[r['entity_id']].add(r['ciqcompanyid'])
    mapping = {}; g_entities = defaultdict(set)
    for e, cs in ids.items():
        if len(cs) != 1:
            continue
        gs = companies[next(iter(cs))]
        if len(gs) == 1 and len(reverse[next(iter(gs))]) == 1:
            mapping[e] = next(iter(gs)); g_entities[next(iter(gs))].add(e)
    by_g = {g: sorted(groups[e], key=rating_stamp) for e, g in mapping.items() if len(g_entities[g]) == 1}
    industry = {g: rows[-1]['industrydescription'] or rows[-1]['sectordescription'] for g, rows in by_g.items() if rows}
    return by_g, industry


def state_at(rows, t):
    """Latest visible symbol/outlook/watch at t, plus any D/SD in (t-730d, t] and ordinal one year earlier."""
    visible = [r for r in rows if rating_stamp(r) <= t]
    if not visible:
        return None
    last = visible[-1]; earlier = [r for r in visible if rating_stamp(r) <= t - timedelta(days=365)]
    symbol = last['ratingsymbol']
    prior = earlier[-1]['ratingsymbol'] if earlier else None
    ordinal = ORDER.index(symbol) if symbol in ORDER else None
    prior_ordinal = ORDER.index(prior) if prior in ORDER else None
    return {'live': symbol in LIVE_SYMBOLS, 'ordinal': ordinal,
            'downgraded': float(ordinal is not None and prior_ordinal is not None and ordinal > prior_ordinal),
            'notches': float(ordinal - prior_ordinal) if ordinal is not None and prior_ordinal is not None else 0.,
            'default_730': float(any(r['ratingsymbol'] in DEFAULT_SYMBOLS and rating_stamp(r) > t - timedelta(days=730) for r in visible)),
            'negative_outlook': float(last['outlook'] == 'Negative'), 'watch_negative': float(last['creditwatch'] == 'Watch Neg')}


def neighbor_block(states):
    rated = [s for s in states if s and s['ordinal'] is not None]
    ords = [s['ordinal'] for s in rated]
    return [math.log1p(len(rated)), float(np.mean(ords)) if ords else math.nan, float(max(ords)) if ords else math.nan,
            math.log1p(sum(s['downgraded'] for s in states if s)), math.log1p(sum(s['default_730'] for s in states if s)),
            math.log1p(sum(s['negative_outlook'] for s in states if s))]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='global_contagion_config.json')
    args = parser.parse_args(); config = json.loads((ROOT / 'experiments' / args.config).read_text())
    directory = ROOT / 'data/processed' / f"global_rating_{config['panel_freshness_days']}"
    for item in json.loads((directory / 'manifest.json').read_text())['outputs']:
        if hashlib.sha256((directory / item['file']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Panel mismatch')
    fin_names = list(financial_features({})) + ['financial_age_days']
    with (directory / 'nodes.csv').open(newline='') as stream:
        nodes = {(r['gvkey'], r['pointdate']): [float(r[k]) for k in fin_names] for r in csv.DictReader(stream)}
    with (directory / 'labels.csv').open(newline='') as stream:
        labels = [r for r in csv.DictReader(stream) if r['split'] in ('train', 'validation', 'test')]
    graph = json.loads((directory / config['graph_file']).read_text())
    suppliers = defaultdict(set); customers = defaultdict(set)
    for p, g in graph.items():
        for u, v in g['supplier_customer_edges']:
            suppliers[(v, p)].add(u); customers[(u, p)].add(v)
    by_g, industry = entity_rows(); cache = {}

    def state(g, p):
        if (g, p) not in cache:
            t = datetime.fromisoformat(p) + timedelta(days=1)
            cache[(g, p)] = state_at(by_g[g], t) if g in by_g else None
        return cache[(g, p)]

    peers = defaultdict(list)
    for g, ind in industry.items():
        peers[ind].append(g)
    rows = {'own_events': [], 'supplier': [], 'customer': [], 'climate': []}
    for r in labels:
        g, p = r['gvkey'], r['pointdate']; own = state(g, p)
        rows['own_events'].append([own['negative_outlook'], own['watch_negative'], own['downgraded'], own['notches']] if own else [math.nan] * 4)
        rows['supplier'].append(neighbor_block([state(u, p) for u in suppliers[(g, p)]]))
        rows['customer'].append(neighbor_block([state(v, p) for v in customers[(g, p)]]))
        group = [state(h, p) for h in peers.get(industry.get(g), []) if h != g]
        group = [s for s in group if s and s['live']]
        rows['climate'].append([float(np.mean([s['downgraded'] for s in group])) if group else math.nan,
                                float(np.mean([s['negative_outlook'] for s in group])) if group else math.nan,
                                math.log1p(len(group))])
    split = np.array([r['split'] for r in labels]); y = np.array([int(r['label']) for r in labels])
    fin = np.array([nodes[(r['gvkey'], r['pointdate'])] for r in labels])
    rating = np.array([rating_features(r['rating_symbol'], r['rating_age_days']) for r in labels])
    blocks = {'financial': fin, 'own_rating': rating, **{k: np.array(v, float) for k, v in rows.items()}}
    train = split == 'train'
    for k, x in blocks.items():  # clip with training quantiles only
        low, high = np.nanquantile(x[train], config['clip_quantiles'], axis=0)
        blocks[k] = np.where(np.isnan(x), np.nan, np.clip(x, low, high))
    ladder = {'financial_rating': ['financial', 'own_rating'],
              'plus_own_events': ['financial', 'own_rating', 'own_events'],
              'plus_supply_contagion': ['financial', 'own_rating', 'own_events', 'supplier', 'customer'],
              'plus_industry_climate': ['financial', 'own_rating', 'own_events', 'supplier', 'customer', 'climate']}
    results = {}; predictions = {}
    for name, keys in ladder.items():
        x = np.concatenate([blocks[k] for k in keys], axis=1)
        for family in ('logistic', 'tree'):
            preds = []
            for seed in config['seeds']:
                model = (make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                       LogisticRegression(C=config['logistic_C'], max_iter=3000, random_state=seed))
                         if family == 'logistic' else
                         LGBMClassifier(n_estimators=config['tree_estimators'], num_leaves=config['tree_num_leaves'],
                                        learning_rate=config['tree_learning_rate'], random_state=seed, n_jobs=1, verbosity=-1))
                model.fit(x[train], y[train]); preds.append(model.predict_proba(x)[:, 1])
            label = f'{name}_{family}'; predictions[label] = np.mean(preds, axis=0)
            results[label] = {s: {**evaluate(y[split == s], predictions[label][split == s], config['review_k']),
                                  'ks': ks_statistic(y[split == s], predictions[label][split == s])}
                              for s in ('validation', 'test')}
    test = split == 'test'; groups = np.array([r['entity_id'] for r in labels])[test]
    pairs = [(f'{a}_{f}', f'{b}_{f}') for f in ('logistic', 'tree') for a, b in
             [('plus_own_events', 'financial_rating'), ('plus_supply_contagion', 'plus_own_events'),
              ('plus_industry_climate', 'plus_supply_contagion'), ('plus_industry_climate', 'financial_rating')]]
    comparisons = {f'{a}_vs_{b}': paired_cluster_bootstrap(y[test], predictions[a][test], predictions[b][test], groups, config['bootstrap_draws'])
                   for a, b in pairs}
    coverage = {s: {'rows': int((split == s).sum()),
                    'with_rated_supply_neighbor': float(((blocks['supplier'][split == s, 0] > 0) | (blocks['customer'][split == s, 0] > 0)).mean()),
                    'with_neighbor_downgrade_365': float(((blocks['supplier'][split == s, 3] > 0) | (blocks['customer'][split == s, 3] > 0)).mean())}
                for s in ('train', 'validation', 'test')}
    # Subgroup where related-party contagion can act: rows with at least one rated supply-chain neighbor.
    exposed = test & ((blocks['supplier'][:, 0] > 0) | (blocks['customer'][:, 0] > 0))
    sub_groups = np.array([r['entity_id'] for r in labels])[exposed]
    exposed_eval = {'rows': int(exposed.sum()), 'positive_rows': int(y[exposed].sum()),
                    'positive_entities': len({r['entity_id'] for r, e in zip(labels, exposed) if e and r['label'] == '1'}),
                    'event_rate': float(y[exposed].mean()), 'event_rate_unexposed_test': float(y[test & ~exposed].mean())}
    if len(np.unique(y[exposed])) == 2:
        exposed_eval['ap'] = {m: float(evaluate(y[exposed], predictions[m][exposed], config['review_k'])['ap']) for m in predictions}
        exposed_eval['comparisons'] = {f'{a}_vs_{b}': paired_cluster_bootstrap(y[exposed], predictions[a][exposed], predictions[b][exposed], sub_groups, config['bootstrap_draws'])
                                       for a, b in pairs if 'supply' in a}
    selected = max(results, key=lambda m: results[m]['validation']['ap'])
    # Scorecard on the validation-selected model: grades cut on training-score quantiles, evaluated on test.
    to_score = lambda p: 600 - 50 / math.log(2) * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    score = to_score(predictions[selected]); cuts = np.quantile(score[train], [.01, .05, .20, .50])
    grade = np.digitize(score, cuts); names = ['E', 'D', 'C', 'B', 'A']
    grades = [{'grade': names[k], 'test_rows': int((test & (grade == k)).sum()), 'test_events': int(y[test & (grade == k)].sum()),
               'test_event_rate': float(y[test & (grade == k)].mean()) if (test & (grade == k)).any() else None,
               'test_event_entities': len({r['entity_id'] for r, t, g in zip(labels, test, grade) if t and g == k and r['label'] == '1'})}
              for k in range(5)]
    order = np.argsort(-predictions[selected][test]); yt = y[test][order]
    capture = {f'top_{int(q*100)}pct': float(yt[:max(1, int(len(yt) * q))].sum() / yt.sum()) for q in (.01, .05, .10, .20)}
    keys = ladder[selected.rsplit('_', 1)[0]]
    names_all = sum(([f'{k}:{n}' for n in (fin_names if k == 'financial' else ['rating_ordinal', 'log_rating_age'] if k == 'own_rating'
                     else OWN_EVENT_NAMES if k == 'own_events' else NEIGHBOR_NAMES if k in ('supplier', 'customer') else CLIMATE_NAMES)]
                     for k in keys), [])
    x_sel = np.concatenate([blocks[k] for k in keys], axis=1)
    lr = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                       LogisticRegression(C=config['logistic_C'], max_iter=3000, random_state=config['seeds'][0])).fit(x_sel[train], y[train])
    coefficients = sorted(({'feature': n, 'standardized_coefficient': float(c)} for n, c in zip(names_all, lr[-1].coef_[0][:len(names_all)])),
                          key=lambda d: -abs(d['standardized_coefficient']))
    np.savez_compressed(ROOT / 'data/processed' / f"global_rating_{config['panel_freshness_days']}" / 'contagion_predictions.npz',
                        y=y, split=split, **predictions)
    # Descriptive event rates across all supervised rows (train+validation+test); rows are dependent quarters.
    def rate(mask):
        return {'rows': int(mask.sum()), 'events': int(y[mask].sum()), 'event_rate': float(y[mask].mean()) if mask.any() else None,
                'event_entities': len({r['entity_id'] for r, m in zip(labels, mask) if m and r['label'] == '1'})}
    ordinal = rating[:, 0]; ev = blocks['own_events']; sup, cus = blocks['supplier'], blocks['customer']
    descriptive = {'rating_bucket': {'投资级(BBB-及以上)': rate(ordinal <= 9), 'BB档': rate((ordinal >= 10) & (ordinal <= 12)),
                                     'B档': rate((ordinal >= 13) & (ordinal <= 15)), 'CCC至C': rate(ordinal >= 16)},
                   'own_negative_outlook': {'是': rate(ev[:, 0] > 0), '否': rate(~(ev[:, 0] > 0))},
                   'own_watch_negative': {'是': rate(ev[:, 1] > 0), '否': rate(~(ev[:, 1] > 0))},
                   'own_downgraded_365': {'是': rate(ev[:, 2] > 0), '否': rate(~(ev[:, 2] > 0))},
                   'supply_link': {'有供应链关系': rate((sup[:, 0] > 0) | (cus[:, 0] > 0) | np.array([bool(suppliers[(r['gvkey'], r['pointdate'])] or customers[(r['gvkey'], r['pointdate'])]) for r in labels])),
                                   '无供应链关系': rate(~np.array([bool(suppliers[(r['gvkey'], r['pointdate'])] or customers[(r['gvkey'], r['pointdate'])]) for r in labels]))},
                   'neighbor_downgraded_365': {'有关联方近一年降级': rate((sup[:, 3] > 0) | (cus[:, 3] > 0)),
                                               '有评级关联方但无降级': rate(((sup[:, 0] > 0) | (cus[:, 0] > 0)) & ~((sup[:, 3] > 0) | (cus[:, 3] > 0)))},
                   'neighbor_negative_outlook': {'有关联方负面展望': rate((sup[:, 5] > 0) | (cus[:, 5] > 0))}}
    report = {'created_at': datetime.utcnow().isoformat() + 'Z', 'config': config, 'coverage': coverage, 'results': results,
              'selected_by_validation': selected, 'scorecard': {'cuts_train_quantiles': [.01, .05, .20, .50], 'score_cuts': cuts.tolist(), 'grades': grades, 'test_capture': capture}, 'selected_coefficients': coefficients[:12], 'exposed_test_subgroup': exposed_eval, 'descriptive_event_rates': descriptive, 'paired_comparisons': comparisons,
              'feature_names': {'own_events': OWN_EVENT_NAMES, 'supplier': NEIGHBOR_NAMES, 'customer': NEIGHBOR_NAMES, 'climate': CLIMATE_NAMES},
              'claims': {'loan_pd': False, 'neighbor_future_events_used': False,
                         'analysis_status': 'Added after primary global test inspection; exploratory, not new independent validation'}}
    (ROOT / 'experiments' / 'global_contagion_results.json').write_text(json.dumps(report, indent=2) + '\n')
    for k, v in results.items():
        print(f"{k:40s} val AP {v['validation']['ap']:.4f} test AP {v['test']['ap']:.4f} AUC {v['test']['roc_auc']:.4f} KS {v['test']['ks']:.4f}")
    print('selected', selected, json.dumps(coverage)); print(json.dumps(descriptive, ensure_ascii=False, indent=1)); print(json.dumps(grades)); print(capture); print(coefficients[:10])
    for k, v in comparisons.items():
        print(k, round(v['ap_difference'], 4), [round(a, 4) for a in v['ap_difference_ci95']])


if __name__ == '__main__':
    main()
