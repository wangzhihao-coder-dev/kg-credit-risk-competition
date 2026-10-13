"""Business-facing analytics for the competition report, computed from saved test-period predictions.

Per test quarter, clients are ranked by each model and a review budget (top k%) is applied, mimicking a bank's
quarterly early-warning review. Outputs: gains curves, review effort needed for a target capture rate, early-warning
lead time, clients caught only with the knowledge graph, an anonymised contagion case, and the measured debt of the
distressed firms each model flags early. Debt is DLC + DLTT (USD or CAD millions, not converted) from the point-in-time
Compustat snapshot that defines each panel node; a firm flagged in several quarters counts once, at its first flagged
quarter. Converting debt into avoided loss needs an LGD (supervisory value) and a loss-reduction share (scenario
assumption); both are labelled with their type in experiments/business_value.json.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from audit_rating_history import ROOT
from build_distress_panel import FEATURE_NAMES
from run_distress_models import neighbours

MODELS = {'graph': 'sage_nb', 'no_graph': 'mlp', 'own_scorecard': 'gbdt_own_unlisted'}  # graph: non-backtracking GraphSAGE
SNAPSHOT = ROOT / 'data/raw/wrds/snapshot_population_2015_2023'
REVIEW_BUDGET_SHARE = 0.05
CASE_OWN_RANK_FLOOR = 0.25  # case shown in the report: on the graph model's list, outside the own-information model's top 25%
LGD = 0.45
LOSS_REDUCTION_SCENARIOS = (0.2, 0.3, 0.5)
ASSUMPTIONS = {
    'review_budget_share': {'value': REVIEW_BUDGET_SHARE, 'type': 'design_choice',
                            'source': '每季度复核按模型风险分排名前5%的客户；与报告其他实测指标同一口径'},
    'lgd': {'value': LGD, 'type': 'regulatory',
            'source': '巴塞尔协议内部评级法初级法：无合格抵押的高级（非次级）公司债权监管LGD 45%（Basel II 第287段）；'
                      '巴塞尔III最终版（CRE32）对非金融企业下调为40%，对金融机构保持45%'},
    'loss_reduction_if_flagged_early': {'values': list(LOSS_REDUCTION_SCENARIOS), 'central': 0.3, 'type': 'scenario_assumption',
                                        'source': '情景假设：提前列入复核名单后，通过压降、追加担保、重组等措施减少的损失比例；无数据依据'},
}


def snapshot_debt(keys):
    """Debt (DLC + DLTT, millions) and currency for each (gvkey, pointdate) panel node, from the hash-verified snapshot.

    Uses the same rows as build_distress_panel (UPD 2/3, fiscal data date on or before the point date), so the figures
    are the ones visible at the observation date. Missing components leave the debt as NaN rather than zero.
    """
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text())
    for item in manifest['files']:
        h = hashlib.sha256()
        with (SNAPSHOT / item['file']).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        if h.hexdigest() != item['sha256']:
            raise ValueError('Snapshot mismatch')
    wanted = set(keys); found = {}
    with (SNAPSHOT / 'financials.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            k = (r['gvkey'], r['pointdate'])
            if k not in wanted or r['UPD'] not in ('2.0000', '3.0000') or not r['datadate']:
                continue
            if r['datadate'] > r['pointdate']:
                raise ValueError('Financial data after the observation date: ' + str(k))
            parts = [float(r[c]) if r[c] not in ('', None) else math.nan for c in ('DLC', 'DLTT')]
            found[k] = (sum(parts), r['curcd'])
    missing = wanted - set(found)
    if missing:
        raise ValueError(f'{len(missing)} panel nodes without a snapshot row')
    return np.array([found[k][0] for k in keys]), np.array([found[k][1] for k in keys])


def first_flagged_rows(gvkey, pointdate, positive, flagged):
    """{firm: row of its earliest quarter that is both distressed and on the review list}."""
    first = {}
    for i in np.flatnonzero(positive & flagged):
        g = gvkey[i]
        if g not in first or pointdate[i] < pointdate[first[g]]:
            first[g] = i
    return first


def debt_difference_ci(firms, rows_a, rows_b, debt, draws=2000, seed=0):
    """Firm-resampled 95% interval of total caught debt A - B (debt of a firm not caught, or missing, counts as 0).

    With debt = 1 for every row it is the interval of the difference in caught firm counts.
    """
    value = lambda rows, g: float(np.nan_to_num(debt[rows[g]])) if g in rows else 0.
    diff = np.array([value(rows_a, g) - value(rows_b, g) for g in firms])
    rng = np.random.default_rng(seed)
    totals = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].sum(axis=1)
    return [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))]


def first_flagged_debt(gvkey, pointdate, positive, flagged, debt, currency):
    """Firm-level debt of distressed firms that a review list catches.

    positive: distressed client-quarters; flagged: rows on the review list. Each firm counts once, with the debt of
    the earliest quarter in which it is both distressed and flagged. Debt NaN = missing (counted, never summed as 0).
    """
    first = first_flagged_rows(gvkey, pointdate, positive, flagged)
    out = {'firms': len(first), 'firms_missing_debt': 0, 'debt_m': defaultdict(float), 'firms_by_currency': defaultdict(int)}
    for i in first.values():
        if np.isnan(debt[i]):
            out['firms_missing_debt'] += 1
        else:
            out['debt_m'][str(currency[i])] += float(debt[i]); out['firms_by_currency'][str(currency[i])] += 1
    out['debt_m'] = dict(sorted(out['debt_m'].items())); out['firms_by_currency'] = dict(sorted(out['firms_by_currency'].items()))
    out['debt_total_nominal_m'] = sum(out['debt_m'].values())
    return out


def expected_random_debt(gvkey, pointdate, positive, list_share_by_row, debt, currency):
    """Expected first_flagged_debt when each quarter's review list is a random draw of the same size.

    list_share_by_row: probability that a row is on its quarter's list (list size / clients in that quarter).
    A firm is first caught at its j-th distressed quarter with probability p_j * prod_{i<j}(1 - p_i).
    """
    rows = defaultdict(list)
    for i in np.flatnonzero(positive):
        rows[gvkey[i]].append(i)
    out = {'firms': 0.0, 'firms_missing_debt': 0.0, 'debt_m': defaultdict(float)}
    for lst in rows.values():
        not_yet = 1.0
        for i in sorted(lst, key=lambda i: pointdate[i]):
            prob = not_yet * list_share_by_row[i]; not_yet *= 1 - list_share_by_row[i]
            out['firms'] += prob
            if np.isnan(debt[i]):
                out['firms_missing_debt'] += prob
            else:
                out['debt_m'][str(currency[i])] += prob * float(debt[i])
    out['debt_m'] = dict(sorted(out['debt_m'].items()))
    out['debt_total_nominal_m'] = sum(out['debt_m'].values())
    return out


def main():
    panel_raw = np.load(ROOT / 'data/processed/distress_panel/panel.npz'); panel = {k: panel_raw[k] for k in panel_raw.files}
    pred = np.load(ROOT / 'data/processed/distress_panel/predictions_distress_own_unlisted.npz')
    y, split, sup = pred['y'], pred['split'], pred['supervised']
    test = sup & (split == 'test'); quarters = sorted(set(panel['pointdate'][test]))
    scores = {k: pred[v] for k, v in MODELS.items()}
    neighbour_risk = pred['gbdt_own_full']  # full-information risk of each firm as a counterparty
    shares = [i / 100 for i in range(1, 31)]
    flagged = {k: np.zeros(len(y), dtype=bool) for k in scores}; quarter_rank = {k: np.full(len(y), np.nan) for k in scores}
    gains = {k: np.zeros(len(shares)) for k in scores}; gains['random'] = np.array(shares)
    events_total = 0; per_quarter = []; list_share = np.zeros(len(y))
    for q in quarters:
        idx = np.flatnonzero(test & (panel['pointdate'] == q)); yq = y[idx]; events_total += yq.sum()
        per_quarter.append({'quarter': q, 'clients': len(idx), 'events': int(yq.sum())})
        list_share[idx] = int(round(REVIEW_BUDGET_SHARE * len(idx))) / len(idx)
        for k, s in scores.items():
            order = idx[np.argsort(-s[idx])]; quarter_rank[k][order] = np.arange(len(order)) / len(order)
            for j, share in enumerate(shares):
                gains[k][j] += y[order[:int(round(share * len(order)))]].sum()
            flagged[k][order[:int(round(REVIEW_BUDGET_SHARE * len(order)))]] = True
    for k in scores:
        gains[k] = gains[k] / events_total
    budget = REVIEW_BUDGET_SHARE; j5 = shares.index(budget)
    precision = {k: float(y[flagged[k] & test].mean()) for k in scores}
    base_rate = float(y[test].mean())
    effort = {k: next((s for s, g in zip(shares, gains[k]) if g >= 0.5), None) for k in scores}

    # Early-warning lead: for each distressed firm, earliest flagged quarter among its positive quarters
    # (positive quarters are the observation points within 12 months before the event; last one ~0-3 months before).
    firms = defaultdict(list)
    for i in np.flatnonzero(test & (y == 1)):
        firms[panel['gvkey'][i]].append(i)
    lead = {k: [] for k in scores}
    for rows in firms.values():
        rows = sorted(rows, key=lambda i: panel['pointdate'][i])
        for k in scores:
            hit = [len(rows) - 1 - pos for pos, i in enumerate(rows) if flagged[k][i]]
            lead[k].append(max(hit) * 3 + 1.5 if hit else None)  # months before the last pre-event quarter, mid-quarter
    lead_summary = {k: {'firms_flagged_share': float(np.mean([v is not None for v in vals])),
                        'flagged_6m_or_earlier_share': float(np.mean([v is not None and v >= 6 for v in vals])),
                        'median_lead_months_when_flagged': float(np.median([v for v in vals if v is not None])) if any(v is not None for v in vals) else None}
                    for k, vals in lead.items()}

    # Clients the graph model flags that the own-information model does not (and vice versa), among distressed firms.
    pos_firm = lambda k: {panel['gvkey'][i] for i in np.flatnonzero(test & (y == 1) & flagged[k])}
    only_graph = pos_firm('graph') - pos_firm('no_graph'); only_own = pos_firm('no_graph') - pos_firm('graph')

    # Anonymised case: distressed client on the graph model's review list that the own-information model ranks outside
    # its top CASE_OWN_RANK_FLOOR of the same quarter (illustration; the report states this selection rule).
    rel = neighbours(panel); recent = panel['recent_distress']; ret = panel['X'][:, FEATURE_NAMES.index('ret_12m')]
    late = panel['X'][:, FEATURE_NAMES.index('late_filing_365')]
    incoming = defaultdict(list)
    for r, e in rel.items():
        if r == 'customer_weight':
            continue
        for s, t in e:
            incoming[t].append((r, s))
    candidates = [i for i in np.flatnonzero(test & (y == 1) & flagged['graph']) if quarter_rank['no_graph'][i] > CASE_OWN_RANK_FLOOR and incoming.get(i)]
    def strength(i):
        return max(neighbour_risk[s] for _, s in incoming[i])
    case = None
    if candidates:
        i = max(candidates, key=strength)
        nbrs = sorted(incoming[i], key=lambda t: -neighbour_risk[t[1]])[:8]
        case = {'quarter': str(panel['pointdate'][i]), 'graph_rank_pct': float(quarter_rank['graph'][i]), 'own_rank_pct': float(quarter_rank['no_graph'][i]),
                'graph_p': float(scores['graph'][i]), 'own_p': float(scores['no_graph'][i]),
                'neighbours': [{'relation': r, 'risk_p': float(neighbour_risk[s]), 'recent_distress': int(recent[s]),
                                'ret_12m': None if np.isnan(ret[s]) else float(ret[s]), 'late_filing': bool(late[s] > 0)} for r, s in nbrs],
                'neighbour_count': len(incoming[i]), 'selection_rule': f'graph top {REVIEW_BUDGET_SHARE:.0%}, own-information rank below top {CASE_OWN_RANK_FLOOR:.0%}',
                'candidates': len(candidates)}

    # Descriptive: distress rate of clients by counterparty signals visible at the observation date.
    def any_neighbour(flag):
        out = np.zeros(len(y), dtype=bool)
        for t, lst in incoming.items():
            out[t] = any(flag[s] for _, s in lst)
        return out
    has_nb = np.zeros(len(y), dtype=bool); has_nb[list(incoming)] = True
    signals = {'关联方近一年发生困境': any_neighbour(recent > 0), '关联方股价一年跌超50%': any_neighbour(np.nan_to_num(ret, nan=0) < -0.5),
               '关联方延迟披露': any_neighbour(late > 0)}
    descriptive = {'无关联方': float(y[test & ~has_nb].mean()), '有关联方但无上述信号': float(y[test & has_nb & ~np.any(list(signals.values()), axis=0)].mean())}
    descriptive.update({k: float(y[test & v].mean()) for k, v in signals.items()})
    descriptive_rows = {k: int((test & v).sum()) for k, v in signals.items()}

    # Measured debt of the distressed firms each review list catches (firm-level, first flagged quarter).
    debt, currency = snapshot_debt(list(zip(panel['gvkey'], panel['pointdate'])))
    positive = test & (y == 1)
    caught = {k: first_flagged_debt(panel['gvkey'], panel['pointdate'], positive, flagged[k], debt, currency) for k in scores}
    caught['random'] = expected_random_debt(panel['gvkey'], panel['pointdate'], positive, list_share, debt, currency)
    # Denominator: every distressed firm, at its earliest distressed test quarter (as if flagged at the first chance).
    universe = first_flagged_debt(panel['gvkey'], panel['pointdate'], positive, positive, debt, currency)
    for v in caught.values():
        v['capture_by_amount'] = v['debt_total_nominal_m'] / universe['debt_total_nominal_m']
        v['capture_by_amount_usd'] = v['debt_m'].get('USD', 0.) / universe['debt_m']['USD']
    increment = {k: caught['graph']['debt_total_nominal_m'] - caught[k]['debt_total_nominal_m'] for k in ('no_graph', 'own_scorecard', 'random')}
    # Debt is concentrated in a few large firms: report a firm-resampled interval and the gap without the 10 largest.
    universe_rows = first_flagged_rows(panel['gvkey'], panel['pointdate'], positive, positive)
    rows = {k: first_flagged_rows(panel['gvkey'], panel['pointdate'], positive, flagged[k]) for k in scores}
    largest = sorted(universe_rows, key=lambda g: -np.nan_to_num(debt[universe_rows[g]]))[:10]
    caught_debt = lambda k, keep: sum(float(np.nan_to_num(debt[i])) for g, i in rows[k].items() if keep(g))
    universe_debt = np.array([np.nan_to_num(debt[i]) for i in universe_rows.values()])
    concentration = {
        'median_firm_debt_m': float(np.nanmedian([debt[i] for i in universe_rows.values()])),
        'largest10_share_of_all_distressed_debt': float(np.sort(universe_debt)[::-1][:10].sum() / universe_debt.sum()),
        'largest10_caught': {k: sum(g in rows[k] for g in largest) for k in scores},
        'graph_minus_model_ci95_m': {k: debt_difference_ci(sorted(universe_rows), rows['graph'], rows[k], debt) for k in ('no_graph', 'own_scorecard')},
        'graph_minus_model_firms_ci95': {k: debt_difference_ci(sorted(universe_rows), rows['graph'], rows[k], np.ones(len(debt))) for k in ('no_graph', 'own_scorecard')},
        'graph_minus_model_excluding_largest10_m': {k: caught_debt('graph', lambda g: g not in largest) - caught_debt(k, lambda g: g not in largest)
                                                    for k in ('no_graph', 'own_scorecard')}}
    scenario = {k: {f'{int(r * 100)}%': v['debt_total_nominal_m'] * LGD * r for r in LOSS_REDUCTION_SCENARIOS} for k, v in caught.items()}
    measured_debt = {
        'definition': '测试期每季度按模型风险分取前5%客户；困境企业（未来12个月破产或业绩性摘牌）按企业去重，取首次被列入名单季度的'
                      '债务余额 DLC+DLTT（观察时点可见的 Compustat 时点年度财务）。分母为全部困境企业在其最早一个困境观察季度的债务。'
                      '随机抽查为每季度随机抽取同样数量客户时的期望值。',
        'unit': '百万（报表币种：USD 或 CAD，未换算；合计为名义金额相加）',
        'source': 'data/raw/wrds/snapshot_population_2015_2023/financials.csv',
        'source_sha256': {i['file']: i['sha256'] for i in json.loads((SNAPSHOT / 'manifest.json').read_text())['files']},
        'scope_limits': '企业对全部债权人的总有息债务，不是单家银行的敞口；样本为北美上市企业',
        'all_distressed': universe, 'cad_share_of_all_distressed_debt': universe['debt_m'].get('CAD', 0.) / universe['debt_total_nominal_m'],
        'test_period': '2021–2023（12个季度观察点）；金额为测试期合计，不是年值',
        'by_model': caught, 'graph_minus_model_m': increment, 'concentration': concentration}
    scenario_loss = {'formula': '可规避损失 = 实测债务 × LGD × 提前处置减损比例',
                     'lgd': LGD, 'loss_reduction_scenarios': list(LOSS_REDUCTION_SCENARIOS),
                     'loss_avoided_m_by_model': scenario,
                     'graph_minus_own_scorecard_m': {r: scenario['graph'][r] - scenario['own_scorecard'][r] for r in scenario['graph']},
                     'graph_minus_own_scorecard_m_lgd40': {r: (scenario['graph'][r] - scenario['own_scorecard'][r]) * 0.40 / LGD for r in scenario['graph']}}
    report = {'assumptions': ASSUMPTIONS, 'base_rate': base_rate, 'test_quarters': len(quarters), 'avg_clients_per_quarter': float(np.mean([q['clients'] for q in per_quarter])),
              'review_shares': shares, 'gains': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in gains.items()},
              'precision_at_budget': precision, 'lift_at_budget': {k: v / base_rate for k, v in precision.items()},
              'review_share_for_half_of_events': effort, 'lead_time': lead_summary,
              'distressed_firms': len(firms), 'caught_only_with_graph': len(only_graph), 'caught_only_without_graph': len(only_own),
              'case': case, 'descriptive_distress_rate': descriptive, 'descriptive_rows': descriptive_rows,
              'measured_debt': measured_debt, 'scenario_loss': scenario_loss}
    (ROOT / 'experiments/business_value.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + '\n')
    print(json.dumps({k: report[k] for k in ('precision_at_budget', 'lift_at_budget', 'review_share_for_half_of_events', 'lead_time',
                                             'distressed_firms', 'caught_only_with_graph', 'caught_only_without_graph', 'measured_debt', 'scenario_loss')},
                     indent=1, ensure_ascii=False, default=float))
    print(json.dumps(case, indent=1, default=float))


if __name__ == '__main__':
    main()
