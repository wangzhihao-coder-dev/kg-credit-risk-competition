"""Point-in-time panel for the competition task: 12-month bankruptcy / distress of all North-American listed firms.

Nodes: every (gvkey, quarter-end) with a fresh annual financial snapshot (UPD 2/3, data date <= point, age <= 550 days).
Node features (visible at t = point + 1 day): financial ratios and size, CRSP market signals from months ending <= point,
own risk events (Audit Analytics litigation / bankruptcy history, NT late filings, restatements) and, when rated,
the S&P rating state. Relations per quarter: supplier->customer (disclosure + 365 days, cumulative), shared
BoardEx people, shared >=5% 13F holders (index-scale holders excluded). Labels: bankruptcy (AA bankruptcy notification
start/file date or CRSP delisting code 574) and distress (bankruptcy or CRSP performance delisting) within (t, t+365];
rows with a bankruptcy in the previous 730 days are not supervised. Outputs go to data/processed/distress_panel/.
"""
import csv, gzip, hashlib, json, math
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import combinations

import numpy as np

from audit_rating_history import ROOT, read_verified
from build_related_graph import FILING_LAG_DAYS, PASSIVE_THRESHOLD, role_active, verified_rows
from global_graph import visible_supply_edges
from global_panel import financial_features
from run_global_contagion import entity_rows, state_at

EVENTS = ROOT / 'data/raw/wrds/distress_events_2012_2024'
OUT = ROOT / 'data/processed/distress_panel'
FRESHNESS = 550
HORIZON, MATURITY, LOOKBACK = 365, 30, 730
DATA_END = date(2024, 12, 31)  # CRSP delisting coverage of the export
DISTRESS_EXCLUDED_CODES = {570, 571, 572, 573, 575}  # company-requested delistings are not distress
FIN_NAMES = list(financial_features({})) + ['financial_age_days', 'log_assets', 'log_sales']
MARKET_NAMES = ['ret_12m', 'vol_12m', 'log_mcap', 'penny_stock', 'drawdown_12m', 'ret_3m']
EVENT_NAMES = ['lit_def_new_365', 'lit_def_open', 'late_filing_365', 'late_10k_365', 'restatement_365',
               'restatement_negative_365', 'prior_bankruptcy']
RATING_NAMES = ['rated', 'rating_ordinal', 'rating_downgraded_365', 'rating_negative_outlook']
FEATURE_NAMES = FIN_NAMES + MARKET_NAMES + EVENT_NAMES + RATING_NAMES


def event_rows(name):
    manifest = json.loads((EVENTS / 'manifest.json').read_text())
    item = next(f for f in manifest['files'] if f['name'] == name)
    if hashlib.sha256((EVENTS / item['file']).read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Source mismatch: ' + name)
    return csv.DictReader(gzip.open(EVENTS / item['file'], 'rt', newline=''))


def permno_key(value):
    """CRSP identifiers arrive as '25881.0' in the link table and '25881' in msf/msedelist."""
    return str(int(float(value)))


def market_features(series, point):
    """series: sorted [(month_end, ret, price, shares)]; uses months ending on or before the point date."""
    k = bisect_right([s[0] for s in series], point)
    window = [s for s in series[max(0, k - 12):k] if s[1] is not None]
    if len(window) < 6 or (point - series[k - 1][0]).days > 62:
        return [math.nan] * len(MARKET_NAMES)
    rets = np.array([s[1] for s in window]); wealth = np.cumprod(1 + rets)
    last = series[k - 1]; mcap = abs(last[2]) * last[3] if last[2] and last[3] else math.nan
    return [float(wealth[-1] - 1), float(rets.std() * math.sqrt(12)), math.log(mcap) if mcap and mcap > 0 else math.nan,
            float(abs(last[2]) < 1) if last[2] else math.nan, float(1 - wealth[-1] / max(1.0, wealth.max())),
            float(np.prod(1 + rets[-3:]) - 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/raw/wrds/snapshot_population_2015_2023'
    manifest = json.loads((source / 'manifest.json').read_text())
    for item in manifest['files']:
        h = hashlib.sha256()
        with (source / item['file']).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        if h.hexdigest() != item['sha256']:
            raise ValueError('Snapshot mismatch')
    rows, sales = {}, {}
    with (source / 'financials.csv').open(newline='') as stream:
        for original in csv.DictReader(stream):
            r = {k.lower(): v for k, v in original.items()}
            if r['upd'] not in ('2.0000', '3.0000') or not r['datadate']:
                continue
            point = date.fromisoformat(r['pointdate']); age = (point - date.fromisoformat(r['datadate'])).days
            if not 0 <= age <= FRESHNESS or point + timedelta(days=1 + HORIZON + MATURITY) > DATA_END:
                continue
            f = financial_features(r)
            value = lambda k: float(r[k]) if r[k] not in ('', None) else math.nan
            sales[(r['gvkey'], r['pointdate'])] = value('sale')
            rows[(r['gvkey'], r['pointdate'])] = list(f.values()) + [age, math.log1p(max(value('at'), 0)) if math.isfinite(value('at')) else math.nan,
                                                                     math.log1p(max(value('sale'), 0)) if math.isfinite(value('sale')) else math.nan]
    present = defaultdict(set)
    for g, p in rows:
        present[p].add(g)
    universe = {g for g, _ in rows}

    # Identifier bridges and events.
    ciks = {r['cik'].lstrip('0'): r['gvkey'] for r in verified_rows('comp_company_ids') if r['gvkey'] in universe and r['cik']}
    links = defaultdict(list)
    for r in event_rows('crsp_ccm_link'):
        if r['gvkey'] in universe:
            links[r['gvkey']].append((permno_key(r['permno']), date.fromisoformat(r['linkdt']), date.fromisoformat(r['linkenddt']) if r['linkenddt'] else date(2099, 1, 1)))
    permno_gvkey = defaultdict(list)
    for g, ls in links.items():
        for permno, start, end in ls:
            permno_gvkey[permno].append((g, start, end))
    monthly = defaultdict(list)
    for r in event_rows('crsp_msf'):
        if permno_key(r['permno']) in permno_gvkey:
            num = lambda k: float(r[k]) if r[k] not in ('', None) else None
            monthly[permno_key(r['permno'])].append((date.fromisoformat(r['date']), num('ret'), num('prc'), num('shrout')))
    for s in monthly.values():
        s.sort()
    bankruptcies, distress = defaultdict(set), defaultdict(set)
    for r in verified_rows('aa_bankruptcy'):
        g = ciks.get(r['cik'].lstrip('0'))
        d = r['bank_begin_date'] or r['file_date']
        if g and d and not r['subsid_name']:
            bankruptcies[g].add(date.fromisoformat(d)); distress[g].add(date.fromisoformat(d))
    for r in event_rows('crsp_delist'):
        code = int(float(r['dlstcd'])); d = date.fromisoformat(r['dlstdt'])
        for g, start, end in permno_gvkey.get(permno_key(r['permno']), []):
            if start <= d <= end + timedelta(days=365):
                if code == 574:
                    bankruptcies[g].add(d)
                if code == 574 or (550 <= code <= 599 and code not in DISTRESS_EXCLUDED_CODES):
                    distress[g].add(d)
    litigation = defaultdict(list)
    for r in verified_rows('aa_litigation_parties'):
        g = ciks.get(r['cik'].lstrip('0'))
        if g and r['create_date'] and r['defendant'] == '1.0':
            litigation[g].append((date.fromisoformat(r['create_date']), r['legal_case_key'],
                                  date.fromisoformat(r['case_end_date']) if r['case_end_date'] else None))
    late = defaultdict(list)
    for r in event_rows('aa_late_filings'):
        g = ciks.get(r['cik'].lstrip('0'))
        if g and r['file_date']:
            late[g].append((date.fromisoformat(r['file_date']), '10-K' in (r['form_fkey'] or '') or r['ten_k_trans_report'] in ('1', '1.0', 'True')))
    restate = defaultdict(list)
    for r in event_rows('aa_restatements'):
        g = ciks.get(r['cik'].lstrip('0'))
        if g and r['file_date']:
            change = float(r['change_cum_net_income']) if r['change_cum_net_income'] else 0.
            restate[g].append((date.fromisoformat(r['file_date']), change < 0))
    by_g, _ = entity_rows()

    # Node features.
    keys = sorted(rows, key=lambda k: (k[1], k[0])); X = np.full((len(keys), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i, (g, p) in enumerate(keys):
        point = date.fromisoformat(p); t = point + timedelta(days=1)
        permno = next((pm for pm, s, e in links.get(g, []) if s <= point <= e), None)
        market = market_features(monthly[permno], point) if permno in monthly else [math.nan] * len(MARKET_NAMES)
        lit = [c for c in litigation.get(g, []) if c[0] <= point]
        lt = [x for x in late.get(g, []) if point - timedelta(days=365) < x[0] <= point]
        rs = [x for x in restate.get(g, []) if point - timedelta(days=365) < x[0] <= point]
        events = [math.log1p(len({c[1] for c in lit if c[0] > point - timedelta(days=365)})),
                  math.log1p(len({c[1] for c in lit if c[2] is None or c[2] > point})),
                  math.log1p(len(lt)), float(any(x[1] for x in lt)), math.log1p(len(rs)), float(any(x[1] for x in rs)),
                  float(any(d <= point - timedelta(days=LOOKBACK) for d in bankruptcies.get(g, [])))]
        st = state_at(by_g[g], datetime.combine(t, datetime.min.time())) if g in by_g else None
        rating = [float(bool(st and st['live'])), float(st['ordinal']) if st and st['ordinal'] is not None else math.nan,
                  st['downgraded'] if st else 0., st['negative_outlook'] if st else 0.]
        X[i] = rows[(g, p)] + market + events + rating

    # Labels.
    labels = np.full((len(keys), 2), -1, dtype=np.int8); supervised = np.zeros(len(keys), dtype=bool)
    recent_distress = np.zeros(len(keys), dtype=np.int8)  # realized event visible at t, used only as a neighbor signal
    for i, (g, p) in enumerate(keys):
        t = date.fromisoformat(p) + timedelta(days=1)
        recent_distress[i] = int(any(t - timedelta(days=365) <= d < t for d in distress.get(g, [])))
        if any(t - timedelta(days=LOOKBACK) <= d < t for d in bankruptcies.get(g, [])):
            continue
        supervised[i] = True
        labels[i, 0] = int(any(t <= d < t + timedelta(days=HORIZON) for d in bankruptcies.get(g, [])))
        labels[i, 1] = int(any(t <= d < t + timedelta(days=HORIZON) for d in distress.get(g, [])))

    # Relations per quarter.
    index = {k: i for i, k in enumerate(keys)}
    supply_rows, _ = read_verified(ROOT / 'data/raw/wrds/supply_chain_history_2015_2023', 'history.csv')
    link_bdx = {r['companyid']: r['gvkey'] for r in verified_rows('boardex_link') if r['preferred'] == '1.0' and r['gvkey'] in universe}
    roles = [r for r in verified_rows('boardex_employment') if r['companyid'] in link_bdx]
    cusips = {r['cusip'][:8]: r['gvkey'] for r in verified_rows('comp_security_ids') if r['gvkey'] in universe and r['cusip']}
    stakes = defaultdict(lambda: defaultdict(set))
    for r in verified_rows('tr13f_block_stakes'):
        g = cusips.get(r['cusip'])
        if g and r['fdate'] == r['rdate'] and float(r['pct']) <= 1.0:
            stakes[r['rdate']][r['mgrno']].add(g)
    edges = {'supply': [], 'people': [], 'investor': []}; stats = {}; supply_weight = []
    for p in sorted(present):
        point = date.fromisoformat(p); t = point + timedelta(days=1); gs = present[p]
        sup = visible_supply_edges(supply_rows, gs, point, 365)
        by_person = defaultdict(set)
        for r in roles:
            g = link_bdx[r['companyid']]
            if g in gs and role_active(r, t):
                by_person[r['directorid']].add(g)
        people = {tuple(sorted(e)) for v in by_person.values() if len(v) > 1 for e in combinations(sorted(v), 2)}
        visible = [d for d in stakes if date.fromisoformat(d) + timedelta(days=FILING_LAG_DAYS) <= t and date.fromisoformat(d) >= t - timedelta(days=365)]
        investors = set()
        if visible:
            for m, held in stakes[max(visible)].items():
                held = held & gs
                if len(held) < PASSIVE_THRESHOLD:
                    investors.update(tuple(sorted(e)) for e in combinations(sorted(held), 2))
        edges['supply'] += [(index[(u, p)], index[(v, p)]) for u, v in sup]
        # Customer dependence of the supplier: latest visible disclosed sales to that customer / supplier's own sales
        # (both in millions, Compustat segment convention). Unknown or non-positive values stay NaN.
        latest, pair_set = {}, set(sup)
        for r in supply_rows:
            d = date.fromisoformat(r['srcdate'])
            if d + timedelta(days=365) <= point and r['salecs'] and (r['gvkey'], r['cgvkey']) in pair_set and (latest.get((r['gvkey'], r['cgvkey'])) or (date.min,))[0] < d:
                latest[(r['gvkey'], r['cgvkey'])] = (d, float(r['salecs']))
        for u, v in sup:
            own_sales = sales.get((u, p), math.nan); disclosed = latest.get((u, v), (None, math.nan))[1]
            supply_weight.append(min(disclosed / own_sales, 1.0) if own_sales > 0 and disclosed >= 0 else math.nan)
        edges['people'] += [(index[(u, p)], index[(v, p)]) for u, v in people]
        edges['investor'] += [(index[(u, p)], index[(v, p)]) for u, v in investors]
        stats[p] = {'nodes': len(gs), 'supply': len(sup), 'people': len(people), 'investor': len(investors)}
    np.savez_compressed(OUT / 'panel.npz', X=X, labels=labels, supervised=supervised, recent_distress=recent_distress,
                        supply_weight=np.array(supply_weight, dtype=np.float32), gvkey=np.array([k[0] for k in keys]), pointdate=np.array([k[1] for k in keys]),
                        **{f'edges_{k}': np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in edges.items()})
    digest = hashlib.sha256((OUT / 'panel.npz').read_bytes()).hexdigest()
    by_year = defaultdict(lambda: [0, 0, 0, 0])
    for i, (g, p) in enumerate(keys):
        if supervised[i]:
            y = by_year[p[:4]]; y[0] += 1; y[1] += int(labels[i, 0]); y[2] += int(labels[i, 1])
    report = {'rules': __doc__.strip(), 'feature_names': FEATURE_NAMES, 'nodes': len(keys), 'supervised': int(supervised.sum()),
              'bankrupt_firms': len(bankruptcies), 'distress_firms': len(distress), 'snapshots': stats,
              'by_year_rows_bankruptcy_distress': {k: v[:3] for k, v in sorted(by_year.items())},
              'market_coverage': float(np.isfinite(X[:, FEATURE_NAMES.index('ret_12m')]).mean()),
              'panel_sha256': digest, 'event_manifest_sha256': hashlib.sha256((EVENTS / 'manifest.json').read_bytes()).hexdigest()}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'experiments/distress_panel_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('nodes', 'supervised', 'bankrupt_firms', 'distress_firms', 'market_coverage', 'by_year_rows_bankruptcy_distress')}, indent=1))


if __name__ == '__main__':
    main()
