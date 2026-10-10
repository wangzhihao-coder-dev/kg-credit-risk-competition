"""Point-in-time people/investor relations, litigation events and layered credit-deterioration labels.

Inputs: data/raw/wrds/related_party_2014_2023 (verified against its manifest), the 550-day panel and
the ratings history. Every relation and event is restricted to what is visible at the observation
timestamp (quarter-end + 1 day):
- people link: both companies have the same BoardEx individual in an active role at t
  (end 9999 = current; end 9000 = ended on an unknown date, treated as active for at most 3 years after start);
- investor link: both companies held at >= 5% by the same 13F manager in the latest report date
  with rdate + 45 days <= t and rdate >= t - 365; managers with >= 50 such positions in that quarter
  (index-scale holders) are excluded in the primary graph because they connect almost every firm;
- litigation: Audit Analytics record create_date <= t (case start dates are often back-filled later);
- bankruptcy notification: file_date <= t.
"""
import csv, gzip, hashlib, json, math
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import combinations

from audit_rating_history import ROOT, read_verified
from rating_state import DEFAULT_SYMBOLS, LIVE_SYMBOLS, rating_stamp

RAW = ROOT / 'data/raw/wrds/related_party_2014_2023'
OUT = ROOT / 'data/processed/global_rating_550'
ORDER = 'AAA AA+ AA AA- A+ A A- BBB+ BBB BBB- BB+ BB BB- B+ B B- CCC+ CCC CCC- CC C'.split()
UNKNOWN_END_YEARS = 3
FILING_LAG_DAYS = 45
PASSIVE_THRESHOLD = 50


def verified_rows(name):
    manifest = json.loads((RAW / 'manifest.json').read_text())
    item = next(f for f in manifest['files'] if f['name'] == name)
    path = RAW / item['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Source mismatch: ' + name)
    return csv.DictReader(gzip.open(path, 'rt', newline=''))


def role_active(row, t):
    start = date.fromisoformat(row['datestartrole']) if row['datestartrole'] else None
    end = row['dateendrole']
    if start is None or start > t:
        return False
    if not end or end.startswith('9999'):
        return True
    if end.startswith('9000'):
        return t <= start + timedelta(days=365 * UNKNOWN_END_YEARS)
    return t <= date.fromisoformat(end)


def layered_target(rows, observation, start_symbol, cutoff, horizon=180, maturity=30):
    """Severe (>=2 notches, into CCC+ or below, or D/SD) and mild (>=1 notch or D/SD) deterioration.
    NR/unknown before the first deterioration censors the row, mirroring recorded_event_target."""
    end = observation + timedelta(days=horizon)
    if cutoff < end + timedelta(days=maturity):
        return None, None
    o0 = ORDER.index(start_symbol)
    future = sorted((rating_stamp(r), r['ratingsymbol']) for r in rows if observation < rating_stamp(r) <= end)
    severe = mild = 0
    for _, symbol in future:
        if symbol in DEFAULT_SYMBOLS:
            return 1, 1
        if symbol not in LIVE_SYMBOLS:
            return (severe or None), (mild or None)
        o = ORDER.index(symbol)
        mild = max(mild, int(o > o0))
        severe = max(severe, int(o >= o0 + 2 or (o >= 16 > o0)))
    return severe, mild


def main():
    nodes = defaultdict(set)
    with (OUT / 'nodes.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            nodes[r['pointdate']].add(r['gvkey'])
    universe = set().union(*nodes.values())
    link = {}
    for r in verified_rows('boardex_link'):
        if r['preferred'] == '1.0' and r['gvkey'] in universe:
            link[r['companyid']] = r['gvkey']
    roles = [r for r in verified_rows('boardex_employment') if r['companyid'] in link]
    ciks = {r['cik'].lstrip('0'): r['gvkey'] for r in verified_rows('comp_company_ids') if r['gvkey'] in universe and r['cik']}
    cusips = {r['cusip'][:8]: r['gvkey'] for r in verified_rows('comp_security_ids') if r['gvkey'] in universe and r['cusip']}
    stakes = defaultdict(lambda: defaultdict(set))  # rdate -> mgrno -> gvkeys
    for r in verified_rows('tr13f_block_stakes'):
        g = cusips.get(r['cusip'])
        if g and r['fdate'] == r['rdate'] and float(r['pct']) <= 1.0:
            stakes[r['rdate']][r['mgrno']].add(g)
    litigation = defaultdict(list)
    for r in verified_rows('aa_litigation_parties'):
        g = ciks.get(r['cik'].lstrip('0'))
        if g and r['create_date']:
            litigation[g].append((date.fromisoformat(r['create_date']), r['legal_case_key'], r['defendant'] == '1.0',
                                  date.fromisoformat(r['case_end_date']) if r['case_end_date'] else None))
    bankruptcy = defaultdict(list)
    for r in verified_rows('aa_bankruptcy'):
        g = ciks.get(r['cik'].lstrip('0'))
        if g and r['file_date']:
            bankruptcy[g].append(date.fromisoformat(r['file_date']))

    graph, events, stats = {}, [], {}
    for p in sorted(nodes):
        t = date.fromisoformat(p) + timedelta(days=1); present = nodes[p]
        by_person = defaultdict(set)
        for r in roles:
            g = link[r['companyid']]
            if g in present and role_active(r, t):
                by_person[r['directorid']].add(g)
        people = sorted({tuple(sorted(e)) for gs in by_person.values() if len(gs) > 1 for e in combinations(sorted(gs), 2)})
        visible = [d for d in stakes if date.fromisoformat(d) + timedelta(days=FILING_LAG_DAYS) <= t
                   and date.fromisoformat(d) >= t - timedelta(days=365)]
        investors, passive = set(), 0
        if visible:
            for m, gs in stakes[max(visible)].items():
                gs = gs & present
                if len(gs) >= PASSIVE_THRESHOLD:
                    passive += 1; continue
                investors.update(tuple(sorted(e)) for e in combinations(sorted(gs), 2))
        graph[p] = {'people': [list(e) for e in people], 'investors': [list(e) for e in sorted(investors)]}
        stats[p] = {'nodes': len(present), 'people_edges': len(people), 'investor_edges': len(investors),
                    'investor_report_date': max(visible) if visible else None, 'passive_managers_excluded': passive}
        for g in present:
            cases = [c for c in litigation.get(g, []) if c[0] <= t]
            new = {c[1] for c in cases if c[2] and c[0] > t - timedelta(days=365)}
            open_ = {c[1] for c in cases if c[2] and (c[3] is None or c[3] > t)}
            new_any = {c[1] for c in cases if c[0] > t - timedelta(days=365)}
            bank = sum(t - timedelta(days=730) < d <= t for d in bankruptcy.get(g, []))
            if new or open_ or new_any or bank:
                events.append({'gvkey': g, 'pointdate': p, 'lit_def_new_365': len(new), 'lit_def_open': len(open_),
                               'lit_any_new_365': len(new_any), 'bankruptcy_730': bank})
    (OUT / 'related_graph.json').write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    with (OUT / 'node_events.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['gvkey', 'pointdate', 'lit_def_new_365', 'lit_def_open', 'lit_any_new_365', 'bankruptcy_730'])
        writer.writeheader(); writer.writerows(events)

    history, _ = read_verified(ROOT / 'data/raw/wrds/ratings_history_2015_2023', 'history.csv')
    groups = defaultdict(list)
    for r in history:
        if (r['ratingtypecode'], r['orgdebttypecode'], r['globalornationalscaleind'], r['longtermflag']) == ('FCLONG', 'ICR', 'G', '1'):
            groups[r['entity_id']].append(r)
    cutoff = datetime(2023, 12, 31, 23, 59, 59); layered = []; counts = defaultdict(lambda: defaultdict(int))
    with (OUT / 'labels.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if r['split'] == 'purged':
                continue
            severe, mild = layered_target(groups[r['entity_id']], datetime.fromisoformat(r['snapshot_date']), r['rating_symbol'], cutoff)
            layered.append({'entity_id': r['entity_id'], 'gvkey': r['gvkey'], 'pointdate': r['pointdate'], 'split': r['split'],
                            'y_default': r['label'], 'y_severe': '' if severe is None else severe, 'y_mild': '' if mild is None else mild})
            for name, v in (('default', int(r['label'])), ('severe', severe), ('mild', mild)):
                if v is not None:
                    counts[r['split']][name + '_rows'] += 1; counts[r['split']][name + '_positive'] += v
    with (OUT / 'labels_layered.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layered[0])); writer.writeheader(); writer.writerows(layered)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'rules': __doc__.strip(), 'unknown_end_years': UNKNOWN_END_YEARS, 'filing_lag_days': FILING_LAG_DAYS,
              'passive_threshold': PASSIVE_THRESHOLD, 'linked_boardex_companies': len(link), 'roles_used': len(roles),
              'graph_gvkeys_with_cik': len(ciks), 'graph_gvkeys_with_cusip': len(set(cusips.values())),
              'snapshots': stats, 'event_rows': len(events), 'label_counts': {k: dict(v) for k, v in counts.items()},
              'outputs': [{'file': f, 'sha256': digest(OUT / f)} for f in ('related_graph.json', 'node_events.csv', 'labels_layered.csv')],
              'source_manifest_sha256': digest(RAW / 'manifest.json')}
    (OUT / 'related_graph_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'experiments/related_graph_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'snapshots': len(stats), 'mean_people_edges': sum(s['people_edges'] for s in stats.values()) / len(stats),
                      'mean_investor_edges': sum(s['investor_edges'] for s in stats.values()) / len(stats),
                      'event_rows': len(events), 'labels': report['label_counts']}, indent=1))


if __name__ == '__main__':
    main()
