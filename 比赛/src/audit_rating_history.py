"""Reconcile downloaded multi-year actions and expose censoring prerequisites."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read_verified(directory,filename):
    manifest=json.loads((directory/'manifest.json').read_text())
    for item in manifest['files']:
        body=(directory/item['file']).read_bytes()
        if len(body)!=item['bytes'] or hashlib.sha256(body).hexdigest()!=item['sha256']:
            raise ValueError('Source mismatch')
    with (directory/filename).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    return rows,manifest['files']


def main():
    raw=ROOT/'data/raw/wrds'
    rows,sources=read_verified(raw/'ratings_history_2015_2023','history.csv')
    probe,_=read_verified(raw/'ratings_2020_population','population.csv')
    yearly=defaultdict(list)
    for row in rows:yearly[row['ratingdate'][:4]].append(row)
    def signatures(rs):return Counter(tuple(sorted(row.items())) for row in rs)
    historic2020=signatures(yearly['2020']);standalone=signatures(probe)
    summary=[]
    for year,rs in sorted(yearly.items()):
        uk=[r for r in rs if r['countrycode']=='GBR']
        defaults=[r for r in rs if r['ratingsymbol'] in ('D','SD')]
        summary.append({'year':int(year),'rows':len(rs),'entities':len({r['entity_id'] for r in rs}),
                        'uk_rows':len(uk),'uk_entities':len({r['entity_id'] for r in uk}),
                        'default_symbol_entities':len({r['entity_id'] for r in defaults}),
                        'uk_default_symbol_entities':len({r['entity_id'] for r in defaults if r['countrycode']=='GBR'}),
                        'NR_rows':sum(r['ratingsymbol']=='NR' for r in rs)})
    events=defaultdict(list)
    for row in rows:events[row['entity_id']].append(row)
    first_observed_defaults=[]
    for rs in events.values():
        ds=[r['ratingdate'] for r in rs if r['ratingsymbol'] in ('D','SD')]
        if ds:first_observed_defaults.append(min(ds))
    report={
        'source_files':sources,'rows':len(rows),'entities':len(events),
        'nonempty_company_ids':len({r['company_id'] for r in rows if r['company_id']}),
        'uk_entities':len({r['entity_id'] for r in rows if r['countrycode']=='GBR'}),
        'yearly':summary,
        'default_symbol_entities_all_types':len(first_observed_defaults),
        'first_observed_default_year_counts':dict(sorted(Counter(d[:4] for d in first_observed_defaults).items())),
        'rating_type_counts':dict(Counter(r['ratingtypename'] for r in rows)),
        'NR_entities':len({r['entity_id'] for r in rows if r['ratingsymbol']=='NR'}),
        '2020_reconciliation':{'exact_multiset_match':historic2020==standalone,
                               'history_extra_rows':sum((historic2020-standalone).values()),
                               'standalone_extra_rows':sum((standalone-historic2020).values())},
        'training_eligible':False,
        'next_contract':'Restrict validated issuer rating types; reconstruct last observed rating per type at each observation, handle withdrawal/NR and re-entry; define coverage and first observed outcome explicitly before negative labels.',
        'boundaries':['2015 history start causes left censoring; first observed D/SD is not first-ever default',
                      'Annual action counts are not full active-issuer counts',
                      'Entity metadata/country joined from current table: historical validity unverified',
                      'NR meaning, rating type codes and public availability require dictionary checks',
                      'Graph crosswalk and historical relations not established by rating rows'],
    }
    output=ROOT/'experiments/wrds_rating_history_audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'entities':len(events),'default_symbol_entities':len(first_observed_defaults),
                      '2020_reconciliation':report['2020_reconciliation'],'yearly':summary}))


if __name__=='__main__':main()
