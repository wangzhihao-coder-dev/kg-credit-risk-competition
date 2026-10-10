"""Run on WRDS Jupyter (~/kg_export). Distress-event and market exports for all North-American public firms.

Same conventions as 02_export.py: each SQL is written verbatim to out2/sql/<name>.sql and hashed in out2/manifest.json.
"""
import gzip, hashlib, json, os, time
from datetime import datetime, timezone

import pandas as pd
import wrds
from sqlalchemy import text

QUERIES = {
    # Late-filing notices (NT 10-K / NT 10-Q): the closest public-firm analogue of "overdue"; file_date is the disclosure date.
    'aa_late_filings': """
select n.nt_notify_key, n.company_fkey as cik, n.file_date, n.pe_date_num, n.ten_k_trans_report, n.form_fkey,
       n.part4_3_check
from audit.feed20_nt n
where n.file_date between date '2012-01-01' and date '2024-12-31'
""",
    # Restatement disclosures; file_date is the disclosure date, change_cum_net_income the cumulative income effect.
    'aa_restatements': """
select r.restatement_filing_key, r.restatement_notification_fkey, r.company_fkey as cik, r.file_date, r.form_fkey,
       r.first_announcement, r.change_cum_net_income, r.change_cum_stockholder_equity
from audit.f39_restatement_filings r
where r.file_date between date '2012-01-01' and date '2024-12-31'
""",
    # Going-concern flags (audit.f74_feed_support74_going_concern) are not licensed for this account
    # (permission denied on schema audit_acct_os), so they are not exported.
    # CRSP/Compustat link history (primary links only).
    'crsp_ccm_link': """
select l.gvkey, l.lpermno as permno, l.linktype, l.linkprim, l.linkdt, l.linkenddt
from crsp.ccmxpf_lnkhist l
where l.linktype in ('LU', 'LC') and l.linkprim in ('P', 'C')
""",
    # Monthly stock file for linked securities, 2014-2024.
    'crsp_msf': """
select m.permno, m.date, m.ret, m.prc, m.shrout
from crsp.msf m
where m.date between date '2014-01-01' and date '2024-12-31'
  and m.permno in (select distinct lpermno from crsp.ccmxpf_lnkhist where linktype in ('LU', 'LC') and linkprim in ('P', 'C'))
""",
    # Delistings with codes (500-599 performance-related, 574 bankruptcy).
    'crsp_delist': """
select d.permno, d.dlstdt, d.dlstcd, d.dlret
from crsp.msedelist d
where d.dlstdt between date '2014-01-01' and date '2024-12-31'
""",
}


def main():
    db = wrds.Connection(wrds_username=os.environ['WRDS_USERNAME'])
    os.makedirs('out2/sql', exist_ok=True)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'wrds_username': os.environ['WRDS_USERNAME'], 'files': []}
    digest = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    for name, sql in QUERIES.items():
        sql_path = f'out2/sql/{name}.sql'
        open(sql_path, 'w').write(sql.strip() + '\n')
        path, rows, started = f'out2/{name}.csv.gz', 0, time.time()
        with db.engine.connect() as conn, gzip.open(path, 'wt', newline='') as stream:
            for k, chunk in enumerate(pd.read_sql_query(text(sql), conn, chunksize=250000)):
                chunk.to_csv(stream, index=False, header=(k == 0)); rows += len(chunk)
        manifest['files'].append({'name': name, 'file': os.path.basename(path), 'rows': rows, 'bytes': os.path.getsize(path),
                                  'sha256': digest(path), 'sql_file': f'sql/{name}.sql', 'sql_sha256': digest(sql_path),
                                  'seconds': round(time.time() - started, 1)})
        json.dump(manifest, open('out2/manifest.json', 'w'), indent=2)
        print(name, rows, 'rows', round(time.time() - started, 1), 's', flush=True)
    print('EXPORT2 DONE')


if __name__ == '__main__':
    main()
