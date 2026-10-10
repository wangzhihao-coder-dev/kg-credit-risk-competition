"""Run on WRDS Jupyter (~/kg_export). Server-side filtered exports for the related-party knowledge graph.

Every SQL statement below is also written verbatim to out/sql/<name>.sql and hashed in out/manifest.json.
Universe filters are deliberately broad (all BoardEx-linked companies, all >=5% 13F stakes, all litigated
companies); the final restriction to the 11,117 North-American graph GVKEYs happens locally.
"""
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import wrds
from sqlalchemy import text

QUERIES = {
    # People links: board/executive roles overlapping 2014-2023 at every BoardEx company that WRDS links to CRSP/Compustat.
    'boardex_employment': """
select e.companyid, e.directorid, e.datestartrole, e.dateendrole, e.datestartroleflag, e.dateendroleflag,
       e.rowtype, e.brdposition, e.ned, e.rolename
from boardex_na.na_wrds_dir_profile_emp e
where e.companyid in (select distinct companyid from wrdsapps_link_crsp_comp_bdx.bdxcrspcomplink)
  and (e.dateendrole is null or e.dateendrole >= date '2014-01-01')
  and (e.datestartrole is null or e.datestartrole <= date '2023-12-31')
""",
    'boardex_link': """
select permco, gvkey, companyid, score, preferred, duplicate
from wrdsapps_link_crsp_comp_bdx.bdxcrspcomplink
""",
    # Investor links: institutional positions of at least 5% of shares outstanding, report dates 2014-2023.
    # shrout1 is in millions, shrout2 in thousands (TR s34 documentation); prefer shrout1 when positive.
    'tr13f_block_stakes': """
select s.rdate, s.fdate, s.mgrno, s.mgrname, s.typecode, s.cusip, s.shares, s.shrout1, s.shrout2,
       s.shares / (case when s.shrout1 > 0 then s.shrout1 * 1e6 else s.shrout2 * 1e3 end) as pct
from tr_13f.s34 s
where s.rdate between date '2014-01-01' and date '2023-12-31'
  and (case when s.shrout1 > 0 then s.shrout1 * 1e6 else s.shrout2 * 1e3 end) > 0
  and s.shares >= 0.05 * (case when s.shrout1 > 0 then s.shrout1 * 1e6 else s.shrout2 * 1e3 end)
""",
    # Litigation events: one row per company party per case (company_fkey = SEC CIK).
    'aa_litigation_parties': """
select p.legal_party_key, p.legal_case_key, p.company_fkey as cik, p.defendant, p.plaintiff, p.is_lead,
       p.is_debtor, p.is_creditor, p.third_party, p.rel_defendant, p.is_nom_defendant, p.been_terminated,
       p.terminated_date, p.case_start_date, p.case_end_date, p.exp_start_date, p.exp_end_date,
       p.create_date, p.change_date, p.claim_dollars, p.settlement_amt, p.settlement_dollars, p.poten_exp_dollars
from audit_corp_legal.feed14_company_legal_party_feed p
where p.company_fkey is not null
  and coalesce(p.case_start_date, p.create_date) between date '2010-01-01' and date '2023-12-31'
""",
    'aa_litigation_cases': """
select c.legal_case_key, c.case_start_date, c.case_end_date, c.exp_start_date, c.exp_end_date, c.outcome,
       c.create_date, c.change_date, c.claim_dollars, c.settlement_dollars, c.court_type_code
from audit_corp_legal.feed13_legal_case_feed c
where coalesce(c.case_start_date, c.create_date) between date '2010-01-01' and date '2023-12-31'
""",
    'aa_case_to_category': "select * from audit_corp_legal.f13_lit_legal_case_to_category",
    'aa_category': "select * from audit_corp_legal.f13_lit_legal_category",
    # Bankruptcy notifications (company_fkey = CIK); file_date is the disclosure date.
    'aa_bankruptcy': """
select b.bank_key, b.company_fkey as cik, b.subsid_name, b.bankruptcy_type, b.bank_begin_date, b.bank_end_date,
       b.file_date, b.form_fkey, b.court_type_code
from audit_corp_legal.feed21_bankruptcy_notification b
""",
    # Identifier bridges for local GVKEY matching (no names).
    'comp_company_ids': "select gvkey, cik, fic, loc, sic, gind from comp.company",
    'comp_security_ids': "select gvkey, iid, cusip, tic, excntry, dldtei from comp.security",
}


def main():
    db = wrds.Connection(wrds_username=os.environ['WRDS_USERNAME'])
    os.makedirs('out/sql', exist_ok=True)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'wrds_username': os.environ['WRDS_USERNAME'], 'files': []}
    for name, sql in QUERIES.items():
        sql_path = f'out/sql/{name}.sql'
        open(sql_path, 'w').write(sql.strip() + '\n')
        path, rows, started = f'out/{name}.csv.gz', 0, time.time()
        with db.engine.connect() as conn, gzip.open(path, 'wt', newline='') as stream:
            for k, chunk in enumerate(pd.read_sql_query(text(sql), conn, chunksize=250000)):
                chunk.to_csv(stream, index=False, header=(k == 0)); rows += len(chunk)
        digest = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
        manifest['files'].append({'name': name, 'file': os.path.basename(path), 'rows': rows, 'bytes': os.path.getsize(path),
                                  'sha256': digest(path), 'sql_file': f'sql/{name}.sql', 'sql_sha256': digest(sql_path),
                                  'seconds': round(time.time() - started, 1)})
        json.dump(manifest, open('out/manifest.json', 'w'), indent=2)
        print(name, rows, 'rows', round(time.time() - started, 1), 's', flush=True)
    print('EXPORT DONE')


if __name__ == '__main__':
    main()
