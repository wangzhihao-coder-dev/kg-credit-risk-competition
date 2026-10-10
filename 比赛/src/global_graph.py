"""Recorded supply graph under explicit delayed cumulative-disclosure assumption."""
from datetime import date, timedelta


def visible_supply_edges(rows,nodes,pointdate,lag_days=365,retention_days=None):
    if lag_days<0:raise ValueError('Negative visibility lag')
    if retention_days is not None and retention_days<0:raise ValueError('Negative retention')
    return sorted({(r['gvkey'],r['cgvkey']) for r in rows
                   if r['gvkey']!=r['cgvkey'] and r['gvkey'] in nodes and r['cgvkey'] in nodes
                   and date.fromisoformat(r['srcdate'])+timedelta(days=lag_days)<=pointdate
                   and (retention_days is None or (pointdate-date.fromisoformat(r['srcdate'])).days<=lag_days+retention_days)})


def directed_messages(supplier,customer,direction='both'):
    if direction not in ('both','forward','reverse'):raise ValueError('Unknown direction')
    messages=[]
    if direction in ('both','forward'):messages.append((supplier,customer,0))
    if direction in ('both','reverse'):messages.append((customer,supplier,1))
    return messages
