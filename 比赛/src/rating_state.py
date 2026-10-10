"""Reconstruct observed issuer states, without asserting public availability."""
from datetime import datetime

LIVE_SYMBOLS = set('AAA AA+ AA AA- A+ A A- BBB+ BBB BBB- BB+ BB BB- B+ B B- CCC+ CCC CCC- CC C'.split())
DEFAULT_SYMBOLS = {'D','SD'}


def rating_stamp(row):
    if not row.get('ratingtime'):
        raise ValueError('Cannot order missing rating time')
    return datetime.fromisoformat(row['ratingdate']+'T'+row['ratingtime'])


def observed_state(rows, observation):
    visible = [row for row in rows if rating_stamp(row) <= observation]
    if not visible:
        return {'status':'no_observed_history'}
    latest = max(rating_stamp(row) for row in visible)
    symbols = {row['ratingsymbol'] for row in visible if rating_stamp(row)==latest}
    if len(symbols)!=1:
        return {'status':'conflicting_latest_state'}
    symbol = next(iter(symbols))
    prior_default = any(row['ratingsymbol'] in DEFAULT_SYMBOLS for row in visible)
    status = 'prior_observed_default' if prior_default else ('active_observed' if symbol in LIVE_SYMBOLS else 'unrated_or_unknown')
    return {'status':status,'symbol':symbol,'last_rating_stamp':latest.isoformat(),
            'days_since_last_rating':(observation-latest).days}


def recorded_event_target(rows, observation, database_cutoff, horizon_days=180, maturity_days=30):
    """Retrospective recorded-rating endpoint; does not certify loan non-default."""
    from datetime import timedelta
    if horizon_days<=0 or maturity_days<0:
        raise ValueError('Invalid horizon')
    end=observation+timedelta(days=horizon_days)
    mature=end+timedelta(days=maturity_days)
    def result(status,label=None,event_time=None):
        return {'status':status,'label':label,'event_time':event_time,
                'endpoint':'recorded_issuer_D_SD','certified_loan_default':False}
    if database_cutoff<mature:return result('incomplete_window')
    state=observed_state(rows,observation)
    if state['status']!='active_observed':return result(state['status'])
    future={}
    for row in rows:
        stamp=rating_stamp(row)
        if observation<stamp<=mature:future.setdefault(stamp,set()).add(row['ratingsymbol'])
    if any(len(symbols)>1 for symbols in future.values()):return result('conflicting_future_state')
    for stamp,symbols in sorted(future.items()):
        symbol=next(iter(symbols))
        if symbol not in LIVE_SYMBOLS|DEFAULT_SYMBOLS:return result('censored',event_time=stamp.isoformat())
        if symbol in DEFAULT_SYMBOLS and stamp<=end:return result('recorded_default',1,stamp.isoformat())
    return result('recorded_event_free',0)
