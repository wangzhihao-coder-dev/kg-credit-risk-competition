"""Currency-scale invariant financial ratios; missing inputs stay missing."""
import math


def financial_features(row):
    def value(key):
        try:
            x=float(row.get(key,''))
            return x if math.isfinite(x) else math.nan
        except (TypeError,ValueError):return math.nan
    def ratio(numerator,denominator):
        return numerator/denominator if math.isfinite(denominator) and denominator>0 else math.nan
    assets=value('at');sales=value('sale')
    return {'liabilities_assets':ratio(value('lt'),assets),
            'debt_assets':ratio(value('dlc')+value('dltt'),assets),
            'cash_assets':ratio(value('che'),assets),
            'current_assets_assets':ratio(value('act'),assets),
            'current_liabilities_assets':ratio(value('lct'),assets),
            'sales_assets':ratio(sales,assets),
            'ebit_assets':ratio(value('ebit'),assets),
            'operating_cashflow_assets':ratio(value('oancf'),assets),
            'current_ratio':ratio(value('act'),value('lct')),
            'interest_coverage':ratio(value('ebit'),value('xint')),
            'ebit_margin':ratio(value('ebit'),sales)}


def rating_features(symbol,age_days):
    ordered='AAA AA+ AA AA- A+ A A- BBB+ BBB BBB- BB+ BB BB- B+ B B- CCC+ CCC CCC- CC C'.split()
    if symbol not in ordered or float(age_days)<0:raise ValueError('Invalid observed live rating')
    return [float(ordered.index(symbol)),math.log1p(float(age_days))]


def entity_partition(gvkey):
    import hashlib
    bucket=int(hashlib.sha256(str(gvkey).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<60 else ('validation' if bucket<80 else 'test')
