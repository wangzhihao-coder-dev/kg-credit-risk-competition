"""Fixed review-budget metrics and paired component-level uncertainty."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def evaluate(y,p,k=50):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if len(np.unique(y))!=2 or len(y)!=len(p) or not np.isfinite(p).all():
        raise ValueError('Evaluation requires finite paired predictions and both classes')
    p=np.clip(p,1e-7,1-1e-7);k=min(k,len(y))
    top=np.argsort(-p,kind='stable')[:k]
    ece=0.
    for left in np.linspace(0,1,11)[:-1]:
        mask=(p>=left)&(p<left+.1)
        if mask.any():ece+=float(mask.mean()*abs(y[mask].mean()-p[mask].mean()))
    return {'n':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()),
            'ap':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),
            'ece10':ece,'k':k,'recall_at_k':float(y[top].sum()/y.sum()),
            'precision_at_k':float(y[top].mean())}


def paired_cluster_bootstrap(y,p_new,p_base,groups,repeats=300,seed=20261008):
    y=np.asarray(y);p_new=np.asarray(p_new);p_base=np.asarray(p_base);groups=np.asarray(groups)
    unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(seed);differences=[];auc_differences=[]
    for _ in range(repeats):
        ids=np.concatenate([members[i] for i in rng.integers(0,len(members),size=len(members))])
        if len(np.unique(y[ids]))<2:continue
        differences.append(average_precision_score(y[ids],p_new[ids])-average_precision_score(y[ids],p_base[ids]))
        auc_differences.append(roc_auc_score(y[ids],p_new[ids])-roc_auc_score(y[ids],p_base[ids]))  # same draws, no extra randomness
    if not differences:raise ValueError('No two-class bootstrap draws')
    return {'ap_difference':float(average_precision_score(y,p_new)-average_precision_score(y,p_base)),
            'ap_difference_ci95':[float(x) for x in np.quantile(differences,[.025,.975])],
            'auc_difference':float(roc_auc_score(y,p_new)-roc_auc_score(y,p_base)),
            'auc_difference_ci95':[float(x) for x in np.quantile(auc_differences,[.025,.975])],
            'clusters':len(unique),'largest_cluster':int(max(len(x) for x in members)),
            'successful_draws':len(differences),'method':'paired graph-component resampling; retrospective sample only'}
