"""Prespecified bivariate tests with effect sizes, intervals, and explicit exclusions."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from .data import fingerprint, engine_fingerprint
from .planner import validate_plan

SEED=42


def analyze(df, plan):
    plan=validate_plan(plan,df)
    x,y=plan['predictor'],plan['outcome']
    data=df[[x,y]].dropna()
    if len(data)<12:
        raise ValueError('At least 12 complete rows are required for this comparison.')
    result={'plan':plan,'n':len(data),'excluded_rows':len(df)-len(data),
            'dataset_sha256':fingerprint(df),'seed':SEED,'engine_version':'2.1.0',
            'engine_sha256':engine_fingerprint(),
            'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Exploratory association. No causal claim; independent observations assumed.'}
    if data[x].nunique()<2 or data[y].nunique()<2:
        raise ValueError('Both variables must vary in the complete rows.')
    if plan['method']=='welch':
        groups=sorted(data[x].unique(),key=str)
        a,b=[data.loc[data[x]==g,y].to_numpy(dtype=float) for g in groups]
        if min(len(a),len(b))<5:
            raise ValueError('Each group needs at least 5 complete observations.')
        test=stats.ttest_ind(a,b,equal_var=False)
        ci=test.confidence_interval(confidence_level=.95)
        if not np.isfinite(test.pvalue) or not np.isfinite(ci.low):
            raise ValueError('The test is undefined for these groups; inspect variance.')
        result.update(test='Welch two-sample t-test',effect=float(a.mean()-b.mean()),
            effect_label=f'Mean {y}: {groups[0]} minus {groups[1]}',
            statistic=float(test.statistic),p_value=float(test.pvalue),
            ci95=[float(ci.low),float(ci.high)],
            groups=[{'group':str(g),'n':len(v),'mean':float(v.mean()),'sd':float(v.std(ddof=1))}
                    for g,v in zip(groups,[a,b])],
            assumptions=['Independent rows; no repeated customers or clustered sampling.',
                         'Mean comparison; small skewed samples need extra care.',
                         'For binary outcomes the effect is a difference in observed rates.'])
    elif plan['method']=='spearman':
        test=stats.spearmanr(data[x],data[y])
        if not np.isfinite(test.statistic):
            raise ValueError('Correlation is undefined.')
        values=data[[x,y]].to_numpy()
        rng=np.random.default_rng(SEED)
        estimates=[]
        for _ in range(300):
            sample=values[rng.integers(0,len(values),len(values))]
            if len(np.unique(sample[:,0]))>1 and len(np.unique(sample[:,1]))>1:
                estimates.append(stats.spearmanr(sample[:,0],sample[:,1]).statistic)
        lo,hi=np.quantile(estimates,[.025,.975])
        result.update(test='Spearman rank correlation',effect=float(test.statistic),
            effect_label=f'Rank association: {x} with {y}',statistic=float(test.statistic),
            p_value=float(test.pvalue),ci95=[float(lo),float(hi)],
            interval_method='300 paired percentile bootstrap resamples',
            assumptions=['Independent rows.','Association can be nonlinear; it does not show causation.',
                         'Approximate p-value; ties and small samples affect calibration.'])
    else:
        table=pd.crosstab(data[x],data[y])
        if table.shape[0]>20 or table.shape[1]>20:
            raise ValueError('Categorical test is limited to 20 levels per variable. Choose another comparison.')
        chi,p,dof,expected=stats.chi2_contingency(table,correction=False)
        sparse=bool((expected<5).any())
        if table.shape==(2,2) and sparse:
            odds,p=stats.fisher_exact(table.to_numpy())
            testname='Fisher exact test'; statistic=float(odds)
        else:
            testname='Pearson chi-square independence test';statistic=float(chi)
        result.update(test=testname,statistic=statistic if np.isfinite(statistic) else None,
            effect=float(np.sqrt(chi/(len(data)*min(table.shape[0]-1,table.shape[1]-1)))),
            effect_label="Cramer's V (association strength)",p_value=float(p),ci95=None,
            sparse_expected=sparse,table=table.to_dict(),
            assumptions=['Independent rows.','Chi-square approximation needs adequate expected counts.',
                         'No confidence interval is supplied for this effect in the MVP.'])
    result['finding']=f"{result['effect_label']} = {result['effect']:.4f}; p = {result['p_value']:.4g}."
    return result


def holm_adjust(pvalues):
    """Familywise correction for the disclosed set of stratum tests."""
    order=np.argsort(pvalues)
    adjusted=np.zeros(len(pvalues))
    previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(pvalues)-rank)*pvalues[index])
        adjusted[index]=min(1.,previous)
    return adjusted.tolist()
