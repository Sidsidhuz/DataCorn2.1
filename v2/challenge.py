"""Independent numerical reviewer: evidence checks, permutation test, strata tests."""
import numpy as np
import pandas as pd
from scipy import stats
from .statistics import analyze, holm_adjust


def challenge(df, finding, quality, confounder=None):
    plan=finding['plan']
    checks=[]; limitations=list(finding['assumptions'])
    def add(name,status,detail):
        checks.append({'check':name,'status':status,'detail':detail})
    p=finding['p_value']
    alpha=plan.get('alpha',.05)
    add('Statistical support','pass' if p<alpha else 'caution',
        f'Two-sided p={p:.4g} at alpha={alpha:g}. This is exploratory, not confirmatory evidence.')
    ci=finding['ci95']
    if ci:
        add('Interval uncertainty','pass' if ci[0]*ci[1]>0 else 'caution',f'95% interval: [{ci[0]:.4f}, {ci[1]:.4f}].')
    threshold=plan.get('effect_threshold',0)
    if threshold:
        robust=bool(ci and (ci[0]>=threshold or ci[1]<=-threshold))
        add('Practical importance','pass' if robust else 'caution',
            f'User-defined minimum effect: {threshold:g}. The point estimate is {finding["effect"]:.4f}; '+
            ('the entire interval exceeds the minimum magnitude.' if robust else 'the available interval does not establish an effect beyond this threshold.'))
    add('Missing observations','caution' if finding['excluded_rows'] else 'pass',
        f"{finding['excluded_rows']} rows excluded by complete-case analysis; missingness may bias the result.")
    add('Possible leakage','caution' if quality['risks'] else 'pass',
        f"{len(quality['risks'])} columns flagged. Feature timing still requires human review.")
    if quality['duplicates']:
        add('Row independence','caution',f"{quality['duplicates']} duplicate rows found; test p-values may be optimistic.")
    if finding.get('sparse_expected') and 'Fisher' not in finding['test']:
        add('Expected cell counts','fail','Sparse contingency cells: chi-square approximation may be unreliable.')
    permutation_p=None
    if plan['method']=='welch':
        x,y=plan['predictor'],plan['outcome']
        data=df[[x,y]].dropna()
        groups=sorted(data[x].unique(),key=str)
        a,b=[data.loc[data[x]==g,y].to_numpy(dtype=float) for g in groups]
        # Studentized statistic handles unequal variance better than a raw mean permutation.
        def statistic(u,v):
            se=np.sqrt(np.var(u,ddof=1)/len(u)+np.var(v,ddof=1)/len(v))
            return (np.mean(u)-np.mean(v))/se if se>0 else 0.
        perm=stats.permutation_test((a,b),statistic,n_resamples=499,
                                   alternative='two-sided',rng=np.random.default_rng(42))
        permutation_p=float(perm.pvalue)
        add('Permutation sensitivity','pass' if (p<alpha)==(permutation_p<alpha) else 'caution',
            f'Studentized two-sided permutation p={permutation_p:.4g} (499 resamples, seed 42). Exchangeability is an additional assumption.')
    strata=[]; skipped=[]; reversal=False
    if confounder and confounder not in [plan['predictor'],plan['outcome']]:
        s=df[confounder]
        if pd.api.types.is_numeric_dtype(s) and s.nunique()>5:
            labels=pd.qcut(s,4,duplicates='drop')
        else:
            labels=s
        if labels.nunique()>8:
            skipped.append('Confounder has more than 8 levels. Choose a smaller segmentation.')
        else:
            for label in sorted(labels.dropna().unique(),key=str):
                part=df.loc[labels==label]
                try:
                    sub=analyze(part,plan)
                    # Effects must use the same method and direction.
                    if sub['plan']['method']!=plan['method']:
                        raise ValueError('Test changes within this stratum.')
                    strata.append({'stratum':str(label),'n':sub['n'],'effect':sub['effect'],
                                   'p_value':sub['p_value']})
                except ValueError as exc:
                    skipped.append(f'{label}: {exc}')
        if strata:
            for row,adjusted in zip(strata,holm_adjust([r['p_value'] for r in strata])):
                row['holm_p']=adjusted
            if plan['method'] in ('welch','spearman') and len(strata)>=2:
                reversal=all(r['effect']*finding['effect']<0 for r in strata)
            add('Alternative explanation','fail' if reversal else 'caution',
                f'{len(strata)} strata checked using {confounder}. '+
                ('Every tested stratum reverses the aggregate direction: a confounding warning. '
                 if reversal else 'Stratified comparisons do not rule out unmeasured confounding. ')+
                'Stratum p-values use Holm correction; these are exploratory tests.')
        else:
            add('Alternative explanation','caution','No valid stratum comparisons were available. '+ ' '.join(skipped))
        limitations.append(f'{int(labels.isna().sum())} rows lack a confounder value; {len(skipped)} strata could not be tested.')
    else:
        add('Alternative explanation','caution','No confounder selected. No adjusted or stratified claim is available.')
    limitations.extend(['A small p-value does not establish business importance or causation.',
                        'Repeated questions on the same data create selection bias. Confirm on fresh data.',
                        'No automatic causal identification or comprehensive leakage detection.'])
    verdict='Challenged' if any(c['status']=='fail' for c in checks) else 'Qualified' if any(c['status']=='caution' for c in checks) else 'Supported within assumptions'
    action=(f"Do not recommend changing {plan['predictor']} based on the aggregate comparison. Investigate the {confounder} mix and design a prospective test."
            if reversal else 'Treat this as a hypothesis. Review assumptions and business effect size, then validate on independent data.')
    return {'verdict':verdict,'checks':checks,'strata':strata,'skipped_strata':skipped,
            'reversal':reversal,'permutation_p':permutation_p,'limitations':limitations,'recommended_action':action}
