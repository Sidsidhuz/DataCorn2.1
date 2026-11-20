import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from v2.data import demo_data, load_bytes, fingerprint
from v2.planner import propose, validate_plan
from v2.quality import profile
from v2.statistics import analyze, holm_adjust
from v2.challenge import challenge
from v2.modeling import train, explain_scenario
from v2.reporting import evidence_package, json_report, pdf_report, excel_report


@pytest.fixture
def df():
    return demo_data()


def test_demo_is_reproducible(df):
    assert fingerprint(df)==fingerprint(demo_data())
    assert len(df)==600


def test_planner_is_honest_and_validated(df):
    p=propose('Does premium support increase customer churn?',df)
    assert p['source']=='Offline · rule-based'
    assert (p['predictor'],p['outcome'])==('support_plan','churn')
    with pytest.raises(ValueError):
        validate_plan({**p,'predictor':'not_a_column'},df)


def test_leakage_flags(df):
    risks={r['column']:r['reasons'] for r in profile(df,'churn')['risks']}
    assert 'customer_id' in risks
    assert 'Exact target copy' in risks['closed_account_flag']
    assert 'monthly_spend' not in risks


def test_welch_matches_independent_scipy(df):
    result=analyze(df,propose('Does premium support increase customer churn?',df))
    a=df.loc[df.support_plan=='Premium','churn'];b=df.loc[df.support_plan=='Standard','churn']
    reference=stats.ttest_ind(a,b,equal_var=False)
    assert result['effect']==pytest.approx(.2)
    assert result['p_value']==pytest.approx(reference.pvalue)
    assert result['ci95'][0]==pytest.approx(reference.confidence_interval().low)


def test_critic_finds_actual_reversal(df):
    finding=analyze(df,propose('Does premium support increase customer churn?',df))
    c=challenge(df,finding,profile(df,'churn'),'customer_segment')
    assert c['reversal'] and c['verdict']=='Challenged'
    assert len(c['strata'])==2
    assert all(s['effect']==pytest.approx(-.1) for s in c['strata'])
    assert 0<c['permutation_p']<=1
    assert all(s['holm_p']>=s['p_value'] for s in c['strata'])


def test_missing_values_are_disclosed(df):
    df.loc[:9,'churn']=np.nan
    result=analyze(df,propose('Does premium support increase customer churn?',df))
    assert result['excluded_rows']==10 and result['n']==590


def test_constant_and_small_groups_rejected(df):
    df['churn']=1
    with pytest.raises(ValueError):
        analyze(df,propose('Compare support and churn',df))


def test_holm_known_values():
    assert holm_adjust([.01,.04,.03])==pytest.approx([.03,.06,.06])


def test_duplicate_headers_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        load_bytes(b'a,a\n1,2\n','example.csv')


def test_spearman_and_sparse_fisher():
    df=pd.DataFrame({'x':np.arange(40),'y':np.arange(40)+np.sin(np.arange(40))})
    r=analyze(df,{'predictor':'x','outcome':'y'})
    assert r['effect']>.99 and r['ci95'] is not None
    cats=pd.DataFrame({'a':['a']*11+['b']*3,'b':['yes']*10+['no']+['yes']+['no']*2})
    r=analyze(cats,{'predictor':'a','outcome':'b'})
    assert r['test']=='Fisher exact test'


def test_training_guardrails_and_real_shap(df):
    b=train(df,'churn','classification')
    assert 'closed_account_flag' not in b['features']
    assert 'customer_id' not in b['features']
    assert b['train_rows']+b['test_rows']==600
    assert len(b['leaderboard'])==6
    assert b['winner']==b['leaderboard'][0]['model']
    s=explain_scenario(b,b['example'].iloc[0].to_dict())
    assert 0<=s['probability']<=1
    assert s['additivity_error']<1e-5
    assert s['contributions']


def test_regression_model_and_explanation():
    rng=np.random.default_rng(7)
    x=rng.normal(size=150)
    frame=pd.DataFrame({'x':x,'category':np.where(x>0,'a','b'),'sales':3*x+rng.normal(0,.1,150)})
    b=train(frame,'sales','regression')
    s=explain_scenario(b,b['example'].iloc[0].to_dict(),0)
    assert b['holdout']['r2']>.9
    assert s['additivity_error']<1e-5


def test_reports_contain_computed_evidence(df):
    f=analyze(df,propose('Does premium support increase customer churn?',df))
    c=challenge(df,f,profile(df,'churn'),'customer_segment')
    package=evidence_package(f,c,profile(df,'churn'))
    assert json.loads(json_report(package))['challenge']['reversal']
    assert pdf_report(package).startswith(b'%PDF')
    assert excel_report(package).startswith(b'PK')
