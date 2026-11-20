import json
import pytest
import httpx
from v2.data import demo_data
from v2.planner import propose


def test_llm_contract_sends_schema_not_rows(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-key-not-real')
    seen={}
    def fake_post(url,**kwargs):
        seen.update(kwargs)
        return httpx.Response(200,request=httpx.Request('POST',url),json={'choices':[{'message':{'content':json.dumps({
            'predictor':'support_plan','outcome':'churn','confounder':'customer_segment','reason':'Compare observed churn rates.'})}}]})
    monkeypatch.setattr(httpx,'post',fake_post)
    result=propose('Does support plan relate to churn?',demo_data(),True)
    assert result['source']=='LLM · schema only'
    payload=json.loads(seen['json']['messages'][1]['content'])
    assert set(payload)=={'question','schema'}
    assert all(set(c)=={'column','dtype','unique'} for c in payload['schema'])
    assert 'C0001' not in json.dumps(payload)


def test_missing_key_does_not_fake_ai(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='not configured'):
        propose('Explain churn',demo_data(),True)


def test_api_failure_is_reported(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-key-not-real')
    def failed(url,**kwargs):
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx,'post',failed)
    with pytest.raises(httpx.ConnectError):
        propose('Explain churn',demo_data(),True)
