import pandas as pd
import pytest

from v2.data import demo_data
from v2.planner import propose
from v2.training import train
from v2.modeling import explain_scenario


@pytest.mark.parametrize('question,outcome,predictor', [
    ('Compare revenue by region', 'revenue', 'region'),
    ('Compare sales by region', 'sales', 'region'),
    ('Compare churn by region', 'churn', 'region'),
])
def test_explicit_outcome_overrides_other_conventional_targets(question, outcome, predictor):
    df = pd.DataFrame({'churn': [0, 1] * 10, 'region': ['east', 'west'] * 10,
                       'revenue': list(range(20)), 'sales': list(range(20, 40))})
    plan = propose(question, df)
    assert plan['outcome'] == outcome
    assert plan['predictor'] == predictor


def test_column_tokens_work_with_spaces_and_punctuation():
    df = pd.DataFrame({'churn': [0, 1] * 10, 'Monthly Spend': range(20), 'region': ['a', 'b'] * 10})
    assert propose('Compare monthly spend and churn', df)['predictor'] == 'Monthly Spend'


@pytest.mark.parametrize('columns', [[], ['only']])
def test_insufficient_schema_has_actionable_error(columns):
    with pytest.raises(ValueError, match='two distinct columns'):
        propose('Compare the data', pd.DataFrame(columns=columns))


def test_invalid_class_is_not_silently_clamped():
    model = train(demo_data(), 'churn', 'classification', algorithms=['Logistic regression'])
    values = model['example'].iloc[0].to_dict()
    for index in (-1, len(model['classes']), 1.2, True):
        with pytest.raises(ValueError, match='valid output class'):
            explain_scenario(model, values, index)


def test_missing_model_serializer_does_not_crash_investigations(monkeypatch):
    import v2.artifacts as artifacts
    from v2.service import WorkspaceService
    monkeypatch.setattr(artifacts, 'sio', None)
    assert WorkspaceService is not None
    with pytest.raises(ValueError, match='requires skops'):
        artifacts.serialize_model({})
    with pytest.raises(ValueError, match='requires skops'):
        artifacts.deserialize_model({})
