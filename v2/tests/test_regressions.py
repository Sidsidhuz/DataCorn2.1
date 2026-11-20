from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from sklearn.datasets import load_wine
from v2.training import train
from v2.modeling import explain_scenario
from v2.artifacts import serialize_model, deserialize_model
from v2.storage import Store
from v2.data import demo_data
from v2.planner import propose
from v2.quality import profile
from v2.statistics import analyze
from v2.challenge import challenge
from v2.reporting import evidence_package, excel_report


def test_real_multiclass_dataset_and_shap():
    data = load_wine(as_frame=True)
    frame = data.frame
    bundle = train(frame, 'target', 'classification', algorithms=['Logistic regression'])
    explanation = explain_scenario(bundle, bundle['example'].iloc[0].to_dict(), class_index=2)
    assert bundle['holdout']['macro_f1'] > .7
    assert explanation['unit'] == 'decision score'
    assert explanation['additivity_error'] < 1e-5


def test_random_forest_artifact_is_supported():
    rng = np.random.default_rng(12)
    x = rng.uniform(-3, 3, 200)
    frame = pd.DataFrame({'x': x, 'target': x*x + rng.normal(0, .1, len(x))})
    bundle = train(frame, 'target', 'regression', algorithms=['Random forest'])
    assert bundle['winner'] == 'Random forest'
    artifact, digest, metadata = serialize_model(bundle)
    restored = deserialize_model({'status': 'completed', 'artifact': artifact, 'artifact_sha256': digest, 'metadata': metadata,
                                  'background': Store.frame_bytes(bundle['background']), 'example': Store.frame_bytes(bundle['example'])})
    explanation = explain_scenario(restored, restored['example'].iloc[0].to_dict())
    assert explanation['additivity_error'] < 1e-5


def test_bad_scenario_input_is_not_silently_imputed():
    bundle = train(demo_data(), 'churn', 'classification', algorithms=['Logistic regression'])
    values = bundle['example'].iloc[0].to_dict()
    with pytest.raises(ValueError, match='finite numeric'):
        explain_scenario(bundle, {**values, 'monthly_spend': 'invalid'})
    with pytest.raises(ValueError, match='exactly'):
        explain_scenario(bundle, {})


def test_excel_formula_text_stays_text():
    df = demo_data()
    finding = analyze(df, propose('Compare support and churn', df))
    finding['plan']['question'] = ' \t=HYPERLINK("https://example.com","click")'
    review = challenge(df, finding, profile(df, 'churn'), 'customer_segment')
    book = load_workbook(BytesIO(excel_report(evidence_package(finding, review, profile(df, 'churn')))))
    cell = book['Evidence']['B2']
    assert cell.data_type == 's' and cell.value.startswith("'")


def test_cancellation_wins_before_atomic_publication(tmp_path):
    store = Store(tmp_path); df = demo_data()
    from v2.data import fingerprint
    dataset = store.save_dataset('owner', 'demo', df, fingerprint(df))
    identifier, _ = store.create_job('owner', dataset, {})
    store.update_job('owner', identifier, status='running')
    store.cancel_job('owner', identifier)
    assert not store.publish_job('owner', identifier, metadata={}, artifact=b'ignored', digest='ignored', background=b'', example=b'')
    assert store.job('owner', identifier)['artifact'] is None
