"""Exercise the real import/service/model/report path with uploaded file bytes.

These tests do not simulate browser clicks or claim HTTP/browser verification.
"""
import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from v2.config import Settings
from v2.data import demo_data
from v2.errors import NotFound
from v2.modeling import explain_scenario
from v2.planner import propose
from v2.reporting import json_report, pdf_report, excel_report
from v2.service import WorkspaceService
from v2.storage import Store
from v2.training import train


@pytest.mark.parametrize('extension', ['csv', 'xlsx'])
def test_uploaded_customer_file_full_workflow(tmp_path, extension):
    original = demo_data()
    sample = tmp_path / ('uploaded_customers.' + extension)
    if extension == 'csv':
        original.to_csv(sample, index=False)
    else:
        original.to_excel(sample, index=False)
    settings = Settings(mode='local', data_dir=tmp_path / 'workspace')
    store = Store(settings.data_dir)
    service = WorkspaceService(store, settings)
    identifier = service.import_dataset('test-analyst', sample.read_bytes(), sample.name)
    assert service.import_dataset('test-analyst', sample.read_bytes(), sample.name) == identifier
    df, meta = store.dataset('test-analyst', identifier)
    assert len(df) == 600 and list(df) == list(original)
    assert service.diagnose('test-analyst', identifier)['quality']['rows'] == 600
    plan = propose('Compare support plan and churn', df)
    run_id, initial = service.start_investigation('test-analyst', identifier, plan)
    assert initial['challenge'] is None
    assert abs(initial['finding']['effect']) == pytest.approx(.2)
    with pytest.raises(ValueError, match='Challenge'):
        service.export_package('test-analyst', run_id)
    reviewed = service.review_investigation('test-analyst', run_id)
    assert reviewed['challenge']['reversal']
    model = train(df, 'churn', 'classification')
    assert len(model['leaderboard']) == 6
    model['dataset_sha256'] = meta['fingerprint']
    values = model['example'].iloc[0].to_dict()
    scenario = explain_scenario(model, values)
    assert 0 <= scenario['probability'] <= 1
    assert scenario['additivity_error'] < 1e-5
    altered = dict(values)
    altered['monthly_spend'] = float(df['monthly_spend'].median()) + 10
    changed = explain_scenario(model, altered)
    assert changed['additivity_error'] < 1e-5
    # Reopen saved investigation through a completely new service instance.
    reopened = WorkspaceService(Store(settings.data_dir), settings)
    assert reopened.store.run('test-analyst', run_id)['evidence']['challenge']['reversal']
    package = reopened.export_package('test-analyst', run_id, model)
    document = json.loads(json_report(package))
    assert document['run_id'] == run_id
    assert document['finding']['dataset_sha256'] == meta['fingerprint']
    assert document['model']['winner'] == model['winner']
    pdf = pdf_report(package)
    assert pdf.startswith(b'%PDF') and pdf.rstrip().endswith(b'%%EOF')
    workbook = load_workbook(BytesIO(excel_report(package)))
    assert 'Evidence' in workbook.sheetnames
    # Preserve local sample and generated exports as test evidence.
    (tmp_path / 'evidence.json').write_bytes(json_report(package))
    (tmp_path / 'evidence.pdf').write_bytes(pdf)
    (tmp_path / 'evidence.xlsx').write_bytes(excel_report(package))
    with pytest.raises(NotFound):
        reopened.store.dataset('different-user', identifier)
    backup = reopened.store.backup(tmp_path / 'backup.sqlite3')
    assert backup.stat().st_size > 0
    reopened.store.delete_dataset('test-analyst', identifier)
    with pytest.raises(NotFound):
        reopened.store.run('test-analyst', run_id)


def test_uploaded_regression_file_and_invalid_input(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=180)
    frame = pd.DataFrame({'spend': x, 'region': np.where(x > 0, 'east', 'west'),
                          'revenue': 4*x + rng.normal(0, .15, len(x))})
    sample = tmp_path / 'revenue.csv'
    frame.to_csv(sample, index=False)
    store = Store(tmp_path / 'workspace')
    service = WorkspaceService(store, Settings(mode='local', data_dir=store.root))
    identifier = service.import_dataset('test-analyst', sample.read_bytes(), sample.name)
    df, _ = store.dataset('test-analyst', identifier)
    plan = propose('Compare spend and revenue', df)
    assert plan['method'] == 'spearman'
    _, evidence = service.investigate('test-analyst', identifier, plan)
    assert evidence['finding']['effect'] > .9
    model = train(df, 'revenue', 'regression')
    assert model['holdout']['r2'] > .9
    values = model['example'].iloc[0].to_dict()
    assert explain_scenario(model, values)['additivity_error'] < 1e-5
    with pytest.raises(ValueError, match='finite numeric'):
        explain_scenario(model, {**values, 'spend': 'invalid'})
    with pytest.raises(ValueError):
        service.import_dataset('test-analyst', b'a,a\n1,2', 'bad.csv')
