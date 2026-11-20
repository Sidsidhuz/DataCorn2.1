from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import time
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook
from v2.auth import create_user, authenticate, password_hash, verify_password
from v2.config import Settings
from v2.data import demo_data, fingerprint, load_bytes
from v2.errors import ProductError, NotFound, JobCancelled
from v2.jobs import JobRunner
from v2.planner import propose
from v2.service import WorkspaceService
from v2.storage import Store
from v2.training import train
from v2.artifacts import serialize_model, deserialize_model
from v2.modeling import explain_scenario
from v2.reporting import excel_report


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def service(store):
    return WorkspaceService(store, Settings(mode='local', data_dir=store.root))


def test_modes_fail_closed(monkeypatch):
    monkeypatch.delenv('AUTOINSIGHT_MODE', raising=False)
    assert Settings.from_env().mode == 'demo'
    assert not Settings.from_env().uploads_allowed
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'unexpected')
    with pytest.raises(ValueError):
        Settings.from_env()


def test_demo_rejects_uploads(store):
    demo = WorkspaceService(store, Settings(mode='demo', data_dir=store.root))
    with pytest.raises(ProductError, match='disabled'):
        demo.import_dataset('demo:abc', b'a,b\n1,2', 'data.csv')


def test_owner_isolation_and_roundtrip(store, service):
    identifier = service.demo('alice')
    frame, metadata = Store(store.root).dataset('alice', identifier)
    assert fingerprint(frame) == metadata['fingerprint'] == fingerprint(demo_data())
    with pytest.raises(NotFound):
        store.dataset('bob', identifier)
    with pytest.raises(NotFound):
        store.delete_dataset('bob', identifier)
    assert not store.list_datasets('bob')


def test_concurrent_import_is_idempotent(store, service):
    with ThreadPoolExecutor(max_workers=4) as executor:
        identifiers = list(executor.map(lambda _: service.demo('alice'), range(4)))
    assert len(set(identifiers)) == 1
    assert len(store.list_datasets('alice')) == 1


def test_saved_investigation_resumes_and_exports(service, store):
    identifier = service.demo('alice')
    plan = propose('Does premium support increase churn?', demo_data())
    run_id, initial = service.start_investigation('alice', identifier, plan)
    assert initial['challenge'] is None
    with pytest.raises(ProductError, match='Complete'):
        service.export_package('alice', run_id)
    evidence = service.review_investigation('alice', run_id)
    assert evidence['challenge']['reversal']
    fresh = WorkspaceService(Store(store.root), service.settings)
    assert fresh.export_package('alice', run_id)['run_id'] == run_id
    with pytest.raises(NotFound):
        fresh.export_package('bob', run_id)


def test_delete_cascades_and_backup_is_consistent(service, store, tmp_path):
    identifier = service.demo('alice')
    run_id, _ = service.investigate('alice', identifier, propose('Compare support and churn', demo_data()))
    backup = store.backup(tmp_path / 'backup.sqlite3')
    assert backup.exists() and backup.stat().st_size > 0
    store.delete_dataset('alice', identifier)
    with pytest.raises(NotFound):
        store.run('alice', run_id)


def test_passwords_hashed_and_accounts_rate_limited(store):
    user_id = create_user(store, 'analyst', 'a-strong-test-password')
    assert user_id and authenticate(store, 'analyst', 'a-strong-test-password')['id'] == user_id
    with store.connect() as db:
        encoded = db.execute('SELECT password_hash FROM users').fetchone()[0]
    assert 'a-strong-test-password' not in encoded
    assert verify_password('a-strong-test-password', encoded)
    assert not verify_password('incorrect', encoded)
    for _ in range(5):
        with pytest.raises(ProductError):
            authenticate(store, 'analyst', 'incorrect', now=100)
    with pytest.raises(ProductError) as exc:
        authenticate(store, 'analyst', 'a-strong-test-password', now=101)
    assert exc.value.code == 'rate_limited'
    assert authenticate(store, 'analyst', 'a-strong-test-password', now=161)['id'] == user_id


def test_excel_duplicate_and_compressed_workbook_limits():
    book = Workbook(); sheet = book.active; sheet.append(['a', 'a'])
    for i in range(12):
        sheet.append([i, i+1])
    stream = BytesIO(); book.save(stream)
    with pytest.raises(ValueError, match='duplicate'):
        load_bytes(stream.getvalue(), 'duplicate.xlsx')
    with pytest.raises(ValueError, match='two columns'):
        load_bytes(('a\n' + '\n'.join(str(i) for i in range(15))).encode(), 'one.csv')


def test_group_split_is_disjoint_and_time_split_is_ordered():
    rng = np.random.default_rng(4)
    size = 180
    frame = pd.DataFrame({'customer': np.repeat(np.arange(30), 6), 'x': rng.normal(size=size),
                          'timestamp': pd.date_range('2024-01-01', periods=size), 'y': rng.normal(size=size)})
    grouped = train(frame, 'y', 'regression', ['x'], split_strategy='group', split_column='customer', algorithms=['Ridge regression'])
    assert grouped['split']['group_overlap'] == 0
    ordered = train(frame, 'y', 'regression', ['x'], split_strategy='temporal', split_column='timestamp', algorithms=['Ridge regression'])
    assert ordered['split']['training_ends'] < ordered['split']['holdout_starts']


def test_cancellation_is_cooperative_and_does_not_publish():
    def cancel(progress, message):
        if progress > .05:
            raise JobCancelled()
    with pytest.raises(JobCancelled):
        train(demo_data(), 'churn', 'classification', progress=cancel)


def test_safe_model_roundtrip_and_integrity():
    bundle = train(demo_data(), 'churn', 'classification', algorithms=['Logistic regression'])
    artifact, digest, metadata = serialize_model(bundle)
    job = {'status': 'completed', 'artifact': artifact, 'artifact_sha256': digest, 'metadata': metadata,
           'background': Store.frame_bytes(bundle['background']), 'example': Store.frame_bytes(bundle['example'])}
    restored = deserialize_model(job)
    original = explain_scenario(bundle, bundle['example'].iloc[0].to_dict())
    actual = explain_scenario(restored, restored['example'].iloc[0].to_dict())
    assert actual['probability'] == pytest.approx(original['probability'])
    with pytest.raises(ProductError, match='integrity'):
        deserialize_model({**job, 'artifact': artifact + b'changed'})


def test_background_job_idempotency_and_owner_boundary(service, store):
    dataset_id = service.demo('alice')
    runner = JobRunner(store)
    config = {'target': 'churn', 'task': 'classification', 'features': ['support_plan', 'customer_segment', 'tenure_months'], 'algorithms': ['Logistic regression']}
    try:
        identifier = runner.submit('alice', dataset_id, config)
        assert runner.submit('alice', dataset_id, config) == identifier
        runner.futures[identifier].result(timeout=30)
        job = store.job('alice', identifier)
        assert job['status'] == 'completed', job['message']
        assert runner.load('alice', identifier)['winner']
        assert runner.submit('alice', dataset_id, config) == identifier
        with pytest.raises(NotFound):
            runner.load('bob', identifier)
        with pytest.raises(NotFound):
            runner.cancel('bob', identifier)
    finally:
        runner.shutdown()


def test_stale_worker_is_reported(store, service):
    dataset = service.demo('alice')
    identifier, _ = store.create_job('alice', dataset, {'target': 'churn'})
    future_cutoff = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    store.recover_stale_jobs(future_cutoff)
    assert store.job('alice', identifier)['status'] == 'interrupted'


def test_practical_threshold_is_not_ignored(service):
    dataset = service.demo('alice')
    plan = {**propose('Compare support and churn', demo_data()), 'effect_threshold': .5, 'alpha': .01}
    _, evidence = service.investigate('alice', dataset, plan)
    check = next(c for c in evidence['challenge']['checks'] if c['check'] == 'Practical importance')
    assert check['status'] == 'caution'
    assert 'alpha=0.01' in evidence['challenge']['checks'][0]['detail']

def test_valid_uploads_roundtrip_and_deduplicate(service, store):
    df = demo_data()
    csv = df.to_csv(index=False).encode()
    first = service.import_dataset('alice', csv, 'customers.csv')
    second = service.import_dataset('alice', csv, 'renamed.csv')
    assert first == second
    restored, _ = store.dataset('alice', first)
    assert len(restored) == 600 and list(restored.columns) == list(df.columns)
    book = Workbook(); sheet = book.active
    sheet.append(['number', 'segment'])
    for i in range(20):
        sheet.append([i, 'a' if i % 2 else 'b'])
    raw = BytesIO(); book.save(raw)
    excel_id = service.import_dataset('alice', raw.getvalue(), 'values.xlsx')
    excel, _ = store.dataset('alice', excel_id)
    assert len(excel) == 20 and excel['number'].iloc[-1] == 19


def test_archive_entry_limit_is_checked_before_excel_parser():
    import zipfile
    raw = BytesIO()
    with zipfile.ZipFile(raw, 'w', zipfile.ZIP_DEFLATED) as archive:
        for index in range(1001):
            archive.writestr(str(index), '')
    with pytest.raises(ValueError, match='expands'):
        load_bytes(raw.getvalue(), 'oversized.xlsx')


def test_artifact_types_are_never_automatically_trusted(monkeypatch):
    import v2.artifacts as artifacts
    bundle = train(demo_data(), 'churn', 'classification', algorithms=['Logistic regression'])
    monkeypatch.setattr(artifacts.sio, 'get_untrusted_types', lambda **kwargs: ['unreviewed.module.Type'])
    with pytest.raises(ProductError, match='unsupported'):
        serialize_model(bundle)
