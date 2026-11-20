from pathlib import Path
import time
from streamlit.testing.v1 import AppTest
from v2.auth import create_user
from v2.storage import Store

APP = str(Path(__file__).parents[1] / 'app.py')


def by_label(elements, label):
    return next(item for item in elements if item.label == label)


def page(app, label):
    return by_label(app.button, label).click().run()


def test_redesigned_navigation_and_dataset_switch(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOINSIGHT_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'local')
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and not app.error
    for destination in ['Investigate', 'Data health', 'Models', 'Evidence', 'History', 'Overview']:
        page(app, destination)
        assert not app.exception and not app.error, destination
    page(app, 'Investigate')
    by_label(app.button, 'Propose investigation').click().run()
    by_label(app.button, 'Confirm plan & run analysis').click().run()
    assert app.session_state['active_run_id']
    from v2.service import WorkspaceService
    from v2.config import Settings
    service = WorkspaceService(Store(tmp_path), Settings.from_env())
    raw = ('region,revenue\n' + '\n'.join(f"{'east' if i % 2 else 'west'},{i + 10}" for i in range(30))).encode()
    identifier = service.import_dataset('local', raw, 'second_dataset.csv')
    app.run()
    by_label(app.selectbox, 'Active dataset').set_value(identifier).run()
    assert app.session_state['dataset_id'] == identifier
    assert 'active_run_id' not in app.session_state
    assert 'evidence' not in app.session_state
    assert not app.exception and not app.error


def test_product_workflow_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOINSIGHT_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'local')
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception and not app.error
    page(app, 'Investigate')
    by_label(app.button, 'Propose investigation').click().run()
    by_label(app.button, 'Confirm plan & run analysis').click().run()
    assert app.session_state['evidence']['challenge'] is None
    by_label(app.button, 'Challenge My Finding').click().run()
    assert app.session_state['evidence']['challenge']['reversal']
    page(app, 'Models')
    by_label(app.checkbox, 'I have reviewed feature availability and chosen a split that matches how predictions will be used.').check().run()
    by_label(app.button, 'Train & compare models').click().run()
    identifier = app.session_state['active_job_id']
    store = Store(tmp_path)
    deadline = time.monotonic() + 30
    while store.job('local', identifier)['status'] in ('queued', 'running') and time.monotonic() < deadline:
        time.sleep(.05)
    assert store.job('local', identifier)['status'] == 'completed'
    app.run()
    assert not app.exception and not app.error
    by_label(app.button, 'Predict scenario & show SHAP').click().run()
    assert app.session_state['scenario']['contributions']
    page(app, 'Evidence')
    by_label(app.button, 'Prepare evidence downloads').click().run()
    assert app.session_state['exports']['pdf'].startswith(b'%PDF')
    assert not app.exception and not app.error
    # An unrelated rerun must keep prepared downloads available.
    app.run()
    assert app.session_state['exports']['pdf'].startswith(b'%PDF')
    fresh = AppTest.from_file(APP, default_timeout=30).run()
    page(fresh, 'History')
    by_label(fresh.button, 'Open').click().run()
    assert fresh.session_state['evidence']['challenge']['reversal']
    page(fresh, 'Models')
    by_label(fresh.button, 'Open model').click().run()
    fresh.run()
    assert fresh.session_state['model']['winner']
    assert not fresh.exception and not fresh.error


def test_public_demo_does_not_offer_uploads(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOINSIGHT_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'demo')
    app = AppTest.from_file(APP).run()
    source = by_label(app.selectbox, 'Dataset source')
    assert 'Upload dataset' not in source.options
    assert app.session_state['_owner'].startswith('demo:')
    assert not app.exception and not app.error


def test_edited_analysis_hides_old_finding_until_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOINSIGHT_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'local')
    app = AppTest.from_file(APP, default_timeout=30).run()
    page(app, 'Investigate')
    by_label(app.button, 'Propose investigation').click().run()
    by_label(app.button, 'Confirm plan & run analysis').click().run()
    original_id = app.session_state['active_run_id']
    by_label(app.selectbox, 'Significance threshold').set_value(.01).run()
    assert not app.exception
    assert any('settings changed' in item.value for item in app.warning)
    assert not any(button.label == 'Challenge My Finding' for button in app.button)
    assert app.session_state['active_run_id'] == original_id
    by_label(app.button, 'Confirm plan & run analysis').click().run()
    assert app.session_state['active_run_id'] != original_id
    assert app.session_state['evidence']['finding']['plan']['alpha'] == .01
    assert any(button.label == 'Challenge My Finding' for button in app.button)


def test_private_mode_requires_account_and_login(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOINSIGHT_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('AUTOINSIGHT_MODE', 'private')
    app = AppTest.from_file(APP).run()
    assert any('setup is incomplete' in item.value for item in app.error)
    assert not app.metric
    create_user(Store(tmp_path), 'analyst', 'strong-private-test-password')
    app.run()
    by_label(app.text_input, 'Username').set_value('analyst')
    by_label(app.text_input, 'Password').set_value('strong-private-test-password')
    by_label(app.button, 'Sign in').click().run()
    assert app.session_state['identity']['username'] == 'analyst'
    assert not app.exception and not app.error
    by_label(app.button, 'Sign out').click().run()
    assert any(item.label == 'Sign in' for item in app.button)
    assert not app.metric
