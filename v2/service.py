"""Application use-cases shared by the UI and a future agent/API adapter."""
import logging
import uuid
from .config import Settings
from .data import demo_data, load_bytes, validate, fingerprint
from .errors import ProductError
from .quality import profile
from .statistics import analyze
from .challenge import challenge
from .reporting import evidence_package, json_report

logger = logging.getLogger('autoinsight.service')


class WorkspaceService:
    def __init__(self, store, settings):
        self.store = store
        self.settings = settings

    def import_dataset(self, owner, raw, filename):
        if not self.settings.uploads_allowed or owner.startswith('demo:'):
            raise ProductError('Uploads are disabled in the public demonstration.', 'uploads_disabled')
        df = load_bytes(raw, filename)
        return self.store.save_dataset(owner, filename, df, fingerprint(df), self.settings.max_datasets)

    def demo(self, owner):
        df = demo_data()
        return self.store.save_dataset(owner, 'Customer retention · synthetic demo', df, fingerprint(df), self.settings.max_datasets)

    def investigate(self, owner, dataset_id, plan):
        identifier, _ = self.start_investigation(owner, dataset_id, plan)
        return identifier, self.review_investigation(owner, identifier)

    def start_investigation(self, owner, dataset_id, plan):
        df, meta = self.store.dataset(owner, dataset_id)
        finding = analyze(df, plan)
        quality = profile(df, finding['plan']['outcome'])
        evidence = {'finding': finding, 'quality': quality, 'challenge': None}
        identifier = self.store.save_run(owner, dataset_id, finding['plan']['question'], evidence)
        logger.info('investigation_completed run_id=%s', identifier)
        return identifier, evidence

    def review_investigation(self, owner, identifier):
        run = self.store.run(owner, identifier)
        if run['evidence']['challenge'] is not None:
            return run['evidence']
        df, _ = self.store.dataset(owner, run['dataset_id'])
        finding = run['evidence']['finding']
        if fingerprint(df) != finding['dataset_sha256']:
            raise ProductError('Dataset content changed. Run a fresh investigation.', 'stale_evidence')
        quality = profile(df, finding['plan']['outcome'])
        review = challenge(df, finding, quality, finding['plan'].get('confounder'))
        evidence = evidence_package(finding, review, quality)
        self.store.update_run(owner, identifier, evidence)
        return evidence

    def export_package(self, owner, run_id, model=None):
        run = self.store.run(owner, run_id)
        package = run['evidence']
        if package['challenge'] is None:
            raise ProductError('Complete Challenge My Finding before exporting this investigation.')
        if model is not None:
            if model.get('dataset_sha256') != package['finding']['dataset_sha256']:
                raise ProductError('This model belongs to a different dataset. Select a matching model.')
            package = evidence_package(package['finding'], package['challenge'], package['quality'], model)
        package['run_id'] = run_id
        return package

    def diagnose(self, owner, dataset_id):
        frame, meta = self.store.dataset(owner, dataset_id)
        return {'dataset': meta, 'quality': profile(frame)}
