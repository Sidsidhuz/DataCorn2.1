"""Bounded background workers. Streamlit APIs are never used from worker threads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
import threading
import time
from .artifacts import serialize_model, deserialize_model
from .errors import JobCancelled, ProductError
from .training import train
from .contracts import TrainingRequest

logger = logging.getLogger('autoinsight.jobs')


class JobRunner:
    def __init__(self, store, max_workers=2, timeout=180):
        self.store = store
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='autoinsight-training')
        self.futures = {}
        self.lock = threading.Lock()
        self.recover()

    def recover(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self.timeout+60)).isoformat()
        self.store.recover_stale_jobs(cutoff)

    def submit(self, owner, dataset_id, config):
        self.recover()
        frame, _ = self.store.dataset(owner, dataset_id)
        config = TrainingRequest.parse(config, frame.columns).to_dict()
        identifier, created = self.store.create_job(owner, dataset_id, config, self.max_workers)
        if created:
            with self.lock:
                self.futures = {k: v for k, v in self.futures.items() if not v.done()}
                self.futures[identifier] = self.executor.submit(self._execute, owner, identifier)
        return identifier

    def _execute(self, owner, identifier):
        started = time.monotonic()
        try:
            job = self.store.job(owner, identifier)
            df, meta = self.store.dataset(owner, job['dataset_id'])
            self.store.update_job(owner, identifier, status='running', message='Preparing raw data', progress=.01)
            def progress(value, message):
                current = self.store.job(owner, identifier)
                if current['cancel_requested']:
                    raise JobCancelled()
                if time.monotonic() - started > self.timeout:
                    raise ProductError('Training exceeded the time budget. Use fewer rows or simpler models.', 'timeout')
                self.store.update_job(owner, identifier, progress=float(value), message=message)
            config = job['config']
            bundle = train(df, config['target'], config['task'], config['features'],
                           split_strategy=config.get('split_strategy', 'random'),
                           split_column=config.get('split_column'), algorithms=config.get('algorithms'),
                           progress=progress, timeout=self.timeout)
            bundle['dataset_sha256'] = meta['fingerprint']
            progress(.98, 'Saving the validated model artifact')
            artifact, digest, metadata = serialize_model(bundle)
            # One transaction publishes metadata and the complete model together.
            if self.store.job(owner, identifier)['cancel_requested']:
                raise JobCancelled()
            published = self.store.publish_job(owner, identifier, metadata=metadata, artifact=artifact, digest=digest,
                background=self.store.frame_bytes(bundle['background']), example=self.store.frame_bytes(bundle['example']))
            if not published:
                raise JobCancelled()
            logger.info('training_completed job_id=%s duration_seconds=%.2f', identifier, time.monotonic()-started)
        except JobCancelled:
            self.store.update_job(owner, identifier, status='cancelled', message='Cancelled; no partial model published')
        except ValueError as exc:
            self.store.update_job(owner, identifier, status='failed', message=str(exc)[:600])
            logger.info('training_validation_failed job_id=%s', identifier)
        except Exception:
            # Do not include dataset rows, credentials, or raw exception messages in logs.
            self.store.update_job(owner, identifier, status='failed', message='Training could not finish. Retry with a smaller dataset; contact the administrator with this job ID.')
            logger.error('training_internal_failure job_id=%s', identifier)

    def cancel(self, owner, identifier):
        self.store.cancel_job(owner, identifier)

    def load(self, owner, identifier):
        return deserialize_model(self.store.job(owner, identifier))

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
