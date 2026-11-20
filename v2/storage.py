"""SQLite repository. Every data lookup and mutation is scoped to an owner."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import uuid
import pandas as pd
from .errors import NotFound, ProductError


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class Store:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'workspace.sqlite3'
        with self.connect() as db:
            if db.execute('PRAGMA user_version').fetchone()[0] not in (0, 1):
                raise ProductError('This database needs a compatible application version. Restore the matching release.', 'schema_version')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    username TEXT PRIMARY KEY, count INTEGER NOT NULL, last_attempt REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, frame BLOB NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(owner, fingerprint));
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, dataset_id TEXT NOT NULL,
                    title TEXT NOT NULL, evidence TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, dataset_id TEXT NOT NULL,
                    config TEXT NOT NULL, config_hash TEXT NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0, message TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0, metadata TEXT,
                    artifact BLOB, artifact_sha256 TEXT, background BLOB, example BLOB,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS dataset_owner ON datasets(owner, created_at);
                CREATE INDEX IF NOT EXISTS run_owner ON runs(owner, dataset_id);
                CREATE INDEX IF NOT EXISTS job_owner ON jobs(owner, dataset_id, config_hash);
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA busy_timeout=10000')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def frame_bytes(df):
        out = BytesIO()
        df.to_parquet(out, index=False)
        return out.getvalue()

    def save_dataset(self, owner, name, df, fingerprint, limit=20):
        if not owner or not name.strip() or len(name) > 180:
            raise ProductError('Use a dataset name between 1 and 180 characters.')
        blob = self.frame_bytes(df)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT id FROM datasets WHERE owner=? AND fingerprint=?', (owner, fingerprint)).fetchone()
            if existing:
                return existing['id']
            if db.execute('SELECT count(*) FROM datasets WHERE owner=?', (owner,)).fetchone()[0] >= limit:
                raise ProductError('Your workspace is full. Delete a saved dataset before adding another.', 'quota')
            identifier = uuid.uuid4().hex
            db.execute('INSERT INTO datasets VALUES (?,?,?,?,?,?)',
                       (identifier, owner, name.strip(), fingerprint, blob, utcnow()))
        return identifier

    def list_datasets(self, owner):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,name,fingerprint,created_at FROM datasets WHERE owner=? ORDER BY created_at DESC', (owner,))]

    def dataset(self, owner, identifier):
        with self.connect() as db:
            row = db.execute('SELECT * FROM datasets WHERE id=? AND owner=?', (identifier, owner)).fetchone()
        if row is None:
            raise NotFound()
        return pd.read_parquet(BytesIO(row['frame'])), {k: row[k] for k in ('id', 'name', 'fingerprint', 'created_at')}

    def delete_dataset(self, owner, identifier):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM jobs WHERE owner=? AND dataset_id=? AND status IN ('queued','running')", (owner, identifier)).fetchone():
                raise ProductError('Cancel or finish active training before deleting this dataset.')
            if db.execute('DELETE FROM datasets WHERE owner=? AND id=?', (owner, identifier)).rowcount != 1:
                raise NotFound()

    def save_run(self, owner, dataset_id, title, evidence):
        self.dataset(owner, dataset_id)
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT count(*) FROM runs WHERE owner=? AND dataset_id=?', (owner, dataset_id)).fetchone()[0] >= 100:
                raise ProductError('This dataset has reached its 100-investigation limit. Export evidence and use a new dataset workspace.', 'quota')
            db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?)', (identifier, owner, dataset_id, title[:250], encode(evidence), utcnow()))
        return identifier

    def list_runs(self, owner, dataset_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,title,created_at FROM runs WHERE owner=? AND dataset_id=? ORDER BY created_at DESC LIMIT 100', (owner, dataset_id))]

    def run(self, owner, identifier):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE owner=? AND id=?', (owner, identifier)).fetchone()
        if row is None:
            raise NotFound()
        return {**dict(row), 'evidence': json.loads(row['evidence'])}

    def update_run(self, owner, identifier, evidence):
        with self.connect() as db:
            if db.execute('UPDATE runs SET evidence=? WHERE owner=? AND id=?', (encode(evidence), owner, identifier)).rowcount != 1:
                raise NotFound()

    def create_job(self, owner, dataset_id, config, max_workers=2):
        self.dataset(owner, dataset_id)
        config_text = encode(config)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        now = utcnow()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT id FROM jobs WHERE owner=? AND dataset_id=? AND config_hash=? AND status IN ('queued','running','completed') ORDER BY created_at DESC LIMIT 1", (owner, dataset_id, config_hash)).fetchone()
            if row:
                return row['id'], False
            if db.execute('SELECT count(*) FROM jobs WHERE owner=? AND dataset_id=?', (owner, dataset_id)).fetchone()[0] >= 30:
                raise ProductError('This dataset has reached its 30-training-job limit. Reopen an existing model or remove the dataset.', 'quota')
            if db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= max_workers:
                raise ProductError('All training slots are busy. Please try again after an active job finishes.', 'busy')
            if db.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone():
                raise ProductError('You already have an active training job.', 'busy')
            identifier = uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,owner,dataset_id,config,config_hash,status,message,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)',
                       (identifier, owner, dataset_id, config_text, config_hash, 'queued', 'Waiting for a training worker', now, now))
        return identifier, True

    def job(self, owner, identifier):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE owner=? AND id=?', (owner, identifier)).fetchone()
        if row is None:
            raise NotFound()
        return {**dict(row), 'config': json.loads(row['config']), 'metadata': json.loads(row['metadata']) if row['metadata'] else None}

    def list_jobs(self, owner, dataset_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,status,message,created_at FROM jobs WHERE owner=? AND dataset_id=? ORDER BY created_at DESC LIMIT 30', (owner, dataset_id))]

    def update_job(self, owner, identifier, **changes):
        allowed = {'status', 'progress', 'message', 'metadata', 'artifact', 'artifact_sha256', 'background', 'example', 'cancel_requested'}
        if not changes or set(changes) - allowed:
            raise ProductError('Invalid job update.')
        if 'metadata' in changes:
            changes['metadata'] = encode(changes['metadata'])
        changes['updated_at'] = utcnow()
        with self.connect() as db:
            query = 'UPDATE jobs SET ' + ','.join(f'{k}=?' for k in changes) + ' WHERE owner=? AND id=?'
            if db.execute(query, (*changes.values(), owner, identifier)).rowcount != 1:
                raise NotFound()

    def recover_stale_jobs(self, cutoff):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='interrupted',message='The worker stopped before publishing a model. Start a new job.',updated_at=? WHERE status IN ('queued','running') AND updated_at<?", (utcnow(), cutoff))

    def publish_job(self, owner, identifier, *, metadata, artifact, digest, background, example):
        with self.connect() as db:
            return db.execute("UPDATE jobs SET status='completed',progress=1,message='Model ready',metadata=?,artifact=?,artifact_sha256=?,background=?,example=?,updated_at=? WHERE owner=? AND id=? AND cancel_requested=0 AND status='running'",
                (encode(metadata), artifact, digest, background, example, utcnow(), owner, identifier)).rowcount == 1

    def cancel_job(self, owner, identifier):
        self.job(owner, identifier)
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel_requested=1,message='Cancellation requested; finishing the current fit',updated_at=? WHERE owner=? AND id=? AND status IN ('queued','running')", (utcnow(), owner, identifier))

    def backup(self, destination):
        target = Path(destination).resolve()
        if target == self.path:
            raise ProductError('Backup destination must differ from the live database.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, sqlite3.connect(target) as output:
            source.backup(output)
        return target
