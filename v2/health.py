"""Deployment readiness probe: HTTP liveness plus workspace/account readiness."""
import argparse
import json
import os
import tempfile
import urllib.request
from .config import Settings
from .storage import Store


def check(settings):
    store = Store(settings.data_dir)
    with store.connect() as db:
        schema = db.execute('PRAGMA user_version').fetchone()[0]
        accounts = db.execute('SELECT count(*) FROM users').fetchone()[0]
        db.execute('SELECT 1').fetchone()
    with tempfile.TemporaryFile(dir=settings.data_dir) as probe:
        probe.write(b'readiness'); probe.flush()
    if settings.mode == 'private' and accounts == 0:
        raise ValueError('Private mode requires a provisioned account.')
    return {'ready': True, 'mode': settings.mode, 'schema_version': schema}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8501')))
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/_stcore/health', timeout=3) as response:
            if response.status != 200:
                raise ValueError('HTTP liveness failed.')
        print(json.dumps(check(Settings.from_env())))
    except Exception as exc:
        print(json.dumps({'ready': False, 'error_type': type(exc).__name__}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
