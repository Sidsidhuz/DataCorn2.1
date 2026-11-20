# Operations and release runbook

## Install and verify

Use Python 3.13. Install v2/requirements-dev.txt, then:

```powershell
python -m pytest v2/tests -q
python -m pip_audit -r v2/requirements.lock
python -m v2.launch --mode local
```

Runtime requirements.txt points to the exact production lock. Human-maintained
dependency bounds are in requirements.in. Changing the lock requires the full
tests, serialization roundtrip, dependency audit and container checks again.

## Container

```sh
docker build -f v2/Dockerfile -t autoinsight:2.1 .
docker volume create autoinsight-data
docker run --rm -p 8501:8501 -v autoinsight-data:/data autoinsight:2.1
```

This starts a synthetic public demo. The image runs as UID 10001. Bind mounts
must be writable by that user; do not solve permissions by running as root.
The image build context excludes local databases, secrets, virtualenvs,
generated reports and test caches.

For private deployment, provision an account in the mounted volume using
the management CLI, then set AUTOINSIGHT_MODE=private. Alternatively use
first-account bootstrap secrets. Put the server behind the hosting provider's
HTTPS proxy and retain /data across deployments. The included Render recipe
is demo-only; private data needs a persistent volume and a reviewed host plan.

## Health and logs

The Streamlit /_stcore/health endpoint is process liveness. The container
uses python -m v2.health to additionally check the database, writable
workspace storage, schema, and a provisioned account in private mode.
Readiness does not prove successful model training, disk capacity headroom,
or restore readiness. Smoke-test those workflows separately.

Application log records identify operation, run/job ID and duration, or an
error reference/type. They intentionally omit passwords, dataset rows and
raw stack traces. User-correctable validation failures have clear messages;
unexpected failures show a reference instead of infrastructure internals.

## Backup and recovery

```powershell
python -m v2.manage backup backups/autoinsight.sqlite3
```

The backup uses SQLite's online backup API, which includes a consistent
snapshot even when WAL is active. It contains business data and account
hashes: protect it with the same access and encryption controls as live data.
Use the deployment host/volume's encryption and access controls; the app does
not claim application-level encryption at rest.

To restore: stop the application, keep a copy of the current data directory,
place the verified backup at AUTOINSIGHT_DATA_DIR/workspace.sqlite3 in a new
directory, then start the matching application/dependency release. Never
replace a live database with an older copy while workers are running. Check
account login, dataset counts, a saved investigation, and a saved-model
prediction after restore. Do not load externally supplied model/database files.

Models require the recorded scikit-learn version. If dependencies change,
retrain or run the matching older release against a copied workspace. The
repository rejects unknown schema versions rather than silently downgrading.

## Retention and capacity

```powershell
python -m v2.manage prune-demo --days 30
```

This removes expired synthetic-demo datasets and cascades their runs/models,
leaving private/local owners untouched. Run it through the host's operations
schedule if deploying publicly. It is not automatically scheduled by this
repository. Deleted SQLite pages may be reused; disk compaction requires an
administrator's offline maintenance window. Monitor volume capacity and keep
at least enough space for a backup plus normal WAL growth.

## Training failure recovery

- Cancel cooperatively through the UI; cancellation may wait for a fit.
- Failed/cancelled jobs can be retried. Successful identical configurations
  reopen their saved artifact.
- Restarted workers do not resume half-trained estimators. Expired jobs become
  interrupted during startup, submission, or UI status polling; users explicitly start a new attempt.
- The 180-second budget is checked between fits. It is not a hard OS process
  kill. Use an external process/job system for strict compute isolation.

## Release gate before November 20

1. Full automated suite and dependency audit pass for the pinned release.
2. Linux container builds, runs non-root, and passes its liveness check.
3. On the actual host: complete investigation → challenge → model → SHAP →
   report, then restart and reopen the saved investigation/model.
4. Verify demo mode rejects uploads; private mode requires login and isolates
   two test accounts; remove test data afterward through normal controls.
5. Verify restore from a real backup in an isolated deployment.
6. Check desktop/mobile navigation and uploads; record a narrated walkthrough.
7. Add the live URL to the application only after signed-out reachability is
   verified. No current document should imply a public deployment already exists.
