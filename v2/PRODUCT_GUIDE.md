# DataCorn 2.1 — Product foundation

This release turns the initial evidence demo into a persistent, bounded
analyst workspace. Ollama integration is deliberately deferred. Current
question planning is explicitly rule-based; the interface makes no LLM calls.

## Start locally

Use Python 3.13 and run from the repository root:

```powershell
python -m venv .venv-v2
.venv-v2\Scripts\Activate.ps1
python -m pip install -r v2/requirements.txt
python -m v2.launch --mode local
```

The local launcher binds to 127.0.0.1. Open its printed address. In local mode,
one analyst workspace is shared by sessions on this computer. Do not expose
this mode to a network or deploy it behind a public proxy.

## Product experience

| Page | Purpose |
|---|---|
| Overview | Dataset summary, quick actions, and starting point |
| Investigate | Question, editable plan, significance/business thresholds, finding, statistical challenge |
| Data health | Structure, missingness, duplicates, and leakage review |
| Models | Explicit task/features/models, random/group/time validation, background progress, cancellation, saved models, scenario SHAP |
| Evidence | Persistent PDF/JSON/Excel downloads with computed evidence |
| History | Reopen investigations; remove datasets and their associated work |

An investigation is saved after its initial test. Challenge My Finding runs
the independent numerical review and updates that saved investigation.
Reports require the challenge to finish. Changing form controls never
silently changes previously saved results.

Training runs in bounded worker threads. You can navigate while it runs.
Progress identifies real validation folds. Cancellation is cooperative and
takes effect between fits; an individual estimator fit cannot be killed
mid-operation. No partial artifact is published.

Repeating the same dataset/configuration opens the existing completed model
instead of silently retraining against the same holdout. Deliberately
changing features, algorithms, or split settings creates a new job. Repeated
tuning still creates selection bias and is disclosed in model limitations.

## What is persisted

The application-owned SQLite database contains:

- Dataset metadata and compressed Parquet data.
- Confirmed investigation evidence and numerical challenges.
- Training configuration, lifecycle, progress and failure status.
- Complete application-trained model artifacts, reference rows and provenance.
- Private account password hashes and temporary failed-login counters.

Models use skops, not imported pickle files. Only default-supported types
and fixed, reviewed NumPy dtype / scikit-learn tree state types are allowed. Unknown types are
rejected; artifact contents never expand the allowlist. Saved models have a
SHA-256 integrity check and require the recorded scikit-learn version.

The default directory is `.autoinsight/` under the repository. Override it
with `AUTOINSIGHT_DATA_DIR`. Preserve this directory across application
restarts/deployments. A container without a persistent volume loses its
workspace when replaced, even though browser refreshes are durable while
that container remains alive.

## Deployment modes

| Mode | Who can enter | Uploads | Appropriate use |
|---|---|---|---|
| demo (default) | Any visitor, isolated session workspace | Disabled | Public synthetic demonstration |
| local | Analyst on loopback | Enabled | Personal laptop workflow |
| private | Admin-provisioned login | Enabled | Controlled private deployment behind HTTPS |

There is no public self-registration. Accounts are provisioned by the
administrator. Private mode refuses to expose the workspace until an
account exists. Passwords use salted PBKDF2-SHA256 with 600,000 iterations;
sign-in attempts are throttled, sessions expire after eight hours, and
sign-out clears session state. Every repository lookup/mutation is scoped
to the authenticated owner.

To create an account interactively:

```powershell
python -m v2.manage create-user analyst
python -m v2.launch --mode private
```

The first command reads passwords through a hidden terminal prompt. An
administrator can alternatively set AUTOINSIGHT_ADMIN_USER and
AUTOINSIGHT_ADMIN_PASSWORD as deployment secrets for first-account bootstrap.
Remove those bootstrap secrets after the account exists; changing them does
not silently change an existing password.

Private mode binds to all interfaces for a deployment proxy. Run it only
behind HTTPS with a protected, persistent data volume. This is managed
password authentication, not enterprise SSO/MFA or a compliance certification.

## Data and compute boundaries

- CSV/XLSX: 10 MB upload, 20,000 rows, 80 columns, at least 12 rows/two columns.
- Headers: unique after trimming, non-empty, at most 128 characters.
- Excel: first worksheet, cached formula values, archive expansion bounded
  to 50 MB/1,000 entries; external workbook links are not retained.
- Training: 40–5,000 labeled rows; three validation folds; bounded candidates.
- Two global training slots; one active job per owner; 180-second cooperative
  budget. A single fit can run longer before the next checkpoint notices.
- Twenty datasets per owner; 100 investigations and 30 training jobs per
  dataset. These limits are enforced by the repository.
- Random validation assumes independent rows. Group validation separates
  entities in both CV and holdout. Temporal validation uses expanding time
  blocks and keeps equal timestamps on one side of each boundary.

## Statistical interpretation

The analysis remains exploratory. Welch and Spearman report 95% intervals;
the selected alpha is separately disclosed. Categorical effect intervals
are not implemented. Holm correction covers displayed stratum comparisons,
not every question a user tries. A descriptive reversal warns of confounding
and does not establish a causal benefit. What-if/SHAP describes model
behaviour, not a real-world intervention.

## Deployment and release gates

The non-root Docker image and Render configuration default to demo mode.
Never add real uploads to a public demonstration by changing only a UI label.
See RUNBOOK.md for backups, private deployment, recovery and readiness.
See release/VALIDATION.md for observed checks and remaining verification.

The current product scope is a small, single-process analyst deployment.
It is not a horizontally distributed SaaS. Enterprise identity, centralized
job workers, cloud object storage, external API contracts, billing, and
autonomous AI execution are intentionally outside this release.
