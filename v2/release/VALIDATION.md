# AutoInsight 2.1 release validation

Verified on November 15, 2026. Release classification: **local product candidate;
production sign-off pending**.

## Observed checks

| Check | Observed result |
|---|---|
| Automated suite | 45 passed in 15.60 seconds |
| Environment | Windows / Python 3.13 |
| Dependency consistency | No broken requirements found |
| Vulnerability audit | 53 pinned runtime dependencies; none skipped; zero known vulnerabilities reported |
| Live local server | http://127.0.0.1:8503 |
| HTTP liveness + storage/account readiness | Successful local-mode readiness probe |
| Source whitespace check | git diff --check passed |

The dependency audit JSON is stored alongside this record. Its result is a
point-in-time advisory check, not a guarantee that every dependency is secure.

## What the tests cover

- Independent SciPy comparison, effect/interval/exclusions, actual Simpson
  reversal, permutation sensitivity, Holm adjustment, and practical thresholds.
- Real bundled Wine multiclass data and genuine SHAP additivity.
- Classification/regression pipelines; random/group/chronological validation.
- Linear and random-forest artifact roundtrips; checksum/version/type boundaries.
- Dataset/run/model ownership boundaries, concurrent idempotent imports, quotas,
  cancellation before atomic publication, and stale job reporting.
- Account hashing, sign-in throttling, private-mode gating, sign-out and readiness.
- Valid CSV/XLSX imports, malformed/duplicate headers, compressed archive limits,
  invalid scenario values, and Excel formula-text protection.
- Full Streamlit workflow: question → finding → challenge → background training
  → scenario SHAP → report downloads → unrelated rerun → new session → saved
  investigation/model reopening.
- Public demo mode rejects uploads and does not expose an upload control.
- LLM request contracts remain mocked legacy checks. The product UI currently
  uses the explicitly labeled guided planner; Ollama has not been integrated.

## Browser verification

The final local product was opened in the browser. Its overview, navigation,
plan controls, saved finding, and statistical challenge were exercised.
Desktop (1280px) and mobile (390px) layouts were inspected. Mobile
navigation opened correctly and the question/plan controls remained readable.
These are basic responsive checks, not an exhaustive device test.
Programmatic UI tests cover the complete modeling and export journey.

## Remaining production gates

- Docker is installed but its daemon is not running. The Linux image build,
  non-root process and container startup have not been verified locally.
  CI includes a container build/start gate, but no remote CI run is claimed.
- No public deployment URL has been created. Host HTTPS, persistent volume,
  credentials, external reachability and hosted workflows remain unverified.
- The backup API has a local automated check; an isolated host restore drill
  and operational backup schedule still need completion.
- No independent security review, load test, enterprise SSO/MFA, or compliance
  certification is claimed.
- Remote source publishing remains awaiting the user's explicit authorization
  after the earlier automatic upload review rejected that action.

## Release commands

Run from the repository root with the pinned environment:

    python -m pytest v2/tests -q
    python -m pip_audit -r v2/requirements.lock
    python -m v2.health --port 8503

Docker and hosted checks must pass before changing the release classification.
