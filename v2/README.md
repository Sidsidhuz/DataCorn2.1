# DataCorn 2.1 — Evidence Studio

A persistent workspace for investigating business questions, challenging conclusions with actual statistical checks, training models, and explaining predictions.

The product foundation adds owner-scoped storage, saved model artifacts, background jobs, explicit validation splits, managed private accounts, and safe public-demo defaults. Ollama integration is deferred.

## Start locally

Use Python 3.13 from the repository root:

```powershell
python -m venv .venv-v2
.venv-v2\Scripts\Activate.ps1
python -m pip install -r v2/requirements.txt
python -m v2.launch --mode local
```

Local mode binds to loopback. The default deployment mode is synthetic-only demo; private uploads require private mode and an administrator-provisioned login.

## Documentation

- [Product guide](PRODUCT_GUIDE.md): workflows, persistence, modes, and limits.
- [Current architecture](PRODUCT_ARCHITECTURE.md): services, ownership, job lifecycle, and future AI boundary.
- [Operations runbook](RUNBOOK.md): deployment, accounts, backup, restore, and release gates.
- [Validation record](release/VALIDATION.md): observed tests, dependency audit, and incomplete checks.
- [Interview teach-back](EXPLAIN_IT.md): explain the statistical critic and model-validation decisions.
- [Original source audit](ARCHITECTURE.md): historical review of the original DataCorn repository.

## Validate

```powershell
python -m pip install -r v2/requirements-dev.txt
python -m pytest v2/tests -q
python -m pip_audit -r v2/requirements.lock
```

requirements.txt installs the exact runtime lock. requirements.in records direct dependency bounds. CI also builds and starts the non-root container.

## The signature demonstration

The synthetic dataset shows premium churn at 50% versus standard at 30% overall. Inside both customer segments, the difference reverses to -10 percentage points. Challenge My Finding runs real tests, reports Holm-adjusted stratum p-values, and warns about confounding without claiming a causal benefit. Neither adjusted stratum p-value is below 0.05 in this example.

Training compares baseline/linear/forest candidates using training-only validation, then evaluates the selected winner on the reserved holdout. Saved models support real SHAP and what-if prediction. Evidence exports preserve computed results, source/data hashes, settings, versions, and limitations.

This is a small, single-process analyst product foundation. Public deployment, host backup/restore, Linux container verification, and external security review must be verified separately before describing it as production-ready.
