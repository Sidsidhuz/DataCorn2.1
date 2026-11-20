> Historical MVP record. The current product release is documented in [PRODUCT_GUIDE.md](PRODUCT_GUIDE.md), [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md), and [release/VALIDATION.md](release/VALIDATION.md).

# Delivery plan — November 20, 2026 deadline

All dates below are in India Standard Time. Status must describe observed
evidence, not planned work as though it were completed.

| Date | Outcome | Acceptance / status |
|---|---|---|
| Nov 15 | Audit original; build first MVP; engine and UI checks | Implemented; see VALIDATION.md for actual results |
| Nov 16 | Host the app and configure optional real LLM | Account sign-in/terms step required; key configuration and live model call unverified |
| Nov 17 | Test one external public dataset; inspect missingness and feature timing | Pending; synthetic demo alone does not establish general performance |
| Nov 18 | Mobile browser QA, deployed upload/report checks, failure cases | Pending; preserve the working guided demo |
| Nov 19 | Feature freeze; record narrated demo; practise teach-back | Pending; leave time to fix only demonstrated defects |
| Nov 20 | Submit working source, accessible demo URL, short recording, summary | User submits the application after checking links |

## MVP boundary

P0 is a defensible end-to-end investigation, statistical critic, quality
checks, real predictive validation, genuine SHAP, and downloadable evidence.
Do not block submission on a chat agent framework, SQL generation, multiagent
debates, database connectors, a model registry, or production authentication.

## Incremental acceptance gates

1. **Inspect:** check dataset hash, missingness, duplicate count, known leaked
   column in the demo, and a legitimate strongly predictive regression feature.
2. **Plan:** confirm valid schema mappings; reject unknown/same columns; label
   offline logic clearly. Mock the LLM API for contract tests, then verify a
   real call after a key is configured.
3. **Investigate:** reproduce Welch in SciPy; verify the interval and exclusions;
   test categorical and correlation paths and invalid small samples.
4. **Challenge:** reproduce the aggregate +0.20 and both segment -0.10 effects;
   show real stratum p-values, Holm correction, and permutation sensitivity.
5. **Model:** compare fixed candidates on training CV; check baseline and final
   holdout; prove copy/identifier removal; test classification and regression.
6. **Explain:** edit a scenario, compute a fresh prediction, validate SHAP
   additivity, and explain probability versus log-odds units.
7. **Report:** confirm PDF, JSON, and workbook reflect the same evidence;
   inspect page rendering and escaping of user-controlled strings.
8. **Deploy:** verify startup, guided workflow, upload, reports, and optional
   LLM error handling on the actual hosted URL; inspect with a signed-out
   browser so an interviewer can reach it.

## Interview deliverables

- Short README with one run command and known limits.
- Source branch and CI checks, with the original project preserved on main.
- Live URL only after the actual host reports success and the workflow works.
- A 90-second guided demo plus a 3–5 minute implementation explanation.
- An evidence PDF and matching JSON from the deterministic demo.
- A candid explanation of AI tools used, what was tested, and what Sidharth
  personally reviewed and can modify without assistance.

## Demonstration recovery

If the host is sleeping, wake it before recording. Keep a locally running
copy and the generated evidence package. If the API is unavailable, select
the labeled offline planner and explain the distinction; do not claim that
the fallback is an LLM. If SHAP fails, show the explicit error and the working
prediction without inventing attribution values.
