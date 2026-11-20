> Historical MVP record. The current product release is documented in [PRODUCT_GUIDE.md](PRODUCT_GUIDE.md), [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md), and [release/VALIDATION.md](release/VALIDATION.md).

# Validation record

Verified November 15, 2026. These results apply to the synthetic demo and
the test cases in this branch; they are not claims about a live deployment.

## Automated checks

`python -m pytest v2/tests -q`: **17 passed** on Windows / Python 3.13.
The final observed run completed in 8.66 seconds. Dependencies from the
tested environment are recorded in `requirements-tested.txt`; the portable
deployment input is `requirements.txt`.

Coverage includes independent SciPy comparison, data fingerprints, column
validation, copy/identifier flags, missingness, invalid test inputs, Holm
adjustment, Spearman and Fisher paths, classification/regression training,
real SHAP additivity, all export formats, mocked LLM request/failure
contracts, and the complete guided UI through Streamlit AppTest.

## Observed synthetic demonstration

| Evidence | Observed value |
|---|---:|
| Dataset rows | 600 |
| Missing cells | 6 |
| Premium aggregate churn | 0.50 |
| Standard aggregate churn | 0.30 |
| Aggregate difference | +0.20 |
| Welch two-sided p-value | 4.599e-07 |
| 95% interval | [0.12297, 0.27703] |
| Studentized permutation p-value | 0.004 |
| Established segment difference | -0.10 |
| Established Holm-adjusted p-value | 0.06979 |
| New segment difference | -0.10 |
| New segment Holm-adjusted p-value | 0.14210 |
| CV-selected model | Logistic regression |
| Training / holdout rows | 480 / 120 |
| Holdout accuracy | 0.74167 |
| Holdout macro F1 | 0.73502 |
| Holdout ROC-AUC | 0.70023 |

The within-segment effects reverse direction descriptively; neither adjusted
stratum p-value is below 0.05. The app therefore warns about confounding and
does not claim a statistically proven beneficial causal effect within groups.

## Browser / document verification

- Local Streamlit server starts at `http://127.0.0.1:8502` and serves the UI.
- Browser inspection confirmed the polished workspace, proposed schema
  mapping, initial computed comparison, and Challenge My Finding reversal.
- The complete model, scenario-SHAP and report actions passed AppTest.
- The generated two-page PDF was rendered to PNG and visually inspected.
  Final regenerated pages were checked again after formatting/provenance edits.
- The final JSON includes dataset and engine hashes, dependency versions,
  parameters, exclusions, effect, interval, tests, and limitations.

## Still unverified / remaining steps

- Real LLM inference: no configured API key or running local Ollama endpoint.
  Only the HTTP contract and error paths have been tested with mocks.
- Public hosting: Streamlit's sign-in screen requires the user to accept its
  terms. No public live URL is claimed.
- Linux deployment, Docker build and GitHub CI: configuration supplied;
  mark as successful only after their actual jobs complete.
- Mobile browser, external public dataset, authenticated multiuser use, and
  narrated recording: pending. The deadline plan lists these separately.

## Fix caught by testing

The first regression test exposed a false positive: a legitimate strongly
predictive numeric feature was automatically excluded for high correlation.
Correlation now creates a human-review warning rather than an automatic
exclusion. Exact target copies and identifier/post-outcome suspects remain
excluded. This is a concrete example of why heuristic warnings need tests
and domain review.
