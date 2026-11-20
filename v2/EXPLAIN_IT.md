# Explain it in your own words

## A 45-second project explanation

“DataCorn started as my no-code machine-learning project. I extended it
into a data investigation tool. A user asks a business question, reviews the
proposed variables, and runs a statistical analysis. The special feature is
Challenge My Finding. It checks the evidence and compares the result within
segments to find possible alternative explanations. In my synthetic churn
demo, the overall result reverses within both customer segments. I also
compare predictive models, keep the final test data separate from model
selection, and explain scenario predictions with SHAP. The report includes
the actual test results and limitations.”

If you have configured and verified the LLM, add: “The language model maps
the question to a constrained plan using schema metadata. Python computes
the statistics.” If you have not, say the demo uses the offline planner.

## Understand these six functions first

1. `planner.propose`: reads a question, maps it to known columns, and returns
   a plan. It does not read a question and magically know the answer.
2. `statistics.analyze`: selects a permitted test, removes incomplete pairs,
   and returns the effect, uncertainty, and statistical evidence.
3. `challenge.challenge`: checks the existing evidence, runs another numerical
   sensitivity check, and tests the comparison inside segments.
4. `modeling.train`: splits raw rows, fits preprocessing in training folds,
   compares CV scores, then evaluates the winner on held-out rows.
5. `modeling.explain_scenario`: makes a new prediction and calculates SHAP
   contributions using the fitted model. It also checks that they add up.
6. `reporting.evidence_package`: collects the same computed records that the
   UI shows. The report does not ask an LLM to invent a business conclusion.

## Why can the demo reverse?

| Segment | Premium churn | Standard churn |
|---|---:|---:|
| New customers | 144 / 240 = 60% | 42 / 60 = 70% |
| Established customers | 6 / 60 = 10% | 48 / 240 = 20% |
| All customers | 150 / 300 = 50% | 90 / 300 = 30% |

Most premium customers are new customers, the high-churn segment. Most
standard customers are established customers, the low-churn segment. Mixing
the groups changes the overall comparison. This is Simpson's paradox.
It warns us to investigate the customer mix. It does not prove the causal
effect of premium support; other differences may still explain the result.

## Questions you should be ready to answer

**Why not let the LLM calculate everything?**
An LLM can misread data or produce unsupported numbers. Here it proposes a
validated plan. SciPy and scikit-learn do the computations.

**What is a p-value?**
Under a specified null hypothesis and assumptions, it measures how unusual
the observed test statistic or a more extreme one would be. It is not the
probability that my conclusion is true.

**Does statistical significance mean business value?**
No. We also need the size of the effect, uncertainty, costs, and context.

**Why Welch's test?**
It compares two means without assuming equal population variances. The demo
uses 0/1 churn, so the mean is the churn rate. The sample sizes are disclosed.

**What does the permutation check do?**
It repeatedly rearranges the observations between groups and compares a
studentized statistic. We use a fixed seed and disclose 499 resamples. It is
a sensitivity check with assumptions, not proof that bias has disappeared.

**Why Holm correction?**
Testing several strata creates more chances for a false positive. Holm
adjusts the displayed family of stratum-test p-values. It does not cover
every repeated question a user might ask.

**What changed from the original ML implementation?**
The old cleaning service learned imputation values before the split, and the
old model comparison chose a winner using test scores. V2 fits preprocessing
inside training CV and evaluates only the selected winner on the holdout.

**What is SHAP showing?**
How the fitted model attributes a prediction to input features relative to a
reference. The units depend on the model: probability for this random forest,
log odds for binary logistic regression. It does not show causal influence.

**What did you reuse?**
The original FeatureService is imported directly. Profiling, SHAP and report
construction use the original project's approaches in smaller pure functions.
The original backend, storage models, and UI remain available.

## Practise independently before claiming ownership

- Change the demo's segment counts, predict how the aggregate changes, then run it.
- Remove `closed_account_flag` and confirm that other predictors still work.
- Add missing values and explain why the analysis and model handle them differently.
- Upload a small unfamiliar dataset and manually confirm the correct task and columns.
- Find the three-fold CV call and the single holdout-prediction call in code.
- Change a model depth and explain why tuning must not be guided by holdout scores.
- Open the JSON and trace one displayed p-value back to `statistics.analyze`.

## AI development disclosure

AI assistance helped audit the repository, implement modules, design the UI,
and write/run tests. Numerical results come from executed Python, not AI
claims. Before submission, describe the parts you personally reviewed,
modified, tested and can explain. Do not present unreviewed generated code
as independently authored work or claim unverified live AI/deployment tests.
