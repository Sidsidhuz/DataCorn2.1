# Available models

Both tasks offer five selectable trained models in addition to an always-included baseline.

| Classification | Regression |
| --- | --- |
| Logistic regression | Ridge regression |
| Random forest classifier | Random forest regressor |
| Decision tree classifier | Decision tree regressor |
| Extra trees classifier | Extra trees regressor |
| K-nearest neighbors classifier | K-nearest neighbors regressor |

All candidates use the existing fold-local preprocessing, three-fold training validation and reserved holdout. The winner is selected by macro F1 for classification or negative mean absolute error for regression. Model saving still uses skops with the existing reviewed-type restrictions.

Linear models use Linear SHAP, tree models use Tree SHAP, and nearest-neighbor models use seeded permutation SHAP. Class explanations report either probability or the linear model's decision units, with additivity checked before display. Nearest-neighbor explanations may take longer with many encoded fields. Selecting fewer candidates reduces training time.

The per-model regression suite covers training, request validation, serialization, reopening, prediction and explanations for all ten task/model combinations, including both output classes for binary classification.
