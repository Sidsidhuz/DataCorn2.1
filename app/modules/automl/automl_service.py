from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)

# Model imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from app.config import settings
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.trained_model import TrainedModel
from app.models.trained_model import TrainedModel
from app.modules.feature_engineering.feature_service import FeatureService
from app.modules.explainability.shap_service import SHAPService


class AutoMLService:
    @staticmethod
    def infer_problem_type(df: pd.DataFrame, target_col: str) -> str:
        """Infures whether the training task is classification or regression."""
        target_series = df[target_col].dropna()
        if target_series.dtype in ["object", "category", "bool"]:
            return "classification"
        
        # Low cardinality integer targets -> classification
        if pd.api.types.is_integer_dtype(target_series) and target_series.nunique() <= 20:
            return "classification"
            
        return "regression"

    @staticmethod
    def run_training_job_sync(db: Session, job_id: int) -> None:
        """
        Executes the AutoML training job: loads data, preprocesses, trains multiple model candidates,
        evaluates metrics, serializes pipelines, logs to DB, and selects the best model.
        """
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        try:
            # 1. Load dataset
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if not dataset:
                raise ValueError("Dataset not found")

            # Load cleaned dataset
            cleaned_path = settings.data_path / f"cleaned/{dataset.id}/cleaned.csv"
            if not cleaned_path.exists():
                raise ValueError(f"Cleaned dataset not found at {cleaned_path}. Run cleaning first.")

            df = pd.read_csv(cleaned_path)
            target_col = job.target_column

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset.")

            # 2. Preprocess Dataframe (Datetime extraction & ID removal)
            metadata = dataset.get_schema_metadata()
            dtypes = metadata.get("dtypes", {})
            df_prepped, updated_dtypes = FeatureService.preprocess_dataframe_pandas(df, target_col, dtypes)

            # Drop missing rows in target
            df_prepped = df_prepped.dropna(subset=[target_col])

            X = df_prepped.drop(columns=[target_col])
            y = df_prepped[target_col]

            # Encode string/categorical target labels to integers
            label_encoder = None
            is_string_target = (
                y.dtype == "object"
                or y.dtype.name == "category"
                or pd.api.types.is_string_dtype(y)
                or not np.issubdtype(y.dtype, np.number)
            )
            if is_string_target:
                label_encoder = LabelEncoder()
                y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
                print(f"[AutoML] Label-encoded target '{target_col}': {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
            else:
                print(f"[AutoML] Target '{target_col}' dtype={y.dtype}, no encoding needed.")

            # Determine problem type
            problem_type = job.problem_type
            if not problem_type or problem_type == "auto":
                problem_type = AutoMLService.infer_problem_type(df, target_col)
                job.problem_type = problem_type
                db.commit()

            # Split data (80/20 split)
            if problem_type == "classification":
                # Ensure stratification is possible (class count > 1)
                unique_classes = y.nunique()
                if unique_classes < 2:
                    raise ValueError(f"Target column '{target_col}' has only {unique_classes} class. Classification requires >= 2 classes.")
                
                # Check class counts for stratification
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            feature_cols = X.columns.tolist()

            # Get base ColumnTransformer preprocessor
            preprocessor = FeatureService.get_preprocessing_pipeline(feature_cols, updated_dtypes, X_train)

            # 3. Setup candidate models and tuning grids
            config = job.get_config()
            requested_algs = config.get("algorithms")

            candidates = AutoMLService._get_algorithms(problem_type, requested_algs)

            trained_models_list = []

            for alg_name, (model_class, param_grid) in candidates.items():
                try:
                    # Setup Pipeline
                    pipeline = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("predictor", model_class),
                        ]
                    )

                    # Short Hyperparameter Tuning
                    search = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_grid,
                        n_iter=3,
                        cv=3,
                        random_state=42,
                        n_jobs=1,  # Keep local execution single-threaded for safety
                    )
                    
                    search.fit(X_train, y_train)
                    best_pipeline = search.best_estimator_

                    # Evaluate on test set
                    y_pred = best_pipeline.predict(X_test)
                    
                    metrics = {}
                    if problem_type == "classification":
                        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
                        metrics["precision"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                        metrics["recall"] = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
                        metrics["f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                        
                        # Compute ROC-AUC if possible
                        try:
                            if hasattr(best_pipeline, "predict_proba"):
                                y_prob = best_pipeline.predict_proba(X_test)
                                if len(np.unique(y_test)) == 2:
                                    # Binary
                                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                                else:
                                    # Multiclass
                                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
                            else:
                                metrics["roc_auc"] = metrics["f1"]
                        except Exception:
                            metrics["roc_auc"] = metrics["f1"]
                            
                        primary_metric = "f1"
                    else:
                        metrics["r2"] = float(r2_score(y_test, y_pred))
                        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
                        primary_metric = "r2"

                    # 4. Save model to database (create record first to get ID)
                    db_model = TrainedModel(
                        training_job_id=job.id,
                        algorithm=alg_name,
                        primary_metric_value=metrics[primary_metric],
                        artifact_path="pending",
                        is_best=False,
                    )
                    db_model.set_hyperparameters(search.best_params_)
                    db_model.set_metrics(metrics)
                    db.add(db_model)
                    db.commit()
                    db.refresh(db_model)

                    # Update and save the model pipeline artifact
                    model_dir = settings.models_dir / str(db_model.id)
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    relative_artifact_path = f"models/{db_model.id}/model.joblib"
                    absolute_artifact_path = settings.data_path / relative_artifact_path

                    # Save pipeline (fittedColumnTransformer + fitted Predictor)
                    # We also save feature columns and problem type inside model artifact for prediction service simplicity
                    model_payload = {
                        "pipeline": best_pipeline,
                        "feature_cols": feature_cols,
                        "target_col": target_col,
                        "problem_type": problem_type,
                        "dtypes": updated_dtypes,
                        "label_encoder": label_encoder,
                    }
                    joblib.dump(model_payload, absolute_artifact_path)

                    db_model.artifact_path = relative_artifact_path
                    db.commit()
                    db.refresh(db_model)

                    trained_models_list.append(db_model)

                except Exception as e:
                    # Log failure of a single algorithm but continue others
                    print(f"Algorithm {alg_name} failed: {str(e)}")
                    continue

            if not trained_models_list:
                raise ValueError("All candidate algorithms failed to train.")

            # 5. Determine the best model and set is_best = True
            # Sort by primary metric value descending (ROC-AUC/F1/R^2 are all higher is better)
            trained_models_list.sort(key=lambda m: m.primary_metric_value or 0.0, reverse=True)
            best_model = trained_models_list[0]
            best_model.is_best = True
            
            job.status = "completed"
            db.commit()

            # 6. Generate Global SHAP Explanation for the best model automatically
            try:
                # Re-load payload of the best model to generate SHAP
                from app.modules.prediction.predict_service import PredictionService
                payload = PredictionService.load_model_payload(best_model.id)
                predictor = payload["pipeline"]
                
                # Transform X_train using the pipeline's preprocessor
                preprocessor = predictor.named_steps.get("preprocessor")
                model_step = predictor.named_steps.get("predictor")
                if preprocessor and model_step:
                    X_transformed = preprocessor.transform(X_train)
                    feature_names = preprocessor.get_feature_names_out()
                    # Convert to dataframe for SHAPService
                    if not isinstance(X_transformed, pd.DataFrame):
                        if hasattr(X_transformed, "toarray"):
                            X_transformed = X_transformed.toarray()
                        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
                    
                    SHAPService.generate_global_explanation(
                        dataset_id=dataset.id,
                        model_id=best_model.id,
                        predictor=model_step,
                        X_preprocessed=X_transformed,
                        feature_names=list(feature_names)
                    )
            except Exception as e:
                print(f"[AutoML] Global SHAP generation failed for job {job.id}: {str(e)}")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            raise e
        finally:
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

    @staticmethod
    def _get_algorithms(problem_type: str, requested: list[str] | None) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
        """Returns the classifier/regressor classes and their hyperparameter search spaces."""
        
        # Helper to filter candidates
        def filter_algs(all_algs):
            if not requested:
                return all_algs
            return {k: v for k, v in all_algs.items() if k in requested}

        if problem_type == "classification":
            all_algs = {
                "logistic_regression": (
                    LogisticRegression(max_iter=1000, random_state=42),
                    {"predictor__C": [0.1, 1.0, 10.0]}
                ),
                "random_forest": (
                    RandomForestClassifier(random_state=42),
                    {"predictor__n_estimators": [50, 100], "predictor__max_depth": [None, 5, 10]}
                ),
                "xgboost": (
                    xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__max_depth": [3, 6], "predictor__n_estimators": [50, 100]}
                ),
                "lightgbm": (
                    lgb.LGBMClassifier(random_state=42, verbose=-1),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__max_depth": [-1, 5, 10], "predictor__n_estimators": [50, 100]}
                ),
                "catboost": (
                    CatBoostClassifier(random_state=42, verbose=0),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__depth": [4, 6], "predictor__iterations": [50, 100]}
                ),
            }
            return filter_algs(all_algs)
        else:
            all_algs = {
                "linear_regression": (
                    LinearRegression(),
                    {"predictor__fit_intercept": [True, False]}
                ),
                "random_forest": (
                    RandomForestRegressor(random_state=42),
                    {"predictor__n_estimators": [50, 100], "predictor__max_depth": [None, 5, 10]}
                ),
                "xgboost": (
                    xgb.XGBRegressor(random_state=42),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__max_depth": [3, 6], "predictor__n_estimators": [50, 100]}
                ),
                "lightgbm": (
                    lgb.LGBMRegressor(random_state=42, verbose=-1),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__max_depth": [-1, 5, 10], "predictor__n_estimators": [50, 100]}
                ),
                "catboost": (
                    CatBoostRegressor(random_state=42, verbose=0),
                    {"predictor__learning_rate": [0.05, 0.1], "predictor__depth": [4, 6], "predictor__iterations": [50, 100]}
                ),
            }
            return filter_algs(all_algs)
