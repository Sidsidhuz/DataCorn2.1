"""Reproducible training with explicit splits, fold-local preprocessing, and cancellation."""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, get_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from app.modules.feature_engineering.feature_service import FeatureService
from .quality import profile
from .errors import JobCancelled

MODEL_NAMES = {
    'classification': ['Logistic regression', 'Random forest', 'Decision tree', 'Extra trees', 'K-nearest neighbors'],
    'regression': ['Ridge regression', 'Random forest', 'Decision tree', 'Extra trees', 'K-nearest neighbors'],
}


def model_candidates(classification):
    if classification:
        return {'Baseline': DummyClassifier(strategy='prior'),
                'Logistic regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random forest': RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1),
                'Decision tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=42),
                'Extra trees': ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1),
                'K-nearest neighbors': KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=1)}
    return {'Baseline': DummyRegressor(strategy='mean'),
            'Ridge regression': Ridge(alpha=1),
            'Random forest': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1),
            'Decision tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42),
            'Extra trees': ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=1),
            'K-nearest neighbors': KNeighborsRegressor(n_neighbors=5, algorithm='brute', n_jobs=1)}


def train(df, target, task, requested_features=None, *, split_strategy='random',
          split_column=None, algorithms=None, progress=None, timeout=180):
    started = time.monotonic()
    def checkpoint(value, message):
        if time.monotonic() - started > timeout:
            raise ValueError('Training exceeded the time budget. Use fewer rows or simpler models.')
        if progress:
            progress(value, message)
    checkpoint(.02, 'Validating target and split strategy')
    if task not in ('classification', 'regression') or target not in df:
        raise ValueError('Choose a valid target and an explicit classification or regression task.')
    if split_strategy not in ('random', 'group', 'temporal'):
        raise ValueError('Choose random, group, or temporal splitting.')
    if split_strategy != 'random' and (split_column not in df or split_column == target):
        raise ValueError('Choose a valid group or time column distinct from the target.')
    raw = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if not 40 <= len(raw) <= 5000:
        raise ValueError('Training requires 40–5,000 labeled rows.')
    chosen = list(df.columns) if requested_features is None else requested_features
    if any(c not in df for c in chosen):
        raise ValueError('A selected predictor does not exist in this dataset.')
    features = list(dict.fromkeys(c for c in chosen if c != target and (split_strategy == 'random' or c != split_column)))
    if not features:
        raise ValueError('Choose at least one predictor other than the target and split column.')
    obvious = {r['column'] for r in profile(raw[features], None)['risks'] if r['exclude']}
    dedupe_features = [c for c in features if c not in obvious]
    clean = raw.drop_duplicates(subset=list(dict.fromkeys(dedupe_features + [target] + ([split_column] if split_strategy != 'random' else []))))
    duplicates_removed = len(raw) - len(clean)
    if len(clean) < 40:
        raise ValueError('At least 40 distinct modeling rows are required after deduplication.')
    if split_strategy == 'temporal':
        times = pd.to_datetime(clean[split_column], errors='coerce', utc=True)
        if times.isna().any() or times.nunique() < 8:
            raise ValueError('Temporal splitting requires valid timestamps and at least eight distinct times.')
        order = np.argsort(times.to_numpy(), kind='stable')
        clean = clean.iloc[order]
        timeline = times.iloc[order]
        unique_times = timeline.drop_duplicates().to_numpy()
        cutoff = unique_times[max(1, int(len(unique_times) * .8))]
        train_idx = np.flatnonzero((timeline < cutoff).to_numpy())
        test_idx = np.flatnonzero((timeline >= cutoff).to_numpy())
    elif split_strategy == 'group':
        groups = clean[split_column]
        if groups.isna().any() or groups.nunique() < 5:
            raise ValueError('Group splitting needs at least five non-missing groups.')
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(clean, groups=groups))
    else:
        preliminary = clean[target].astype(str) if task == 'classification' else clean[target]
        if task == 'classification' and (not 2 <= preliminary.nunique() <= 10 or preliminary.value_counts().min() < 6):
            raise ValueError('Classification needs 2–10 classes with at least six rows in each.')
        train_idx, test_idx = train_test_split(np.arange(len(clean)), test_size=.2, random_state=42,
                                             stratify=preliminary if task == 'classification' else None)
    X = clean[features].copy()
    for c in X:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    y = clean[target].astype(str) if task == 'classification' else pd.to_numeric(clean[target], errors='coerce')
    if y.isna().any() or y.nunique() < 2 or (task == 'classification' and y.nunique() > 10):
        raise ValueError('The selected target is not valid for this task.')
    Xtrain, Xtest = X.iloc[train_idx], X.iloc[test_idx]
    ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]
    if min(len(Xtrain), len(Xtest)) < 8:
        raise ValueError('This split leaves too few training or holdout rows.')
    if task == 'classification' and (ytrain.nunique() < 2 or ytrain.value_counts().min() < 3):
        raise ValueError('The training split needs at least three rows in each class. Review the split.')
    train_quality = profile(pd.concat([Xtrain, clean[target].iloc[train_idx]], axis=1), target)
    blocked = {r['column'] for r in train_quality['risks'] if r['exclude']}
    kept = [c for c in features if c not in blocked]
    if not kept:
        raise ValueError('All selected predictors were flagged. Select legitimate pre-outcome features.')
    Xtrain, Xtest = Xtrain[kept], Xtest[kept]
    dtypes = {c: 'numeric' if pd.api.types.is_numeric_dtype(Xtrain[c]) else 'categorical' for c in kept}
    preprocessor = FeatureService.get_preprocessing_pipeline(kept, dtypes, Xtrain)
    classification = task == 'classification'
    candidates = model_candidates(classification)
    if algorithms is not None:
        if not algorithms or set(algorithms) - set(candidates):
            raise ValueError('Select at least one supported model.')
        candidates = {k: v for k, v in candidates.items() if k in set(algorithms) | {'Baseline'}}
    scoring = 'f1_macro' if classification else 'neg_mean_absolute_error'
    scorer = get_scorer(scoring)
    metadata = {'strategy': split_strategy, 'column': split_column if split_strategy != 'random' else None}
    if split_strategy == 'group':
        train_groups = clean[split_column].iloc[train_idx]
        test_groups = clean[split_column].iloc[test_idx]
        if train_groups.nunique() < 3:
            raise ValueError('At least three training groups are needed for group cross-validation.')
        folds = list(GroupKFold(n_splits=3).split(Xtrain, ytrain, train_groups))
        metadata.update(train_group_count=int(train_groups.nunique()), holdout_group_count=int(test_groups.nunique()), group_overlap=0)
    elif split_strategy == 'temporal':
        training_times = timeline.iloc[train_idx].reset_index(drop=True)
        blocks = np.array_split(training_times.drop_duplicates().to_numpy(), 4)
        folds = [(np.flatnonzero(training_times.isin(np.concatenate(blocks[:i]))), np.flatnonzero(training_times.isin(blocks[i]))) for i in range(1, 4)]
        metadata.update(training_ends=str(training_times.max()), holdout_starts=str(timeline.iloc[test_idx].min()))
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if classification else KFold(n_splits=3, shuffle=True, random_state=42)
        folds = list(cv.split(Xtrain, ytrain))
    leaderboard, pipelines = [], {}
    total = len(candidates) * 3 + 1
    step = 0
    for name, estimator in candidates.items():
        pipe = Pipeline([('preprocessor', clone(preprocessor)), ('predictor', estimator)])
        model_started = time.perf_counter()
        scores = []
        try:
            for fold, (fit_idx, val_idx) in enumerate(folds, 1):
                checkpoint(.08 + .78 * step / total, f'{name}: validation fold {fold} of 3')
                if classification and ytrain.iloc[fit_idx].nunique() < 2:
                    raise ValueError('A training fold contains only one class. Use a larger dataset or review the split.')
                fitted = clone(pipe).fit(Xtrain.iloc[fit_idx], ytrain.iloc[fit_idx])
                scores.append(float(scorer(fitted, Xtrain.iloc[val_idx], ytrain.iloc[val_idx])))
                step += 1
        except JobCancelled:
            raise
        except ValueError as exc:
            raise ValueError(f'{name} could not be validated: {exc}') from exc
        if not np.isfinite(scores).all():
            raise ValueError(f'{name} produced invalid validation scores.')
        leaderboard.append({'model': name, 'cv_score': float(np.mean(scores)), 'cv_std': float(np.std(scores)),
                            'seconds': round(time.perf_counter() - model_started, 2)})
        pipelines[name] = pipe
    leaderboard.sort(key=lambda row: row['cv_score'], reverse=True)
    winner = leaderboard[0]['model']
    checkpoint(.9, 'Fitting the selected model and evaluating the reserved holdout')
    pipeline = pipelines[winner].fit(Xtrain, ytrain)
    pred = pipeline.predict(Xtest)
    classes = pipeline.named_steps['predictor'].classes_.tolist() if classification else []
    if classification:
        metrics = {'accuracy': float(accuracy_score(ytest, pred)), 'macro_f1': float(f1_score(ytest, pred, average='macro', zero_division=0))}
        if len(classes) == 2 and ytest.nunique() == 2 and set(ytest).issubset(classes):
            metrics['roc_auc'] = float(roc_auc_score((ytest == classes[1]).astype(int), pipeline.predict_proba(Xtest)[:, 1]))
    else:
        metrics = {'mae': float(mean_absolute_error(ytest, pred)), 'r2': float(r2_score(ytest, pred))}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError('Holdout metrics are undefined; review target variance and split.')
    checkpoint(.97, 'Preparing model provenance')
    return {'pipeline': pipeline, 'winner': winner, 'task': task, 'target': target, 'features': kept,
            'excluded_features': [r for r in train_quality['risks'] if r['exclude']],
            'review_flags': train_quality['risks'], 'leaderboard': leaderboard, 'holdout': metrics,
            'cv_metric': scoring, 'train_rows': len(Xtrain), 'test_rows': len(Xtest),
            'missing_target_rows': len(df) - len(raw), 'duplicates_removed': duplicates_removed,
            'background': Xtrain.sample(min(80, len(Xtrain)), random_state=42), 'example': Xtest.head(1),
            'classes': classes, 'seed': 42, 'split': metadata,
            'limitations': ['Validation estimates depend on the chosen split and sampling process.',
               'Repeated feature or model tuning on displayed holdout scores invalidates final-test independence.',
               'Feature availability and domain-specific leakage still require human review.']}
