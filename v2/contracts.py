"""Validated application contracts, independent of the UI and any future LLM."""
from dataclasses import asdict, dataclass
from .errors import ProductError


@dataclass(frozen=True)
class TrainingRequest:
    target: str
    task: str
    features: tuple[str, ...]
    algorithms: tuple[str, ...]
    split_strategy: str = 'random'
    split_column: str | None = None

    @classmethod
    def parse(cls, value, columns):
        allowed = {'target', 'task', 'features', 'algorithms', 'split_strategy', 'split_column'}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ProductError('Training configuration contains unsupported fields.')
        target, task = value.get('target'), value.get('task')
        if not isinstance(target, str) or target not in columns or not isinstance(task, str) or task not in {'classification', 'regression'}:
            raise ProductError('Choose a valid target and prediction task.')
        strategy = value.get('split_strategy', 'random')
        split_column = value.get('split_column') if strategy != 'random' else None
        if not isinstance(strategy, str) or strategy not in {'random', 'group', 'temporal'}:
            raise ProductError('Unknown validation strategy.')
        if strategy != 'random' and (not isinstance(split_column, str) or split_column not in columns or split_column == target):
            raise ProductError('Choose a valid group or timestamp column.')
        features = value.get('features')
        if not isinstance(features, (list, tuple)) or not features or any(not isinstance(f, str) or f not in columns for f in features):
            raise ProductError('Choose valid predictor columns.')
        features = tuple(sorted(set(features) - {target, split_column}))
        if not features:
            raise ProductError('Choose at least one predictor other than the target and split column.')
        from .training import MODEL_NAMES
        available = {'Baseline', *MODEL_NAMES[task]}
        algorithms = value.get('algorithms', sorted(available - {'Baseline'}))
        if not isinstance(algorithms, (list, tuple)) or not algorithms or any(not isinstance(a, str) or a not in available for a in algorithms):
            raise ProductError('Choose at least one model supported by the prediction task.')
        return cls(target, task, features, tuple(sorted(set(algorithms))), strategy, split_column)

    def to_dict(self):
        result = asdict(self)
        result['features'] = list(self.features)
        result['algorithms'] = list(self.algorithms)
        return result
