import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from v2.training import train, MODEL_NAMES
from v2.contracts import TrainingRequest
from v2.artifacts import serialize_model, deserialize_model
from v2.storage import Store
from v2.modeling import explain_scenario


@pytest.mark.parametrize('task,name', [(task, name) for task, names in MODEL_NAMES.items() for name in names])
def test_each_model_trains_persists_predicts_and_explains(task, name):
    if task == 'classification':
        x, y = make_classification(n_samples=120, n_features=4, n_informative=3, n_redundant=0, class_sep=2, random_state=21)
    else:
        x, y = make_regression(n_samples=120, n_features=4, noise=5, random_state=21)
    df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
    df['target'] = y
    config = TrainingRequest.parse({'target': 'target', 'task': task, 'features': ['a', 'b', 'c', 'd'], 'algorithms': [name]}, df.columns)
    bundle = train(df, config.target, config.task, list(config.features), algorithms=list(config.algorithms))
    assert bundle['winner'] == name
    assert {row['model'] for row in bundle['leaderboard']} == {'Baseline', name}
    artifact, digest, metadata = serialize_model(bundle)
    loaded = deserialize_model({'status': 'completed', 'artifact': artifact, 'artifact_sha256': digest,
                                'metadata': metadata, 'background': Store.frame_bytes(bundle['background']),
                                'example': Store.frame_bytes(bundle['example'])})
    np.testing.assert_array_equal(bundle['pipeline'].predict(bundle['example']), loaded['pipeline'].predict(bundle['example']))
    for index in ([0, 1] if task == 'classification' else [0]):
        scenario = explain_scenario(loaded, loaded['example'].iloc[0].to_dict(), index)
        assert scenario['contributions']
        assert scenario['additivity_error'] < 1e-5
        if task == 'classification':
            assert scenario['unit'] in ('probability', 'log odds')
            assert 0 <= scenario['probability'] <= 1
