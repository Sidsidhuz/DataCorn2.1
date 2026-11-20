import pytest
from v2.contracts import TrainingRequest
from v2.errors import ProductError


def test_training_contract_canonicalizes_feature_choices():
    value = {'target': 'y', 'task': 'classification', 'features': ['b', 'a', 'b'], 'algorithms': ['Random forest']}
    request = TrainingRequest.parse(value, ['a', 'b', 'y'])
    assert request.features == ('a', 'b')
    assert request.to_dict()['features'] == ['a', 'b']


def test_unknown_operations_and_invalid_columns_are_rejected():
    base = {'target': 'y', 'task': 'classification', 'features': ['a']}
    for value in ({**base, 'execute_code': 'arbitrary'}, {**base, 'features': []}, {**base, 'features': ['missing']}, {**base, 'algorithms': ['Unknown model']}):
        with pytest.raises(ProductError):
            TrainingRequest.parse(value, ['a', 'y'])

def test_private_readiness_requires_a_provisioned_account(tmp_path):
    from v2.config import Settings
    from v2.health import check
    from v2.auth import create_user
    from v2.storage import Store
    settings = Settings(mode='private', data_dir=tmp_path)
    with pytest.raises(ValueError, match='provisioned'):
        check(settings)
    create_user(Store(tmp_path), 'operator', 'long-test-only-password')
    assert check(settings)['ready']


def test_demo_readiness_does_not_require_an_account(tmp_path):
    from v2.config import Settings
    from v2.health import check
    assert check(Settings(mode='demo', data_dir=tmp_path))['ready']
