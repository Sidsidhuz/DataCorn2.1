"""Persist only application-trained pipelines with skops' default trusted types."""
import hashlib
from importlib.metadata import version
from io import BytesIO
import pandas as pd
try:
    import skops.io as sio
except ImportError:
    sio = None
from .errors import ProductError

# Reviewed NumPy descriptor and scikit-learn fitted tree state container.
# Never expand this list from artifact contents or user input.
REVIEWED_TYPES = {'numpy.dtype', 'sklearn.tree._tree.Tree'}


def serialize_model(bundle):
    if sio is None:
        raise ProductError('Model saving requires skops. Install the pinned v2 requirements in the Python environment used to launch this app.', 'missing_dependency')
    artifact = sio.dumps(bundle['pipeline'])
    unknown = sio.get_untrusted_types(data=artifact)
    if set(unknown) - REVIEWED_TYPES:
        raise ProductError('This model contains unsupported serialization types. No artifact was saved.', 'unsupported_artifact')
    metadata = {k: v for k, v in bundle.items() if k not in {'pipeline', 'background', 'example'}}
    metadata['versions'] = {name: version(name) for name in ['scikit-learn', 'numpy', 'skops']}
    return artifact, hashlib.sha256(artifact).hexdigest(), metadata


def deserialize_model(job):
    if sio is None:
        raise ProductError('Opening saved models requires skops. Install the pinned v2 requirements in the Python environment used to launch this app.', 'missing_dependency')
    if job['status'] != 'completed' or not job['artifact']:
        raise ProductError('The model is not ready yet.')
    if hashlib.sha256(job['artifact']).hexdigest() != job['artifact_sha256']:
        raise ProductError('The saved model failed its integrity check. Train a new model.', 'artifact_integrity')
    metadata = job['metadata']
    if metadata['versions']['scikit-learn'] != version('scikit-learn'):
        raise ProductError('This model was created with a different scikit-learn version. Retrain it with this release.', 'artifact_version')
    if set(sio.get_untrusted_types(data=job['artifact'])) - REVIEWED_TYPES:
        raise ProductError('The artifact contains unsupported types and cannot be opened.', 'unsupported_artifact')
    return {**metadata, 'pipeline': sio.loads(job['artifact'], trusted=sorted(REVIEWED_TYPES)),
            'background': pd.read_parquet(BytesIO(job['background'])),
            'example': pd.read_parquet(BytesIO(job['example']))}
