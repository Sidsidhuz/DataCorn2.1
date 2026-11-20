"""Prediction and real SHAP explanations over trained model bundles."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from .training import train


def explain_scenario(bundle, values, class_index=1):
    import shap
    if set(values) != set(bundle['features']):
        raise ValueError('Supply exactly the predictors used by this model.')
    pipeline=bundle['pipeline']; prep=pipeline.named_steps['preprocessor']; model=pipeline.named_steps['predictor']
    row=pd.DataFrame([values],columns=bundle['features'])
    for c in bundle['features']:
        if pd.api.types.is_numeric_dtype(bundle['background'][c]):
            row[c]=pd.to_numeric(row[c],errors='coerce')
            if not np.isfinite(row[c].iloc[0]):
                raise ValueError(f'Enter a finite numeric value for {c}.')
    prediction=pipeline.predict(row)[0]
    classification=bundle['task']=='classification'
    if classification and (not isinstance(class_index, (int, np.integer)) or isinstance(class_index, bool)
                           or not 0 <= class_index < len(bundle['classes'])):
        raise ValueError('Choose a valid output class.')
    index=int(class_index) if classification else 0
    probability=float(pipeline.predict_proba(row)[0,index]) if classification else None
    transformed=prep.transform(row); background=prep.transform(bundle['background'])
    names=prep.get_feature_names_out().tolist()
    if isinstance(model,(DummyClassifier,DummyRegressor)):
        return {'prediction':str(prediction),'probability':probability,'class':bundle['classes'][index] if classification else None,
                'contributions':[],'note':'Baseline model ignores all predictors; there is no feature attribution.'}
    if isinstance(model,(RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,DecisionTreeClassifier,DecisionTreeRegressor)):
        explanation=shap.TreeExplainer(model)(transformed)
        unit='probability' if classification else 'target units'
    elif isinstance(model,(KNeighborsClassifier,KNeighborsRegressor)):
        output = (lambda x: model.predict_proba(x)[:, index]) if classification else model.predict
        explanation=shap.PermutationExplainer(output, background, seed=42)(transformed, max_evals=max(2 * transformed.shape[1] + 1, 101))
        unit='probability' if classification else 'target units'
    else:
        explanation=shap.LinearExplainer(model,background)(transformed)
        unit=('log odds' if len(bundle['classes'])==2 else 'decision score') if classification else 'target units'
    values_shap=explanation.values[0]
    base=np.asarray(explanation.base_values)[0]
    if values_shap.ndim==2:
        values_shap=values_shap[:,index]; base=np.asarray(base)[index]
    elif classification and unit in ('log odds', 'decision score') and len(bundle['classes'])==2 and index==0:
        values_shap=-values_shap; base=-base
    contributions=[{'feature':name,'SHAP':float(v)} for name,v in zip(names,values_shap)]
    contributions.sort(key=lambda r:abs(r['SHAP']),reverse=True)
    total=float(base+sum(r['SHAP'] for r in contributions))
    if unit=='probability':
        expected=probability
    elif unit in ('log odds','decision score'):
        decision=np.asarray(model.decision_function(transformed))[0]
        expected=float(decision[index]) if np.ndim(decision) else float(decision)*(1 if index==1 else -1)
    else:
        expected=float(prediction)
    if not np.isclose(total,expected,rtol=1e-5,atol=1e-6):
        raise ValueError('The explanation failed its additivity check. The prediction was not attributed.')
    return {'prediction':str(prediction),'probability':probability,
            'class':bundle['classes'][index] if classification else None,
            'base_value':float(base),'reconstructed_output':total,'model_output':expected,
            'additivity_error':abs(total-expected),'unit':unit,'contributions':contributions,
            'note':'Real SHAP attribution for this model. Editing a feature is a model scenario, not a causal intervention.'}
