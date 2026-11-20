import numpy as np
import pandas as pd
import streamlit as st
from v2.modeling import explain_scenario
from v2.quality import profile
from v2.training import MODEL_NAMES
from v2.artifacts import sio
from .common import header, failure, panel_title


def job_monitor(ctx, identifier):
    try:
        ctx.runner.recover()
        job = ctx.store.job(ctx.owner, identifier)
    except Exception as exc:
        failure(exc, 'Training status'); return
    if job['status'] in ('queued', 'running'):
        st.progress(float(job['progress']), text=job['message'])
        st.caption('You can move to another page while training continues. Cancellation takes effect between model fits.')
        if st.button('Cancel training', key='cancel_' + identifier):
            ctx.runner.cancel(ctx.owner, identifier)
        return
    if job['status'] == 'completed':
        if st.session_state.get('model_job_id') != identifier:
            try:
                st.session_state['model'] = ctx.runner.load(ctx.owner, identifier)
                st.session_state['model_job_id'] = identifier
                st.session_state.pop('scenario', None)
                st.rerun()
            except Exception as exc:
                failure(exc, 'Opening saved model')
    else:
        st.warning(job['message'])
    st.caption('Job ' + identifier[:12] + ' · ' + job['status'])


def render(ctx):
    header('MODEL LAB', 'Predictions you can inspect.',
           'Choose predictors and a validation strategy. Compare against a baseline, reserve a final holdout, then explain individual scenarios.')
    configure, evaluate = st.tabs(['01 / Configure training', '02 / Results & scenarios'])
    with configure:
        inputs, notes = st.columns([2, 1], gap='large')
        with notes:
            with st.container(border=True):
                panel_title('model', 'Validation protocol')
                st.write('1. Choose the target and predictors.')
                st.write('2. Match the split to the prediction use case.')
                st.write('3. Compare against a baseline.')
                st.write('4. Evaluate the winner on held-out data.')
                st.caption('Results and individual explanations are in the Results & scenarios tab.')
        with inputs:
            a, b = st.columns(2)
            names = ctx.df.columns.tolist()
            target = a.selectbox('Prediction target', names, index=names.index('churn') if 'churn' in names else len(names)-1, key='model_target')
            task = b.selectbox('Task', ['classification', 'regression'], key='model_task')
            health = profile(ctx.df, target)
            obvious = {r['column'] for r in health['risks'] if r['exclude']}
            features = st.multiselect('Candidate predictors', [c for c in names if c != target],
                                     default=[c for c in names if c != target and c not in obvious], key='model_features_' + target)
            split = st.selectbox('Validation strategy', ['random', 'group', 'temporal'],
                                 format_func=lambda v: {'random': 'Independent rows · random holdout', 'group': 'Repeated entities · group holdout', 'temporal': 'Time-dependent data · future holdout'}[v], key='model_split')
            split_column = st.selectbox('Group / timestamp column', [c for c in names if c != target], key='model_split_column') if split != 'random' else None
            available = MODEL_NAMES[task]
            algorithms = st.multiselect('Candidate models · baseline always included', available, default=available, key='model_algorithms_' + task)
            st.caption('Three training-only validation folds select the winner. Preprocessing is fitted within each fold. Group/time columns are excluded from predictors for those split strategies.')
            acknowledged = st.checkbox('I have reviewed feature availability and chosen a split that matches how predictions will be used.', key='model_acknowledged')
            if sio is None:
                st.warning('Model saving is unavailable because skops is missing from this Python environment. Install the pinned v2 requirements before training. Investigations and evidence reports remain available.')
            if st.button('Train & compare models', type='primary', disabled=not acknowledged or sio is None):
                try:
                    if not features or not algorithms:
                        raise ValueError('Choose at least one predictor and one model.')
                    config = {'target': target, 'task': task, 'features': sorted(features), 'algorithms': sorted(algorithms),
                              'split_strategy': split, 'split_column': split_column}
                    identifier = ctx.runner.submit(ctx.owner, ctx.dataset_id, config)
                    # Keep the opened model and exports if validation/submission fails.
                    for key in ('active_job_id', 'model', 'model_job_id', 'scenario', 'exports'):
                        st.session_state.pop(key, None)
                    st.session_state['active_job_id'] = identifier
                    st.success('Training submitted. Open Results & scenarios to follow progress and inspect the result.')
                except Exception as exc:
                    failure(exc, 'Training')
    with evaluate:
        if not st.session_state.get('active_job_id') and not st.session_state.get('model'):
            st.info('No model is open. Configure training or open a saved model below.')
        identifier = st.session_state.get('active_job_id')
        if identifier:
            active = ctx.store.job(ctx.owner, identifier)['status'] in ('queued', 'running')
            st.fragment(job_monitor, run_every=1 if active else None)(ctx, identifier)
        model = st.session_state.get('model')
        if model:
            display_model(ctx, model)
        st.divider()
        with st.expander('Saved models and training jobs'):
            jobs = ctx.store.list_jobs(ctx.owner, ctx.dataset_id)
            if not jobs:
                st.caption('No training jobs yet.')
            for job in jobs:
                a, b = st.columns([4, 1])
                a.caption(f"{job['created_at'][:16]} · {job['status']} · {job['message']}")
                if job['status'] == 'completed' and b.button('Open model', key='open_model_' + job['id']):
                    st.session_state['active_job_id'] = job['id']
                    st.session_state.pop('model_job_id', None)
                    st.session_state.pop('scenario', None)
                    st.rerun()


def display_model(ctx, model):
    st.divider()
    st.success('Selected by training validation: ' + model['winner'])
    st.caption(f"Target: {model['target']} · Task: {model['task']} · Split: {model['split']['strategy']}")
    st.dataframe(pd.DataFrame(model['leaderboard']), hide_index=True, width='stretch',
                 column_config={k: st.column_config.NumberColumn(k, format='%.4f') for k in ('cv_score', 'cv_std')})
    st.caption('Selection metric: ' + model['cv_metric'] + '. Higher scores are better; negative MAE is in target units.')
    st.markdown('**Final holdout · evaluated after selecting the winner**')
    cols = st.columns(len(model['holdout']))
    for col, (metric, value) in zip(cols, model['holdout'].items()):
        col.metric(metric.replace('_', ' ').title(), f'{value:.3f}')
    st.caption(f"Training: {model['train_rows']} · Holdout: {model['test_rows']} · Modeling duplicates removed: {model['duplicates_removed']} · Missing targets excluded: {model['missing_target_rows']}")
    if model['excluded_features']:
        st.warning('Training-only scan excluded: ' + ', '.join(r['column'] for r in model['excluded_features']))
    with st.expander('Validation details and limitations'):
        st.json(model['split'])
        for line in model['limitations']:
            st.write('• ' + line)
    st.subheader('Explore a what-if scenario')
    st.caption('Edit model inputs and compare with the original example. A scenario is not a causal intervention.')
    with st.form('scenario_form_' + st.session_state['model_job_id']):
        values = {}; columns = st.columns(2)
        for i, name in enumerate(model['features']):
            series = model['background'][name]; example = model['example'][name].iloc[0]
            key = 'what_' + st.session_state['model_job_id'] + '_' + name
            if pd.api.types.is_numeric_dtype(series):
                values[name] = columns[i % 2].number_input(name, value=float(example if pd.notna(example) else series.median()), key=key)
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if pd.notna(example) and str(example) not in options:
                    options.append(str(example))
                values[name] = columns[i % 2].selectbox(name, options, index=options.index(str(example)) if str(example) in options else 0, key=key)
        selected = st.selectbox('Explain class', model['classes'], index=min(1, len(model['classes'])-1)) if model['classes'] else None
        predict = st.form_submit_button('Predict scenario & show SHAP', type='primary')
    if predict:
        st.session_state.pop('scenario', None)
        try:
            with st.spinner('Computing prediction and real SHAP attribution…'):
                index = model['classes'].index(selected) if selected is not None else 0
                result = explain_scenario(model, values, index)
                baseline = explain_scenario(model, model['example'].iloc[0].to_dict(), index)
            st.session_state['scenario'] = {**result, 'baseline': baseline, 'inputs': values}
        except Exception as exc:
            failure(exc, 'Prediction explanation')
    scenario = st.session_state.get('scenario')
    if scenario:
        if scenario['probability'] is not None:
            delta = (scenario['probability'] - scenario['baseline']['probability']) * 100
            st.metric('Predicted probability · class ' + str(scenario['class']), f"{scenario['probability']:.1%}", f'{delta:+.1f} percentage points vs original')
        else:
            st.metric('Predicted outcome', scenario['prediction'])
        if scenario['contributions']:
            st.bar_chart(pd.DataFrame(scenario['contributions']).head(12).set_index('feature')['SHAP'], horizontal=True, color='#5065d9')
            st.caption(f"Units: {scenario['unit']} · Base + SHAP: {scenario['reconstructed_output']:.4f} · Additivity error: {scenario['additivity_error']:.2g}")
        st.caption(scenario['note'])
        for name, value in scenario['inputs'].items():
            series = model['background'][name]
            if isinstance(value, (int, float)) and (value < series.min() or value > series.max()):
                st.warning(name + ' is outside the reference sample range. Interpret cautiously.')
