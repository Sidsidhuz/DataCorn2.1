import pandas as pd
import streamlit as st
from v2.planner import propose, validate_plan
from .common import header, failure, display_finding, display_review, navigate, workflow_steps, panel_title


def render(ctx):
    header('WORKSPACE / INVESTIGATE', 'Investigation workspace',
           'Turn one business question into a reproducible comparison. Confirm the variables before running; challenge the result before acting.')
    steps = st.empty()
    editor, results = st.columns([1, 1.25], gap='large')
    with editor:
        panel_title('search', 'Question & method', 'Define the comparison you want to test')
        st.caption('Guided planning uses column-name matching. Review the suggested variables before running.')
        with st.form('business_question'):
            default_question = 'Does premium support increase customer churn?' if {'support_plan', 'churn'}.issubset(ctx.df.columns) else ''
            question = st.text_area('Business question', value=default_question, placeholder='What business relationship would you like to investigate?', max_chars=2000, height=90, key='question_text')
            proposed = st.form_submit_button('Propose investigation', type='primary')
        if proposed:
            for key in list(st.session_state):
                if key.startswith('plan_'):
                    del st.session_state[key]
            for key in ('plan', 'active_run_id', 'evidence', 'exports'):
                st.session_state.pop(key, None)
            try:
                st.session_state['plan'] = propose(question, ctx.df, use_llm=False)
                # Reset mounted browser widgets as well as Python session state.
                st.session_state['question_plan_revision'] = st.session_state.get('question_plan_revision', 0) + 1
            except Exception as exc:
                failure(exc, 'Planning')
        plan = st.session_state.get('plan')
        if plan:
            st.markdown('**Review the investigation plan**')
            st.caption(plan.get('reason', 'Verify the variables and assumptions.'))
            names = ctx.df.columns.tolist()
            revision = st.session_state.get('question_plan_revision', 0)
            # Inputs outside the submit form keep dependent column choices current.
            a = b = c = st.container()
            predictor = a.selectbox('Compare / predictor', names, index=names.index(plan['predictor']), key=f'plan_{revision}_predictor')
            outcome = b.selectbox('Outcome', [n for n in names if n != predictor],
                                  index=([n for n in names if n != predictor].index(plan['outcome']) if plan['outcome'] != predictor else 0), key=f'plan_{revision}_outcome')
            confs = ['None'] + [n for n in names if n not in (predictor, outcome)]
            confounder = c.selectbox('Challenge within segments', confs,
                                    index=confs.index(plan['confounder']) if plan.get('confounder') in confs else 0, key=f'plan_{revision}_confounder')
            a = b = st.container()
            alpha = a.selectbox('Significance threshold', [.05, .01, .10], format_func=lambda v: f'{v:.0%}', key=f'plan_{revision}_alpha')
            minimum = b.number_input('Minimum meaningful effect (absolute units)', min_value=0., value=0., help='Optional business threshold in mean-difference, correlation, or Cramer’s V units. Zero disables this check.', key=f'plan_{revision}_minimum')
            selected = validate_plan({**plan, 'predictor': predictor, 'outcome': outcome,
                                     'confounder': None if confounder == 'None' else confounder,
                                     'alpha': alpha, 'effect_threshold': minimum}, ctx.df)
            method = {'welch': 'Welch two-group mean comparison', 'spearman': 'Spearman rank correlation', 'chi_square': 'Categorical independence test'}[selected['method']]
            st.info(f'Planned method: {method}. Missing comparison rows are excluded and counted. Analysis assumes independent observations.')
            if st.button('Confirm plan & run analysis', type='primary'):
                for key in ('active_run_id', 'evidence', 'exports'):
                    st.session_state.pop(key, None)
                try:
                    with st.status('Computing the investigation', expanded=True) as status:
                        st.write('Validating selected columns and sample size')
                        identifier, evidence = ctx.service.start_investigation(ctx.owner, ctx.dataset_id, selected)
                        st.write('Statistical result and dataset fingerprint saved')
                        st.session_state.update(active_run_id=identifier, evidence=evidence)
                        status.update(label='Initial analysis saved', state='complete', expanded=False)
                except Exception as exc:
                    failure(exc, 'Investigation')
    with steps.container():
        workflow_steps(2 if st.session_state.get('evidence') else 1 if st.session_state.get('plan') else 0)
    with results:
        panel_title('report', 'Evidence canvas', 'Computed results appear here after analysis')
        evidence = st.session_state.get('evidence')
        if evidence:
            if plan:
                saved = evidence['finding']['plan']
                fields = ('predictor', 'outcome', 'confounder', 'alpha', 'effect_threshold')
                if any(selected.get(key) != saved.get(key) for key in fields):
                    st.warning('The analysis settings changed. Run the analysis again to review or challenge the updated comparison. Your previous investigation is still saved in History.')
                    return
            st.divider()
            display_finding(evidence['finding'])
            if plan:
                actual = evidence['finding']['plan']
                st.caption(f"Saved comparison: {actual['predictor']} → {actual['outcome']}. Changing controls does not change this saved result until you run again.")
            if evidence['challenge'] is None:
                st.markdown('**Could the conclusion be misleading?**')
                st.write('Run independent uncertainty, permutation, and segment checks. The final report requires this review.')
                if st.button('Challenge My Finding', type='primary'):
                    try:
                        with st.spinner('Running the statistical reviewer…'):
                            st.session_state['evidence'] = ctx.service.review_investigation(ctx.owner, st.session_state['active_run_id'])
                        st.rerun()
                    except Exception as exc:
                        failure(exc, 'Statistical review')
            else:
                display_review(evidence['challenge'])
                if st.button('Open evidence report'):
                    navigate('Evidence')
        else:
            st.markdown('<div class="empty-state canvas-empty"><strong>Your evidence starts here</strong><p>Propose a question, confirm the variables, and run the analysis. Results and independent review will appear in this panel.</p></div>', unsafe_allow_html=True)
