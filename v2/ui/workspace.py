import json
import pandas as pd
import streamlit as st
from v2.quality import profile
from v2.reporting import json_report, pdf_report, excel_report
from .common import header, navigate, failure, panel_title


def overview(ctx):
    title, action = st.columns([4, 1], vertical_alignment='center')
    with title:
        header('WORKSPACE / OVERVIEW', 'Analysis overview', 'Your dataset, recent work, and next investigation in one place.')
    with action:
        if st.button('New investigation', type='primary', icon=':material/add:', width='stretch'):
            navigate('Investigate')
    runs = ctx.store.list_runs(ctx.owner, ctx.dataset_id)
    missing = int(ctx.df.isna().sum().sum())
    complete = 1 - missing / max(ctx.df.size, 1)
    for col, label, value in zip(st.columns(4),
                                 ['Total records', 'Available fields', 'Cell completeness', 'Investigations'],
                                 [f'{len(ctx.df):,}', str(len(ctx.df.columns)), f'{complete:.1%}', str(len(runs))]):
        col.metric(label, value)
    st.write('')
    main, aside = st.columns([2.2, 1], gap='large')
    with main:
        with st.container(border=True):
            panel_title('search', 'Investigation activity', 'Saved work for the active dataset')
            if not runs:
                st.markdown('<div class="empty-state"><strong>No investigations yet</strong><p>Start with a comparison or relationship. DataCorn keeps the method and evidence with each result.</p></div>', unsafe_allow_html=True)
                if st.button('Start investigation', icon=':material/arrow_forward:'):
                    navigate('Investigate')
            for run in runs[:4]:
                text, button = st.columns([5, 1])
                text.write(run['title'])
                text.caption(run['created_at'][:16].replace('T', ' ') + ' UTC')
                if button.button('Open', key='recent_' + run['id']):
                    record = ctx.store.run(ctx.owner, run['id'])
                    st.session_state.update(active_run_id=run['id'], evidence=record['evidence'])
                    st.session_state.pop('plan', None); st.session_state.pop('exports', None)
                    navigate('Investigate')
            if runs and st.button('View all investigations', icon=':material/history:'):
                navigate('History')
        with st.container(border=True):
            panel_title('data', 'Dataset explorer', ctx.meta['name'])
            data, fields = st.tabs(['Records', 'Field summary'])
            with data:
                st.dataframe(ctx.df.head(50), hide_index=True, width='stretch', height=300)
                st.caption('Preview of the first 50 records')
            with fields:
                st.dataframe(pd.DataFrame(profile(ctx.df)['columns']), hide_index=True, width='stretch')
    with aside:
        with st.container(border=True):
            panel_title('data', 'Data readiness', 'Completeness and structure')
            st.progress(complete, text=f'{complete:.1%} of cells populated')
            st.caption(f'{missing:,} missing cells | {int(ctx.df.duplicated().sum()):,} duplicate rows')
            st.caption('Completeness does not guarantee data quality. Review feature timing and missing values.')
            if st.button('Review data health', icon=':material/fact_check:', width='stretch'):
                navigate('Data health')
        with st.container(border=True):
            panel_title('model', 'Model workspace', 'Train, compare, and explain')
            st.write('Build a baseline and compare validated models. Inspect individual predictions with SHAP.')
            if st.button('Open model lab', icon=':material/arrow_forward:', width='stretch'):
                navigate('Models')
        with st.expander('Dataset provenance'):
            st.caption('SHA-256 ' + ctx.meta['fingerprint'])
            st.caption('Saved evidence refers to this dataset fingerprint.')
    if ctx.settings.mode == 'demo':
        st.info('Synthetic demo workspace. Uploads are disabled in this mode.')


def data_health(ctx):
    header('WORKSPACE / DATA HEALTH', 'Data quality review',
           'Inspect structure and missingness. Leakage flags are prompts for domain review; they cannot establish when a feature became available.')
    names = ctx.df.columns.tolist()
    target = st.selectbox('Target for leakage review', names, index=names.index('churn') if 'churn' in names else 0)
    quality = profile(ctx.df, target)
    missing, duplicates, flagged = st.columns(3)
    missing.metric('Missing cells', f'{int(ctx.df.isna().sum().sum()):,}')
    duplicates.metric('Duplicate rows', f"{quality['duplicates']:,}")
    flagged.metric('Columns to review', len(quality['risks']))
    st.write('')
    flags, columns, preview = st.tabs(['Quality review', 'Column profile', 'Data preview'])
    with flags:
        panel_title('data', 'Feature review', 'Heuristics to guide your review, not a quality certification')
        if quality['risks']:
            rows = [{'Column': r['column'], 'Treatment': 'Exclude from model' if r['exclude'] else 'Review; retained', 'Reason': '; '.join(r['reasons'])} for r in quality['risks']]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        else:
            st.success("No heuristic leakage flags. Feature timing still needs a person's review.")
        if quality['duplicates']:
            st.warning(f"{quality['duplicates']} exact duplicate rows. Statistical analysis preserves them and discloses the independence concern; modeling deduplicates selected observations.")
        st.caption(quality['caveat'])
    with columns:
        st.dataframe(pd.DataFrame(quality['columns']), hide_index=True, width='stretch')
    with preview:
        st.dataframe(ctx.df.head(50), hide_index=True, width='stretch')
    if ctx.settings.mode == 'demo':
        st.download_button('Download synthetic CSV', ctx.df.to_csv(index=False), 'datacorn_demo.csv', 'text/csv', on_click='ignore')


def evidence(ctx):
    header('WORKSPACE / EVIDENCE', 'Evidence & exports',
           'Export the confirmed question, computed findings, independent challenge, provenance, and limitations. Raw dataset rows are omitted.')
    run_id = st.session_state.get('active_run_id')
    if not run_id:
        st.info('Open a saved investigation from History, or run a new one.')
        if st.button('Start an investigation'):
            navigate('Investigate')
        return
    record = ctx.store.run(ctx.owner, run_id)
    if record['evidence']['challenge'] is None:
        st.warning('This investigation has an initial finding but has not been challenged yet.')
        if st.button('Complete statistical review'):
            st.session_state['evidence'] = record['evidence']; navigate('Investigate')
        return
    model = st.session_state.get('model')
    include_model = st.checkbox('Include the currently opened model evaluation', value=bool(model), disabled=model is None)
    try:
        package = ctx.service.export_package(ctx.owner, run_id, model if include_model else None)
    except Exception as exc:
        failure(exc, 'Preparing evidence'); return
    review = package['challenge']
    summary, contents = st.columns([2, 1], gap='large')
    with summary:
        with st.container(border=True):
            panel_title('report', 'Decision brief', 'Question, review status, and recommended action')
            st.write(package['finding']['plan']['question'])
            st.info('Decision status: ' + review['verdict'])
            st.write(review['recommended_action'])
    with contents:
        with st.container(border=True):
            panel_title('report', 'Export contents')
            st.write('PDF / Business brief')
            st.write('Excel / Review workbook')
            st.write('JSON / Computed evidence')
            st.caption('Includes methods and limitations. Raw data rows are omitted.')
    if model and include_model:
        st.caption(f"Included model: {model['winner']} · target {model['target']} · {model['split']['strategy']} split")
    signature = run_id + ':' + (st.session_state.get('model_job_id', '') if include_model else '')
    if st.button('Prepare evidence downloads', type='primary', icon=':material/download:'):
        st.session_state.pop('exports', None)
        try:
            with st.spinner('Building evidence documents…'):
                exports = {'signature': signature, 'pdf': pdf_report(package), 'json': json_report(package), 'xlsx': excel_report(package)}
            st.session_state['exports'] = exports
        except Exception as exc:
            failure(exc, 'Report generation')
    exports = st.session_state.get('exports')
    if exports and exports['signature'] == signature:
        a, b, c = st.columns(3)
        a.download_button('Business brief · PDF', exports['pdf'], 'datacorn_evidence.pdf', 'application/pdf', on_click='ignore', width='stretch')
        b.download_button('Evidence trail · JSON', exports['json'], 'datacorn_evidence.json', 'application/json', on_click='ignore', width='stretch')
        c.download_button('Review workbook · Excel', exports['xlsx'], 'datacorn_evidence.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', on_click='ignore', width='stretch')
    with st.expander('Inspect the computed evidence'):
        st.json(json.loads(json_report(package)))


def history(ctx):
    header('WORKSPACE / HISTORY', 'Investigation history',
           'Reopen saved investigations and model results. Completed models survive an application restart when the same data volume is retained.')
    runs = ctx.store.list_runs(ctx.owner, ctx.dataset_id)
    if not runs:
        st.info('No investigations saved for this dataset yet.')
    search = st.text_input('Search investigations', placeholder='Filter by question...', key='history_search')
    filtered = [run for run in runs if search.casefold() in run['title'].casefold()]
    if runs and not filtered:
        st.info('No investigations match your search.')
    for run in filtered:
        with st.container(border=True):
            a, b = st.columns([5, 1])
            a.write('**' + run['title'] + '**')
            a.caption(run['created_at'][:19].replace('T', ' ') + ' UTC · ' + run['id'][:12])
            if b.button('Open', key='open_run_' + run['id']):
                record = ctx.store.run(ctx.owner, run['id'])
                st.session_state.update(active_run_id=run['id'], evidence=record['evidence'])
                st.session_state.pop('plan', None); st.session_state.pop('exports', None)
                navigate('Investigate')
    st.divider()
    with st.expander('Remove the active dataset and its saved work'):
        st.write('This removes the dataset, its investigations, training jobs, and model artifacts from this workspace. Download any evidence you want to keep first.')
        confirmed = st.checkbox('I understand this removes the active dataset and its saved work.', key='confirm_delete_dataset')
        if st.button('Delete dataset', disabled=not confirmed):
            try:
                ctx.store.delete_dataset(ctx.owner, ctx.dataset_id)
                st.session_state.pop('dataset_id', None)
                from .common import activate_dataset
                activate_dataset(None)
                navigate('Overview')
            except Exception as exc:
                failure(exc, 'Dataset removal')
