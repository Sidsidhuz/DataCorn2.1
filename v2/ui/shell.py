"""Authentication, workspace selection, navigation, and runtime composition."""
import logging
import html
import secrets
import time
from pathlib import Path
import base64
import streamlit as st
from v2.auth import authenticate, bootstrap_from_environment
from v2.config import Settings
from v2.jobs import JobRunner
from v2.service import WorkspaceService
from v2.storage import Store
from .common import Context, activate_dataset, failure, render_theme, navigate, icon
from . import investigate, models, workspace

PAGES = ['Overview', 'Investigate', 'Data health', 'Models', 'Evidence', 'History']
LOGO = Path(__file__).resolve().parents[1] / 'assets' / 'datacorn-logo.png'


def render_logo(sign_in=False):
    encoded = base64.b64encode(LOGO.read_bytes()).decode('ascii')
    style = 'product-logo sign-in-logo' if sign_in else 'product-logo'
    st.markdown(f'<div class="{style}"><img src="data:image/png;base64,{encoded}" alt="DataCorn — blue and teal unicorn analytics logo"></div>', unsafe_allow_html=True)


@st.cache_resource
def runtime(data_dir, workers, timeout):
    store = Store(data_dir)
    return store, JobRunner(store, workers, timeout)


def identity(settings, store):
    if settings.mode == 'private':
        bootstrap_from_environment(store)
        auth = st.session_state.get('identity')
        if auth and auth['expires_at'] <= time.time():
            st.session_state.clear(); auth = None
        if not auth:
            render_logo(sign_in=True)
            st.markdown('<p class="eyebrow">PRIVATE WORKSPACE</p>', unsafe_allow_html=True)
            st.title('Welcome to DataCorn.')
            st.write('Sign in to open your datasets, investigations, and models.')
            with store.connect() as db:
                provisioned = db.execute('SELECT count(*) FROM users').fetchone()[0]
            if not provisioned:
                st.error('Private workspace setup is incomplete. The administrator must provision an account before this deployment can be used.')
                st.stop()
            with st.form('sign_in'):
                username = st.text_input('Username', max_chars=64)
                password = st.text_input('Password', type='password', max_chars=1024)
                submit = st.form_submit_button('Sign in', type='primary')
            if submit:
                try:
                    auth = authenticate(store, username, password)
                    st.session_state.clear(); st.session_state['identity'] = auth
                    st.rerun()
                except Exception as exc:
                    failure(exc, 'Sign-in')
            st.stop()
        return auth['id'], auth['username']
    if settings.mode == 'local':
        return 'local', 'Local analyst'
    if 'demo_identity' not in st.session_state:
        st.session_state['demo_identity'] = 'demo:' + secrets.token_hex(16)
    return st.session_state['demo_identity'], 'Demo visitor'


def render():
    product_logger = logging.getLogger('autoinsight')
    if not product_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        product_logger.addHandler(handler)
        product_logger.setLevel(logging.INFO)
        product_logger.propagate = False
    st.set_page_config(page_title='DataCorn | Evidence Studio', page_icon=':material/analytics:', layout='wide')
    render_theme()
    try:
        settings = Settings.from_env()
        store, runner = runtime(str(settings.data_dir), settings.max_workers, settings.job_timeout)
        owner, display_name = identity(settings, store)
    except Exception as exc:
        failure(exc, 'Workspace startup'); st.stop()
    if st.session_state.get('_owner') != owner:
        for key in list(st.session_state):
            if key not in ('identity', 'demo_identity'):
                del st.session_state[key]
        st.session_state['_owner'] = owner
    service = WorkspaceService(store, settings)
    datasets = store.list_datasets(owner)
    available_ids = {d['id'] for d in datasets}
    if st.session_state.get('dataset_id') not in available_ids:
        identifier = datasets[0]['id'] if datasets else service.demo(owner)
        activate_dataset(identifier)
        datasets = store.list_datasets(owner)
    if 'navigate_to' in st.session_state:
        st.session_state['page'] = st.session_state.pop('navigate_to')
    page = st.session_state.get('page', 'Overview')
    if page not in PAGES:
        page = 'Overview'
        st.session_state['page'] = page
    with st.sidebar:
        render_logo()
        st.caption('WORKSPACE')
        nav_icons = ['dashboard', 'search', 'table_view', 'model_training', 'description', 'history']
        for destination, symbol in zip(PAGES, nav_icons):
            if st.button(destination, key='nav_' + destination, icon=f':material/{symbol}:',
                         type='primary' if page == destination else 'secondary', width='stretch'):
                navigate(destination)
        st.divider()
        st.markdown('<div class="rail-note">EVIDENCE FIRST<br><span>Investigate. Validate. Explain.</span></div>', unsafe_allow_html=True)
        st.caption(display_name)
        st.caption(settings.mode.title() + ' workspace | v2.1')
        if settings.mode == 'private' and st.button('Sign out', icon=':material/logout:', width='stretch'):
            st.session_state.clear(); st.rerun()
    with st.container(key='dataset_toolbar'):
        selector, management = st.columns([4, 1], vertical_alignment='bottom')
        mapping = {d['id']: d['name'] for d in datasets}
        current = st.session_state['dataset_id']
        with selector:
            chosen = st.selectbox('Active dataset', list(mapping), index=list(mapping).index(current),
                                  format_func=lambda value: mapping[value], key='dataset_choice')
            if chosen != current:
                activate_dataset(chosen); st.rerun()
        with management:
            with st.popover('Manage data', icon=':material/database:', width='stretch'):
                choices = ['Saved datasets', 'Guided demo'] + (['Upload dataset'] if settings.uploads_allowed else [])
                source = st.selectbox('Dataset source', choices, key='dataset_source')
                if source == 'Upload dataset':
                    upload = st.file_uploader('CSV or XLSX | up to 10 MB', type=['csv', 'xlsx'])
                    if st.button('Import dataset', disabled=upload is None, type='primary', icon=':material/upload:'):
                        try:
                            identifier = service.import_dataset(owner, upload.getvalue(), upload.name)
                            activate_dataset(identifier); st.rerun()
                        except Exception as exc:
                            failure(exc, 'Dataset import')
                elif source == 'Guided demo':
                    st.caption('600 synthetic customers with an aggregate comparison that reverses within segments.')
                    if st.button('Load guided demo'):
                        activate_dataset(service.demo(owner)); st.rerun()
                else:
                    st.caption('Select a dataset in the toolbar. Saved investigations are available in History.')
                st.caption('Data stays in this workspace. No external AI requests are made by guided planning.')
    try:
        frame, meta = store.dataset(owner, st.session_state['dataset_id'])
        ctx = Context(settings, store, service, runner, owner, meta['id'], frame, meta)
        st.markdown(f'<div class="workspace-bar"><span>WORKSPACE / <strong>{html.escape(page)}</strong></span><span>{html.escape(meta["name"])} &nbsp; · &nbsp; {len(frame):,} rows · {len(frame.columns)} columns</span><span class="status-pill">{html.escape(settings.mode.title())} workspace</span></div>', unsafe_allow_html=True)
        if page == 'Overview':
            workspace.overview(ctx)
        elif page == 'Investigate':
            investigate.render(ctx)
        elif page == 'Data health':
            workspace.data_health(ctx)
        elif page == 'Models':
            models.render(ctx)
        elif page == 'Evidence':
            workspace.evidence(ctx)
        else:
            workspace.history(ctx)
        st.divider()
        st.caption('DataCorn · Every finding has a method, an evidence trail, and limits. Statistical association does not establish causation.')
    except Exception as exc:
        failure(exc, 'Workspace')
