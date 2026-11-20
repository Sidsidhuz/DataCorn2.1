from dataclasses import dataclass
import html
import logging
import uuid
import pandas as pd
import streamlit as st
from v2.errors import ProductError

logger = logging.getLogger('autoinsight.ui')


@dataclass
class Context:
    settings: object
    store: object
    service: object
    runner: object
    owner: str
    dataset_id: str
    df: pd.DataFrame
    meta: dict


def failure(exc, context='Operation'):
    if isinstance(exc, ValueError):
        st.error(str(exc))
    else:
        reference = uuid.uuid4().hex[:10]
        logger.error('operation_failed reference=%s operation=%s exception_type=%s', reference, context, type(exc).__name__)
        st.error(f'{context} could not finish. Please retry. Reference: {reference}')


def navigate(page):
    st.session_state['navigate_to'] = page
    st.rerun()


def activate_dataset(identifier):
    for key in list(st.session_state):
        if key in {'dataset_choice', 'plan', 'active_run_id', 'evidence', 'active_job_id', 'model', 'model_job_id', 'scenario', 'exports'} or key.startswith(('question_', 'plan_', 'model_', 'what_')):
            del st.session_state[key]
    st.session_state['dataset_id'] = identifier


def render_theme():
    from pathlib import Path
    css = Path(__file__).with_name('theme.css').read_text(encoding='utf-8')
    st.markdown('<style>' + css + '</style>', unsafe_allow_html=True)


def workflow_steps(active):
    labels = ['Ask a question', 'Confirm the plan', 'Review the evidence']
    items = ''.join(
        f'<div class="step{ " active" if i == active else ""}"><b>{i + 1:02}</b>{label}</div>'
        for i, label in enumerate(labels)
    )
    st.markdown('<div class="steps" aria-label="Investigation steps">' + items + '</div>', unsafe_allow_html=True)


def icon(name):
    # Inline vector assets work offline; labels carry meaning for screen readers.
    paths = {
        'brand': '<path d="M5 5h8a7 7 0 0 1 0 14H5z"/><path d="M9 9h4a3 3 0 0 1 0 6H9z"/>',
        'search': '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5M7 10h6m-3-3v6"/>',
        'data': '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
        'model': '<path d="m12 3 9 5v8l-9 5-9-5V8zM3 8l9 5 9-5M12 13v8"/>',
        'report': '<path d="M6 3h9l4 4v14H6zM15 3v5h4M9 12h7M9 16h5"/>',
    }
    return '<svg class="dc-icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' + paths.get(name, paths['data']) + '</svg>'


def panel_title(symbol, title, detail=''):
    st.markdown('<div class="panel-heading">' + icon(symbol) + '<div><strong>' + html.escape(title) +
                '</strong><span>' + html.escape(detail) + '</span></div></div>', unsafe_allow_html=True)


def header(kicker, title, description):
    st.markdown(f'<p class="eyebrow">{html.escape(kicker)}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="intro">{html.escape(description)}</p>', unsafe_allow_html=True)


def display_finding(finding):
    st.subheader('What the data says')
    st.caption('Saved question: ' + finding['plan']['question'])
    st.write(finding['finding'])
    a, b, c = st.columns(3)
    a.metric('Observed effect', f"{finding['effect']:+.3f}")
    b.metric('Two-sided p-value', f"{finding['p_value']:.4g}")
    c.metric('Complete observations', f"{finding['n']:,}")
    st.caption(finding['test'] + ' · ' + finding['scope'])
    if finding.get('groups'):
        chart = pd.DataFrame(finding['groups']).set_index('group')
        st.bar_chart(chart['mean'], color='#5065d9')
        st.dataframe(chart, width='stretch', column_config={'mean': st.column_config.NumberColumn('Mean / observed rate', format='%.4f'), 'sd': st.column_config.NumberColumn('Std. deviation', format='%.4f')})
    if finding['ci95']:
        st.caption(f"95% interval [{finding['ci95'][0]:.4f}, {finding['ci95'][1]:.4f}] · {finding['excluded_rows']} incomplete rows excluded")


def display_review(review):
    st.subheader('Challenge My Finding')
    if review['reversal']:
        st.warning('CHALLENGED · The observed direction reverses within the tested segments.')
    else:
        st.info(review['verdict'] + ' · Review assumptions before making a decision.')
    st.write(review['recommended_action'])
    if review['strata']:
        strata = pd.DataFrame(review['strata'])
        st.bar_chart(strata.set_index('stratum')['effect'], color='#9b72cb')
        st.dataframe(strata, width='stretch', hide_index=True, column_config={k: st.column_config.NumberColumn(k.replace('_', ' ').title(), format='%.4f') for k in ['effect', 'p_value', 'holm_p']})
        st.caption('Same effect direction as the aggregate comparison. Holm adjustment covers these stratum tests only. A reversal is a confounding warning, not causal proof.')
    for check in review['checks']:
        st.markdown(f'<div class="check"><span class="tag">{html.escape(check["status"].upper())}</span><b>{html.escape(check["check"])}</b><p>{html.escape(check["detail"])}</p></div>', unsafe_allow_html=True)
    with st.expander('Assumptions, limitations, and skipped comparisons'):
        for line in review['limitations'] + review['skipped_strata']:
            st.write('• ' + line)
