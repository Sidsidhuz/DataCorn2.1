from __future__ import annotations

import time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── App Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataCorn — Local AutoML & Explainable AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/api/v1"

# ── Dark Glassmorphism CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* ── Base: Dark background ────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
    .stApp, .main, section.main {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #0f1117 !important;
        color: #f1f5f9 !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #13151f !important;
        border-right: 1px solid rgba(124,58,237,0.2) !important;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #94a3b8 !important; }

    /* ── Headings ─────────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; font-weight: 700 !important; }
    p, span, label, li, td, th, div { color: #e2e8f0; }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li { color: #e2e8f0 !important; }

    /* ── Form labels ──────────────────────────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label,
    .stTextInput label, .stNumberInput label,
    .stTextArea label, .stDateInput label,
    .stSlider label, .stCheckbox label,
    .stRadio label, .stFileUploader label { color: #cbd5e1 !important; font-weight: 600 !important; }

    [data-testid="stRadio"] label span,
    [data-testid="stCheckbox"] label span,
    .stRadio > div > label,
    .stCheckbox > label { color: #cbd5e1 !important; }

    /* ── Dropdowns / Selects (closed state) ─────────────────────────────── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        color: #f1f5f9 !important;
        background-color: #1e2130 !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div { color: #f1f5f9 !important; background-color: #1e2130 !important; }

    /* ── Dropdown popup / popover (the open menu) ────────────────────────── */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    ul[data-baseweb="menu"] {
        background-color: #1a1d2e !important;
        border: 1px solid rgba(124,58,237,0.35) !important;
        border-radius: 12px !important;
        box-shadow: 0 16px 48px rgba(0,0,0,0.6) !important;
    }
    /* Individual option items */
    [data-baseweb="option"],
    [role="option"],
    li[role="option"] {
        background-color: #1a1d2e !important;
        color: #e2e8f0 !important;
    }
    [data-baseweb="option"]:hover,
    [role="option"]:hover,
    li[role="option"]:hover,
    [aria-selected="true"][role="option"] {
        background-color: rgba(124,58,237,0.2) !important;
        color: #f1f5f9 !important;
    }
    /* Listbox container */
    [role="listbox"] {
        background-color: #1a1d2e !important;
    }
    /* Select value text */
    [data-baseweb="select"] [data-baseweb="tag"],
    [data-baseweb="select"] input { color: #f1f5f9 !important; background-color: transparent !important; }

    /* ── Streamlit Header / Toolbar (top-right area) ─────────────────────── */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stApp > header {
        background-color: #0f1117 !important;
        border-bottom: 1px solid rgba(124,58,237,0.15) !important;
    }
    /* Toolbar icons and buttons */
    [data-testid="stToolbar"],
    [data-testid="stToolbar"] * {
        color: #94a3b8 !important;
        fill: #94a3b8 !important;
    }
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] a {
        color: #94a3b8 !important;
        background: transparent !important;
    }
    [data-testid="stToolbar"] button:hover {
        color: #a78bfa !important;
        background: rgba(124,58,237,0.1) !important;
    }
    /* Top decoration stripe */
    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
        height: 3px !important;
    }
    /* Deploy button specifically */
    [data-testid="stDeployButton"],
    [data-testid="stDeployButton"] * {
        color: #94a3b8 !important;
        background-color: rgba(124,58,237,0.1) !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
        border-radius: 8px !important;
    }

    /* ── Input fields ─────────────────────────────────────────────────────── */
    input, textarea, select {
        color: #f1f5f9 !important;
        background-color: #1e2130 !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #1e2130 !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
        color: #f1f5f9 !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] button[role="tab"] {
        color: #64748b !important; font-weight: 600; background: transparent !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #7c3aed !important; border-bottom: 2px solid #7c3aed;
    }

    /* ── Metrics ──────────────────────────────────────────────────────────── */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #94a3b8 !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #a78bfa !important; }
    [data-testid="stMetricDelta"] { color: #22c55e !important; }

    /* ── Dataframe ────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] * { color: #e2e8f0 !important; background-color: #1a1d2e !important; }
    .dataframe th, .dataframe td { color: #e2e8f0 !important; background-color: #1a1d2e !important; }

    /* ── Alert boxes ──────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        background-color: #1e2130 !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
        color: #e2e8f0 !important;
    }

    /* ── Spinner ──────────────────────────────────────────────────────────── */
    [data-testid="stStatusWidget"] * { color: #e2e8f0 !important; }

    /* ══════════════════════════════════════════════════════════════════════
       Premium Dark Glassmorphism Components
    ══════════════════════════════════════════════════════════════════════ */

    .main-title {
        background: linear-gradient(135deg, #a78bfa, #7c3aed, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        line-height: 1.1;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ── Metric Cards ─────────────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .metric-card:hover {
        border-color: rgba(124,58,237,0.4);
        box-shadow: 0 8px 32px rgba(124,58,237,0.12);
        transform: translateY(-2px);
    }
    .metric-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.72rem;
        color: #64748b !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 2rem;
        color: #f1f5f9 !important;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    /* ── Glass Cards ──────────────────────────────────────────────────────── */
    .custom-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-title-bar {
        display: inline-block;
        width: 4px; height: 18px;
        background: linear-gradient(180deg, #7c3aed, #3b82f6);
        border-radius: 2px;
        margin-right: 8px;
        vertical-align: middle;
    }

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
    }
    .stButton > button p { color: #ffffff !important; }

    /* ── Status Badges ────────────────────────────────────────────────────── */
    .badge {
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: inline-block;
    }
    .badge-success { background: rgba(34,197,94,0.15); color: #4ade80 !important; border: 1px solid rgba(34,197,94,0.25); }
    .badge-info    { background: rgba(59,130,246,0.15); color: #60a5fa !important; border: 1px solid rgba(59,130,246,0.25); }
    .badge-warning { background: rgba(234,179,8,0.15);  color: #fbbf24 !important; border: 1px solid rgba(234,179,8,0.25); }
    .badge-danger  { background: rgba(239,68,68,0.15);  color: #f87171 !important; border: 1px solid rgba(239,68,68,0.25); }

    /* ── Progress bar ─────────────────────────────────────────────────────── */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
    }

    /* ── File uploader ────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzone"] span { color: #cbd5e1 !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(124,58,237,0.05) !important;
        border: 2px dashed rgba(124,58,237,0.4) !important;
        border-radius: 14px !important;
    }

    /* ── Algorithm Toggle Cards ───────────────────────────────────────────── */
    .alg-grid { display: flex; flex-direction: column; gap: 10px; }
    .alg-card {
        display: flex; align-items: center; gap: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        transition: all 0.2s ease;
    }
    .alg-card.selected {
        background: rgba(124,58,237,0.12);
        border-color: rgba(124,58,237,0.5);
        box-shadow: 0 0 20px rgba(124,58,237,0.12);
    }
    .alg-card-icon {
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
    }
    .alg-card-name { font-weight: 700; font-size: 0.88rem; color: #f1f5f9 !important; }
    .alg-card-desc { font-size: 0.72rem; color: #64748b !important; margin-top: 2px; }

    /* ── Rank labels ──────────────────────────────────────────────────────── */
    .rank-1 { color: #fbbf24; font-weight: 800; }
    .rank-2 { color: #94a3b8; font-weight: 800; }
    .rank-3 { color: #cd7c54; font-weight: 800; }
    .rank-n { color: #475569; font-weight: 600; }

    /* ── Prediction result card ───────────────────────────────────────────── */
    .pred-result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .pred-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
        color: #94a3b8 !important;
    }
    .pred-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .pred-yes  { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); }
    .pred-yes .pred-value { color: #f87171 !important; }
    .pred-no   { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); }
    .pred-no .pred-value { color: #4ade80 !important; }
    .pred-num  { background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3); }
    .pred-num .pred-value { color: #a78bfa !important; }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper Functions ───────────────────────────────────────────────────────────
def api_request(method, endpoint, **kwargs):
    try:
        url = f"{BACKEND_URL}{endpoint}"
        r = requests.request(method, url, **kwargs)
        if r.status_code in [200, 201, 202]:
            if "application/json" in r.headers.get("Content-Type", ""):
                return r.json()
            return r.content
        else:
            try:
                err = r.json()
                st.error(f"API Error: {err.get('message', r.text)}")
            except Exception:
                st.error(f"API Error ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend API: {str(e)}")
        return None


def dark_plotly_layout(**kwargs):
    """Return common dark-theme Plotly layout kwargs."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    base.update(kwargs)
    return base


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="padding: 1.5rem 1rem 1rem 1rem; text-align: center;">
        <div style="font-size: 2.2rem; margin-bottom: 6px;">🔮</div>
        <div style="font-size: 1.3rem; font-weight: 800;
                    background: linear-gradient(135deg, #a78bfa, #7c3aed);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">
            DataCorn
        </div>
        <div style="font-size: 0.68rem; color: #475569; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;">
            Local AutoML Platform
        </div>
    </div>
    <hr style="border: none; border-top: 1px solid rgba(124,58,237,0.2); margin: 0 1rem 1rem 1rem;"/>
    """,
    unsafe_allow_html=True,
)

# Liveness Check
try:
    health = requests.get("http://localhost:8000/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.markdown(
            '<div style="text-align:center; margin-bottom:1rem;">'
            '<span class="badge badge-success">● Backend Online</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="text-align:center; margin-bottom:1rem;">'
            '<span class="badge badge-danger">● Backend Offline</span></div>',
            unsafe_allow_html=True,
        )
except Exception:
    st.sidebar.markdown(
        '<div style="text-align:center; margin-bottom:1rem;">'
        '<span class="badge badge-danger">● Backend Offline</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.warning("FastAPI backend is not running. Start it with: `uvicorn app.main:app --reload`")

NAV_ICONS = {
    "Dashboard & History":        "🏠",
    "Upload & Preview":           "📂",
    "Data Cleaning & EDA":        "🧼",
    "AutoML Training Console":    "⚙️",
    "Leaderboard & Explanations": "🏆",
    "Serve Predictions":          "🎯",
    "Export Center":              "📦",
}

menu = st.sidebar.radio(
    "Navigation",
    list(NAV_ICONS.keys()),
    format_func=lambda x: f"{NAV_ICONS[x]}  {x}",
)




# Fetch datasets history globally
datasets = api_request("GET", "/datasets")
datasets_list = datasets if datasets else []

# ── PAGE: Dashboard & History ──────────────────────────────────────────────────
if menu == "Dashboard & History":
    st.markdown("<h1 class='main-title'>DataCorn Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Your local AutoML & Explainable AI workspace — zero cloud, zero cost.</p>", unsafe_allow_html=True)

    # ── Metric Cards Row ──
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, "📂", "background:rgba(124,58,237,0.15);", "Datasets Uploaded",  str(len(datasets_list))),
        (col2, "⚙️", "background:rgba(59,130,246,0.15);",  "AutoML Runs",         str(len(datasets_list))),
        (col3, "🤖", "background:rgba(16,185,129,0.15);",  "Active Algorithms",   "5"),
        (col4, "🏠", "background:rgba(234,179,8,0.15);",   "Deployment Target",   "Local"),
    ]
    for col, icon, icon_bg, title, value in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon" style="{icon_bg}">{icon}</div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── History Table ──
    st.markdown(
        '<div class="custom-card">'
        '<div class="card-title"><span class="card-title-bar"></span>Datasets & Job History</div>',
        unsafe_allow_html=True,
    )

    if not datasets_list:
        st.info("No datasets uploaded yet. Head over to 'Upload & Preview' to get started!")
    else:
        history_data = []
        for d in datasets_list:
            history_data.append({
                "ID":          d["id"],
                "Filename":    d["filename"],
                "Rows":        d["rows"] or "Pending",
                "Columns":     d["columns"] or "Pending",
                "Status":      d["status"],
                "Uploaded At": pd.to_datetime(d["uploaded_at"]).strftime("%Y-%m-%d %H:%M"),
            })
        # Build custom HTML table (st.dataframe canvas ignores dark theme CSS)
        STATUS_COLORS = {
            "cleaned":    ("#22c55e", "rgba(34,197,94,0.12)"),
            "profiled":   ("#60a5fa", "rgba(59,130,246,0.12)"),
            "uploaded":   ("#fbbf24", "rgba(234,179,8,0.12)"),
            "failed":     ("#f87171", "rgba(239,68,68,0.12)"),
        }
        rows_html = ""
        for d in history_data:
            status = d["Status"]
            txt_col, bg_col = STATUS_COLORS.get(status, ("#94a3b8", "rgba(148,163,184,0.12)"))
            badge = (
                f'<span style="background:{bg_col};color:{txt_col};border:1px solid {txt_col}33;'
                f'padding:2px 10px;border-radius:9999px;font-size:0.72rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.06em;">{status}</span>'
            )
            rows_html += (
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<td style="padding:12px 16px;color:#94a3b8;font-size:0.8rem;">{d["ID"]}</td>'
                f'<td style="padding:12px 16px;color:#f1f5f9;font-weight:600;">{d["Filename"]}</td>'
                f'<td style="padding:12px 16px;color:#e2e8f0;">{d["Rows"]}</td>'
                f'<td style="padding:12px 16px;color:#e2e8f0;">{d["Columns"]}</td>'
                f'<td style="padding:12px 16px;">{badge}</td>'
                f'<td style="padding:12px 16px;color:#64748b;font-size:0.85rem;">{d["Uploaded At"]}</td>'
                f'</tr>'
            )
        st.markdown(
            f"""
            <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:2px solid rgba(124,58,237,0.3);">
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">ID</th>
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Filename</th>
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Rows</th>
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Columns</th>
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Status</th>
                        <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Uploaded At</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Upload & Preview ─────────────────────────────────────────────────────
elif menu == "Upload & Preview":
    st.markdown("<h1 class='main-title'>Upload & Profile Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Upload raw data, check structural integrity, and view statistical profiles.</p>", unsafe_allow_html=True)

    st.markdown(
        '<div class="custom-card">'
        '<div class="card-title"><span class="card-title-bar"></span>Upload CSV or Excel File</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if st.button("⬆️  Upload & Process"):
            with st.spinner("Uploading and analyzing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = api_request("POST", "/datasets", files=files)
                if res:
                    st.success(f"Successfully processed '{res['filename']}'! Assigned ID: {res['id']}.")
                    st.balloons()
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if datasets_list:
        st.markdown(
            '<div class="custom-card">'
            '<div class="card-title"><span class="card-title-bar"></span>Dataset Profile Viewer</div>',
            unsafe_allow_html=True,
        )
        d_options = {
            f"{d['filename']} (ID: {d['id']})": d["id"]
            for d in datasets_list
            if d["status"] not in ["failed", "uploaded"]
        }

        if not d_options:
            st.info("No validated datasets available. Please upload a dataset first.")
        else:
            selected_d_id_key = st.selectbox("Select dataset to view profile", list(d_options.keys()))
            d_id = d_options[selected_d_id_key]

            profile = api_request("GET", f"/datasets/{d_id}/profile")
            if profile:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows",            profile["row_count"])
                m2.metric("Columns",         profile["column_count"])
                m3.metric("Duplicate Rows",  profile["duplicates"])
                m4.metric("Avg Missingness", f"{np.mean([float(x.replace('%','')) for x in profile['missing_values'].values()]):.2f}%")

                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("### Column Schema & Statistics")

                stats_rows = []
                dtypes   = profile["dtypes"]
                missing  = profile["missing_values"]
                outliers = profile["outlier_counts"]
                summary  = profile["summary_stats"]

                for col in dtypes.keys():
                    col_stats = summary.get(col, {})
                    stats_rows.append({
                        "Column Name":  col,
                        "Data Type":    dtypes[col],
                        "Missing %":    missing[col],
                        "Outliers":     outliers.get(col, 0),
                        "Unique Count": col_stats.get("unique_count", "N/A"),
                        "Mean":   f"{col_stats['mean']:.4f}"   if "mean"   in col_stats else "N/A",
                        "Median": f"{col_stats['median']:.4f}" if "median" in col_stats else "N/A",
                        "Min":    f"{col_stats['min']:.4f}"    if "min"    in col_stats else "N/A",
                        "Max":    f"{col_stats['max']:.4f}"    if "max"    in col_stats else "N/A",
                    })
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Data Cleaning & EDA ──────────────────────────────────────────────────
elif menu == "Data Cleaning & EDA":
    st.markdown("<h1 class='main-title'>Data Cleaning & Exploratory Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Configure feature imputation, deduplicate rows, and explore correlations.</p>", unsafe_allow_html=True)

    d_options = {
        f"{d['filename']} (ID: {d['id']})": d
        for d in datasets_list
        if d["status"] not in ["failed", "uploaded"]
    }

    if not d_options:
        st.info("Please upload and validate a dataset first.")
    else:
        selected_d = st.selectbox("Select dataset to configure and clean", list(d_options.keys()))
        dataset = d_options[selected_d]
        d_id = dataset["id"]

        st.markdown(
            '<div class="custom-card">'
            '<div class="card-title"><span class="card-title-bar"></span>Data Imputation Configuration</div>',
            unsafe_allow_html=True,
        )
        profile = api_request("GET", f"/datasets/{d_id}/profile")

        if profile:
            dtypes      = profile["dtypes"]
            missing_vals = profile["missing_values"]
            cols_with_missing = [c for c, pct in missing_vals.items() if float(pct.replace("%", "")) > 0.0]
            imputation_overrides = {}

            if not cols_with_missing:
                st.success("No missing values detected! You can proceed directly to clean the dataset.")
            else:
                st.write("Configure imputation strategy for columns containing missing values:")
                for col in cols_with_missing:
                    col_type    = dtypes[col]
                    col_missing = missing_vals[col]
                    if col_type == "numeric":
                        strategy = st.selectbox(
                            f"Column '{col}' ({col_type}, {col_missing} missing)",
                            ["median", "mean", "constant"],
                            key=f"impute_{col}",
                        )
                    else:
                        strategy = st.selectbox(
                            f"Column '{col}' ({col_type}, {col_missing} missing)",
                            ["mode", "constant"],
                            key=f"impute_{col}",
                        )
                    imputation_overrides[col] = strategy

            if st.button("🧼  Apply Cleaning & Deduplication"):
                with st.spinner("Executing cleaning pipeline..."):
                    res = api_request("POST", f"/datasets/{d_id}/clean", json=imputation_overrides)
                    if res:
                        st.success("Dataset cleaned and deduplicated successfully!")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # EDA Visualizations
        if dataset["status"] == "cleaned":
            st.markdown(
                '<div class="custom-card">'
                '<div class="card-title"><span class="card-title-bar"></span>Interactive EDA Visualizations</div>',
                unsafe_allow_html=True,
            )
            eda_data = api_request("GET", f"/datasets/{d_id}/eda")

            if eda_data:
                tab1, tab2 = st.tabs(["📊  Correlation Matrix", "📈  Column Distributions"])

                with tab1:
                    corr = eda_data["correlation"]
                    if not corr["columns"]:
                        st.info("No numeric features to correlate.")
                    else:
                        fig = go.Figure(data=go.Heatmap(
                            z=corr["matrix"],
                            x=corr["columns"],
                            y=corr["columns"],
                            colorscale="RdBu",
                            zmin=-1, zmax=1,
                            text=np.round(corr["matrix"], 2),
                            texttemplate="%{text}",
                            colorbar=dict(tickfont=dict(color="#e2e8f0")),
                        ))
                        fig.update_layout(**dark_plotly_layout(height=520, title="Pearson Correlation Matrix"))
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    dist = eda_data["distributions"]
                    col_to_plot = st.selectbox("Select column to view distribution", list(dist.keys()))
                    if col_to_plot:
                        col_dist = dist[col_to_plot]
                        if col_dist["type"] == "numeric":
                            bin_edges  = col_dist["bin_edges"]
                            counts     = col_dist["counts"]
                            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
                            widths      = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges) - 1)]
                            fig = go.Figure(data=[go.Bar(
                                x=bin_centers, y=counts, width=widths,
                                marker=dict(
                                    color=counts,
                                    colorscale=[[0, "#3b82f6"], [1, "#7c3aed"]],
                                    line=dict(color="rgba(0,0,0,0)"),
                                ),
                            )])
                            fig.update_layout(**dark_plotly_layout(
                                title=f"Distribution of '{col_to_plot}'",
                                xaxis_title=col_to_plot, yaxis_title="Count",
                            ))
                        else:
                            fig = go.Figure(data=[go.Bar(
                                x=col_dist["categories"], y=col_dist["counts"],
                                marker_color="#7c3aed", opacity=0.85,
                            )])
                            fig.update_layout(**dark_plotly_layout(
                                title=f"Value Counts of '{col_to_plot}'",
                                xaxis_title=col_to_plot, yaxis_title="Count",
                            ))
                        st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: AutoML Training Console ─────────────────────────────────────────────
elif menu == "AutoML Training Console":
    st.markdown("<h1 class='main-title'>AutoML Training Console</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Define your modeling target, select candidate algorithms, and trigger hyperparameter searches.</p>", unsafe_allow_html=True)

    d_options = {
        f"{d['filename']} (ID: {d['id']})": d
        for d in datasets_list
        if d["status"] == "cleaned"
    }

    if not d_options:
        st.info("No cleaned datasets available. Please upload, validate, and clean a dataset first.")
    else:
        selected_d = st.selectbox("Select dataset for AutoML training", list(d_options.keys()))
        dataset = d_options[selected_d]
        d_id = dataset["id"]

        profile = api_request("GET", f"/datasets/{d_id}/profile")

        if profile:
            dtypes = profile["dtypes"]

            st.markdown(
                '<div class="custom-card">'
                '<div class="card-title"><span class="card-title-bar"></span>Job Parameters</div>',
                unsafe_allow_html=True,
            )

            target_column = st.selectbox("🎯  Target variable (column to predict)", list(dtypes.keys()))

            inferred_type = "classification"
            if (
                dtypes[target_column] == "numeric"
                and profile["summary_stats"].get(target_column, {}).get("unique_count", 0) > 20
            ):
                inferred_type = "regression"

            problem_type = st.selectbox(
                "Task Type",
                ["auto", "classification", "regression"],
                index=0,
                help=f"Currently inferred: {inferred_type.upper()}",
            )

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── Algorithm selector — visual dark cards ──
            ALG_META = {
                "logistic_regression": ("📊", "Logistic Regression",   "Fast linear baseline for classification",          "background:rgba(59,130,246,0.15);"),
                "linear_regression":   ("📈", "Linear Regression",     "Fast linear baseline for regression",               "background:rgba(16,185,129,0.15);"),
                "random_forest":       ("🌲", "Random Forest",         "Ensemble of decision trees, robust & interpretable","background:rgba(34,197,94,0.15);"),
                "xgboost":             ("⚡", "XGBoost",               "Gradient boosting, winner of many Kaggle comps",   "background:rgba(234,179,8,0.15);"),
                "lightgbm":            ("🚀", "LightGBM",              "High-speed gradient boosting by Microsoft",         "background:rgba(239,68,68,0.15);"),
                "catboost":            ("🐱", "CatBoost",              "Gradient boosting handling categoricals natively",  "background:rgba(124,58,237,0.15);"),
            }

            st.markdown(
                '<div style="font-size:0.8rem; font-weight:700; color:#94a3b8; '
                'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem;">'
                'Select algorithms to benchmark</div>',
                unsafe_allow_html=True,
            )

            selected_algs = []
            alg_keys = list(ALG_META.keys())
            col_a, col_b = st.columns(2)

            for idx, alg in enumerate(alg_keys):
                icon, name, desc, icon_bg = ALG_META[alg]
                col_container = col_a if idx % 2 == 0 else col_b
                with col_container:
                    checked = st.checkbox(f"**{name}**", value=True, key=f"check_{alg}")
                    if checked:
                        selected_algs.append(alg)
                    card_class = "alg-card selected" if checked else "alg-card"
                    st.markdown(
                        f"""
                        <div class="{card_class}" style="margin-top:-0.5rem; margin-bottom:0.5rem; pointer-events:none;">
                            <div class="alg-card-icon" style="{icon_bg}">{icon}</div>
                            <div>
                                <div class="alg-card-name">{name}</div>
                                <div class="alg-card-desc">{desc}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br/>", unsafe_allow_html=True)
            tuning_budget_seconds = st.number_input(
                "⏱  Tuning budget per algorithm (seconds)",
                min_value=10, max_value=300, value=60,
            )

            if st.button("🚀  Start AutoML Training"):
                payload = {
                    "target_column":         target_column,
                    "problem_type":          problem_type if problem_type != "auto" else None,
                    "algorithms":            selected_algs,
                    "tuning_budget_seconds": tuning_budget_seconds,
                }
                with st.spinner("Submitting training job..."):
                    res = api_request("POST", f"/datasets/{d_id}/train", json=payload)
                    if res:
                        st.success(f"Job queued! Assigned Job ID: {res['id']}.")
                        st.session_state["active_job_id"]        = res["id"]
                        st.session_state["training_start_time"]  = time.time()
                        st.session_state["training_algorithms"]  = selected_algs
                        st.session_state["training_budget"]      = tuning_budget_seconds

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Animated Progress Tracker ──
            if "active_job_id" in st.session_state:
                job_id       = st.session_state["active_job_id"]
                algorithms   = st.session_state.get("training_algorithms", [])
                budget_secs  = st.session_state.get("training_budget", 60)
                start_time   = st.session_state.get("training_start_time", time.time())

                n_algs             = max(len(algorithms), 1)
                total_est_secs     = n_algs * budget_secs + 10

                ALG_DISPLAY = {
                    "logistic_regression": "Logistic Regression",
                    "linear_regression":   "Linear Regression",
                    "random_forest":       "Random Forest",
                    "xgboost":             "XGBoost",
                    "lightgbm":            "LightGBM",
                    "catboost":            "CatBoost",
                }

                progress_placeholder = st.empty()

                while True:
                    job_status = api_request("GET", f"/training-jobs/{job_id}")
                    if not job_status:
                        break

                    status  = job_status["status"]
                    elapsed = time.time() - start_time

                    if status == "queued":
                        pct           = 0.04
                        current_phase = "Queuing job…"
                    elif status == "running":
                        raw       = 0.05 + (elapsed / total_est_secs) * 0.90
                        overshoot = max(0, elapsed - total_est_secs) * 0.002
                        pct       = min(raw + overshoot, 0.99)
                        alg_idx   = min(int((elapsed / total_est_secs) * n_algs), n_algs - 1)
                        alg_name  = ALG_DISPLAY.get(algorithms[alg_idx], algorithms[alg_idx]) if algorithms else "models"
                        current_phase = f"Training <b>{alg_name}</b> (algorithm {alg_idx + 1} of {n_algs})"
                    elif status == "completed":
                        pct           = 1.0
                        current_phase = "✅ All done!"
                    elif status == "failed":
                        pct           = 1.0
                        current_phase = "❌ Training failed"
                    else:
                        pct           = 0.0
                        current_phase = status

                    remaining = total_est_secs - elapsed
                    if remaining > 0:
                        mins, secs = divmod(int(remaining), 60)
                        eta_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                    else:
                        eta_str = None

                    steps = ["📂 Preparing data"] + [
                        f"🤖 {ALG_DISPLAY.get(a, a)}" for a in algorithms
                    ] + ["🏁 Finalizing results"]

                    alg_idx_running = int((elapsed / total_est_secs) * n_algs) if status == "running" else -1
                    step_html = ""
                    for i, step in enumerate(steps):
                        if status == "completed":
                            icon, color = "✅", "#22c55e"
                        elif i == 0 and status in ("queued", "running"):
                            icon, color = "✅", "#22c55e"
                        elif status == "running" and 0 < i <= n_algs:
                            alg_step_idx = i - 1
                            if alg_step_idx < alg_idx_running:
                                icon, color = "✅", "#22c55e"
                            elif alg_step_idx == alg_idx_running:
                                icon, color = "⚙️", "#7c3aed"
                            else:
                                icon, color = "⏳", "#475569"
                        elif i == n_algs + 1 and status == "completed":
                            icon, color = "✅", "#22c55e"
                        else:
                            icon, color = "⏳", "#475569"
                        step_html += (
                            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                            f'<span style="font-size:16px;">{icon}</span>'
                            f'<span style="color:{color};font-size:13px;font-weight:500;">{step}</span>'
                            f'</div>'
                        )

                    bar_color = "#ef4444" if status == "failed" else "#7c3aed"
                    bar_pct   = int(pct * 100)
                    eta_display = (
                        f"⏱ Est. remaining: <b>{eta_str}</b>" if (status == "running" and eta_str)
                        else ("⏱ <b>Wrapping up…</b>" if status == "running" else "")
                    )

                    progress_placeholder.markdown(
                        f"""
                        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                                    border-radius:20px;padding:24px;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
                            <div style="font-size:17px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">
                                🔬 AutoML Training Pipeline
                            </div>
                            <div style="font-size:12px;color:#64748b;margin-bottom:20px;">
                                Job ID: <code style="color:#a78bfa;">{job_id}</code>&nbsp;|&nbsp;{eta_display}
                            </div>
                            <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:12px;overflow:hidden;margin-bottom:8px;">
                                <div style="background:linear-gradient(90deg,{bar_color},{bar_color}88);
                                            width:{bar_pct}%;height:100%;border-radius:999px;
                                            transition:width 1.5s ease-in-out;"></div>
                            </div>
                            <div style="display:flex;justify-content:space-between;font-size:12px;color:#64748b;margin-bottom:20px;">
                                <span>{current_phase}</span>
                                <span style="font-weight:700;color:#a78bfa;">{bar_pct}%</span>
                            </div>
                            <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:16px;">
                                <div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                                            letter-spacing:0.08em;margin-bottom:10px;">Pipeline Steps</div>
                                {step_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if status == "completed":
                        time.sleep(0.5)
                        st.success("🎉 AutoML training completed! Go to **Leaderboard & Explanations** to review results.")
                        st.balloons()
                        for k in ["active_job_id", "training_start_time", "training_algorithms", "training_budget"]:
                            del st.session_state[k]
                        break
                    elif status == "failed":
                        st.error(f"AutoML job failed: {job_status.get('error_message')}")
                        del st.session_state["active_job_id"]
                        break

                    time.sleep(2)


# ── PAGE: Leaderboard & Explanations ──────────────────────────────────────────
elif menu == "Leaderboard & Explanations":
    st.markdown("<h1 class='main-title'>Model Leaderboard & Explainable AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Compare candidate algorithms and visualise feature importances via SHAP.</p>", unsafe_allow_html=True)

    d_completed = {
        f"{d['filename']} (ID: {d['id']})": d["id"]
        for d in datasets_list
        if d["status"] == "cleaned"
    }

    if not d_completed:
        st.info("No AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset to view leaderboard", list(d_completed.keys()))
        d_id = d_completed[selected_d]

        all_jobs   = api_request("GET", f"/training-jobs?dataset_id={d_id}") or []
        latest_job = next(
            (j for j in all_jobs if j["status"] in ["completed", "failed", "running", "queued"]),
            None,
        )

        if not latest_job:
            st.warning("No training job found for this dataset.")
        else:
            st.markdown(
                '<div class="custom-card">'
                '<div class="card-title"><span class="card-title-bar"></span>AutoML Performance Leaderboard</div>',
                unsafe_allow_html=True,
            )
            st.write(
                f"Target: **{latest_job['target_column']}** &nbsp;|&nbsp; "
                f"Task: **{latest_job['problem_type'].upper()}**"
            )

            leaderboard = latest_job["leaderboard"]
            if not leaderboard:
                st.info("No models trained successfully for this job.")
            else:
                primary_metric = "F1-Score" if latest_job["problem_type"] == "classification" else "R²"
                rank_labels    = {1: "🥇", 2: "🥈", 3: "🥉"}

                # Build ranked rows
                rows = []
                for rank, item in enumerate(leaderboard, start=1):
                    medal = rank_labels.get(rank, f"#{rank}")
                    rows.append({
                        "Rank":         f"{medal}",
                        "Algorithm":    item["algorithm"].replace("_", " ").title(),
                        primary_metric: round(item["primary_metric_value"], 4),
                        "Status":       "🏆 BEST" if item["is_best"] else "Candidate",
                        "model_id":     item["model_id"],
                    })

                # Custom HTML leaderboard table (st.dataframe ignores dark theme CSS)
                leader_rows_html = ""
                for r in rows:
                    is_best_row = r["Status"] == "🏆 BEST"
                    row_bg = "background:rgba(34,197,94,0.06);" if is_best_row else ""
                    status_html = (
                        '<span style="background:rgba(34,197,94,0.15);color:#4ade80;'
                        'border:1px solid rgba(34,197,94,0.3);padding:2px 10px;border-radius:9999px;'
                        'font-size:0.72rem;font-weight:700;">🏆 BEST</span>'
                        if is_best_row else
                        '<span style="color:#475569;font-size:0.85rem;">Candidate</span>'
                    )
                    leader_rows_html += (
                        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);{row_bg}">'
                        f'<td style="padding:14px 16px;font-size:1.1rem;">{r["Rank"]}</td>'
                        f'<td style="padding:14px 16px;color:#f1f5f9;font-weight:600;">{r["Algorithm"]}</td>'
                        f'<td style="padding:14px 16px;color:#a78bfa;font-weight:700;font-size:1rem;">{r[primary_metric]}</td>'
                        f'<td style="padding:14px 16px;">{status_html}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f"""
                    <div style="overflow-x:auto;margin-bottom:1rem;">
                    <table style="width:100%;border-collapse:collapse;">
                        <thead>
                            <tr style="border-bottom:2px solid rgba(124,58,237,0.35);">
                                <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Rank</th>
                                <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Algorithm</th>
                                <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">{primary_metric}</th>
                                <th style="padding:10px 16px;color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;text-align:left;">Status</th>
                            </tr>
                        </thead>
                        <tbody>{leader_rows_html}</tbody>
                    </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Bar Chart — dark Plotly ──
                st.markdown("<br/>", unsafe_allow_html=True)
                alg_names  = [r["Algorithm"] for r in rows]
                metric_vals = [r[primary_metric] for r in rows]
                is_best    = [r["Status"] == "🏆 BEST" for r in rows]

                bar_colors = [
                    "#22c55e" if best else "#7c3aed" for best in is_best
                ]
                fig = go.Figure(data=[go.Bar(
                    x=alg_names,
                    y=metric_vals,
                    marker=dict(
                        color=bar_colors,
                        line=dict(color="rgba(0,0,0,0)"),
                    ),
                    text=[f"{v:.4f}" for v in metric_vals],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=12, family="Plus Jakarta Sans"),
                )])
                fig.update_layout(**dark_plotly_layout(
                    title=f"Algorithm Comparison — {primary_metric}",
                    yaxis_title=primary_metric,
                    height=400,
                    showlegend=False,
                ))
                fig.update_traces(
                    marker_line_width=0,
                    selector=dict(type="bar"),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ── SHAP Explainability ──
            if leaderboard:
                st.markdown(
                    '<div class="custom-card">'
                    '<div class="card-title"><span class="card-title-bar"></span>Global Feature Importance (SHAP)</div>',
                    unsafe_allow_html=True,
                )
                import os
                project_root      = Path(__file__).resolve().parent.parent
                shap_summary_path = project_root / f"data/plots/{d_id}/shap_summary.png"

                if shap_summary_path.exists():
                    st.image(
                        str(shap_summary_path),
                        caption="Global SHAP Feature Importance Summary",
                        use_container_width=True,
                    )
                else:
                    st.info("SHAP Global Summary plot not found. Try generating a report to trigger it.")
                st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Serve Predictions ────────────────────────────────────────────────────
elif menu == "Serve Predictions":
    st.markdown("<h1 class='main-title'>Model Serving & Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Enter raw feature values to get instant predictions with local SHAP explanations.</p>", unsafe_allow_html=True)

    d_completed = {
        f"{d['filename']} (ID: {d['id']})": d["id"]
        for d in datasets_list
        if d["status"] == "cleaned"
    }

    if not d_completed:
        st.info("No completed AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset/model to serve predictions", list(d_completed.keys()))
        d_id       = d_completed[selected_d]

        model_options = {}
        all_jobs      = api_request("GET", f"/training-jobs?dataset_id={d_id}") or []
        for job_info in all_jobs:
            for m in job_info.get("leaderboard", []):
                model_options[
                    f"{m['algorithm'].replace('_', ' ').title()} (Model ID: {m['model_id']})"
                ] = m["model_id"]

        if not model_options:
            st.warning("No trained models found for this dataset.")
        else:
            selected_model = st.selectbox("Select trained model candidate", list(model_options.keys()))
            model_id       = model_options[selected_model]

            profile = api_request("GET", f"/datasets/{d_id}/profile")

            if profile:
                dtypes     = profile["dtypes"]
                summary    = profile["summary_stats"]
                model_meta = api_request("GET", f"/models/{model_id}")
                job_id     = model_meta["training_job_id"]
                job_info   = api_request("GET", f"/training-jobs/{job_id}")
                target_col = job_info["target_column"]

                st.markdown(
                    '<div class="custom-card">'
                    '<div class="card-title"><span class="card-title-bar"></span>Single Instance Prediction Console</div>',
                    unsafe_allow_html=True,
                )

                input_data = {}
                cols       = st.columns(2)

                for idx, (col, col_type) in enumerate(dtypes.items()):
                    if col == target_col:
                        continue
                    col_stats     = summary.get(col, {})
                    col_container = cols[idx % 2]

                    with col_container:
                        if col_type == "numeric":
                            input_data[col] = st.number_input(
                                f"{col}",
                                min_value=float(col_stats.get("min", 0.0)),
                                max_value=float(col_stats.get("max", 100.0)),
                                value=float(col_stats.get("mean", 50.0)),
                                key=f"pred_{col}",
                            )
                        elif col_type == "boolean":
                            input_data[col] = st.selectbox(f"{col}", [True, False], key=f"pred_{col}")
                        elif col_type == "datetime":
                            input_data[col] = st.date_input(
                                f"{col}", value=pd.Timestamp.now(), key=f"pred_{col}"
                            ).strftime("%Y-%m-%d")
                        else:
                            input_data[col] = st.text_input(
                                f"{col}", value=str(col_stats.get("mode", "")), key=f"pred_{col}"
                            )

                st.markdown("<br/>", unsafe_allow_html=True)

                if st.button("🔮  Generate Prediction"):
                    with st.spinner("Executing inference and calculating SHAP values..."):
                        payload = {"input_data": input_data}
                        res = api_request("POST", f"/models/{model_id}/predict", json=payload)

                        if res:
                            pred = res["prediction"]
                            prob = res["probability"]

                            res_c1, res_c2 = st.columns([1, 2])

                            with res_c1:
                                # ── Prediction Result Card ──
                                pred_str = str(pred).upper()
                                if pred_str in ("TRUE", "YES", "1", "CHURN", "POSITIVE"):
                                    card_cls = "pred-yes"
                                elif pred_str in ("FALSE", "NO", "0", "NO CHURN", "NEGATIVE"):
                                    card_cls = "pred-no"
                                else:
                                    card_cls = "pred-num"

                                st.markdown(
                                    f"""
                                    <div class="pred-result-card {card_cls}">
                                        <div class="pred-label">Prediction</div>
                                        <div class="pred-value">{pred}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                # ── Confidence Bar ──
                                if prob is not None:
                                    conf_pct = int(prob * 100)
                                    bar_col  = "#ef4444" if conf_pct >= 50 else "#22c55e"
                                    st.markdown(
                                        f"""
                                        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                                                    border-radius:16px;padding:1.25rem;margin-top:0.5rem;">
                                            <div style="font-size:0.72rem;font-weight:700;color:#64748b;
                                                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">
                                                Confidence
                                            </div>
                                            <div style="font-size:2rem;font-weight:800;color:#f1f5f9;
                                                        margin-bottom:0.75rem;">{conf_pct}%</div>
                                            <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:8px;overflow:hidden;">
                                                <div style="background:{bar_col};width:{conf_pct}%;height:100%;
                                                            border-radius:999px;transition:width 0.6s ease;"></div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                            with res_c2:
                                st.markdown("**Local SHAP Explanation**")
                                expl = res["explanation"]
                                if expl and expl["plot_path"]:
                                    project_root = Path(__file__).resolve().parent.parent
                                    plot_path    = project_root / f"data/{expl['plot_path']}"
                                    if plot_path.exists():
                                        st.image(str(plot_path), use_container_width=True)
                                    else:
                                        shap_vals = expl["shap_values"]
                                        features  = [x["feature"]      for x in shap_vals][:12]
                                        contribs  = [x["contribution"] for x in shap_vals][:12]
                                        colors    = ["#ef4444" if c >= 0 else "#3b82f6" for c in contribs]
                                        fig = go.Figure(go.Bar(
                                            x=contribs, y=features,
                                            orientation="h",
                                            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
                                            text=[f"{c:+.3f}" for c in contribs],
                                            textposition="outside",
                                            textfont=dict(color="#e2e8f0", size=11),
                                        ))
                                        fig.update_layout(**dark_plotly_layout(
                                            title="Feature Impact (SHAP Values)",
                                            xaxis_title="SHAP Value",
                                            yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.06)"),
                                            height=420,
                                        ))
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("SHAP explanation not available.")

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Batch Predictions ──
                st.markdown(
                    '<div class="custom-card">'
                    '<div class="card-title"><span class="card-title-bar"></span>Batch Predictions</div>',
                    unsafe_allow_html=True,
                )
                batch_file = st.file_uploader(
                    "Upload batch CSV (must match the same feature columns)", type=["csv"], key="batch_upload"
                )
                if batch_file is not None:
                    if st.button("📊  Generate Batch Predictions"):
                        with st.spinner("Computing batch predictions..."):
                            files     = {"file": (batch_file.name, batch_file.getvalue(), "text/csv")}
                            res_bytes = api_request("POST", f"/models/{model_id}/predict-batch", files=files)
                            if res_bytes:
                                st.success("Batch predictions completed!")
                                st.download_button(
                                    label="⬇️  Download Predictions CSV",
                                    data=res_bytes,
                                    file_name="batch_predictions.csv",
                                    mime="text/csv",
                                )
                st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Export Center ────────────────────────────────────────────────────────
elif menu == "Export Center":
    st.markdown("<h1 class='main-title'>Export Center</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Compile and download executive PDF summaries and multi-sheet Excel reports.</p>", unsafe_allow_html=True)

    d_completed = {
        f"{d['filename']} (ID: {d['id']})": d["id"]
        for d in datasets_list
        if d["status"] == "cleaned"
    }

    if not d_completed:
        st.info("No completed AutoML runs found. Complete training first!")
    else:
        selected_d = st.selectbox("Select dataset for report generation", list(d_completed.keys()))
        d_id = d_completed[selected_d]

        col_pdf, col_xlsx = st.columns(2)

        # ── PDF Report ──
        with col_pdf:
            st.markdown(
                '<div class="custom-card" style="height:100%;">'
                '<div class="card-title"><span class="card-title-bar"></span>📄 Executive PDF Report</div>'
                '<p style="color:#64748b;font-size:0.88rem;margin-bottom:1.5rem;">'
                'Statistical profiling tables, correlation heatmaps, model leaderboards, and global SHAP summary charts.'
                '</p>',
                unsafe_allow_html=True,
            )
            if st.button("Generate PDF Report", key="btn_pdf"):
                with st.spinner("Assembling PDF document..."):
                    res = api_request("POST", f"/datasets/{d_id}/reports", json={"report_type": "pdf"})
                    if res:
                        report_id   = res["id"]
                        status_info = None
                        while True:
                            status_info = api_request("GET", f"/reports/{report_id}")
                            if not status_info:
                                break
                            if status_info["status"] == "completed":
                                break
                            elif status_info["status"] == "failed":
                                st.error("PDF generation failed.")
                                break
                            time.sleep(1)
                        if status_info and status_info["status"] == "completed":
                            st.success("PDF report generated!")
                            file_bytes = api_request("GET", f"/reports/{report_id}/download")
                            if file_bytes:
                                st.download_button(
                                    label="⬇️  Download PDF",
                                    data=file_bytes,
                                    file_name="autoinsight_summary.pdf",
                                    mime="application/pdf",
                                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Excel Workbook ──
        with col_xlsx:
            st.markdown(
                '<div class="custom-card" style="height:100%;">'
                '<div class="card-title"><span class="card-title-bar"></span>📊 Excel Workbook</div>'
                '<p style="color:#64748b;font-size:0.88rem;margin-bottom:1.5rem;">'
                'Structured metadata, model comparison parameters, and a cleaned data sheet (up to 1000 rows).'
                '</p>',
                unsafe_allow_html=True,
            )
            if st.button("Generate Excel Report", key="btn_xlsx"):
                with st.spinner("Compiling Excel sheets..."):
                    res = api_request("POST", f"/datasets/{d_id}/reports", json={"report_type": "excel"})
                    if res:
                        report_id   = res["id"]
                        status_info = None
                        while True:
                            status_info = api_request("GET", f"/reports/{report_id}")
                            if not status_info:
                                break
                            if status_info["status"] == "completed":
                                break
                            elif status_info["status"] == "failed":
                                st.error("Excel generation failed.")
                                break
                            time.sleep(1)
                        if status_info and status_info["status"] == "completed":
                            st.success("Excel report generated!")
                            file_bytes = api_request("GET", f"/reports/{report_id}/download")
                            if file_bytes:
                                st.download_button(
                                    label="⬇️  Download Excel",
                                    data=file_bytes,
                                    file_name="autoinsight_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
            st.markdown("</div>", unsafe_allow_html=True)
