"""
PediaSed - Decision Support
Pediatric dental general anesthesia assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PediaSed - Decision Support",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS - Dark sidebar, card layout, clinical aesthetic
# ============================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f4f8 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b2d 0%, #1a2940 100%) !important;
    color: white !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #c8d6e5 !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #c8d6e5 !important;
}

/* ── Hide branding without removing the header that owns sidebar controls ── */
#MainMenu, footer, .stDeployButton {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Streamlit owns the sidebar state and responsive breakpoint. Keep its native
   controls usable after a resize or browser zoom collapses the sidebar. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    pointer-events: auto !important;
}

/* When the sidebar is folded, keep its reopen control anchored to the viewport
   rather than aligned with the centered main-content container. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    position: fixed !important;
    top: 0.75rem !important;
    left: 0.75rem !important;
    z-index: 1001 !important;
}

/* ── Main container padding ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px !important;
}

/* ── Card styling ── */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e8edf2;
    margin-bottom: 16px;
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.card-header .card-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.card-header .card-title {
    font-size: 17px;
    font-weight: 700;
    color: #1a2940;
    margin: 0;
}
.card-header .card-subtitle {
    font-size: 12px;
    color: #7f8c9b;
    margin: 0;
}

/* ── Step indicators ── */
.step-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 24px;
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e8edf2;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
}
.step-num {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    flex-shrink: 0;
}
.step-num.active {
    background: #2563eb;
    color: white;
}
.step-num.inactive {
    background: #e8edf2;
    color: #7f8c9b;
}
.step-label {
    font-size: 14px;
    font-weight: 600;
    color: #1a2940;
}
.step-sublabel {
    font-size: 11px;
    color: #7f8c9b;
}
.step-connector {
    flex: 0.3;
    height: 2px;
    background: #e8edf2;
    margin: 0 8px;
}

/* ── Section labels ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.section-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #2563eb;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
}
.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #1a2940;
}

/* ── Field labels ── */
.field-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 6px;
}
.field-icon {
    font-size: 15px;
}

/* ── Prediction panel ── */
.prediction-panel {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    border: 1px solid #e8edf2;
}
.prediction-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid #f0f4f8;
}
.prediction-header-icon {
    font-size: 18px;
    color: #f59e0b;
}
.prediction-header-text {
    font-size: 16px;
    font-weight: 700;
    color: #1a2940;
}

/* ── Donut chart (CSS) ── */
.donut-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 16px 0 20px 0;
}
.donut {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.donut-inner {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: white;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.donut-value {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
}
.donut-label {
    font-size: 11px;
    color: #7f8c9b;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ── Recommendation box ── */
.rec-box {
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.rec-box.ga {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fecaca;
}
.rec-box.n2o {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
}
.rec-box.local {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
}
.rec-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.rec-icon.ga { background: #fecaca; }
.rec-icon.n2o { background: #fde68a; }
.rec-icon.local { background: #a7f3d0; }
.rec-title {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.3;
}
.rec-subtitle {
    font-size: 11px;
    color: #6b7280;
    margin-top: 2px;
}

/* ── Contributing factors ── */
.factors-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f0f4f8;
}
.factors-title {
    font-size: 14px;
    font-weight: 700;
    color: #1a2940;
}
.factor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f8fafc;
}
.factor-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.factor-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.factor-text {
    font-size: 13px;
    color: #374151;
}
.factor-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
}
.badge-very-high {
    background: #fef2f2;
    color: #dc2626;
}
.badge-high {
    background: #fff7ed;
    color: #ea580c;
}
.badge-moderate {
    background: #fffbeb;
    color: #d97706;
}
.badge-low {
    background: #ecfdf5;
    color: #059669;
}
.badge-minimal {
    background: #f0f4f8;
    color: #6b7280;
}

/* ── Probability breakdown bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-label {
    font-size: 13px;
    font-weight: 600;
    color: #4a5568;
    min-width: 85px;
}
.prob-bar-bg {
    flex: 1;
    height: 24px;
    background: #f0f4f8;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding-left: 10px;
    font-size: 12px;
    font-weight: 700;
    color: white;
    transition: width 0.5s ease;
    min-width: fit-content;
}

/* ── Info box ── */
.info-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: #0369a1;
    margin-top: 16px;
}

/* ── Footer disclaimer ── */
.disclaimer {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: #065f46;
    margin-top: 16px;
}

/* ── Hide Streamlit elements ── */

/* ── Make selectbox/inputs cleaner ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #1a2940 !important;
    background: #f8fafc !important;
    border-radius: 10px !important;
}

/* ── Toggle ── */
.stToggle label span {
    font-size: 14px !important;
}

/* ── Radio buttons as pill toggles ── */
.stRadio > div {
    gap: 0 !important;
}
.stRadio > div[role="radiogroup"] {
    display: flex !important;
    gap: 0 !important;
    background: #f0f4f8;
    border-radius: 10px;
    padding: 3px;
}
.stRadio > div[role="radiogroup"] label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    transition: all 0.2s ease !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
}
.stRadio > div[role="radiogroup"] label[data-checked="true"],
.stRadio > div[role="radiogroup"] label:has(input:checked) {
    background: #2563eb !important;
    color: white !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.3) !important;
}
/* Hide the default radio dot */
.stRadio > div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Reference-design layout overrides
st.markdown("""
<style>
.block-container {
    max-width: 1500px !important;
    padding: .55rem 1.5rem .45rem !important;
}
.stMainBlockContainer > div[data-testid="stVerticalBlock"] {
    gap: .45rem !important;
}
section[data-testid="stSidebar"] {
    min-width: 225px !important;
    max-width: 225px !important;
    height: 100vh !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] > div {
    height: 100vh !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 0 13px !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0 13px 76px !important;
}
.sidebar-brand {
    text-align: center;
    padding: 0 0 13px;
    transform: translateY(-12px);
}
.sidebar-tooth {
    font-size: 54px;
    line-height: 1;
    margin-bottom: 8px;
}
.sidebar-name {
    color: #fff !important;
    font-size: 26px;
    font-weight: 800;
    line-height: 1.1;
}
.sidebar-tagline {
    color: #b7c8dc !important;
    font-size: 13px;
    margin-top: 5px;
}
section[data-testid="stSidebar"] hr {
    margin: 8px 0 12px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
    padding: 0 !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100% !important;
    min-height: 35px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    color: #c8d6e5 !important;
    background: transparent !important;
    font-size: 10px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    color: #fff !important;
    background: #1267e8 !important;
    box-shadow: 0 4px 12px rgba(18, 103, 232, .32) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: inherit !important;
    font-size: 14px !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    position: absolute !important;
    opacity: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] label:has(input[type="radio"]) > div:first-child,
section[data-testid="stSidebar"] input[type="radio"] + div {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p::before {
    content: "i";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    width: 14px;
    height: 14px;
    flex: 0 0 14px;
    border: 1px solid #9fb2c8;
    border-radius: 50%;
    font-family: Arial, sans-serif;
    font-size: 9px;
    font-weight: 700;
    line-height: 1;
    color: #9fb2c8;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:first-of-type p::before {
    content: "+";
    border-color: #fff;
    background: #fff;
    color: #1267e8;
    font-size: 12px;
    font-weight: 800;
}
.sidebar-assist {
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 8px;
    padding: 12px 10px;
    margin-top: 14px;
}
.sidebar-assist-title {
    color: #fff !important;
    font-size: 14px;
    font-weight: 750;
}
.sidebar-assist-copy {
    color: #b8c7da !important;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 5px;
}
.st-key-nav_new button,
.st-key-nav_info button {
    width: 100% !important;
    min-height: 35px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 8px 10px !important;
    border: 0 !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: #c8d6e5 !important;
    box-shadow: none !important;
}
.st-key-nav_new button p,
.st-key-nav_info button p {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    color: inherit !important;
    font-size: 14px !important;
    white-space: nowrap !important;
}
.st-key-nav_new button p::before,
.st-key-nav_info button p::before {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    width: 14px;
    height: 14px;
    flex: 0 0 14px;
    border: 1px solid #9fb2c8;
    border-radius: 50%;
    color: #9fb2c8;
    font-family: Arial, sans-serif;
    font-size: 9px;
    font-weight: 700;
    line-height: 1;
}
.st-key-nav_new button p::before {
    content: "+";
    font-size: 12px;
}
.st-key-nav_info button p::before {
    content: "i";
}
.st-key-nav_new button:hover,
.st-key-nav_info button:hover {
    background: rgba(255,255,255,.06) !important;
    color: #fff !important;
}
.sidebar-profile {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 225px;
    min-height: 68px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 13px;
    border-top: 1px solid rgba(255,255,255,.12);
    background: #10233f;
}
.profile-avatar {
    width: 27px;
    height: 27px;
    flex: 0 0 27px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.2);
    color: #dbe7f4 !important;
    font-size: 12px;
}
.profile-copy { flex: 1; min-width: 0; }
.profile-name {
    color: #fff !important;
    font-size: 11px;
    font-weight: 650;
    white-space: nowrap;
}
.profile-mode {
    color: #8ea2ba !important;
    font-size: 10px;
    margin-top: 2px;
}
.profile-chevron {
    color: #b9c8d9 !important;
    font-size: 11px;
}
.app-heading {
    font-size: 24px;
    line-height: 1.15;
    font-weight: 800;
    color: #10233f;
    margin: 0 0 3px;
}
.app-subheading {
    color: #718096;
    font-size: 11px;
    margin-bottom: 7px;
}
.st-key-reset_assessment [data-testid="stBaseButton-secondary"] {
    position: relative !important;
    z-index: 20 !important;
    width: 100% !important;
    height: 42px;
    white-space: nowrap !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    background: #fff !important;
    border: 1px solid #dfe6ee !important;
    color: #24364b !important;
    box-shadow: 0 2px 8px rgba(15, 35, 63, .04);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}
.st-key-reset_assessment [data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
    border-color: #93c5fd !important;
    background: #fff !important;
    color: #1267e8 !important;
    box-shadow: 0 7px 16px rgba(15, 35, 63, .15) !important;
}
.st-key-reset_assessment [data-testid="stBaseButton-secondary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px rgba(15, 35, 63, .1) !important;
}
.st-key-reset_assessment [data-testid="stBaseButton-secondary"] p {
    white-space: nowrap !important;
    font-size: 11px !important;
    pointer-events: none !important;
}
.st-key-reset_assessment [data-testid="stBaseButton-secondary"] * {
    pointer-events: none !important;
}
.workflow-card, .result-card, .factor-card, .probability-card {
    background: #fff;
    border: 1px solid #e3e9f0;
    border-radius: 12px;
    box-shadow: 0 3px 13px rgba(15, 35, 63, .055);
}
.workflow-card {
    display: flex;
    align-items: center;
    padding: 11px 15px;
    margin: 0 0 8px;
}
.workflow-step {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 10px;
}
.workflow-line {
    width: 50px;
    height: 1px;
    background: #cbd5e1;
    margin: 0 8px;
}
.workflow-num, .section-dot {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #0875f5;
    color: #fff;
    font-weight: 800;
}
.workflow-num { width: 29px; height: 29px; font-size: 12px; }
.section-dot { width: 22px; height: 22px; font-size: 12px; margin-right: 9px; }
.workflow-title { color: #13263f; font-size: 11px; font-weight: 750; }
.workflow-copy { color: #8491a3; font-size: 9px; margin-top: 1px; }
.form-section-title {
    color: #13263f;
    font-size: 15px;
    font-weight: 800;
    margin: 5px 0 4px;
    display: flex;
    align-items: center;
}
.field-caption {
    color: #27384d;
    font-size: 12px;
    font-weight: 650;
    min-height: 19px;
    margin-bottom: 1px;
}
.field-caption .emoji { font-size: 17px; margin-right: 6px; }
.field-caption.two-line-row {
    min-height: 34px;
    font-size: 12px;
    line-height: 1.2;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div {
    border-radius: 8px !important;
}
.segmented-field div[role="radiogroup"] {
    background: #fff !important;
    border: 1px solid #dfe6ee;
    border-radius: 8px;
    padding: 2px !important;
}
.segmented-field div[role="radiogroup"] label {
    min-height: 36px;
}
.st-key-previous_ga div[role="radiogroup"],
.st-key-infection div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    gap: 5px !important;
    padding: 2px !important;
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
.st-key-previous_ga div[role="radiogroup"] label,
.st-key-infection div[role="radiogroup"] label {
    flex: 1 1 50% !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 6px 12px !important;
    border-radius: 6px !important;
    background: #fff !important;
    color: #334155 !important;
}
.st-key-previous_ga div[role="radiogroup"] label:has(input:checked),
.st-key-infection div[role="radiogroup"] label:has(input:checked) {
    background: #0875f5 !important;
    color: #fff !important;
    box-shadow: 0 2px 6px rgba(8, 117, 245, .25) !important;
}
.st-key-previous_ga div[role="radiogroup"] label:has(input:checked) p,
.st-key-infection div[role="radiogroup"] label:has(input:checked) p {
    color: #fff !important;
}
.st-key-previous_ga div[role="radiogroup"] label p,
.st-key-infection div[role="radiogroup"] label p {
    white-space: nowrap !important;
    word-break: keep-all !important;
    overflow-wrap: normal !important;
}
.st-key-previous_ga div[role="radiogroup"] label > div:first-child,
.st-key-infection div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
.completion-banner {
    margin-top: 5px;
    padding: 10px 14px;
    color: #138a5b;
    background: linear-gradient(90deg, #ecfdf5, #f2fff8);
    border: 1px solid #a7efd0;
    border-radius: 9px;
    font-size: 13px;
    font-weight: 650;
}
.st-key-generate_prediction [data-testid="stBaseButton-primary"] {
    display: block;
    width: 280px;
    margin: 7px auto 3px;
    height: 38px;
    background: #0875f5 !important;
    border-radius: 7px !important;
    font-weight: 750;
}
.privacy-note {
    color: #8794a6;
    text-align: center;
    font-size: 10px;
    margin-top: 2px;
}
.result-card { padding: 11px; margin-bottom: 7px; }
.panel-title {
    color: #14263d;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 7px;
}
.prediction-hero {
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 11px;
    border-radius: 10px;
    border: 1px solid var(--accent-border);
    background: var(--accent-bg);
}
.result-donut {
    width: 108px;
    height: 108px;
    flex: 0 0 108px;
    border-radius: 50%;
    display: grid;
    place-items: center;
}
.result-donut-inner {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: #fff;
    display: grid;
    place-content: center;
    text-align: center;
}
.result-percent { font-size: 24px; font-weight: 850; line-height: 1; }
.result-probability-label {
    width: 62px;
    margin: 5px auto 0;
    color: #667085;
    font-size: 8px;
    line-height: 1.1;
    text-align: center;
    white-space: normal;
}
.predicted-label { color: #667085; font-size: 11px; }
.predicted-treatment { font-size: 17px; font-weight: 850; line-height: 1.12; margin: 3px 0 5px; }
.predicted-copy { color: #4b5563; font-size: 11px; line-height: 1.45; }
.clinical-notice {
    color: #3f536d;
    background: #f8fbff;
    border: 1px solid #b9d3f6;
    border-radius: 8px;
    padding: 7px 10px;
    margin-top: 7px;
    font-size: 10px;
    line-height: 1.4;
}
.factor-card, .probability-card { padding: 10px 12px; margin-bottom: 7px; }
.impact-row, .probability-line {
    display: flex;
    align-items: center;
    min-height: 29px;
    border-bottom: 1px solid #edf1f5;
}
.impact-row:last-child, .probability-line:last-child { border-bottom: 0; }
.impact-icon {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    display: grid;
    place-items: center;
    margin-right: 8px;
}
.impact-text { flex: 1; color: #405168; font-size: 11px; }
.impact-pill { border-radius: 12px; padding: 3px 8px; font-size: 9px; font-weight: 750; }
.impact-high { color: #dc2626; background: #fff1f2; }
.impact-moderate { color: #d97706; background: #fff7e6; }
.impact-low { color: #059669; background: #ecfdf5; }
.probability-head {
    display: grid;
    grid-template-columns: 1.25fr 1fr 72px;
    color: #6b7280;
    font-size: 9px;
    padding-bottom: 5px;
    border-bottom: 1px solid #edf1f5;
}
.probability-head span:last-child {
    text-align: right;
    white-space: nowrap;
}
.probability-line { display: grid; grid-template-columns: 1.25fr 1fr 72px; gap: 8px; }
.prob-name { font-size: 10px; font-weight: 750; }
.mini-track { height: 6px; border-radius: 5px; background: #edf1f4; overflow: hidden; }
.mini-fill { height: 100%; border-radius: 5px; }
.prob-value { color: #1f2937; font-size: 10px; font-weight: 800; text-align: right; }
.rounding-note { color: #8a96a6; font-size: 9px; margin-top: 8px; }
.empty-result {
    background: #fff;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    padding: 42px 24px;
    text-align: center;
    color: #748195;
}
@media (max-width: 900px) {
    .workflow-copy, .workflow-line { display: none; }
    .prediction-hero { flex-direction: column; text-align: center; }
}
@media (min-width: 1050px) and (min-height: 650px) {
    .stAppViewContainer,
    section[data-testid="stMain"] {
        height: 100vh !important;
        overflow: hidden !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD THE PRE-TRAINED MULTICLASS MODEL AND ITS FEATURE CONTRACT
# ============================================================
EXPECTED_FEATURE_NAMES = [
    "Age_1_3", "Age_4_6", "Age_7_9", "Age_10_12",
    "ASA_ordinal",
    "Behavior_Agitated", "Behavior_Anxious", "Behavior_Uncooperative",
    "Special_Needs", "Previous_Failed_Attempts",
    "Max_ICDAS", "Num_Teeth_Max_ICDAS",
    "Previous_GA", "Infection",
]


@st.cache_resource
def load_model_assets():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "model_results")
    model = joblib.load(os.path.join(results_dir, "best_multiclass_model.joblib"))
    metadata_path = os.path.join(results_dir, "best_multiclass_model_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    expected_classes = {"GA", "Nitrous", "Local"}
    model_classes = {str(value) for value in model.classes_}
    if model_classes != expected_classes:
        raise ValueError(
            f"Unexpected model classes: {sorted(model_classes)}; "
            f"expected {sorted(expected_classes)}."
        )
    if metadata.get("feature_names") != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            "The saved model feature schema does not match the application input schema."
        )
    return model, metadata

multiclass_model, model_metadata = load_model_assets()

# ============================================================
# SESSION STATE AND NAVIGATION
# ============================================================
WIDGET_DEFAULTS = {
    "age_group": "4-6",
    "asa_class": "ASA I",
    "special_needs": "No",
    "behavior": "Calm",
    "icdas": "4 - Moderate",
    "num_teeth": 3,
    "failed_attempts": "No",
    "previous_ga": "No",
    "infection": "No",
}

if "prediction_ready" not in st.session_state:
    st.session_state.prediction_ready = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "New Assessment"


def reset_assessment():
    for widget_key, default_value in WIDGET_DEFAULTS.items():
        st.session_state[widget_key] = default_value
    st.session_state.prediction_ready = False


with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-tooth">🦷</div>
        <div class="sidebar-name">PediaSed</div>
        <div class="sidebar-tagline">Decision Support</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.container(key="nav_new"):
        if st.button("New Assessment", use_container_width=True):
            st.session_state.active_page = "New Assessment"
    with st.container(key="nav_info"):
        if st.button("Model Information", use_container_width=True):
            st.session_state.active_page = "Model Information"
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-assist">
        <div class="sidebar-assist-title">AI-Assisted Tool</div>
        <div class="sidebar-assist-copy">
            Supports clinical judgment.<br>Not a substitute for clinician decision.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-profile">
        <div class="profile-avatar">♙</div>
        <div class="profile-copy">
            <div class="profile-name">AI in Pediatric Dentistry</div>
            <div class="profile-mode">Research Mode</div>
        </div>
        <div class="profile-chevron">⌄</div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.active_page
active_nav_key = "nav_new" if page == "New Assessment" else "nav_info"
st.markdown(
    f"""
    <style>
    .st-key-{active_nav_key} button {{
        background: #1267e8 !important;
        color: #fff !important;
        box-shadow: 0 4px 12px rgba(18, 103, 232, .32) !important;
    }}
    .st-key-{active_nav_key} button p::before {{
        border-color: #fff !important;
        background: #fff !important;
        color: #1267e8 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if page == "Model Information":
    st.markdown('<div class="app-heading">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subheading">Technical details about the validated multiclass model</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="result-card">
        <b>Dataset:</b> 416 pediatric dental records<br><br>
        <b>Production model:</b> multiclass Random Forest<br><br>
        <b>Classes:</b> General Anesthesia, Nitrous Oxide Sedation, and Local Anesthesia<br><br>
        The result is the class with the highest model-estimated probability.
        Validation is internal and does not replace external clinical validation.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# ASSESSMENT FORM
# ============================================================
form_column, result_column = st.columns([1.75, 1], gap="large")

with form_column:
    header_text, reset_column = st.columns([4.6, 1.4], vertical_alignment="center")
    with header_text:
        st.markdown('<div class="app-heading">New Assessment</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subheading">AI-assisted decision support for pediatric dental sedation and general anesthesia</div>',
            unsafe_allow_html=True,
        )
    with reset_column:
        with st.container(key="reset_assessment"):
            st.button(
                "↻  Reset Assessment",
                use_container_width=True,
                on_click=reset_assessment,
            )

    st.markdown("""
    <div class="workflow-card">
      <div class="workflow-step"><span class="workflow-num">1</span><div><div class="workflow-title">Patient Profile</div><div class="workflow-copy">Child & baseline information</div></div></div>
      <span class="workflow-line"></span>
      <div class="workflow-step"><span class="workflow-num">2</span><div><div class="workflow-title">Clinical Risk Factors</div><div class="workflow-copy">Key clinical indicators</div></div></div>
      <span class="workflow-line"></span>
      <div class="workflow-step"><span class="workflow-num">3</span><div><div class="workflow-title">AI Output</div><div class="workflow-copy">Prediction & explanation</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="form-section-title"><span class="section-dot">1</span>Patient Profile</div>',
        unsafe_allow_html=True,
    )
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="field-caption"><span class="emoji">📅</span>Age Group</div>', unsafe_allow_html=True)
        age_group = st.selectbox(
            "Age Group",
            ["1-3", "4-6", "7-9", "10-12", ">12"],
            format_func=lambda value: f"{value} years" if value != ">12" else "Older than 12 years",
            key="age_group",
            label_visibility="collapsed",
        )
    with p2:
        st.markdown('<div class="field-caption"><span class="emoji">🏥</span>ASA Classification</div>', unsafe_allow_html=True)
        asa_class = st.selectbox(
            "ASA",
            ["ASA I", "ASA II", "ASA III", "ASA IV"],
            key="asa_class",
            label_visibility="collapsed",
        )
    with p3:
        st.markdown('<div class="field-caption"><span class="emoji">👤</span>Special Needs</div>', unsafe_allow_html=True)
        special_needs_choice = st.selectbox(
            "Special Needs",
            ["No", "Yes"],
            key="special_needs",
            label_visibility="collapsed",
        )

    st.markdown(
        '<div class="form-section-title"><span class="section-dot">2</span>Clinical Risk Factors</div>',
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown('<div class="field-caption"><span class="emoji">😊</span>Behavior</div>', unsafe_allow_html=True)
        selected_behavior = st.selectbox(
            "Behavior",
            ["Calm", "Agitated", "Anxious", "Uncooperative"],
            format_func=lambda value: {
                "Calm": "🟢  Calm",
                "Agitated": "🔴  Agitated",
                "Anxious": "🟠  Anxious",
                "Uncooperative": "🔴  Uncooperative",
            }[value],
            key="behavior",
            label_visibility="collapsed",
        )
    with r2:
        st.markdown('<div class="field-caption"><span class="emoji">🦷</span>ICDAS Max</div>', unsafe_allow_html=True)
        icdas_selection = st.selectbox(
            "ICDAS",
            ["2 - Early", "3 - Moderate", "4 - Moderate", "5 - Severe", "6 - Severe"],
            key="icdas",
            label_visibility="collapsed",
        )
    with r3:
        st.markdown('<div class="field-caption"><span class="emoji">🔢</span>Number of Teeth at Max ICDAS</div>', unsafe_allow_html=True)
        num_teeth = st.number_input(
            "Teeth",
            min_value=1,
            max_value=14,
            key="num_teeth",
            label_visibility="collapsed",
        )

    r4, r5, r6 = st.columns(3)
    with r4:
        st.markdown('<div class="field-caption two-line-row"><span class="emoji">📝</span>Previous Failed Dental<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Attempts</div>', unsafe_allow_html=True)
        prev_failed_opt = st.selectbox(
            "Failed attempts",
            ["No", "Yes"],
            key="failed_attempts",
            label_visibility="collapsed",
        )
    with r5:
        st.markdown('<div class="field-caption two-line-row"><span class="emoji">💉</span>Previous General Anesthesia</div>', unsafe_allow_html=True)
        previous_ga_choice = st.radio(
            "Previous GA",
            ["No", "Yes"],
            horizontal=True,
            key="previous_ga",
            label_visibility="collapsed",
        )
    with r6:
        st.markdown('<div class="field-caption two-line-row"><span class="emoji">🔬</span>Presence of Infection</div>', unsafe_allow_html=True)
        infection_choice = st.radio(
            "Infection",
            ["No", "Yes"],
            horizontal=True,
            key="infection",
            label_visibility="collapsed",
        )

    st.markdown(
        '<div class="completion-banner">✓ &nbsp; All required fields completed.</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="generate_prediction"):
        if st.button("✨  Generate AI Prediction", type="primary", use_container_width=True):
            st.session_state.prediction_ready = True
    st.markdown(
        '<div class="privacy-note">🔒 &nbsp; Patient inputs are processed only for this assessment.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# FEATURE ENCODING AND MODEL OUTPUT
# ============================================================
age_encoding = {
    "1-3": [1, 0, 0, 0],
    "4-6": [0, 1, 0, 0],
    "7-9": [0, 0, 1, 0],
    "10-12": [0, 0, 0, 1],
    ">12": [0, 0, 0, 0],
}
asa_class_clean = asa_class.replace("ASA ", "")
asa_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
max_icdas = int(icdas_selection[0])
special_needs = special_needs_choice == "Yes"
previous_failed = prev_failed_opt == "Yes"
previous_ga = previous_ga_choice == "Yes"
infection = infection_choice == "Yes"

features = (
    age_encoding[age_group]
    + [asa_map[asa_class_clean]]
    + [
        int(selected_behavior == "Agitated"),
        int(selected_behavior == "Anxious"),
        int(selected_behavior == "Uncooperative"),
    ]
    + [int(special_needs), int(previous_failed)]
    + [max_icdas, num_teeth]
    + [int(previous_ga), int(infection)]
)
X_input = pd.DataFrame(
    [features],
    columns=model_metadata["feature_names"],
    dtype=float,
)
class_probabilities = dict(
    zip(
        [str(value) for value in multiclass_model.classes_],
        multiclass_model.predict_proba(X_input)[0],
    )
)
predicted_class = max(class_probabilities, key=class_probabilities.get)

raw_percentages = np.array(
    [
        class_probabilities["Nitrous"],
        class_probabilities["GA"],
        class_probabilities["Local"],
    ]
) * 100
display_percentages = np.floor(raw_percentages).astype(int)
remaining_points = 100 - int(display_percentages.sum())
if remaining_points:
    order = np.argsort(-(raw_percentages - display_percentages))
    display_percentages[order[:remaining_points]] += 1
n2o_pct, ga_pct, local_pct = display_percentages.tolist()


def clinical_factors():
    behavior_impact = {
        "Calm": ("😊", "Calm behavior", "Low impact", "impact-low", "#ecfdf5"),
        "Agitated": ("😤", "Agitated behavior", "High impact", "impact-high", "#fff1f2"),
        "Anxious": ("😰", "Anxious behavior", "Moderate impact", "impact-moderate", "#fff7e6"),
        "Uncooperative": ("😣", "Uncooperative behavior", "High impact", "impact-high", "#fff1f2"),
    }
    factors = [behavior_impact[selected_behavior]]
    if previous_failed:
        factors.append(("📝", "Previous failed dental attempt", "High impact", "impact-high", "#fff7e6"))
    if num_teeth >= 5:
        factors.append(("🦷", f"{num_teeth} teeth at maximum ICDAS", "High impact", "impact-high", "#ecfeff"))
    elif num_teeth >= 3:
        factors.append(("🦷", f"{num_teeth} teeth at maximum ICDAS", "Moderate impact", "impact-moderate", "#ecfeff"))
    asa_impact = "Low impact" if asa_map[asa_class_clean] == 1 else "Moderate impact"
    asa_class_name = "impact-low" if asa_map[asa_class_clean] == 1 else "impact-moderate"
    factors.append(("🏥", f"ASA {asa_class_clean} classification", asa_impact, asa_class_name, "#ecfdf5"))
    factors.append(
        ("🔬", "Infection present" if infection else "No infection",
         "High impact" if infection else "Low impact",
         "impact-high" if infection else "impact-low",
         "#eff6ff")
    )
    return factors


with result_column:
    if not st.session_state.prediction_ready:
        st.markdown("""
        <div class="empty-result">
            <div style="font-size:30px; margin-bottom:10px;">✨</div>
            <b>AI Prediction</b><br>
            <span style="font-size:11px;">Complete the assessment and generate a prediction to view the result.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        treatment_content = {
            "GA": {
                "name": "General Anesthesia",
                "color": "#dc2626",
                "bg": "#fff7f7",
                "border": "#fecaca",
                "pct": ga_pct,
                "copy": "This pathway has the highest model-estimated probability for the provided clinical profile.",
            },
            "Nitrous": {
                "name": "Nitrous Oxide Sedation",
                "color": "#e58a00",
                "bg": "#fffbf2",
                "border": "#f7dda5",
                "pct": n2o_pct,
                "copy": "This pathway has the highest model-estimated probability for the provided clinical profile.",
            },
            "Local": {
                "name": "Conventional Treatment (Local)",
                "color": "#059669",
                "bg": "#f3fcf8",
                "border": "#a7f3d0",
                "pct": local_pct,
                "copy": "This pathway has the highest model-estimated probability for the provided clinical profile.",
            },
        }
        selected = treatment_content[predicted_class]
        st.markdown(f"""
        <div class="result-card">
          <div class="panel-title">✨ &nbsp; AI Prediction</div>
          <div class="prediction-hero" style="--accent-bg:{selected['bg']}; --accent-border:{selected['border']};">
            <div class="result-donut" style="background:conic-gradient({selected['color']} 0 {selected['pct'] * 3.6}deg,#edf1f4 {selected['pct'] * 3.6}deg 360deg);">
              <div class="result-donut-inner">
                <div class="result-percent" style="color:{selected['color']};">{selected['pct']}%</div>
                <div class="result-probability-label">Model<br>probability</div>
              </div>
            </div>
            <div>
              <div class="predicted-label">Predicted Treatment</div>
              <div class="predicted-treatment" style="color:{selected['color']};">{selected['name']}</div>
              <div class="predicted-copy">{selected['copy']}</div>
            </div>
          </div>
          <div class="clinical-notice">ⓘ &nbsp; This recommendation is intended to support clinical judgment and should not replace the clinician's decision.</div>
        </div>
        """, unsafe_allow_html=True)

        factor_rows = "".join(
            f"""<div class="impact-row">
              <span class="impact-icon" style="background:{background};">{icon}</span>
              <span class="impact-text">{text}</span>
              <span class="impact-pill {impact_class}">{impact}</span>
            </div>"""
            for icon, text, impact, impact_class, background in clinical_factors()
        )
        st.markdown(f"""
        <div class="factor-card">
          <div class="panel-title">Key clinical factors in this assessment &nbsp;ⓘ</div>
          {factor_rows}
        </div>
        """, unsafe_allow_html=True)

        probability_rows = [
            ("Nitrous Oxide Sedation (N₂O)", n2o_pct, "#f59e0b"),
            ("General Anesthesia (GA)", ga_pct, "#dc2626"),
            ("Conventional Treatment (Local)", local_pct, "#10b981"),
        ]
        probability_html = "".join(
            f"""<div class="probability-line">
              <span class="prob-name" style="color:{color};">{name}</span>
              <span class="mini-track"><span class="mini-fill" style="display:block;width:{percentage}%;background:{color};"></span></span>
              <span class="prob-value">{percentage}%</span>
            </div>"""
            for name, percentage, color in probability_rows
        )
        st.markdown(f"""
        <div class="probability-card">
          <div class="panel-title">Prediction Probabilities &nbsp;ⓘ</div>
          <div class="probability-head"><span>Treatment Option</span><span></span><span>Probability</span></div>
          {probability_html}
          <div class="rounding-note">Displayed percentages are rounded and total 100%.</div>
        </div>
        """, unsafe_allow_html=True)
