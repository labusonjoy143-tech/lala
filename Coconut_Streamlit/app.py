"""Run with: python -m streamlit run app.py"""
import hashlib
import importlib
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from coconut import CoconutWorkbench

st.set_page_config(page_title="Coconut Supply and Inflation Analysis", page_icon="🥥", layout="wide")

# Load the violet stylesheet from common Streamlit Cloud and local layouts.
# Streamlit Cloud may launch the app from the repository root or a subfolder.
APP_DIR = Path(__file__).resolve().parent
_CSS_CANDIDATES = [
    APP_DIR / ".streamlit" / "style.css",
    Path.cwd() / ".streamlit" / "style.css",
    Path.cwd() / "Coconut_Streamlit" / ".streamlit" / "style.css",
    APP_DIR.parent / ".streamlit" / "style.css",
]
CSS_PATH = next((candidate for candidate in _CSS_CANDIDATES if candidate.is_file()), None)
if CSS_PATH is not None:
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:
    # Keep the dashboard styled even if the separate CSS file was not committed.
    st.markdown(
        """<style>
        :root { color-scheme: light; }
        .stApp { background: #FAF9FD; color: #24163B; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#32105F,#5B21B6); }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }
        .stButton > button { background: #6D28D9; color: white; border: 0; border-radius: 10px; }
        .stButton > button:hover { background: #5B21B6; color: white; border: 0; }
        [data-testid="stMetric"] { background: #F0EAFF; border: 1px solid #DDD0FF; padding: 14px; border-radius: 14px; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #6D28D9; border-bottom-color: #6D28D9; }
        h1, h2, h3 { color: #4C1D95; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.info("Using the built-in violet theme. Add .streamlit/style.css to your repository root for the full stylesheet.")

SCREENS = {
    "1 · Data quality": "views.data_quality",
    "2 · Explore production": "views.explore",
    "3 · Inflation analysis": "views.inflation",
    "4 · Monthly prediction": "views.monthly",
    "5 · Annual forecast": "views.annual",
    "6 · Other data mining": "views.mining",
    "7 · Management dashboard": "views.dashboard",
    "8 · Download results": "views.downloads",
}


def upload_sidebar():
    st.sidebar.title("🥥 Coconut Supply & Inflation")
    st.sidebar.caption("Upload once, then run the modules you need.")
    uploaded = st.sidebar.file_uploader("Upload coconut data", type=["xlsx", "csv"], key="upload")
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        sheet = 0
        usable = True
        if uploaded.name.lower().endswith(".xlsx"):
            try:
                if st.session_state.get("sheet_digest") != digest:
                    with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as book:
                        st.session_state["sheet_names"] = list(book.sheet_names)
                    st.session_state["sheet_digest"] = digest
                sheet = st.sidebar.selectbox("Worksheet", st.session_state["sheet_names"], key="sheet")
            except Exception as exc:
                usable = False
                st.sidebar.error(f"Cannot read this Excel workbook: {exc}")
        if st.sidebar.button("Load and clean data", type="primary", disabled=not usable, key="load"):
            try:
                candidate = CoconutWorkbench()
                candidate.load_bytes(content, uploaded.name, sheet_name=sheet)
            except Exception as exc:
                st.sidebar.error(f"Data were not loaded: {exc}")
            else:
                st.session_state["workbench"] = candidate
                for key in ["decision_scenario", "scenario_signature", "export_bytes", "task_runs"]:
                    st.session_state.pop(key, None)
                st.session_state["loaded_signature"] = (digest, str(sheet))
                st.sidebar.success("Data cleaned. Choose a module below.")
        if "workbench" in st.session_state and st.session_state.get("loaded_signature") != (digest, str(sheet)):
            st.sidebar.info("A different file or worksheet is selected. Click Load and clean data to use it.")
    wb = st.session_state.get("workbench")
    if wb is not None:
        st.sidebar.caption(f"Loaded: {wb.source_name}\n\nLatest month: {wb.audit['last_month']}")
    return wb


def main():
    wb = upload_sidebar()
    selected = st.sidebar.radio("Module", list(SCREENS), key="module")
    st.sidebar.caption("Analyses remain available while this session is open. Reloading data resets model results.")
    if wb is None:
        st.title("Coconut Supply and Local Inflation Analysis")
        st.write("Upload an Excel workbook or CSV using the left panel and click **Load and clean data**.")
        st.info("Start with data quality and exploration. Run monthly or annual models when needed, "
                "review inflation analysis, then open the management dashboard.")
        with st.expander("Expected columns"):
            st.code("Year, Month, Barangay, Harvested Area (ha), Harvest (kg)\n"
                    "Optional: Rainfall (mm), Temperature (°C), prices, Inflation Rate (%), Weather")
            st.write("Use one record per barangay and month. Month may be a name, abbreviation, or number.")
        return
    from views.common import source_note
    source_note(wb)
    importlib.import_module(SCREENS[selected]).render(wb)


if __name__ == "__main__":
    main()
