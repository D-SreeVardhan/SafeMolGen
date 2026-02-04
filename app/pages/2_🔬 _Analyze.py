"""Analysis page for Streamlit multipage app."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import analyze_page, load_pipeline

st.set_page_config(page_title="Analyze - SafeMolGen", page_icon="🔬", layout="wide")
st.title("🔬 Analyze Molecule")

pipeline = load_pipeline()
analyze_page(pipeline)
