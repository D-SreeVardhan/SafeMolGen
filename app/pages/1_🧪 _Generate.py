"""Generation page for Streamlit multipage app."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import generate_page, load_pipeline

st.set_page_config(page_title="Generate - SafeMolGen", page_icon="🧪", layout="wide")
st.title("🧪 Generate New Molecules")

pipeline = load_pipeline()
target_success = st.slider("Target Success", 0.1, 0.5, 0.25)
max_iterations = st.slider("Max Iterations", 1, 20, 10)
generate_page(pipeline, target_success, max_iterations)
