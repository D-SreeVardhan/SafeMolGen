"""Generation page for Streamlit multipage app."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import generate_page, load_pipeline
from app.components.property_input import property_input

st.set_page_config(page_title="Generate - SafeMolGen", page_icon="🧪", layout="wide")
st.title("🧪 Generate New Molecules")

pipeline = load_pipeline()
with st.sidebar:
    st.markdown("### Generation & safety")
    target_success = st.slider("Target Success", 0.1, 0.5, 0.25)
    max_iterations = st.slider("Max Iterations", 1, 20, 10)
    safety_threshold = st.slider("Safety threshold", 0.05, 0.5, 0.2, 0.05)
    require_no_structural_alerts = st.checkbox("Require no structural alerts", value=False)
    st.markdown("---")
    property_input()

property_targets = st.session_state.get("property_targets")
generate_page(
    pipeline,
    target_success,
    max_iterations,
    safety_threshold=safety_threshold,
    require_no_structural_alerts=require_no_structural_alerts,
    property_targets=property_targets,
)
