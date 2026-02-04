"""Session state helpers."""

import streamlit as st


def init_state():
    if "generation_result" not in st.session_state:
        st.session_state["generation_result"] = None
