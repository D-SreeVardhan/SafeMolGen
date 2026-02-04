"""Model caching utilities for Streamlit."""

import streamlit as st


def cache_resource(func):
    return st.cache_resource(show_spinner=False)(func)
