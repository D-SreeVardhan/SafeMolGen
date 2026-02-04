"""Property input widget for constraints."""

import streamlit as st


def property_input():
    st.subheader("🎯 Target Properties")
    logp = st.slider("LogP (target range)", 0.0, 6.0, (2.0, 4.0), 0.1)
    mw = st.slider("Molecular Weight (max)", 200, 800, 500, 10)
    qed = st.slider("QED (min)", 0.0, 1.0, 0.5, 0.05)
    return {"logp": logp, "mw": mw, "qed": qed}
