"""Recommendations display component."""

import streamlit as st


def recommendation_panel(recommendations):
    if not recommendations:
        st.info("No recommendations available")
        return
    st.subheader("💡 Recommendations")
    for rec in recommendations:
        severity = rec.get("severity", "medium")
        icon = "🟡 "
        if severity == "critical":
            icon = "🔴 "
        elif severity == "high":
            icon = "🟠 "
        elif severity == "low":
            icon = "🟢 "
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{icon}{rec.get('type', 'Suggestion').title()}**")
                st.write(f"**Issue:** {rec.get('issue', 'N/A')}")
                st.write(f"**Suggestion:** {rec.get('suggestion', 'N/A')}")
            with col2:
                expected = rec.get("expected_improvement", "")
                if expected:
                    st.success(expected)
            st.divider()
