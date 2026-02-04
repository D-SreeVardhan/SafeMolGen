"""Iteration journey visualization component."""

import plotly.graph_objects as go
import streamlit as st


def iteration_timeline(iteration_history):
    if not iteration_history:
        st.info("No iteration history available")
        return
    iterations = [r.iteration if hasattr(r, "iteration") else r["iteration"] for r in iteration_history]
    overall = [
        r.prediction.overall_prob if hasattr(r, "prediction") else r["overall_prob"]
        for r in iteration_history
    ]
    phase1 = [
        r.prediction.phase1_prob if hasattr(r, "prediction") else r["phase1_prob"]
        for r in iteration_history
    ]
    phase2 = [
        r.prediction.phase2_prob if hasattr(r, "prediction") else r["phase2_prob"]
        for r in iteration_history
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=[p * 100 for p in overall], mode="lines+markers", name="Overall"))
    fig.add_trace(go.Scatter(x=iterations, y=[p * 100 for p in phase1], mode="lines+markers", name="Phase I"))
    fig.add_trace(go.Scatter(x=iterations, y=[p * 100 for p in phase2], mode="lines+markers", name="Phase II"))
    fig.add_hline(y=14, line_dash="dot", line_color="gray", annotation_text="Baseline (14%)")
    fig.add_hline(y=25, line_dash="dot", line_color="green", annotation_text="Target (25%)")
    fig.update_layout(
        title="Optimization Journey",
        xaxis_title="Iteration",
        yaxis_title="Success Probability (%)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def iteration_details(iteration_history):
    st.subheader("📜 Iteration Details")
    for idx, result in enumerate(iteration_history):
        if hasattr(result, "prediction"):
            iteration = result.iteration
            smiles = result.smiles
            overall = result.prediction.overall_prob
            improvements = result.improvements
            alerts = result.prediction.structural_alerts
        else:
            iteration = result["iteration"]
            smiles = result["smiles"]
            overall = result["overall_prob"]
            improvements = result.get("improvements", [])
            alerts = result.get("structural_alerts", [])
        with st.expander(f"Iteration {iteration}: {overall:.1%}", expanded=(idx == len(iteration_history) - 1)):
            st.code(smiles, language=None)
            if improvements:
                for imp in improvements:
                    st.success(f"✓ {imp}")
            if alerts:
                st.warning(f"Alerts: {', '.join(alerts)}")
