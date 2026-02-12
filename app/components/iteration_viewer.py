"""Iteration journey visualization component."""

import plotly.graph_objects as go
import streamlit as st

from app.components.molecule_viewer import draw_molecule_2d


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
    st.caption(
        '"Passed safety" = above minimum bar (safety threshold). '
        "The loop keeps using Oracle feedback until Overall reaches the Target (e.g. 25%)."
    )
    if iteration_history:
        st.markdown("**Molecules across iterations** — how the best molecule changes each step:")
        n = len(iteration_history)
        cols = st.columns(n)
        for idx, result in enumerate(iteration_history):
            if hasattr(result, "prediction"):
                iteration = result.iteration
                smiles = result.smiles
                overall = result.prediction.overall_prob
            else:
                iteration = result["iteration"]
                smiles = result["smiles"]
                overall = result["overall_prob"]
            with cols[idx]:
                svg = draw_molecule_2d(smiles, size=(180, 140)) if smiles else None
                if svg:
                    st.image(svg, use_container_width=True)
                st.caption(f"Iter {iteration}: {overall:.1%}")
        st.markdown("---")

    for idx, result in enumerate(iteration_history):
        if hasattr(result, "prediction"):
            iteration = result.iteration
            smiles = result.smiles
            overall = result.prediction.overall_prob
            improvements = result.improvements
            alerts = result.prediction.structural_alerts
            passed_safety = getattr(result, "passed_safety", True)
            used_oracle_feedback = getattr(result, "used_oracle_feedback", False)
        else:
            iteration = result["iteration"]
            smiles = result["smiles"]
            overall = result["overall_prob"]
            improvements = result.get("improvements", [])
            alerts = result.get("structural_alerts", [])
            passed_safety = result.get("passed_safety", True)
            used_oracle_feedback = result.get("used_oracle_feedback", False)
        label = f"Iteration {iteration}: {overall:.1%}"
        if used_oracle_feedback:
            label += " (Oracle feedback used)"
        with st.expander(label, expanded=(idx == len(iteration_history) - 1)):
            col_mol, col_info = st.columns([1, 1])
            with col_mol:
                svg = draw_molecule_2d(smiles, size=(320, 240)) if smiles else None
                if svg:
                    st.image(svg, use_container_width=True)
            with col_info:
                st.code(smiles, language=None)
                if passed_safety:
                    st.caption("✅ Passed safety criteria")
                else:
                    st.caption("⚠️ Did not pass safety → next iteration used Oracle feedback")
                if improvements:
                    for imp in improvements:
                        st.success(f"✓ {imp}")
                if alerts:
                    st.warning(f"Alerts: {', '.join(alerts)}")
