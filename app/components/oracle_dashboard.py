"""Oracle results dashboard component."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def _gauge(prob: float, phase_name: str):
    if prob >= 0.7:
        color = "green"
    elif prob >= 0.4:
        color = "orange"
    else:
        color = "red"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": phase_name},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "lightcoral"},
                    {"range": [40, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightgreen"},
                ],
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def oracle_dashboard(prediction) -> None:
    st.subheader("🔮 Oracle Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(_gauge(prediction.phase1_prob, "Phase I"), use_container_width=True)
    with col2:
        st.plotly_chart(_gauge(prediction.phase2_prob, "Phase II"), use_container_width=True)
    with col3:
        st.plotly_chart(_gauge(prediction.phase3_prob, "Phase III"), use_container_width=True)
    with col4:
        st.plotly_chart(_gauge(prediction.overall_prob, "Overall"), use_container_width=True)

    if prediction.risk_factors:
        st.subheader("⚠ Risk Factors")
        for risk in prediction.risk_factors:
            with st.expander(f"{risk.name} ({risk.category})"):
                st.write(f"**Description:** {risk.description}")
                st.write(f"**Impact:** {risk.impact:.0%}")
                st.write(f"**Source:** {risk.source}")

    if prediction.structural_alerts:
        st.subheader("🚨 Structural Alerts")
        st.table(pd.DataFrame([{"Alert": a} for a in prediction.structural_alerts]))

    if prediction.admet_predictions:
        st.subheader("📊 ADMET Predictions")
        key_endpoints = ["herg", "ames", "dili", "bioavailability_ma", "bbb_martins"]
        data = []
        for ep in key_endpoints:
            if ep in prediction.admet_predictions:
                data.append({"Endpoint": ep.upper(), "Value": prediction.admet_predictions[ep]})
        if data:
            df = pd.DataFrame(data)
            fig = px.bar(
                df,
                x="Endpoint",
                y="Value",
                color="Value",
                color_continuous_scale=["green", "yellow", "red"],
            )
            st.plotly_chart(fig, use_container_width=True)


def comparison_table(predictions):
    data = []
    for p in predictions:
        data.append(
            {
                "SMILES": (p["smiles"][:30] + "...") if len(p["smiles"]) > 30 else p["smiles"],
                "Phase I": f"{p['prediction']['phase1_prob']:.1%}",
                "Phase II": f"{p['prediction']['phase2_prob']:.1%}",
                "Phase III": f"{p['prediction']['phase3_prob']:.1%}",
                "Overall": f"{p['prediction']['overall_prob']:.1%}",
                "Alerts": len(p["prediction"].get("structural_alerts", [])),
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df)
