"""Property input widget for constraints — user-defined targets flow to generator filters and oracle."""

import streamlit as st


def _parse_range(value, default_lo, default_hi):
    """Accept single number or (lo, hi) tuple from sliders."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


def property_input():
    """Render inputs for all property targets; values are used to filter candidates and guide the oracle."""
    st.subheader("🎯 Target Properties (→ Generator & Oracle)")

    st.caption(
        "Set desired ranges for generated molecules. Structural properties (LogP, MW, HBD, HBA, TPSA) "
        "filter candidates before scoring; ADMET-related (Solubility, Protein Binding, Metabolic Stability) "
        "filter after Oracle prediction. Leave defaults or adjust as needed."
    )

    # --- Structural / computed (from molecular structure) ---
    st.markdown("**Structural (computed from structure)**")
    logp = st.slider(
        "LogP (lipophilicity)",
        0.0,
        8.0,
        (2.0, 5.0),
        0.1,
        help="Target range. Often 2–5 for drug-like; from octanol/water or computed.",
    )
    mw = st.slider(
        "Molecular Weight (max, Da)",
        150,
        800,
        500,
        10,
        help="Max MW; typically <500 for oral drugs.",
    )
    hbd = st.number_input(
        "Hydrogen Bond Donors (max)",
        min_value=0,
        max_value=15,
        value=5,
        help="e.g. N-H, O-H; Lipinski ≤5.",
    )
    hba = st.number_input(
        "Hydrogen Bond Acceptors (max)",
        min_value=0,
        max_value=20,
        value=10,
        help="N, O with lone pairs; Lipinski ≤10.",
    )
    tpsa = st.slider(
        "TPSA — Topological Polar Surface Area (max, Å²)",
        0.0,
        200.0,
        140.0,
        5.0,
        help="Often ≤140 for good absorption.",
    )
    qed = st.slider(
        "QED — drug-likeness (min)",
        0.0,
        1.0,
        0.5,
        0.05,
        help="Quantitative Estimate of Drug-likeness.",
    )

    # --- ADMET / experimental-style (used to filter after Oracle prediction) ---
    st.markdown("**ADMET / experimental (from Oracle predictions)**")
    use_solubility = st.checkbox("Constrain Solubility (logS)", value=False, help="Filter by predicted solubility (solubility_aqsoldb).")
    solubility_min = -12.0
    solubility_max = 2.0
    solubility_range = (0.0, 1.0)
    if use_solubility:
        solubility_range = st.slider(
            "Solubility logS (min, max)",
            solubility_min,
            solubility_max,
            (-4.0, 1.0),
            0.5,
            help="Predicted aqueous solubility (log mol/L).",
        )

    use_ppbr = st.checkbox("Constrain Protein Binding (%)", value=False, help="Filter by predicted plasma protein binding (ppbr_az).")
    ppbr_range = None
    if use_ppbr:
        ppbr_range = st.slider(
            "Plasma Protein Binding % (min, max)",
            0.0,
            100.0,
            (50.0, 100.0),
            5.0,
            help="Predicted fraction bound.",
        )

    use_metab = st.checkbox("Constrain Metabolic Stability (clearance)", value=False, help="Filter by predicted hepatocyte clearance (lower = more stable).")
    clearance_max = None
    if use_metab:
        clearance_max = st.number_input(
            "Max Hepatocyte Clearance (log mL/min/kg)",
            min_value=-2.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Lower clearance = more metabolically stable.",
        )

    # --- pKa (no ADMET endpoint; store for display / future use) ---
    st.markdown("**Optional (informational)**")
    use_pka = st.checkbox("Set pKa range (not used in filtering)", value=False, help="pKa is not predicted by current Oracle; stored for reference.")
    pka_range = None
    if use_pka:
        pka_range = st.slider(
            "pKa (min, max)",
            0.0,
            14.0,
            (4.0, 10.0),
            0.5,
            help="Acid/base dissociation; experimental or predicted elsewhere.",
        )

    # Build targets dict (same keys used in pipeline)
    targets = {
        "logp": logp,
        "mw": mw,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "qed": qed,
    }
    if use_solubility:
        targets["solubility"] = solubility_range
    if use_ppbr:
        targets["ppbr"] = ppbr_range
    if use_metab and clearance_max is not None:
        targets["clearance_hepatocyte_max"] = clearance_max
    if use_pka and pka_range is not None:
        targets["pka"] = pka_range

    if "property_targets" not in st.session_state:
        st.session_state["property_targets"] = targets
    st.session_state["property_targets"] = targets
    return targets
