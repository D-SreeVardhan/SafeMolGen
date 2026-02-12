"""SafeMolGen-DrugOracle: Main Streamlit Application."""

import sys
from pathlib import Path

# Add project root first so "app" and "models" resolve when run as streamlit run app/app.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Prevent "app" from resolving to this file's directory (so "app" stays the package)
_app_dir = _PROJECT_ROOT / "app"
sys.path = [p for p in sys.path if Path(p).resolve() != _app_dir]

import streamlit as st

from app.components.molecule_viewer import molecule_card
from app.components.oracle_dashboard import oracle_dashboard, comparison_table
from app.components.iteration_viewer import iteration_timeline, iteration_details
from app.components.recommendation_panel import recommendation_panel
from app.components.property_input import property_input
from app.utils.session_state import init_state
from app.utils.caching import cache_resource
from models.integrated.pipeline import SafeMolGenDrugOracle
from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
import yaml

PROJECT_ROOT = _PROJECT_ROOT

st.set_page_config(
    page_title="SafeMolGen-DrugOracle",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@cache_resource
def load_pipeline(use_rl_model: bool = False):
    generator_path = PROJECT_ROOT / "checkpoints" / "generator_rl" if use_rl_model else PROJECT_ROOT / "checkpoints" / "generator"
    if not generator_path.exists():
        generator_path = PROJECT_ROOT / "checkpoints" / "generator"
    oracle_path = PROJECT_ROOT / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = PROJECT_ROOT / "checkpoints" / "admet" / "best_model.pt"
    if not generator_path.exists() or not oracle_path.exists() or not admet_path.exists():
        return None
    endpoints_cfg = yaml.safe_load(
        (PROJECT_ROOT / "config" / "endpoints.yaml").read_text(encoding="utf-8")
    )
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))
    return SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
    )


def _render_result_into(container, result):
    """Render a DesignResult (partial or final) into a Streamlit container."""
    with container:
        st.subheader("Results")
        st.markdown("### 🏆 Best Molecule")
        molecule_card(result.final_smiles, result.final_prediction)
        oracle_dashboard(result.final_prediction)
        st.markdown("### 📈 Optimization Journey")
        iteration_timeline(result.iteration_history)
        iteration_details(result.iteration_history)
        st.markdown("### 💡 Recommendations")
        recommendation_panel(result.final_prediction.recommendations)
        if not result.target_achieved and result.iteration_history:
            st.caption(f"⏳ Iteration {result.total_iterations} of max — updating as each iteration completes.")


def generate_page(
    pipeline,
    target_success,
    max_iterations,
    top_k=40,
    safety_threshold=0.2,
    require_no_structural_alerts=False,
    property_targets=None,
):
    st.header("🧪 Generate New Molecules")
    if pipeline is None:
        st.warning("⚠ Models not loaded. Train models first.")
        return
    col1, col2 = st.columns([1, 2])
    with col2:
        results_placeholder = st.empty()
        result = st.session_state.get("generation_result")
        if result:
            _render_result_into(results_placeholder, result)
    with col1:
        st.subheader("Generation Settings")
        n_molecules = st.number_input("Number of molecules to return", 1, 20, 5)
        temperature = st.slider("Creativity (temperature)", 0.5, 1.5, 0.75, 0.05)
        top_k_ui = st.slider("Top-K (trim low-prob tokens)", 0, 100, top_k, 10)
        seed_smiles = st.text_input(
            "Seed SMILES (optional scaffold)",
            value="",
            placeholder="e.g. c1ccccc1 for benzene scaffold",
            help="Generated molecules will contain this substructure.",
        )
        seed_smiles = seed_smiles.strip() or None
        if st.button("🚀 Generate", type="primary", use_container_width=True):
            with st.spinner("Generating molecules..."):
                result = pipeline.design_molecule(
                    target_success=target_success,
                    max_iterations=max_iterations,
                    candidates_per_iteration=200,
                    top_k=top_k_ui,
                    safety_threshold=safety_threshold,
                    require_no_structural_alerts=require_no_structural_alerts,
                    use_oracle_feedback=True,
                    property_targets=property_targets,
                    seed_smiles=seed_smiles,
                    on_iteration_done=lambda r: _render_result_into(results_placeholder, r),
                )
            st.session_state["generation_result"] = result
            _render_result_into(results_placeholder, result)


def analyze_page(pipeline):
    st.header("🔬 Analyze Molecule")
    smiles_input = st.text_input(
        "Enter SMILES", value="CC(=O)Oc1ccccc1C(=O)O"
    )
    if st.button("Analyze", type="primary"):
        if pipeline is None:
            st.warning("Models not loaded")
            return
        with st.spinner("Analyzing..."):
            prediction = pipeline.evaluate_molecule(smiles_input)
        if prediction is None:
            st.error("Invalid SMILES")
            return
        molecule_card(smiles_input, prediction, show_3d=True)
        oracle_dashboard(prediction)
        recommendation_panel(prediction.recommendations)


def compare_page(pipeline):
    st.header("📊 Compare Molecules")
    smiles_text = st.text_area(
        "Enter SMILES (one per line)",
        value="CC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccccc1",
        height=150,
    )
    if st.button("Compare", type="primary"):
        if pipeline is None:
            st.warning("Models not loaded")
            return
        smiles_list = [s.strip() for s in smiles_text.split("\n") if s.strip()]
        if len(smiles_list) < 2:
            st.warning("Enter at least 2 molecules")
            return
        with st.spinner("Comparing..."):
            results = pipeline.compare_molecules(smiles_list)
        comparison_table(results)


def about_page():
    st.header("📚 About SafeMolGen-DrugOracle")
    st.markdown(
        """
## Overview
SafeMolGen-DrugOracle is an integrated AI system for intelligent drug design:
1. Generates drug-like molecules using a Transformer-based generator
2. Predicts clinical trial success probability using DrugOracle
3. Guides generation through iterative feedback loops
4. Explains predictions with actionable recommendations
        """
    )


def main():
    init_state()
    st.markdown(
        '<h1 class="main-header">🧬 SafeMolGen-DrugOracle</h1>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        mode = st.radio(
            "Mode",
            ["🧪 Generate", "🔬 Analyze", "📊 Compare", "📚 About"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### Settings")
        target_success = st.slider("Target Success Probability", 0.1, 0.5, 0.25, 0.05)
        max_iterations = st.slider("Max Iterations", 1, 20, 10)
        safety_threshold = st.slider(
            "Safety threshold (min overall prob)",
            0.01,
            0.5,
            0.02,
            0.01,
            help="Below this, next iteration uses Oracle feedback. Use a low value (e.g. 0.02) with current models so the loop can complete.",
        )
        require_no_structural_alerts = st.checkbox(
            "Require no structural alerts to pass safety",
            value=False,
            help="If set, failing alerts also triggers Oracle feedback.",
        )
        use_rl_model = st.checkbox(
            "Use RL model",
            value=False,
            help="Use Oracle-fine-tuned generator (checkpoints/generator_rl) if available.",
        )
        st.caption(
            "Oracle conditioning steers generation when the loaded checkpoint was trained with cond_dim>0; otherwise only avoid-substructure filtering applies."
        )
        st.markdown("---")
        property_input()

    pipeline = load_pipeline(use_rl_model=use_rl_model)
    if pipeline is not None and getattr(pipeline.generator.model, "cond_dim", 0) == 0:
        st.warning(
            "Oracle conditioning is disabled (generator cond_dim=0). "
            "Only avoid-substructure filtering applies across iterations; success probability may not improve. "
            "Train the generator with conditioned pretrain to enable steering."
        )
    property_targets = st.session_state.get("property_targets")
    if mode == "🧪 Generate":
        generate_page(
            pipeline,
            target_success,
            max_iterations,
            safety_threshold=safety_threshold,
            require_no_structural_alerts=require_no_structural_alerts,
            property_targets=property_targets,
        )
    elif mode == "🔬 Analyze":
        analyze_page(pipeline)
    elif mode == "📊 Compare":
        compare_page(pipeline)
    else:
        about_page()


if __name__ == "__main__":
    main()
