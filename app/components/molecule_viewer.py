"""Molecule visualization component using RDKit and optionally py3Dmol."""

from typing import Optional, List

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import py3Dmol
except ImportError:
    py3Dmol = None


def draw_molecule_2d(
    smiles: str,
    highlight_atoms: Optional[List[int]] = None,
    size: tuple = (400, 300),
) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    if highlight_atoms:
        colors = {i: (1, 0, 0) for i in highlight_atoms}
        drawer.DrawMolecule(
            mol, highlightAtoms=highlight_atoms, highlightAtomColors=colors
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_molecule_3d(
    smiles: str,
    highlight_atoms: Optional[List[int]] = None,
    width: int = 400,
    height: int = 300,
) -> None:
    if py3Dmol is None:
        st.info("3D view requires `pip install py3Dmol`. Showing 2D instead.")
        svg = draw_molecule_2d(smiles, highlight_atoms=highlight_atoms, size=(width, height))
        if svg:
            st.image(svg, width="stretch")
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES")
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol_block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}})
    if highlight_atoms:
        for idx in highlight_atoms:
            viewer.addStyle(
                {"serial": int(idx)},
                {"sphere": {"color": "red", "radius": 0.5}},
            )
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    st.components.v1.html(viewer._make_html(), width=width, height=height)


def molecule_card(
    smiles: str,
    prediction=None,
    show_3d: bool = False,
    highlight_alerts: bool = True,
) -> None:
    col1, col2 = st.columns([1, 1])
    with col1:
        highlight_atoms = None
        if highlight_alerts and prediction and prediction.alert_atoms is not None:
            highlight_atoms = list(np.where(prediction.alert_atoms > 0)[0])
        if show_3d:
            draw_molecule_3d(smiles, highlight_atoms=highlight_atoms)
        else:
            svg = draw_molecule_2d(smiles, highlight_atoms=highlight_atoms)
            if svg:
                st.image(svg, width="stretch")
    with col2:
        st.code(smiles, language=None)
        if prediction:
            st.metric("Phase I", f"{prediction.phase1_prob:.1%}")
            st.metric("Phase II", f"{prediction.phase2_prob:.1%}")
            st.metric("Phase III", f"{prediction.phase3_prob:.1%}")
            st.metric("Overall", f"{prediction.overall_prob:.1%}")
