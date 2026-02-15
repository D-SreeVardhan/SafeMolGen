"""2D molecule SVG via RDKit."""
from typing import Optional, List

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


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
