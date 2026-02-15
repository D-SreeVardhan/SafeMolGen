"""RDKit chemistry utilities."""

from typing import Optional, Dict, Any, List
import random

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, DataStructs
from torch_geometric.data import Data


RDLogger.DisableLog("rdApp.error")

# Reactions: replace one H with a substituent. [*:1] is heavy atom, [H:2] is the H we replace.
# Extended set to allow bigger property changes (e.g. CF3, nitrile, carboxyl for ADMET diversity).
_MUTATION_REACTIONS = [
    "[*:1]-[H:2]>>[*:1]-[F]",
    "[*:1]-[H:2]>>[*:1]-[Cl]",
    "[*:1]-[H:2]>>[*:1]-[OH]",
    "[*:1]-[H:2]>>[*:1]-[CH3]",
    "[*:1]-[H:2]>>[*:1]-[OCH3]",
    "[*:1]-[H:2]>>[*:1]-[NH2]",
    "[*:1]-[H:2]>>[*:1]-[CF3]",
    "[*:1]-[H:2]>>[*:1]-[C#N]",
    "[*:1]-[H:2]>>[*:1]-[C(=O)OH]",
    "[*:1]-[H:2]>>[*:1]-[SH]",
    "[*:1]-[H:2]>>[*:1]-[OC(=O)C]",
    "[*:1]-[H:2]>>[*:1]-[NHCH3]",
    "[*:1]-[H:2]>>[*:1]-[OCC]",
]


def validate_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def generate_mutations(
    smiles: str,
    n: int = 25,
    random_seed: Optional[int] = None,
) -> List[str]:
    """Generate up to n valid SMILES that are one small substituent change from the seed.
    Ensures optimization can improve even when the generator is unconditioned (cond_dim=0).
    Uses RDKit reactions to replace one H with F, Cl, OH, CH3, OCH3, NH2.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    rng = random.Random(random_seed)
    seen: set = set()
    out: List[str] = []
    orig_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True) if Chem.MolFromSmiles(smiles) else ""
    reactions = list(_MUTATION_REACTIONS)
    rng.shuffle(reactions)
    for rxn_smarts in reactions:
        if len(out) >= n:
            break
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn is None:
                continue
            RDLogger.DisableLog("rdApp.warning")
            try:
                product_tuples = rxn.RunReactants((mol,))
            finally:
                RDLogger.EnableLog("rdApp.warning")
            for t in product_tuples:
                if len(out) >= n:
                    break
                for p in t:
                    if p is None:
                        continue
                    try:
                        Chem.SanitizeMol(p)
                        s = Chem.MolToSmiles(p, canonical=True, allHsExplicit=False)
                        if not s or s in seen or s == orig_canon:
                            continue
                        if not validate_smiles(s):
                            continue
                        seen.add(s)
                        out.append(s)
                        break
                    except Exception:
                        continue
        except Exception:
            continue
    return out[:n]


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Tanimoto similarity between two SMILES (Morgan fingerprint). Returns 0.0 if either is invalid."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _atom_features(atom: Chem.Atom) -> List[float]:
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hybridization_one_hot = [
        1.0 if atom.GetHybridization() == h else 0.0 for h in hybridization_types
    ]
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        *hybridization_one_hot,
    ]


def _bond_features(bond: Chem.Bond) -> List[float]:
    bond_type = bond.GetBondType()
    return [
        1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def calculate_properties(smiles: str) -> Optional[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "qed": Descriptors.qed(mol),
    }


class MoleculeProcessor:
    """SMILES to graph conversion wrapper."""

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        return smiles_to_graph(smiles)
