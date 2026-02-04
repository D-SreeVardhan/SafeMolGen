"""RDKit chemistry utilities."""

from typing import Optional, Dict, Any, List

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from torch_geometric.data import Data


RDLogger.DisableLog("rdApp.error")


def validate_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


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
