# GNN32 w/ Hyperopt changes

# external validaton on docked complexes version

import os
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdMolTransforms, EState, MolSurf, Crippen
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GeneralConv, global_mean_pool
from scipy.stats import zscore
from scipy.spatial import cKDTree, Voronoi
import networkx as nx
import pickle
from torch_geometric.utils import dropout_adj
from Bio import BiopythonWarning
from scipy.special import sph_harm
warnings.simplefilter('ignore', BiopythonWarning)

def get_partial_charges(mol):
    AllChem.ComputeGasteigerCharges(mol)
    partial_charges = {}
    for atom in mol.GetAtoms():
        partial_charges[atom.GetIdx()] = float(atom.GetProp('_GasteigerCharge'))
    return partial_charges

def get_torsion_angles(mol):
    torsion_angles = {}
    conformer = mol.GetConformer()
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        for neighbor1 in begin_atom.GetNeighbors():
            if neighbor1.GetIdx() != end_atom.GetIdx():
                for neighbor2 in end_atom.GetNeighbors():
                    if neighbor2.GetIdx() != begin_atom.GetIdx():
                        angle = rdMolTransforms.GetDihedralDeg(conformer, neighbor1.GetIdx(), begin_atom.GetIdx(), end_atom.GetIdx(), neighbor2.GetIdx())
                        torsion_angles[(neighbor1.GetIdx(), begin_atom.GetIdx(), end_atom.GetIdx(), neighbor2.GetIdx())] = angle
    return torsion_angles

def get_bond_angles(mol):
    bond_angles = {}
    conformer = mol.GetConformer()
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        for neighbor in begin_atom.GetNeighbors():
            if neighbor.GetIdx() != end_atom.GetIdx():
                angle = rdMolTransforms.GetAngleDeg(conformer, neighbor.GetIdx(), begin_atom.GetIdx(), end_atom.GetIdx())
                bond_angles[(neighbor.GetIdx(), begin_atom.GetIdx(), end_atom.GetIdx())] = angle
        for neighbor in end_atom.GetNeighbors():
            if neighbor.GetIdx() != begin_atom.GetIdx():
                angle = rdMolTransforms.GetAngleDeg(conformer, begin_atom.GetIdx(), end_atom.GetIdx(), neighbor.GetIdx())
                bond_angles[(begin_atom.GetIdx(), end_atom.GetIdx(), neighbor.GetIdx())] = angle
    return bond_angles

def get_bond_lengths(mol):
    bond_lengths = {}
    conformer = mol.GetConformer()
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bond_length = conformer.GetAtomPosition(idx1).Distance(conformer.GetAtomPosition(idx2))
        bond_lengths[(idx1, idx2)] = bond_length
    return bond_lengths

def get_bond_orders(mol):
    bond_orders = {}
    for bond in mol.GetBonds():
        bond_orders[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = bond.GetBondTypeAsDouble()
    return bond_orders

def get_electrostatic_interactions(mol):
    electrostatic_interactions = {}
    partial_charges = get_partial_charges(mol)
    conformer = mol.GetConformer()
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        distance = conformer.GetAtomPosition(idx1).Distance(conformer.GetAtomPosition(idx2))
        interaction = partial_charges[idx1] * partial_charges[idx2] / distance
        electrostatic_interactions[(idx1, idx2)] = interaction
    return electrostatic_interactions

def get_element(element):
    elements = ["C", "H", "N", "O", "S", "B", "P", "Halogen", "Metal"]
    one_hot = [0] * len(elements)
    
    halogens = ["F", "Cl", "Br", "I"]
    metals = [
        "Na", "K", "Ca", "Mg", "Fe", "Zn", "Cu", "Mn", "Co", "Ni", "Cr", "Mo", "V", "Se", "Li",
        "Al", "Si", "As", "Cd", "Pt", "Hg", "Au", "Ti", "W", "Pb", "Ag", "Bi", "Sb", "Tl", "Ru",
        "Rh", "Pd", "Os", "Ir", "Ga", "Ge", "Y", "La", "Ce", "Nd", "Sm", "Dy", "Yb"
    ]
    
    if element in elements:
        one_hot[elements.index(element)] = 1
    elif element in halogens:
        one_hot[elements.index("Halogen")] = 1
    elif element in metals:
        one_hot[elements.index("Metal")] = 1
    
    return one_hot


ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 
    'P': 2.19, 'B': 2.04, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Metals': 1.60, 'Halogens': 3.00  
}

def get_electronegativity(element):
    if element in ['F', 'Cl', 'Br', 'I']:
        return ELECTRONEGATIVITY['Halogens']
    elif element in ['Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:
        return ELECTRONEGATIVITY['Metals']
    else:
        return ELECTRONEGATIVITY.get(element, 0) 




def get_vdw_radius(element):
    vdw_radii = {
        'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.85, 'C': 1.70, 'N': 1.55, 'O': 1.52,
        'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80,
        'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31, 'Sc': 2.30, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05,
        'Mn': 2.05, 'Fe': 2.04, 'Co': 2.00, 'Ni': 2.00, 'Cu': 2.00, 'Zn': 2.10, 'Ga': 1.87, 'Ge': 2.11,
        'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.40, 'Zr': 2.30,
        'Nb': 2.15, 'Mo': 2.10, 'Tc': 2.05, 'Ru': 2.05, 'Rh': 2.00, 'Pd': 2.05, 'Ag': 2.10, 'Cd': 2.18,
        'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68,
        'La': 2.50, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45, 'Pm': 2.43, 'Sm': 2.42, 'Eu': 2.40, 'Gd': 2.38,
        'Tb': 2.37, 'Dy': 2.35, 'Ho': 2.33, 'Er': 2.32, 'Tm': 2.30, 'Yb': 2.28, 'Lu': 2.27, 'Hf': 2.25,
        'Ta': 2.20, 'W': 2.10, 'Re': 2.05, 'Os': 2.00, 'Ir': 2.00, 'Pt': 2.05, 'Au': 2.10, 'Hg': 2.05,
        'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20, 'Fr': 3.48, 'Ra': 2.83,
        'Ac': 2.00, 'Th': 2.40, 'Pa': 2.00, 'U': 1.86, 'Np': 1.90, 'Pu': 1.93, 'Am': 1.97, 'Cm': 2.00,
        'Bk': 2.00, 'Cf': 2.00, 'Es': 2.00, 'Fm': 2.00, 'Md': 2.00, 'No': 2.00, 'Lr': 2.00
    }
    return vdw_radii.get(element, 2.00)

def get_residue_encoding(atom):
    name = atom.GetPDBResidueInfo().GetResidueName()
    if name == "ALA":
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "ARG":
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "ASN":
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "ASP":
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "CYS":
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "GLN":
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "GLU":
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "GLY":
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "HIS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "ILE":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "LEU":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "LYS":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "MET":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == "PHE":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif name == "PRO":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif name == "SER":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif name == "THR":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif name == "TRP":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif name == "TYR":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif name == "VAL":
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def is_hydrogen_bond_donor(atom):
    if atom.GetAtomicNum() in (7, 8):
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  
                return 1
    return 0

def is_hydrogen_bond_acceptor(atom):
    if atom.GetAtomicNum() in (7, 8):  
        return 1
    return 0

def get_hydrophobicity_index(atom):
    if atom.GetAtomicNum() == 6:  
        return 1
    elif atom.GetAtomicNum() in (7, 8):  
        return -1
    else:
        return 0


def normalize_molecular_features(features):
    features = np.array(features)
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  
    normalized_features = (features - min_vals) / range_vals
    return normalized_features


def calculate_fingerprints(mol, radius=2, nBits=1024):
    mfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    httfp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
    hapfp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
    return np.array(httfp), np.array(hapfp), np.array(mfp)

def get_pocket_molecular_features(pdb_file):
    print("Starting pocket_molecular_features function...")
    mol = None

    if os.path.exists(pdb_file):
        try:
            mol = Chem.MolFromPDBFile(pdb_file)
            print(f'Successfully processed PDB file: {pdb_file}')
        except Exception as e:
            print(f'Error processing PDB file: {pdb_file}, Error: {e}')
            return None

    if mol is None:
        print("Molecule could not be loaded from PDB file.")
        return None
    
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print("Molecule sanitized successfully.")
    except Exception as e:
        print(f"Kekulization failed for molecule: {pdb_file}, attempting partial sanitization. Error: {e}")
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION)
            print("Partial sanitization succeeded.")
        except Exception as e:
            print(f"Partial sanitization failed for molecule: {pdb_file}. Error: {e}")
            return None
        
    try:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        print("Property cache updated and ring information initialized.")
    except Exception as e:
        print(f"Failed to update property cache or initialize ring information. Error: {e}")
        return None

    mol_features = [
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcKappa1(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcPBF(mol),
        rdMolDescriptors.CalcAsphericity(mol),
        rdMolDescriptors.CalcEccentricity(mol),
        rdMolDescriptors.CalcExactMolWt(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcInertialShapeFactor(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcChi0n(mol),
        rdMolDescriptors.CalcChi0v(mol),  
        Crippen.MolMR(mol),
        0,
        0,
        0,
        MolSurf.PEOE_VSA1(mol),
        MolSurf.PEOE_VSA2(mol),
        MolSurf.PEOE_VSA3(mol),
        MolSurf.PEOE_VSA4(mol),
        MolSurf.PEOE_VSA5(mol),
        MolSurf.PEOE_VSA6(mol),
        MolSurf.PEOE_VSA7(mol),
        MolSurf.PEOE_VSA8(mol),
        MolSurf.PEOE_VSA9(mol),
        MolSurf.PEOE_VSA10(mol),
        MolSurf.PEOE_VSA11(mol),
        MolSurf.PEOE_VSA12(mol),
        MolSurf.PEOE_VSA13(mol),
        MolSurf.PEOE_VSA14(mol),
        MolSurf.SMR_VSA1(mol),
        MolSurf.SMR_VSA2(mol),
        MolSurf.SMR_VSA3(mol),
        MolSurf.SMR_VSA4(mol),
        MolSurf.SMR_VSA5(mol),
        MolSurf.SMR_VSA6(mol),
        MolSurf.SMR_VSA7(mol),
        MolSurf.SMR_VSA8(mol),
        MolSurf.SlogP_VSA1(mol),
        MolSurf.SlogP_VSA2(mol),
        MolSurf.SlogP_VSA3(mol),
        MolSurf.SlogP_VSA4(mol),
        MolSurf.SlogP_VSA5(mol),
        MolSurf.SlogP_VSA6(mol),
        MolSurf.SlogP_VSA7(mol),
        MolSurf.SlogP_VSA8(mol),
        MolSurf.SlogP_VSA9(mol),
        MolSurf.SlogP_VSA10(mol),
        MolSurf.SlogP_VSA11(mol),
        MolSurf.SlogP_VSA12(mol),
        EState.EState_VSA.VSA_EState1(mol),
        EState.EState_VSA.VSA_EState2(mol),
        EState.EState_VSA.VSA_EState3(mol),
        EState.EState_VSA.VSA_EState4(mol),
        EState.EState_VSA.VSA_EState5(mol),
        EState.EState_VSA.VSA_EState6(mol),
        EState.EState_VSA.VSA_EState7(mol),
        EState.EState_VSA.VSA_EState8(mol),
        EState.EState_VSA.VSA_EState9(mol),
        EState.EState_VSA.VSA_EState10(mol)
    ]

    httfp, hapfp, mfp = calculate_fingerprints(mol)
    mol_features.extend(httfp)
    mol_features.extend(hapfp)
    mol_features.extend(mfp)

    return mol_features

def get_ligand_molecular_features(ligand_file):
    mol = None

    if os.path.exists(ligand_file):
        try:
            mol = Chem.MolFromPDBFile(ligand_file)
            print(f"Successfully processed MOL2 file: {ligand_file}")
        except Exception as e:
            print(f"Error processing MOL2 file: {ligand_file}, Error: {e}")



    if mol is None:
        print("Molecule could not be loaded from either MOL2 or SDF file.")
        return None

    try:
        mol = Chem.AddHs(mol)
        print("Explicit hydrogens added successfully.")
    except Exception as e:
        print(f"Adding explicit hydrogens failed for molecule: {ligand_file}. Error: {e}")
        return None

    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print("Molecule sanitized successfully.")
    except Exception as e:
        print(f"Kekulization failed for molecule: {ligand_file}, attempting partial sanitization. Error: {e}")
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION)
            print("Partial sanitization succeeded.")
        except Exception as e:
            print(f"Partial sanitization failed for molecule: {ligand_file}. Error: {e}")
            return None

    try:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        print("Property cache updated and ring information initialized.")
    except Exception as e:
        print(f"Failed to update property cache or initialize ring information. Error: {e}")
        return None

    mol_features = [
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcKappa1(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcPBF(mol),
        rdMolDescriptors.CalcAsphericity(mol),
        rdMolDescriptors.CalcEccentricity(mol),
        rdMolDescriptors.CalcExactMolWt(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcInertialShapeFactor(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcChi0n(mol),
        rdMolDescriptors.CalcChi0v(mol),    
        Descriptors.MolMR(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.qed(mol),
        Descriptors.PEOE_VSA1(mol),
        Descriptors.PEOE_VSA2(mol),
        Descriptors.PEOE_VSA3(mol),
        Descriptors.PEOE_VSA4(mol),
        Descriptors.PEOE_VSA5(mol),
        Descriptors.PEOE_VSA6(mol),
        Descriptors.PEOE_VSA7(mol),
        Descriptors.PEOE_VSA8(mol),
        Descriptors.PEOE_VSA9(mol),
        Descriptors.PEOE_VSA10(mol),
        Descriptors.PEOE_VSA11(mol),
        Descriptors.PEOE_VSA12(mol),
        Descriptors.PEOE_VSA13(mol),
        Descriptors.PEOE_VSA14(mol),
        Descriptors.SMR_VSA1(mol),
        Descriptors.SMR_VSA2(mol),
        Descriptors.SMR_VSA3(mol),
        Descriptors.SMR_VSA4(mol),
        Descriptors.SMR_VSA5(mol),
        Descriptors.SMR_VSA6(mol),
        Descriptors.SMR_VSA7(mol),
        Descriptors.SMR_VSA8(mol),
        Descriptors.SlogP_VSA1(mol),
        Descriptors.SlogP_VSA2(mol),
        Descriptors.SlogP_VSA3(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA5(mol),
        Descriptors.SlogP_VSA6(mol),
        Descriptors.SlogP_VSA7(mol),
        Descriptors.SlogP_VSA8(mol),
        Descriptors.SlogP_VSA9(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.SlogP_VSA11(mol),
        Descriptors.SlogP_VSA12(mol),
        Descriptors.VSA_EState1(mol),
        Descriptors.VSA_EState2(mol),
        Descriptors.VSA_EState3(mol),
        Descriptors.VSA_EState4(mol),
        Descriptors.VSA_EState5(mol),
        Descriptors.VSA_EState6(mol),
        Descriptors.VSA_EState7(mol),
        Descriptors.VSA_EState8(mol),
        Descriptors.VSA_EState9(mol),
        Descriptors.VSA_EState10(mol)
    ]

    httfp, hapfp, mfp = calculate_fingerprints(mol)
    mol_features.extend(httfp)
    mol_features.extend(hapfp)
    mol_features.extend(mfp)

    return mol_features


def compute_spherical_harmonics(coords, l_max=3):
    harmonics = []
    for coord in coords:
        r, theta, phi = np.linalg.norm(coord), np.arccos(coord[2]/np.linalg.norm(coord)), np.arctan2(coord[1], coord[0])
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                harmonics.append(np.real(sph_harm(m, l, phi, theta)))
                harmonics.append(np.imag(sph_harm(m, l, phi, theta)))
    return harmonics

def docked_protein_pocket_features_to_graph(pdb_file, ligand_file):
    print("Starting protein_pocket_features_to_graph function...")
    print(f"Protein file: {pdb_file}, Ligand file: {ligand_file}")
    mol = None

    if os.path.exists(pdb_file):
        try:
            mol = Chem.MolFromPDBFile(pdb_file)
            print(f'Successfully processed PDB file: {pdb_file}')
        except Exception as e:
            print(f'Error processing PDB file: {pdb_file}, Error: {e}')
            return None

    if mol is None:
        print("Molecule could not be loaded from PDB file.")
        return None

    ligand = Chem.MolFromPDBFile(ligand_file)
    if not ligand:
        print("Failed to load ligand molecule.")
        return None

    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print("Molecule sanitized successfully.")
    except Exception as e:
        print(f"Kekulization failed for molecule: {pdb_file}, attempting partial sanitization. Error: {e}")
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION)
            print("Partial sanitization succeeded.")
        except Exception as e:
            print(f"Partial sanitization failed for molecule: {pdb_file}. Error: {e}")
            return None
        
    try:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        print("Property cache updated and ring information initialized.")
    except Exception as e:
        print(f"Failed to update property cache or initialize ring information. Error: {e}")
        return None

    ligand_coords = [atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()]
    ligand_coords = np.array(ligand_coords)
    ligand_tree = cKDTree(ligand_coords)

    torsion_angles = get_torsion_angles(mol)
    bond_angles = get_bond_angles(mol)
    bond_lengths = get_bond_lengths(mol)
    bond_orders = get_bond_orders(mol)
    electrostatic_interactions = get_electrostatic_interactions(mol)

    vor = Voronoi([atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()])
    # dssp = DSSP(PDBParser().get_structure('X', pdb_file)[0], pdb_file)

    graph = nx.Graph()
    node_features = []
    atom_indices = []
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        element = atom.GetSymbol()
        coord = mol.GetConformer().GetAtomPosition(atom_index)
        coord_list = [coord.x, coord.y, coord.z]

        if ligand_tree.query_ball_point(coord, 15): 
            atom_features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                # atom.GetFormalCharge(),
                # atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                # atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
                # atom.GetIsotope(),
                # atom.GetImplicitValence(),
                atom.GetMass(),
                is_hydrogen_bond_donor(atom),  
                is_hydrogen_bond_acceptor(atom),
                get_hydrophobicity_index(atom),
                get_electronegativity(element)
            ]
            # if (atom.get_parent().get_id(), atom.get_id()) in dssp:
            #     atom_features.append(dssp[(atom.get_parent().get_id(), atom.get_id())][2])  
            # else:
            #     atom_features.append(0) 
            atom_features.extend(get_element(element))
            atom_features.extend(get_residue_encoding(atom))
            atom_features.extend(coord_list)

            voronoi_region = vor.regions[vor.point_region[atom_index]]
            voronoi_features = voronoi_region if len(voronoi_region) <= 10 else voronoi_region[:10]
            voronoi_features += [0] * (10 - len(voronoi_features))
            atom_features.extend(voronoi_features)

            spherical_harmonics = compute_spherical_harmonics([coord_list])
            atom_features.extend(spherical_harmonics)

            while len(atom_features) < 80:
                atom_features.append(0)

            node_features.append(atom_features)
            atom_indices.append(atom_index) 
            graph.add_node(atom_index, features=torch.tensor(atom_features, dtype=torch.float))
    
    print("Nodes added to the graph.")
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}

    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        
        if start_atom in idx_map and end_atom in idx_map:  
            bond_features = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                # bond.GetIsAromatic(),
                # int(bond.GetStereo()),
                # int(bond.GetBondDir()),
                bond_orders[(start_atom, end_atom)],
                bond_lengths[(start_atom, end_atom)],
                # bond_angles.get((start_atom, end_atom), bond_angles.get((end_atom, start_atom), 0)),
                # torsion_angles.get((start_atom, end_atom), torsion_angles.get((end_atom, start_atom), 0)),
                electrostatic_interactions[(start_atom, end_atom)]
            ]
            
            graph.add_edge(idx_map[start_atom], idx_map[end_atom], features=torch.tensor(bond_features, dtype=torch.float))
    print("Edges added to the graph.")

    if len(graph.nodes) == 0:
        print(f"No nodes in graph for {pdb_file}. Returning None.")
        return None
    
    if len(graph.edges) == 0:
        print(f"No edges in graph for {pdb_file}. Returning None.")
        return None
    
    # print("Graph nodes and edges before conversion:")
    # for node in graph.nodes(data=True):
    #     print(f"Node {node[0]} feature length: {len(node[1]['features'])}")
    # for edge in graph.edges(data=True):
    #     print(f"Edge {edge[0]}-{edge[1]}: {edge[2]}")

    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    edge_attr = torch.stack([graph.edges[edge]['features'] for edge in graph.edges], dim=0)
    x = torch.stack([graph.nodes[node]['features'] for node in graph.nodes if 'features' in graph.nodes[node]], dim=0)

    mol_features = get_pocket_molecular_features(pdb_file)
    if mol_features is None:
        return None
    mol_features = normalize_molecular_features([mol_features])[0]

    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=torch.tensor(mol_features, dtype=torch.float))
    
    # print(f"Converted graph: {pyg_graph}")
    # print(f"Edge attributes: {pyg_graph.edge_attr}")
    # print(f"Node attributes: {pyg_graph.x}")
    # print(f"Pocket type: {type(pyg_graph)}")


    node_min = pyg_graph.x.min(dim=0).values
    node_max = pyg_graph.x.max(dim=0).values
    node_range = node_max - node_min
    node_range[node_range == 0] = 1  
    pyg_graph.x = (pyg_graph.x - node_min) / node_range

    pyg_graph.x[torch.isnan(pyg_graph.x)] = 0.0
    pyg_graph.x[torch.isinf(pyg_graph.x)] = 0.0

    edge_min = pyg_graph.edge_attr.min(dim=0).values
    edge_max = pyg_graph.edge_attr.max(dim=0).values
    edge_range = edge_max - edge_min
    edge_range[edge_range == 0] = 1  
    pyg_graph.edge_attr = (pyg_graph.edge_attr - edge_min) / edge_range

    pyg_graph.edge_attr[torch.isnan(pyg_graph.edge_attr)] = 0.0
    pyg_graph.edge_attr[torch.isinf(pyg_graph.edge_attr)] = 0.0


    return pyg_graph


def docked_ligand_features_to_graph(ligand_file):
    print("Starting ligand_features_to_graph function...")
    mol = None

    if os.path.exists(ligand_file):
        try:
            mol = Chem.MolFromPDBFile(ligand_file)
            print(f"Successfully processed ligand file: {ligand_file}")
        except Exception as e:
            print(f"Error processing ligand file: {ligand_file}, Error: {e}")

    if mol is None:
        print("Molecule could not be loaded from either MOL2 or SDF file.")
        return None

    try:
        mol = Chem.AddHs(mol)
        print("Explicit hydrogens added successfully.")
    except Exception as e:
        print(f"Adding explicit hydrogens failed for molecule: {ligand_file}. Error: {e}")
        return None

    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print("Molecule sanitized successfully.")
    except Exception as e:
        print(f"Kekulization failed for molecule: {ligand_file}, attempting partial sanitization. Error: {e}")
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION)
            print("Partial sanitization succeeded.")
        except Exception as e:
            print(f"Partial sanitization failed for molecule: {ligand_file}. Error: {e}")
            return None

    try:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        print("Property cache updated and ring information initialized.")
    except Exception as e:
        print(f"Failed to update property cache or initialize ring information. Error: {e}")
        return None

    torsion_angles = get_torsion_angles(mol)
    bond_angles = get_bond_angles(mol)
    bond_lengths = get_bond_lengths(mol)
    bond_orders = get_bond_orders(mol)
    electrostatic_interactions = get_electrostatic_interactions(mol)

    coords = [atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    vor = Voronoi(coords)

    graph = nx.Graph()
    node_features = []
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        element = atom.GetSymbol()
        coord = mol.GetConformer().GetAtomPosition(atom_index)
        coord_list = [coord.x, coord.y, coord.z]
        residue_encoding = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        atom_features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            # atom.GetFormalCharge(),
            # atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            # atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            # atom.GetIsotope(),
            # atom.GetImplicitValence(),
            atom.GetMass(),
            is_hydrogen_bond_donor(atom),  
            is_hydrogen_bond_acceptor(atom),
            get_hydrophobicity_index(atom),
            get_electronegativity(element)
        ]
        atom_features.extend(get_element(element))
        atom_features.extend(residue_encoding)
        atom_features.extend(coord_list)

        voronoi_region = vor.regions[vor.point_region[atom_index]]
        voronoi_features = voronoi_region if len(voronoi_region) <= 10 else voronoi_region[:10]
        voronoi_features += [0] * (10 - len(voronoi_features))
        atom_features.extend(voronoi_features)

        spherical_harmonics = compute_spherical_harmonics([coord_list])
        atom_features.extend(spherical_harmonics)

        while len(atom_features) < 80:
            atom_features.append(0)

        graph.add_node(atom_index, features=torch.tensor(atom_features, dtype=torch.float))
        node_features.append(atom_features)
    
    print("Nodes added to the graph.")
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            # float(bond.GetIsAromatic()),
            # float(bond.GetStereo()),
            # float(bond.GetBondDir()),
            bond_orders[(start_atom, end_atom)],
            bond_lengths[(start_atom, end_atom)],
            # bond_angles.get((start_atom, end_atom), bond_angles.get((end_atom, start_atom), 0)),
            # torsion_angles.get((start_atom, end_atom), torsion_angles.get((end_atom, start_atom), 0)),
            electrostatic_interactions[(start_atom, end_atom)]
        ]
        graph.add_edge(start_atom, end_atom, features=torch.tensor(bond_features, dtype=torch.float))
    print("Edges added to the graph.")

    x = torch.stack([graph.nodes[node]['features'] for node in graph.nodes], dim=0)
    edge_index = torch.tensor([[edge[0], edge[1]] for edge in graph.edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.stack([graph.edges[edge]['features'] for edge in graph.edges], dim=0)

    mol_features = get_ligand_molecular_features(ligand_file)
    if mol_features is None:
        return None
    mol_features = normalize_molecular_features([mol_features])[0]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=torch.tensor(mol_features, dtype=torch.float))
    # print(f"Converted graph: {data}")
    # print(f"Edge attributes: {data.edge_attr}")
    # print(f"Ligand type: {type(data)}")


    node_min = data.x.min(dim=0).values
    node_max = data.x.max(dim=0).values
    node_range = node_max - node_min
    node_range[node_range == 0] = 1  
    data.x = (data.x - node_min) / node_range

    data.x[torch.isnan(data.x)] = 0.0
    data.x[torch.isinf(data.x)] = 0.0

    edge_min = data.edge_attr.min(dim=0).values
    edge_max = data.edge_attr.max(dim=0).values
    edge_range = edge_max - edge_min
    edge_range[edge_range == 0] = 1  
    data.edge_attr = (data.edge_attr - edge_min) / edge_range

    data.edge_attr[torch.isnan(data.edge_attr)] = 0.0
    data.edge_attr[torch.isinf(data.edge_attr)] = 0.0

    print("Ligand Graph Node Features Normalized Min:", data.x.min(dim=0).values.numpy())
    print("Ligand Graph Node Features Normalized Max:", data.x.max(dim=0).values.numpy())
    print("Ligand Graph Edge Features Normalized Min:", data.edge_attr.min(dim=0).values.numpy())
    print("Ligand Graph Edge Features Normalized Max:", data.edge_attr.max(dim=0).values.numpy())


    return data





def calculate_min_max_features(proteins, ligands):
    print(f"Proteins: {proteins}")
    num_node_features_protein = proteins[0].x.shape[1]
    num_edge_features_protein = proteins[0].edge_attr.shape[1]
    num_node_features_ligand = ligands[0].x.shape[1]
    num_edge_features_ligand = ligands[0].edge_attr.shape[1]

    node_min_protein = np.full(num_node_features_protein, np.inf)
    node_max_protein = np.full(num_node_features_protein, -np.inf)
    edge_min_protein = np.full(num_edge_features_protein, np.inf)
    edge_max_protein = np.full(num_edge_features_protein, -np.inf)

    node_min_ligand = np.full(num_node_features_ligand, np.inf)
    node_max_ligand = np.full(num_node_features_ligand, -np.inf)
    edge_min_ligand = np.full(num_edge_features_ligand, np.inf)
    edge_max_ligand = np.full(num_edge_features_ligand, -np.inf)

    for data in proteins:
        node_min_protein = np.minimum(node_min_protein, data.x.min(dim=0).values.numpy())
        node_max_protein = np.maximum(node_max_protein, data.x.max(dim=0).values.numpy())
        edge_min_protein = np.minimum(edge_min_protein, data.edge_attr.min(dim=0).values.numpy())
        edge_max_protein = np.maximum(edge_max_protein, data.edge_attr.max(dim=0).values.numpy())

    for data in ligands:
        node_min_ligand = np.minimum(node_min_ligand, data.x.min(dim=0).values.numpy())
        node_max_ligand = np.maximum(node_max_ligand, data.x.max(dim=0).values.numpy())
        edge_min_ligand = np.minimum(edge_min_ligand, data.edge_attr.min(dim=0).values.numpy())
        edge_max_ligand = np.maximum(edge_max_ligand, data.edge_attr.max(dim=0).values.numpy())

    # print(f"Protein Node Features Min: {node_min_protein}, Max: {node_max_protein}")
    # print(f"Protein Edge Features Min: {edge_min_protein}, Max: {edge_max_protein}")
    # print(f"Ligand Node Features Min: {node_min_ligand}, Max: {node_max_ligand}")
    # print(f"Ligand Edge Features Min: {edge_min_ligand}, Max: {edge_max_ligand}")

    return node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand

def normalize_features(data, node_min, node_max, edge_min, edge_max):
    node_range = torch.tensor(node_max, dtype=torch.float) - torch.tensor(node_min, dtype=torch.float)
    node_range[node_range == 0] = 1
    data.x = (data.x - torch.tensor(node_min, dtype=torch.float)) / node_range
    
    edge_range = torch.tensor(edge_max, dtype=torch.float) - torch.tensor(edge_min, dtype=torch.float)
    edge_range[edge_range == 0] = 1
    data.edge_attr = (data.edge_attr - torch.tensor(edge_min, dtype=torch.float)) / edge_range

    data.x[torch.isnan(data.x)] = 0.0
    data.x[torch.isinf(data.x)] = 0.0
    data.edge_attr[torch.isnan(data.edge_attr)] = 0.0
    data.edge_attr[torch.isinf(data.edge_attr)] = 0.0

    # print(f"Normalized Node Features Min: {data.x.min(dim=0).values.numpy()}, Max: {data.x.max(dim=0).values.numpy()}")
    # print(f"Normalized Edge Features Min: {data.edge_attr.min(dim=0).values.numpy()}, Max: {data.edge_attr.max(dim=0).values.numpy()}")

    assert data.x.min().item() >= 0 and data.x.max().item() <= 1, "Node features are not properly normalized!"
    assert data.edge_attr.min().item() >= 0 and data.edge_attr.max().item() <= 1, "Edge features are not properly normalized!"

    return data

def calculate_morgan_fingerprint(mol, radius=2, nBits=1024):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

def print_feature_distributions(data_list, feature_type="Node"):
    all_features = torch.cat([data.x if feature_type == "Node" else data.edge_attr for data in data_list], dim=0)
    for i in range(all_features.shape[1]):
        feature_column = all_features[:, i]
        print(f"{feature_type} Feature {i}: min = {feature_column.min().item()}, max = {feature_column.max().item()}, mean = {feature_column.mean().item()}")

def save_graphs(graphs, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(graphs, f)

def load_graphs(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


    
# rfiles = random.choices(subdirectories, k=5315)
# training_files = random.choices(rfiles, k=4252)
# testing_files = [entry for entry in rfiles if entry not in training_files]



def prepare_data(dataset_path, outlier_ids=set(), save_path=None, load_saved=False):
    ligands = []
    proteins = []
    affinities = []
    pdb_ids = []
    ic50_ids = []

    entries = os.listdir(dataset_path)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_path, entry))]

    if load_saved and save_path:
        saved_data = load_graphs(save_path)
        if saved_data:
            proteins, ligands, affinities, pdb_ids = saved_data['proteins'], saved_data['ligands'], saved_data['affinities'], saved_data['pdb_ids']
            


            print(f"Length of proteins: {len(proteins)}")
            print(f"Length of ligands: {len(ligands)}")



            proteins = [proteins[i] for i in range(len(proteins)) if pdb_ids[i] not in ic50_ids]
            ligands = [ligands[i] for i in range(len(ligands)) if pdb_ids[i] not in ic50_ids]
            affinities = [affinities[i] for i in range(len(affinities)) if pdb_ids[i] not in ic50_ids]
            pdb_ids = [pdb_ids[i] for i in range(len(pdb_ids)) if pdb_ids[i] not in ic50_ids]
            


            print(f"Length of proteins after ic50 removal: {len(proteins)}")
            print(f"Length of ligands after ic50 removal: {len(ligands)}")

            assert len(proteins) == len(ligands) == len(affinities) == len(pdb_ids), "Mismatch in lengths of data lists!"

            for i in range(len(proteins)):
                assert pdb_ids[i] not in ic50_ids, f"ic50 id {pdb_ids[i]} should have been removed!"

            print("All checks passed: Lists are synchronized and ic50 ids are excluded.")


            z_scores = zscore(affinities)
            outlier_threshold = 3  
            
            print(f"Z-scores of affinities: {z_scores}")
            print(f"Outlier threshold: {outlier_threshold}")

            non_outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) <= outlier_threshold]
            outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) > outlier_threshold]

            print(f"Outlier indices: {outlier_indices}")
            print(f"Number of outliers: {len(outlier_indices)}")

            proteins = [proteins[i] for i in non_outlier_indices]
            ligands = [ligands[i] for i in non_outlier_indices]
            affinities = [affinities[i] for i in non_outlier_indices]
            pdb_ids = [pdb_ids[i] for i in non_outlier_indices]



            print(f"Number of graphs before outlier removal: {len(saved_data['proteins'])}")
            print(f"Number of graphs after outlier removal: {len(proteins)}")

            node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
            proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
            ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]

            return proteins, ligands, affinities, pdb_ids

    files_to_process = subdirectories

    code = files_to_process[0]
  
    protein_name = code[0:4]
    ligand_file1 = f"{dataset_path}/{protein_name}_CINPA1/CINPA1_out.pdb"
    ligand_name_1 = "CINPA1"
    ligand_file2 = f"{dataset_path}/{protein_name}_CITCO/CITCO_out.pdb"
    ligand_name_2 = "CITCO"
    ligand_file3 = f"{dataset_path}/{protein_name}_clotrimazole/clotrimazole_out.pdb"
    ligand_name_3 = "Clotrimazole"
    ligand_file4 = f"{dataset_path}/{protein_name}_PK11195/PK11195_out.pdb"
    ligand_name_4 = "PK11195"
    ligand_file5 = f"{dataset_path}/{protein_name}_TO901317/TO901317_out.pdb"
    ligand_name_5 = "TO901317"

    pocket_file = f"{dataset_path}/{protein_name}_TO901317/1XNXRECEPTOR.pdb"

    complexes = [(pocket_file, ligand_file1, f"{protein_name}_{ligand_name_1}"), (pocket_file, ligand_file2, f"{protein_name}_{ligand_name_2}"), (pocket_file, ligand_file3, f"{protein_name}_{ligand_name_3}"), (pocket_file, ligand_file4, f"{protein_name}_{ligand_name_4}"), (pocket_file, ligand_file5, f"{protein_name}_{ligand_name_5}")]
    # print(f"COMPLEXES: {complexes}")
    # print(f"Pocket file in prepare data function: {pocket_file}")

    # files2 = subdirectories2

    # code2 = files2[0]

    # protein_name2 = code2[0:4]
    # ligand_file1_2 = f"{dataset_path_2}/{protein_name2}_CINPA1/CINPA1_out.pdb"

    for complex in complexes:
        try:
            pocket_graph = docked_protein_pocket_features_to_graph(complex[0], complex[1])
            ligand_graph = docked_ligand_features_to_graph(complex[1])
            
            if pocket_graph is None or ligand_graph is None:
                raise ValueError("Graph creation failed.")
            
            proteins.append(pocket_graph)
            ligands.append(ligand_graph)
            # affinities.append(affinity_dict[code])
            pdb_id = complex[2]
            # print(f"Complex (this should be purely the ID: {pdb_id}")
            pdb_ids.append(pdb_id)
        except Exception as e:
            print(f"Error processing files for code: {code}, Error: {e}")
            continue

    # node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
    # proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
    # ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]

    print(f"Number of proteins: {len(proteins)}")
    print(f"Number of ligands: {len(ligands)}")
    # print(f"Number of affinities: {len(affinities)}")

    data = {
        'proteins': proteins,
        'ligands': ligands,
        # 'affinities': affinities,
        'pdb_ids': pdb_ids
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    print(f"Codes: {pdb_ids}")

    targets = [7.7, 7.5, 7.2, 7.4, 8.3]


    return proteins, ligands, targets, pdb_ids

def prepare_data_2(dataset_path, outlier_ids=set(), save_path=None, load_saved=False):
    ligands = []
    proteins = []
    affinities = []
    pdb_ids = []
    ic50_ids = []

    entries = os.listdir(dataset_path)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_path, entry))]

    if load_saved and save_path:
        saved_data = load_graphs(save_path)
        if saved_data:
            proteins, ligands, affinities, pdb_ids = saved_data['proteins'], saved_data['ligands'], saved_data['affinities'], saved_data['pdb_ids']
            


            print(f"Length of proteins: {len(proteins)}")
            print(f"Length of ligands: {len(ligands)}")


            proteins = [proteins[i] for i in range(len(proteins)) if pdb_ids[i] not in ic50_ids]
            ligands = [ligands[i] for i in range(len(ligands)) if pdb_ids[i] not in ic50_ids]
            affinities = [affinities[i] for i in range(len(affinities)) if pdb_ids[i] not in ic50_ids]
            pdb_ids = [pdb_ids[i] for i in range(len(pdb_ids)) if pdb_ids[i] not in ic50_ids]
            


            print(f"Length of proteins after ic50 removal: {len(proteins)}")
            print(f"Length of ligands after ic50 removal: {len(ligands)}")

            assert len(proteins) == len(ligands) == len(affinities) == len(pdb_ids), "Mismatch in lengths of data lists!"

            for i in range(len(proteins)):
                assert pdb_ids[i] not in ic50_ids, f"ic50 id {pdb_ids[i]} should have been removed!"

            print("All checks passed: Lists are synchronized and ic50 ids are excluded.")


            z_scores = zscore(affinities)
            outlier_threshold = 3  
            
            print(f"Z scores of affinities: {z_scores}")
            print(f"Outlier threshold: {outlier_threshold}")

            non_outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) <= outlier_threshold]
            outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) > outlier_threshold]

            print(f"Outlier indices: {outlier_indices}")
            print(f"Number of outliers: {len(outlier_indices)}")

            proteins = [proteins[i] for i in non_outlier_indices]
            ligands = [ligands[i] for i in non_outlier_indices]
            affinities = [affinities[i] for i in non_outlier_indices]
            pdb_ids = [pdb_ids[i] for i in non_outlier_indices]



            print(f"Number of graphs before outlier removal: {len(saved_data['proteins'])}")
            print(f"Number of graphs after outlier removal: {len(proteins)}")

            node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
            proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
            ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]

            return proteins, ligands, affinities, pdb_ids

    files_to_process = subdirectories

    code = files_to_process[0]

    print(f"****CODE: {code}")
  
    protein_name = code[0:5]
    ligand_file1 = f"{dataset_path}/{protein_name}_DHT/2piv_C_DHT_out.pdb"
    ligand_name_1 = "DHT"
    ligand_file3 = f"{dataset_path}/{protein_name}_Flutamide/Flutamide_out.pdb"
    ligand_name_3 = "Flutamide"
    ligand_file5 = f"{dataset_path}/{protein_name}_MethylTestosterone/MethylTestosterone_out.pdb"
    ligand_name_5 = "MethylTestosterone"
    ligand_file6 = f"{dataset_path}/{protein_name}_R1881/R1881_out.pdb"
    ligand_name_6 = "R1881"
    ligand_file9 = f"{dataset_path}/{protein_name}_testosterone/testosterone_out.pdb"
    ligand_name_9 = "testosterone"
    ligand_file10 = f"{dataset_path}/{protein_name}_TolfenamicAcid/TolfenamicAcid_out.pdb"
    ligand_name_10 = "TolfenamicAcid"
    ligand_file10 = f"{dataset_path}/{protein_name}_Spironolactone/Spironolactone_out.pdb"
    ligand_name_10 = "Spironolactone"


    pocket_file = f"{dataset_path}/{protein_name}_DHT/NewAR.pdb"

    complexes = [(pocket_file, ligand_file1, f"{protein_name}_{ligand_name_1}"), (pocket_file, ligand_file3, f"{protein_name}_{ligand_name_3}"), (pocket_file, ligand_file5, f"{protein_name}_{ligand_name_5}"), (pocket_file, ligand_file6, f"{protein_name}_{ligand_name_6}"), (pocket_file, ligand_file9, f"{protein_name}_{ligand_name_9}"), (pocket_file, ligand_file10, f"{protein_name}_{ligand_name_10}"), (pocket_file, ligand_file10, f"{protein_name}_{ligand_name_10}")]

    # files2 = subdirectories2

    # code2 = files2[0]

    # protein_name2 = code2[0:4]
    # ligand_file1_2 = f"{dataset_path_2}/{protein_name2}_CINPA1/CINPA1_out.pdb"

    for complex in complexes:
        try:
            pocket_graph = docked_protein_pocket_features_to_graph(complex[0], complex[1])
            ligand_graph = docked_ligand_features_to_graph(complex[1])
            
            if pocket_graph is None or ligand_graph is None:
                raise ValueError("Graph creation failed.")
            
            proteins.append(pocket_graph)
            ligands.append(ligand_graph)
            # affinities.append(affinity_dict[code])
            pdb_id = complex[2]
            # print(f"Complex (this should be purely the ID: {pdb_id}")
            pdb_ids.append(pdb_id)
        except Exception as e:
            print(f"Error processing files for code: {code}, Error: {e}")
            continue
    
    # node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
    # proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
    # ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]

    print(f"Number of proteins: {len(proteins)}")
    print(f"Number of ligands: {len(ligands)}")
    # print(f"Number of affinities: {len(affinities)}")

    data = {
        'proteins': proteins,
        'ligands': ligands,
        # 'affinities': affinities,
        'pdb_ids': pdb_ids
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    print(f"Codes: {pdb_ids}")

    # targets = [7.147, 7.301, 6.158, 6.085, 5.652
    
    targets = [7.7, 7.13, 7.7, 8.49, 7.85, 4.33, 6.14]


    return proteins, ligands, targets, pdb_ids

def prepare_training_data(dataset_path, index_file, outlier_ids=set(), save_path=None, gen_save_path=None, load_saved=True):
    ligands = []
    proteins = []
    affinities = []
    pdb_ids = []

    entries = os.listdir(dataset_path)
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(dataset_path, entry))]

    affinity_dict = {}
    with open(index_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            pdb_id = parts[0].strip()
            if pdb_id not in subdirectories or pdb_id in outlier_ids:
                continue
            if '=' in parts[3] and 'IC50' in parts[3]:
                continue  
            affinity = float(parts[3].strip())
            affinity_dict[pdb_id] = affinity

    affinities_all = list(affinity_dict.values())
    pdb_ids_all = list(affinity_dict.keys())

    z_scores = zscore(affinities_all)
    outlier_threshold = 3  
    outlier_pdb_ids = [pdb_ids_all[i] for i in range(len(pdb_ids_all)) if abs(z_scores[i]) > outlier_threshold]


    if load_saved and save_path and gen_save_path:
        saved_data = load_graphs(save_path)
        gen_saved_data = load_graphs(gen_save_path)
        if saved_data and gen_saved_data:
            proteins, ligands, affinities, pdb_ids = saved_data['proteins'], saved_data['ligands'], saved_data['affinities'], saved_data['pdb_ids']
            
            z_scores = zscore(affinities)
            outlier_threshold = 3  
            
            print(f"Z scores of affinities: {z_scores}")
            print(f"Outlier threshold: {outlier_threshold}")

            non_outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) <= outlier_threshold]
            outlier_indices = [i for i in range(len(affinities)) if abs(z_scores[i]) > outlier_threshold]

            print(f"Outlier indices: {outlier_indices}")
            print(f"Number of outliers: {len(outlier_indices)}")

            proteins = [proteins[i] for i in non_outlier_indices]
            ligands = [ligands[i] for i in non_outlier_indices]
            affinities = [affinities[i] for i in non_outlier_indices]
            pdb_ids = [pdb_ids[i] for i in non_outlier_indices]

            print(f"Number of graphs before outlier removal: {len(saved_data['proteins'])}")
            print(f"Number of graphs after outlier removal: {len(proteins)}")

            node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
            proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
            ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]
            
            gen_proteins, gen_ligands, gen_affinities, gen_pdb_ids = gen_saved_data['proteins'], gen_saved_data['ligands'], gen_saved_data['affinities'], gen_saved_data['pdb_ids']
            
            gen_z_scores = zscore(gen_affinities)
            gen_outlier_threshold = 3  
            
            print(f"Z-scores of affinities: {gen_z_scores}")
            print(f"Outlier threshold: {gen_outlier_threshold}")

            gen_non_outlier_indices = [i for i in range(len(gen_affinities)) if abs(gen_z_scores[i]) <= gen_outlier_threshold]
            gen_outlier_indices = [i for i in range(len(gen_affinities)) if abs(gen_z_scores[i]) > gen_outlier_threshold]

            print(f"Outlier indices: {gen_outlier_indices}")
            print(f"Number of outliers: {len(gen_outlier_indices)}")

            gen_proteins = [gen_proteins[i] for i in gen_non_outlier_indices]
            gen_ligands = [gen_ligands[i] for i in gen_non_outlier_indices]
            gen_affinities = [gen_affinities[i] for i in gen_non_outlier_indices]
            gen_pdb_ids = [gen_pdb_ids[i] for i in gen_non_outlier_indices]

            print(f"Number of graphs before outlier removal: {len(gen_saved_data['proteins'])}")
            print(f"Number of graphs after outlier removal: {len(gen_proteins)}")

            gen_node_min_protein, gen_node_max_protein, gen_edge_min_protein, gen_edge_max_protein, gen_node_min_ligand, gen_node_max_ligand, gen_edge_min_ligand, gen_edge_max_ligand = calculate_min_max_features(gen_proteins, gen_ligands)
            gen_proteins = [normalize_features(data, gen_node_min_protein, gen_node_max_protein, gen_edge_min_protein, gen_edge_max_protein) for data in gen_proteins]
            gen_ligands = [normalize_features(data, gen_node_min_ligand, gen_node_max_ligand, gen_edge_min_ligand, gen_edge_max_ligand) for data in gen_ligands]

            pro = proteins + gen_proteins
            lig = ligands + gen_ligands
            aff = affinities + gen_affinities
            ids = pdb_ids + gen_pdb_ids


            proteins = pro
            ligands = lig
            affinities = aff
            pdb_ids = ids

            print(f"Length of proteins list: {len(proteins)}")
            print(f"Length of ligands list: {len(ligands)}")
            print(f"Length of affinities list: {len(affinities)}")
            print(f"Length of pdb ids list: {len(pdb_ids)}")

            return proteins, ligands, affinities, pdb_ids

    files_to_process = subdirectories

    for code in files_to_process:
        if code in outlier_pdb_ids:
            continue  

        ligand_file = f"{dataset_path}/{code}/{code}_hydrogenated_ligand.pdb"
        pocket_file = f"{dataset_path}/{code}/{code}_hydrogenated_pocket.pdb"

        if code not in affinity_dict:
            continue  
        try:
            pocket_graph = docked_protein_pocket_features_to_graph(pocket_file, ligand_file)
            ligand_graph = docked_ligand_features_to_graph(ligand_file)

            if pocket_graph is None or ligand_graph is None:
                raise ValueError("Graph creation failed.")
            
            proteins.append(pocket_graph)
            ligands.append(ligand_graph)
            affinities.append(affinity_dict[code])
            pdb_ids.append(code)
        
        except Exception as e:
            print(f"Error processing files for code: {code}, Error: {e}")
            continue

    node_min_protein, node_max_protein, edge_min_protein, edge_max_protein, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand = calculate_min_max_features(proteins, ligands)
    proteins = [normalize_features(data, node_min_protein, node_max_protein, edge_min_protein, edge_max_protein) for data in proteins]
    ligands = [normalize_features(data, node_min_ligand, node_max_ligand, edge_min_ligand, edge_max_ligand) for data in ligands]

    print(f"Number of proteins: {len(proteins)}")
    print(f"Number of ligands: {len(ligands)}")
    print(f"Number of affinities: {len(affinities)}")

    data = {
        'proteins': proteins,
        'ligands': ligands,
        'affinities': affinities,
        'pdb_ids': pdb_ids
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    return proteins, ligands, np.array(affinities), pdb_ids



class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_graph_features, edge_dropout_rate=0.1):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GeneralConv(84, hidden_channels, in_edge_channels=6)
        self.conv2 = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=6)
        self.conv3 = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=6)
        self.conv4 = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=6)
        self.fc_graph = torch.nn.Linear(num_graph_features, hidden_channels)
        self.fc_combined = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.edge_dropout_rate = edge_dropout_rate

    def forward(self, x, edge_index, edge_attr, batch, graph_attr):

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.edge_dropout_rate, force_undirected=True, num_nodes=x.size(0))

        x1 = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr).relu() + x1
        x3 = self.conv3(x2, edge_index, edge_attr=edge_attr).relu() + x2
        x4 = self.conv4(x3, edge_index, edge_attr=edge_attr).relu() + x3

        x = global_mean_pool(x3, batch)


        graph_attr = self.fc_graph(graph_attr)

        combined = torch.cat([x, graph_attr], dim=1)
        combined = self.fc_combined(combined).relu()
        combined = F.dropout(combined, p=dropout_rate, training=self.training)
        combined = self.lin(combined)

        return combined


dropout_rate = 0.5