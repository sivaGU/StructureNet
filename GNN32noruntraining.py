# UTILITY FUNCTIONS

import os
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdMolTransforms, EState, MolSurf, Crippen
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GeneralConv, global_mean_pool
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

def protein_pocket_features_to_graph(pdb_file, ligand_file):
    print("Starting protein_pocket_features_to_graph function...")
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

        if ligand_tree.query_ball_point(coord, 4):  
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
    # print(f"Graph attributes: {pyg_graph.graph_attr}")
    # print(f"Pocket type: {type(pyg_graph)}")
    # print("Raw Protein Node Features Before Normalization:", pyg_graph.x)

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


def ligand_features_to_graph(ligand_file):
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

    bond_lengths = get_bond_lengths(mol)
    bond_orders = get_bond_orders(mol)
    electrostatic_interactions = get_electrostatic_interactions(mol)

    coords = [atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    print(f"Coordinates: {coords}")
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
        # print(f"Ligand spherical harmonics: {spherical_harmonics}")
        # print

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

    # pos = nx.spring_layout(graph) 
    # plt.figure(figsize=(8, 8))
    # nx.draw(graph, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=500, edge_color='gray')
    # plt.title(f"Graph visualization for {ligand_file}")
    # plt.show()

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
    # print(f"Graph attributes: {data.graph_attr}")

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

    return data





# rfiles = random.choices(subdirectories, k=5315)
# training_files = random.choices(rfiles, k=4252)
# testing_files = [entry for entry in rfiles if entry not in training_files]



class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_graph_features, edge_dropout_rate=0.1):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GeneralConv(84, hidden_channels, in_edge_channels=6)
        self.conv2 = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=6)
        self.conv3 = GeneralConv(hidden_channels, hidden_channels, in_edge_channels=6)
        self.fc_graph = torch.nn.Linear(num_graph_features, hidden_channels)
        self.fc_combined = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.edge_dropout_rate = edge_dropout_rate

    def forward(self, x, edge_index, edge_attr, batch, graph_attr):
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.edge_dropout_rate, force_undirected=True, num_nodes=x.size(0))

        x1 = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr).relu() + x1
        x3 = self.conv3(x2, edge_index, edge_attr=edge_attr).relu() + x2

        x = global_mean_pool(x3, batch)

        graph_attr = self.fc_graph(graph_attr)
        combined = torch.cat([x, graph_attr], dim=1)
        combined = self.fc_combined(combined).relu()

        combined = F.dropout(combined, p=dropout_rate, training=self.training)
        combined = self.lin(combined)

        return combined


num_hidden_channels = 256
num_graph_features = 10368
dropout_rate = 0.5



def extract_embeddings_2(loader, model, device):
    model.eval()
    embeddings = []
    labels = []
    ids = []

    with torch.no_grad():
        for d in loader:
            print(f"after for d in loader")
            d = d.to(device)
            print(f"after d = d.to(device)")
            print(f"batch before model: {d.batch}")
            out = model(d.x.float(), d.edge_index, d.edge_attr.float(), d.batch, d.graph_attr.float())
            print(f"output: {out}")
            print(f"after out = model")
            print(f"embeddings type{type(embeddings)}")
            embeddings.append(out.cpu().numpy())
            print(f"after embeddings.append(out.cpu().numpy())")

    embeddings = np.vstack(embeddings)
    return embeddings, labels, ids



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(hidden_channels=num_hidden_channels, num_graph_features=num_graph_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00979)
criterion = torch.nn.MSELoss()



