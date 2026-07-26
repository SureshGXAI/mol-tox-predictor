"""
Rich molecular featurization for the D-MPNN.

Replaces the old 6-scalar `atom_features`. Two things change vs. the original:

  1. Atoms get a proper one-hot feature vector (element, degree, charge,
     hybridization, H-count, aromaticity, ring membership, chirality, mass).
  2. Bonds now carry features (type, conjugation, ring, stereo) — the original
     GATConv never saw bond information at all.

We also compute a compact block of normalized RDKit descriptors per molecule,
which get fused into the readout (worth ~+0.05 mean AUC in our experiments).

Everything here is plain RDKit + numpy — no torch_geometric.
"""
from typing import List
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Atom / bond feature vocabularies
# ---------------------------------------------------------------------------
ATOM_LIST = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # B C N O F Si P S Cl Br I
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot(value, choices):
    """One-hot with an extra final slot for 'unknown / out of vocabulary'."""
    vec = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else -1
    vec[idx] = 1
    return vec


def atom_features(atom) -> List[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
        + [atom.GetMass() * 0.01]  # scaled so it sits ~O(1)
    )


def bond_features(bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
    )


# Feature dimensions (kept as module constants so the model can size itself).
ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FDIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


# ---------------------------------------------------------------------------
# Molecular descriptor block (fused into the readout)
# ---------------------------------------------------------------------------
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "NumAromaticRings", "RingCount", "FractionCSP3", "NumHeteroatoms",
    "NumValenceElectrons", "HeavyAtomCount", "NHOHCount", "NOCount", "LabuteASA",
    "BalabanJ", "BertzCT", "Chi0", "Chi1", "HallKierAlpha", "Kappa1", "Kappa2",
    "Kappa3", "qed", "MaxPartialCharge", "MinPartialCharge", "MolMR",
    "NumSaturatedRings", "NumAliphaticRings",
]
_calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def molecule_descriptors(mol) -> np.ndarray:
    vals = np.array(_calc.CalcDescriptors(mol), dtype=np.float64)
    return np.where(np.isfinite(vals), vals, np.nan)  # inf -> nan, imputed later


class DescriptorScaler:
    """Median-impute + standardize. Fit on the TRAIN set only, then persisted
    inside the model checkpoint so inference matches training."""

    def __init__(self):
        self.median = None
        self.mean = None
        self.std = None

    def fit(self, D: np.ndarray):
        self.median = np.nanmedian(D, axis=0)
        Di = self._impute(D)
        self.mean = Di.mean(axis=0)
        self.std = Di.std(axis=0) + 1e-8
        return self

    def _impute(self, D):
        D = D.copy()
        inds = np.where(np.isnan(D))
        D[inds] = np.take(self.median, inds[1])
        return D

    def transform(self, D: np.ndarray) -> np.ndarray:
        return (self._impute(D) - self.mean) / self.std

    def state_dict(self):
        return {"median": self.median, "mean": self.mean, "std": self.std}

    def load_state_dict(self, s):
        self.median, self.mean, self.std = s["median"], s["mean"], s["std"]
        return self


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
class MolGraph:
    """Directed-bond graph for one molecule.

    For each RDKit bond we create TWO directed bonds (v->w and w->v). Each
    directed bond stores: source atom, target atom, its reverse-bond index,
    and a feature vector = [source atom features || bond features] (Chemprop's
    bond-message initialization).
    """

    def __init__(self, mol):
        self.n_atoms = mol.GetNumAtoms()
        self.f_atoms = np.array(
            [atom_features(a) for a in mol.GetAtoms()], dtype=np.float32
        )

        # NOTE: we store BOND-ONLY features here. The source atom's features are
        # concatenated at run time inside the model (from the single f_atoms leaf
        # tensor) so that gradient x input saliency flows through both the readout
        # and the message-passing paths.
        self.f_bonds = []       # [n_bonds, BOND_FDIM]
        self.b2a_source = []    # source atom of each directed bond (v)
        self.b2a_target = []    # target atom of each directed bond (w)
        self.b2revb = []        # reverse directed-bond index

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            b1 = len(self.f_bonds)
            b2 = b1 + 1
            self.f_bonds.append(bf)  # i -> j
            self.f_bonds.append(bf)  # j -> i
            self.b2a_source += [i, j]
            self.b2a_target += [j, i]
            self.b2revb += [b2, b1]

        self.n_bonds = len(self.f_bonds)
        self.f_bonds = np.array(self.f_bonds, dtype=np.float32) if self.n_bonds else \
            np.zeros((0, BOND_FDIM), dtype=np.float32)


def mol_to_molgraph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    return MolGraph(mol), mol


TARGETS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
           "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
