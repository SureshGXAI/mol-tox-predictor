"""
Offline smoke tests — no torch, no GPU, no network. These are what CI runs.
They cover featurization, scaffold splitting, functional-group matching, the
torch-free explainability findings, and the LLM fallback path.
"""
import numpy as np
from rdkit import Chem

from chemistry.features import (mol_to_molgraph, molecule_descriptors,
                                DescriptorScaler, ATOM_FDIM, BOND_FDIM, N_DESCRIPTORS)
from data.splits import scaffold_split, generate_scaffold
from explainability.functional_groups import groups_for_atoms, match_functional_groups
from explainability.saliency_report import build_findings
from llm.report import generate_narrative

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
NIMESULIDE = "CC(=O)Nc1ccc(Oc2ccccc2[N+](=O)[O-])cc1S(C)(=O)=O"


def test_featurization_shapes():
    g, mol = mol_to_molgraph(CAFFEINE)
    assert g.f_atoms.shape[1] == ATOM_FDIM
    assert g.f_bonds.shape[1] == BOND_FDIM
    assert g.n_bonds == 2 * mol.GetNumBonds()  # directed bonds


def test_reverse_bond_consistency():
    g, _ = mol_to_molgraph(NIMESULIDE)
    for b in range(g.n_bonds):
        r = g.b2revb[b]
        assert g.b2a_source[b] == g.b2a_target[r]
        assert g.b2a_target[b] == g.b2a_source[r]


def test_descriptor_scaler_finite():
    D = np.array([molecule_descriptors(mol_to_molgraph(s)[1])
                  for s in [CAFFEINE, NIMESULIDE, "CCO", "c1ccccc1"]])
    T = DescriptorScaler().fit(D).transform(D)
    assert T.shape[1] == N_DESCRIPTORS and np.isfinite(T).all()


def test_scaffold_split_no_overlap():
    smiles = [CAFFEINE, NIMESULIDE, "CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"] * 40
    tr, va, te = scaffold_split(smiles)
    strain = {generate_scaffold(smiles[i]) for i in tr}
    stest = {generate_scaffold(smiles[i]) for i in te}
    assert len(strain & stest) == 0


def test_nitro_toxicophore_detected():
    mol = Chem.MolFromSmiles(NIMESULIDE)
    groups = set(match_functional_groups(mol).keys())
    assert "nitro" in groups and "sulfone" in groups


def test_findings_and_fallback_narrative():
    mol = Chem.MolFromSmiles(NIMESULIDE)
    n = mol.GetNumAtoms()
    probs = [0.1] * 12; probs[2] = 0.82
    findings = build_findings(NIMESULIDE, probs, {"NR-AhR": list(np.linspace(0, 1, n))})
    assert "NR-AhR" in findings["flagged_targets"]
    nar = generate_narrative(findings)   # offline -> fallback
    assert nar["source"] == "fallback" and len(nar["narrative"]) > 0
