"""
SMARTS library of common functional groups and known toxicophores, plus a
matcher that returns which atoms belong to each detected group.

Used by the explainability pipeline to translate "atom 7 is important" into
"the sulfonamide group is important", which is what a medicinal chemist reads.
"""
from rdkit import Chem

# name -> SMARTS. Mix of generic functional groups and structural-alert
# toxicophores frequently discussed in Tox21 / tox literature.
FUNCTIONAL_GROUPS = {
    # --- generic functional groups ---
    "hydroxyl":            "[OX2H]",
    "primary_amine":       "[NX3;H2;!$(NC=O)]",
    "secondary_amine":     "[NX3;H1;!$(NC=O)]",
    "carboxylic_acid":     "[CX3](=O)[OX2H1]",
    "ester":               "[CX3](=O)[OX2H0][#6]",
    "amide":               "[NX3][CX3](=[OX1])",
    "ketone":              "[#6][CX3](=O)[#6]",
    "aldehyde":            "[CX3H1](=O)[#6]",
    "ether":               "[OD2]([#6])[#6]",
    "nitrile":             "[NX1]#[CX2]",
    "sulfonamide":         "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone":             "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "phosphate":           "[PX4](=O)([OX2])([OX2])[OX2]",
    "benzene_ring":        "c1ccccc1",
    "pyridine":            "n1ccccc1",
    "pyrimidine":          "c1cncnc1",
    "imidazole":           "c1cnc[nH]1",
    "furan":               "c1ccoc1",
    "thiophene":           "c1ccsc1",
    # --- structural-alert toxicophores ---
    "nitro":               "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitroso":             "[NX2]=O",
    "aromatic_nitro":      "[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "aryl_halide":         "[c][F,Cl,Br,I]",
    "alkyl_halide":        "[CX4][F,Cl,Br,I]",
    "epoxide":             "[OX2r3]1[#6r3][#6r3]1",
    "aromatic_amine":      "[c][NX3;H2,H1]",
    "azo":                 "[NX2]=[NX2]",
    "hydrazine":           "[NX3][NX3]",
    "aniline":             "c1ccccc1[NX3]",
    "phenol":              "c1ccccc1[OX2H]",
    "michael_acceptor":    "[CX3]=[CX3][CX3]=[OX1]",
    "thiol":               "[SX2H]",
    "isocyanate":          "[NX2]=[CX2]=[OX1]",
    "quinone":             "O=C1C=CC(=O)C=C1",
    "polycyclic_aromatic": "c1ccc2ccccc2c1",
}

_COMPILED = {name: Chem.MolFromSmarts(sm) for name, sm in FUNCTIONAL_GROUPS.items()}


def match_functional_groups(mol):
    """Return {group_name: [tuple(atom_idxs), ...]} for every group present."""
    out = {}
    for name, patt in _COMPILED.items():
        if patt is None:
            continue
        hits = mol.GetSubstructMatches(patt)
        if hits:
            out[name] = [tuple(h) for h in hits]
    return out


def groups_for_atoms(mol, atom_indices):
    """Given a set of important atoms, return the functional groups that
    contain any of them, with the overlap count (how many important atoms fall
    inside the group)."""
    atom_set = set(atom_indices)
    found = []
    for name, matches in match_functional_groups(mol).items():
        for atoms in matches:
            overlap = atom_set & set(atoms)
            if overlap:
                found.append({"group": name, "atoms": list(atoms),
                              "n_important": len(overlap)})
    # de-duplicate identical (group, atoms) and sort by involvement
    seen, uniq = set(), []
    for f in sorted(found, key=lambda d: d["n_important"], reverse=True):
        key = (f["group"], tuple(f["atoms"]))
        if key not in seen:
            seen.add(key); uniq.append(f)
    return uniq
