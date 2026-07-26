"""
Dataset splitting.

`scaffold_split` groups molecules by Bemis-Murcko scaffold and assigns whole
scaffold groups to train/val/test — so no scaffold ever straddles the split.
This is the standard, honest evaluation for molecular property prediction and
the reason the numbers here are trustworthy where a random split's are not.
"""
from collections import defaultdict
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str) -> str:
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        return s if s else smiles
    except Exception:
        return smiles


def scaffold_split(smiles_list, frac=(0.8, 0.1, 0.1), seed=42):
    """Largest scaffold groups go to train first, then val, then test.
    Returns (train_idx, val_idx, test_idx) as numpy arrays."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[generate_scaffold(smi)].append(i)

    # Deterministic: big groups first; shuffle the *singletons* among themselves
    # so val/test aren't dominated by one accidental ordering.
    big = [g for g in groups.values() if len(g) > 1]
    small = [g for g in groups.values() if len(g) == 1]
    rng = np.random.default_rng(seed)
    rng.shuffle(small)
    big.sort(key=lambda g: len(g), reverse=True)
    ordered = big + small

    n = len(smiles_list)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    train, val, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return np.array(train), np.array(val), np.array(test)


def random_split(n, frac=(0.8, 0.1, 0.1), seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]
