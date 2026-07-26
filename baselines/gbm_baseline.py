"""
Gradient-boosting baseline — no PyTorch required.

This is the model that reached 0.833 mean AUC on the honest scaffold split in
our experiments (HistGradientBoosting on ECFP4 + RDKit descriptors), i.e. it
matches/beats the original GAT while being trivially reproducible and fast.

Two good uses:
  1. A strong, dependency-light fallback if you don't want to run the GNN.
  2. An extra member to ensemble with the D-MPNN (tree + GNN errors decorrelate).

Run:
    python -m baselines.gbm_baseline --data tox21.csv
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from chemistry.features import molecule_descriptors, DescriptorScaler, TARGETS
from data.splits import scaffold_split

RDLogger.DisableLog("rdApp.*")


def morgan(mol, nbits=2048, radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(path):
    df = pd.read_csv(path)
    fps, descs, ys, smis = [], [], [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        fps.append(morgan(mol))
        descs.append(molecule_descriptors(mol))
        ys.append([row[t] if not pd.isna(row[t]) else np.nan for t in TARGETS])
        smis.append(row["smiles"])
    return (np.array(fps, np.float32), np.array(descs, np.float64),
            np.array(ys, np.float32), smis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tox21.csv")
    ap.add_argument("--out", default="saved_models/gbm_results.json")
    args = ap.parse_args()

    fp, desc, y, smiles = featurize(args.data)
    tr, va, te = scaffold_split(smiles)
    tr = np.concatenate([tr, va])   # trees don't need our external val set

    scaler = DescriptorScaler().fit(desc[tr])
    X = np.hstack([fp, scaler.transform(desc).astype(np.float32)]).astype(np.float32)

    aucs = {}
    for i, t in enumerate(TARGETS):
        ytr, yte = y[tr, i], y[te, i]
        mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
        Xtr, Ytr = X[tr][mtr], ytr[mtr].astype(int)
        Xte, Yte = X[te][mte], yte[mte].astype(int)
        if len(np.unique(Yte)) < 2:
            aucs[t] = None; continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=48,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=0)
        clf.fit(Xtr, Ytr)
        aucs[t] = float(roc_auc_score(Yte, clf.predict_proba(Xte)[:, 1]))
        print(f"  {t:<14} {aucs[t]:.4f}", flush=True)
    valid = [a for a in aucs.values() if a is not None]
    aucs["MEAN"] = float(np.mean(valid))
    print(f"  {'MEAN':<14} {aucs['MEAN']:.4f}")
    json.dump(aucs, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
