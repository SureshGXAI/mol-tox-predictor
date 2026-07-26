"""
Generate every figure shipped in reports/figures/ :
  - dataset EDA (class balance, missingness, co-occurrence, scaffold sizes)
  - comparison plots (random-vs-scaffold inflation, feature ablation)
  - evaluation plots from the GBM baseline (per-target AUC, ROC, calibration,
    confusion) computed on the honest scaffold split
  - an explainability demo image (functional-group highlights) and a saliency
    heatmap rendering demo

Run from the repo root:  python scripts/make_figures.py --data tox21.csv
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingClassifier

from chemistry.features import molecule_descriptors, DescriptorScaler, TARGETS
from data.splits import scaffold_split
from baselines.gbm_baseline import featurize
from plots import dataset_plots as dp
from plots import eval_plots as ep
from explainability.functional_groups import match_functional_groups, groups_for_atoms
from explainability.visualize import highlight_groups, saliency_heatmap, save_png, molecule_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tox21.csv")
    ap.add_argument("--figdir", default="reports/figures")
    ap.add_argument("--results", default="reports/results_classical.json")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    F = lambda name: os.path.join(args.figdir, name)

    df = pd.read_csv(args.data)
    smiles_all = df["smiles"].tolist()

    print("dataset EDA figures...")
    dp.label_balance(df, TARGETS, F("eda_class_balance.png"))
    dp.missingness(df, TARGETS, F("eda_missingness.png"))
    dp.cooccurrence(df, TARGETS, F("eda_cooccurrence.png"))
    dp.scaffold_sizes(smiles_all, F("eda_scaffold_sizes.png"))

    if os.path.exists(args.results):
        print("comparison figures...")
        results = json.load(open(args.results))
        dp.split_inflation(results, F("compare_split_inflation.png"))
        dp.feature_ablation(results, F("compare_feature_ablation.png"))

    print("GBM eval figures (scaffold split)...")
    fp, desc, y, smiles = featurize(args.data)
    tr, va, te = scaffold_split(smiles); tr = np.concatenate([tr, va])
    scaler = DescriptorScaler().fit(desc[tr])
    X = np.hstack([fp, scaler.transform(desc).astype(np.float32)]).astype(np.float32)

    ytbt, ypbt, aucs = {}, {}, {}
    for i, t in enumerate(TARGETS):
        ytr, yte = y[tr, i], y[te, i]
        mtr, mte = ~np.isnan(ytr), ~np.isnan(yte)
        if len(np.unique(yte[mte])) < 2:
            ytbt[t] = ypbt[t] = None; aucs[t] = None; continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
              max_leaf_nodes=48, l2_regularization=1.0, early_stopping=True,
              validation_fraction=0.15, random_state=0)
        clf.fit(X[tr][mtr], ytr[mtr].astype(int))
        p = clf.predict_proba(X[te][mte])[:, 1]
        ytbt[t] = yte[mte].astype(int); ypbt[t] = p
        from sklearn.metrics import roc_auc_score
        aucs[t] = float(roc_auc_score(ytbt[t], p))
        print(f"  {t:<14} AUC={aucs[t]:.4f}", flush=True)
    aucs["MEAN"] = float(np.mean([a for a in aucs.values() if a is not None]))
    print(f"  MEAN {aucs['MEAN']:.4f}")

    ep.per_target_auc_bar(aucs, F("gbm_per_target_auc.png"),
                          title=f"GBM (ECFP+descriptors) test AUC — scaffold split (mean {aucs['MEAN']:.3f})")
    ep.roc_grid(ytbt, ypbt, TARGETS, F("gbm_roc_grid.png"))
    ep.calibration_grid(ytbt, ypbt, TARGETS, F("gbm_calibration.png"))
    ep.confusion_grid(ytbt, ypbt, TARGETS, F("gbm_confusion.png"))

    print("explainability demo figures...")
    # Nimesulide-like nitro/sulfonamide example rich in toxicophores.
    demo_smiles = "CC(=O)Nc1ccc(Oc2ccccc2[N+](=O)[O-])cc1S(C)(=O)=O"
    mol = Chem.MolFromSmiles(demo_smiles)
    save_png(molecule_png(mol), F("explain_molecule.png"))
    findings = groups_for_atoms(mol, list(range(mol.GetNumAtoms())))
    save_png(highlight_groups(mol, findings), F("explain_functional_groups.png"))
    # Saliency-render demo: use |Gasteiger charge| as a stand-in per-atom signal.
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([abs(float(mol.GetAtomWithIdx(i).GetPropsAsDict().get("_GasteigerCharge", 0.0)))
                        for i in range(mol.GetNumAtoms())])
    charges = np.nan_to_num(charges)
    save_png(saliency_heatmap(mol, charges), F("explain_saliency_render.png"))
    json.dump({"demo_smiles": demo_smiles,
               "detected_groups": sorted({f["group"] for f in findings})},
              open(F("explain_demo.json"), "w"), indent=2)

    print("done. figures in", args.figdir)


if __name__ == "__main__":
    main()
