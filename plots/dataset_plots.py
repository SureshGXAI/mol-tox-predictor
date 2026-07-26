"""Dataset EDA + comparison figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
ACCENT = "#3b7dd8"


def label_balance(df, targets, out):
    tox = [(df[t] == 1).sum() for t in targets]
    tested = [df[t].notna().sum() for t in targets]
    pct = [100 * a / b if b else 0 for a, b in zip(tox, tested)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(targets, pct, color=ACCENT)
    ax.set_ylabel("% positive (toxic)"); ax.set_title("Class imbalance per Tox21 target")
    ax.set_xticklabels(targets, rotation=45, ha="right")
    for i, p in enumerate(pct):
        ax.text(i, p + 0.2, f"{p:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def missingness(df, targets, out):
    frac = [df[t].isna().mean() for t in targets]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(targets, frac, color="#8a8f98")
    ax.set_ylabel("fraction of molecules untested"); ax.set_title("Missing labels per target")
    ax.set_xticklabels(targets, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def cooccurrence(df, targets, out):
    M = df[targets].fillna(0).values
    co = (M.T @ M)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(co, cmap="viridis")
    ax.set_xticks(range(len(targets))); ax.set_yticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_title("Positive-label co-occurrence")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.grid(False); fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def scaffold_sizes(smiles_list, out):
    from collections import Counter
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaf = []
    for s in smiles_list:
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False) or s)
        except Exception:
            scaf.append(s)
    sizes = np.array(sorted(Counter(scaf).values(), reverse=True))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(sizes) + 1), sizes, color=ACCENT)
    ax.set_yscale("log"); ax.set_xlabel("scaffold rank"); ax.set_ylabel("molecules in scaffold (log)")
    ax.set_title(f"Scaffold-group sizes ({len(sizes)} unique scaffolds) — motivates scaffold split")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def split_inflation(results, out):
    """Bar comparison of random vs scaffold for the RF model (from results_classical.json)."""
    rand = results["RF_ecfp_random"]; scaf = results["RF_ecfp_scaffold"]
    targets = [k for k in rand if k != "MEAN"]
    x = np.arange(len(targets)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, [rand[t] for t in targets], w, label=f"random ({rand['MEAN']:.3f})", color="#f0ad4e")
    ax.bar(x + w / 2, [scaf[t] for t in targets], w, label=f"scaffold ({scaf['MEAN']:.3f})", color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Same RF model: random split inflates AUC vs. honest scaffold split")
    ax.legend(); fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def feature_ablation(results, out):
    labels = ["XGB\nECFP", "XGB\nECFP+desc", "HGB\nECFP+desc", "Ensemble\nECFP+desc"]
    keys = ["XGB_ecfp", "XGB_both", "HGB_both", "ENSEMBLE_xgb_hgb_both"]
    vals = [results[k]["MEAN"] for k in keys if k in results]
    labels = labels[:len(vals)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=[ACCENT] * len(vals))
    ax.set_ylim(0.7, 0.87); ax.set_ylabel("mean ROC-AUC (scaffold split)")
    ax.set_title("Descriptors + model choice (all on the honest scaffold split)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out
