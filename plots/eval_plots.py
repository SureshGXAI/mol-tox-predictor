"""Evaluation figures. Each function saves a PNG and returns its path."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.calibration import calibration_curve

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
ACCENT = "#3b7dd8"


def per_target_auc_bar(auc_dict, out, title="Per-target test AUC (scaffold split)"):
    items = [(k, v) for k, v in auc_dict.items() if k != "MEAN" and v is not None]
    items.sort(key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [("#d9534f" if v < 0.7 else "#f0ad4e" if v < 0.8 else ACCENT) for v in vals]
    ax.barh(names, vals, color=colors)
    mean = auc_dict.get("MEAN")
    if mean:
        ax.axvline(mean, color="k", ls="--", lw=1, label=f"mean {mean:.3f}")
        ax.legend(loc="lower right")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("ROC-AUC"); ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out


def roc_grid(y_true_by_target, y_prob_by_target, targets, out):
    n = len(targets); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, t in zip(axes.ravel(), targets):
        yt, yp = y_true_by_target[t], y_prob_by_target[t]
        if yt is None or len(np.unique(yt)) < 2:
            ax.set_visible(False); continue
        fpr, tpr, _ = roc_curve(yt, yp); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=ACCENT, lw=1.8)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
        ax.set_title(f"{t}\nAUC={a:.3f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("ROC curves per target", y=1.005)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out


def calibration_grid(y_true_by_target, y_prob_by_target, targets, out):
    n = len(targets); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, t in zip(axes.ravel(), targets):
        yt, yp = y_true_by_target[t], y_prob_by_target[t]
        if yt is None or len(np.unique(yt)) < 2:
            ax.set_visible(False); continue
        frac, mean_pred = calibration_curve(yt, yp, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", color=ACCENT, lw=1.5, ms=4)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
        ax.set_title(t, fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Reliability (calibration) curves", y=1.005)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out


def confusion_grid(y_true_by_target, y_prob_by_target, targets, out, threshold=0.5):
    n = len(targets); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for ax, t in zip(axes.ravel(), targets):
        yt, yp = y_true_by_target[t], y_prob_by_target[t]
        if yt is None or len(np.unique(yt)) < 2:
            ax.set_visible(False); continue
        cm = confusion_matrix(yt, (np.asarray(yp) >= threshold).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"], fontsize=7)
        ax.set_yticklabels(["neg", "pos"], fontsize=7)
        ax.set_title(t, fontsize=9); ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Confusion matrices @ threshold {threshold}", y=1.005)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out
