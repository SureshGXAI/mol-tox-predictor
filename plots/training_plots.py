"""Training-curve figures (used by train.py; logged to MLflow as artifacts)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})
ACCENT = "#3b7dd8"


def training_curves(history, out):
    """history: {'epoch': [...], 'train_loss': [...], 'val_auc': [...]}"""
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(history["epoch"], history["train_loss"], color="#d9534f", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#d9534f")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_auc"], color=ACCENT, label="val mean AUC")
    ax2.set_ylabel("val mean AUC", color=ACCENT); ax2.grid(False)
    best = max(history["val_auc"])
    ax2.axhline(best, color=ACCENT, ls="--", lw=0.8)
    ax1.set_title(f"Training curve (best val AUC {best:.4f})")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out
