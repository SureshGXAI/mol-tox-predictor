"""
Train the multi-task D-MPNN on Tox21 with honest evaluation + MLflow tracking.

  * Scaffold split into TRAIN / VAL / TEST.
  * Early stopping on VAL; metrics reported on the untouched TEST set.
  * Bond-aware D-MPNN + descriptor fusion; optional --ensemble N.
  * MLflow logs params, per-epoch val AUC, final per-target test AUC, the
    training-curve + ROC + per-target-AUC figures, results.json, and the models.

Usage:
    python train.py --data tox21.csv --epochs 60 --ensemble 3
    mlflow ui --backend-store-uri ./mlruns     # then open http://localhost:5000
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from chemistry.features import mol_to_molgraph, molecule_descriptors, DescriptorScaler
from data.splits import scaffold_split, random_split
from models.dmpnn import Tox21DMPNN, BatchMolGraph, TARGETS
from plots.training_plots import training_curves
from plots.eval_plots import per_target_auc_bar, roc_grid
import tracking.mlflow_utils as mlf


def load_dataset(path):
    df = pd.read_csv(path)
    graphs, descs, ys, smiles = [], [], [], []
    for _, row in df.iterrows():
        g, mol = mol_to_molgraph(row["smiles"])
        if g is None:
            continue
        graphs.append(g); descs.append(molecule_descriptors(mol))
        ys.append([row[t] if not pd.isna(row[t]) else np.nan for t in TARGETS])
        smiles.append(row["smiles"])
    return graphs, np.array(descs), np.array(ys, dtype=np.float32), smiles


def masked_bce(logits, targets, pos_weight):
    total, count = 0.0, 0
    for i in range(targets.shape[1]):
        y = targets[:, i]; mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        total = total + nn.functional.binary_cross_entropy_with_logits(
            logits[mask, i], y[mask], pos_weight=pos_weight[i])
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def predict_probs(model, idx, graphs, desc_t, batch_size, device):
    model.eval()
    out = np.zeros((len(idx), len(TARGETS)), dtype=np.float32)
    for k in range(0, len(idx), batch_size):
        chunk = np.arange(k, min(k + batch_size, len(idx)))
        gi = idx[chunk]
        bmg = BatchMolGraph([graphs[j] for j in gi], device=device)
        out[chunk] = torch.sigmoid(model(bmg, desc_t[gi])).cpu().numpy()
    return out


def per_target_auc(probs, y_np, idx):
    aucs = {}
    for i, t in enumerate(TARGETS):
        yt = y_np[idx, i]; m = ~np.isnan(yt)
        aucs[t] = (float(roc_auc_score(yt[m].astype(int), probs[m, i]))
                   if len(np.unique(yt[m])) >= 2 else None)
    valid = [a for a in aucs.values() if a is not None]
    aucs["MEAN"] = float(np.mean(valid)); return aucs


def train_one(seed, graphs, desc, y_np, tr, va, te, args, device):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = DescriptorScaler().fit(desc[tr])
    desc_t = torch.tensor(scaler.transform(desc).astype(np.float32), device=device)
    y_t = torch.tensor(y_np, device=device)

    pw = []
    for i in range(len(TARGETS)):
        yt = y_np[tr, i]; yt = yt[~np.isnan(yt)]
        pos = (yt == 1).sum(); neg = (yt == 0).sum()
        pw.append(neg / pos if pos > 0 else 1.0)
    pos_weight = torch.tensor(pw, dtype=torch.float, device=device)

    model = Tox21DMPNN(hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                       use_descriptors=not args.no_descriptors).to(device)
    model.desc_mean.copy_(torch.tensor(scaler.mean, dtype=torch.float))
    model.desc_std.copy_(torch.tensor(scaler.std, dtype=torch.float))
    model.desc_median.copy_(torch.tensor(scaler.median, dtype=torch.float))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_val, best_state, no_improve = -1, None, 0
    order = tr.copy()
    for epoch in range(1, args.epochs + 1):
        model.train(); np.random.shuffle(order); ep_loss = 0.0; nb = 0
        for k in range(0, len(order), args.batch_size):
            gi = order[k:k + args.batch_size]
            bmg = BatchMolGraph([graphs[j] for j in gi], device=device)
            opt.zero_grad()
            loss = masked_bce(model(bmg, desc_t[gi]), y_t[gi], pos_weight)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ep_loss += float(loss); nb += 1

        val_auc = per_target_auc(predict_probs(model, va, graphs, desc_t, args.batch_size, device), y_np, va)["MEAN"]
        sched.step(val_auc)
        history["epoch"].append(epoch); history["train_loss"].append(ep_loss / nb)
        history["val_auc"].append(val_auc)
        mlf.log_metrics({f"seed{seed}_train_loss": ep_loss / nb,
                         f"seed{seed}_val_auc": val_auc}, step=epoch)
        if val_auc > best_val:
            best_val = val_auc; no_improve = 0
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        else:
            no_improve += 1
        print(f"  seed {seed} | epoch {epoch:3d} | loss {ep_loss/nb:.4f} | val AUC {val_auc:.4f} | best {best_val:.4f}", flush=True)
        if no_improve >= args.patience:
            print(f"  seed {seed} | early stop @ {epoch}", flush=True); break

    model.load_state_dict(best_state)
    test_probs = predict_probs(model, te, graphs, desc_t, args.batch_size, device)
    return model, test_probs, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tox21.csv")
    ap.add_argument("--out", default="saved_models")
    ap.add_argument("--figdir", default="reports/figures")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    ap.add_argument("--no_descriptors", action="store_true")
    ap.add_argument("--experiment", default="tox21-dmpnn")
    ap.add_argument("--tracking_uri", default="./mlruns")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)
    graphs, desc, y_np, smiles = load_dataset(args.data)
    tr, va, te = (scaffold_split(smiles) if args.split == "scaffold"
                  else random_split(len(graphs)))
    print(f"device={device} split={args.split} train/val/test={len(tr)}/{len(va)}/{len(te)}")

    with mlf.run(args.experiment, run_name=f"{args.split}_ens{args.ensemble}",
                 tracking_uri=args.tracking_uri):
        mlf.log_params({"split": args.split, "ensemble": args.ensemble,
                        "hidden": args.hidden, "depth": args.depth, "dropout": args.dropout,
                        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                        "descriptors": not args.no_descriptors, "n_molecules": len(graphs)})

        ens = np.zeros((len(te), len(TARGETS)), dtype=np.float32)
        t0 = time.time()
        for s in range(args.ensemble):
            print(f"[model {s+1}/{args.ensemble}]", flush=True)
            model, test_probs, history = train_one(1000 + s, graphs, desc, y_np, tr, va, te, args, device)
            ens += test_probs / args.ensemble
            ckpt = os.path.join(args.out, f"dmpnn_seed{s}.pt")
            torch.save(model.state_dict(), ckpt); mlf.log_artifact(ckpt, "checkpoints")
            curve = training_curves(history, os.path.join(args.figdir, f"training_curve_seed{s}.png"))
            mlf.log_artifact(curve, "figures")

        aucs = per_target_auc(ens, y_np, te)
        print("\n=== TEST AUC (%s split, %d models) ===" % (args.split, args.ensemble))
        for t in TARGETS:
            print(f"  {t:<14} {aucs[t]:.4f}" if aucs[t] else f"  {t:<14}  N/A")
        print(f"  {'MEAN':<14} {aucs['MEAN']:.4f}\n  trained in {time.time()-t0:.0f}s")

        mlf.log_metrics({f"test_auc_{t}": v for t, v in aucs.items() if v is not None})
        res_path = os.path.join(args.out, "results_dmpnn.json")
        json.dump(aucs, open(res_path, "w"), indent=2); mlf.log_artifact(res_path)

        bar = per_target_auc_bar(aucs, os.path.join(args.figdir, "dmpnn_per_target_auc.png"),
                                 title=f"D-MPNN test AUC ({args.split} split)")
        mlf.log_artifact(bar, "figures")
        ytbt = {t: (y_np[te, i][~np.isnan(y_np[te, i])].astype(int)) for i, t in enumerate(TARGETS)}
        ypbt = {t: ens[~np.isnan(y_np[te, i]), i] for i, t in enumerate(TARGETS)}
        roc = roc_grid(ytbt, ypbt, TARGETS, os.path.join(args.figdir, "dmpnn_roc_grid.png"))
        mlf.log_artifact(roc, "figures")


if __name__ == "__main__":
    main()
