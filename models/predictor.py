"""
Inference wrapper for the D-MPNN — drop-in replacement for the old
ToxicityPredictor. Same public surface the FastAPI app relies on:

    predictor.predict(smiles)                     -> list[12] probabilities
    predictor.predict_with_saliency(smiles, ...)  -> (probs, {target: [atom scores]})

Supports an ensemble: point MODEL_PATHS at one or more saved seed checkpoints
and their probabilities are averaged.
"""
import glob
import numpy as np
import torch

from chemistry.features import mol_to_molgraph, molecule_descriptors
from models.dmpnn import Tox21DMPNN, BatchMolGraph, TARGETS


class ToxicityPredictor:
    def __init__(self, model_glob="saved_models/dmpnn_seed*.pt",
                 hidden=300, depth=4, use_descriptors=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        for path in sorted(glob.glob(model_glob)):
            m = Tox21DMPNN(hidden=hidden, depth=depth, use_descriptors=use_descriptors)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.to(self.device).eval()
            self.models.append(m)
        if not self.models:
            raise FileNotFoundError(f"No checkpoints matched {model_glob}")
        print(f"Loaded {len(self.models)} D-MPNN model(s)")

    def _prepare(self, smiles):
        g, mol = mol_to_molgraph(smiles)
        if g is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        bmg = BatchMolGraph([g], device=self.device)
        # scale descriptors with the stats stored in the (first) model
        m0 = self.models[0]
        d = molecule_descriptors(mol)
        d = np.where(np.isfinite(d), d, m0.desc_median.cpu().numpy())
        d = (d - m0.desc_mean.cpu().numpy()) / m0.desc_std.cpu().numpy()
        desc = torch.tensor(d[None, :], dtype=torch.float, device=self.device)
        return bmg, desc, mol

    @torch.no_grad()
    def predict(self, smiles):
        bmg, desc, _ = self._prepare(smiles)
        probs = np.mean([torch.sigmoid(m(bmg, desc))[0].cpu().numpy()
                         for m in self.models], axis=0)
        return probs.tolist()

    def predict_with_saliency(self, smiles, target_names=None, threshold=0.35):
        """probs + per-target gradient x input atom saliency (uses model[0])."""
        bmg, desc, _ = self._prepare(smiles)
        probs = self.predict(smiles)
        if target_names:
            idx = [TARGETS.index(t) for t in target_names if t in TARGETS]
        else:
            idx = [i for i, p in enumerate(probs) if p >= threshold]
        if not idx:
            return probs, {}
        _, sal_by_idx = self.models[0].compute_saliency(bmg, desc, idx)
        return probs, {TARGETS[i]: s for i, s in sal_by_idx.items()}


# Instantiate lazily in app.py, e.g.:  predictor = ToxicityPredictor()
