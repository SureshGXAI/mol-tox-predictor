"""
Builds the structured explainability findings that feed both the figures and
the LLM narrative.

`build_findings` is deliberately torch-free: it takes already-computed
probabilities + per-target saliency, so it can be unit-tested and demoed
without a trained network. `explain` is the end-to-end entry point that uses a
(torch) predictor.
"""
from typing import Dict, List
import numpy as np
from rdkit import Chem

from explainability.functional_groups import groups_for_atoms
from chemistry.features import TARGETS

# Same reliability numbers surfaced in the original project, kept for the UI.
TARGET_DESCRIPTIONS = {
    "NR-AR": "Androgen Receptor", "NR-AR-LBD": "Androgen Receptor LBD",
    "NR-AhR": "Aryl Hydrocarbon Receptor", "NR-Aromatase": "Aromatase Enzyme",
    "NR-ER": "Estrogen Receptor", "NR-ER-LBD": "Estrogen Receptor LBD",
    "NR-PPAR-gamma": "PPAR-gamma", "SR-ARE": "Antioxidant Response Element",
    "SR-ATAD5": "DNA Damage Response", "SR-HSE": "Heat Shock Response",
    "SR-MMP": "Mitochondrial Membrane Potential", "SR-p53": "p53 Tumour Suppressor",
}


def top_saliency_atoms(atom_scores, top_n=6):
    order = np.argsort(atom_scores)[::-1][:top_n]
    return [{"atom": int(i), "score": float(atom_scores[i])} for i in order]


def build_findings(smiles: str, probs: List[float],
                   saliency_maps: Dict[str, List[float]],
                   top_n: int = 6, flag_threshold: float = 0.5) -> dict:
    """Assemble a JSON-serializable findings object."""
    mol = Chem.MolFromSmiles(smiles)
    per_target = []
    for t, p in zip(TARGETS, probs):
        entry = {"target": t, "description": TARGET_DESCRIPTIONS.get(t, t),
                 "probability": round(float(p), 4), "flagged": bool(p >= flag_threshold)}
        if t in saliency_maps:
            scores = np.asarray(saliency_maps[t], dtype=float)
            tops = top_saliency_atoms(scores, top_n)
            entry["top_atoms"] = tops
            entry["functional_groups"] = groups_for_atoms(
                mol, [a["atom"] for a in tops])
        per_target.append(entry)

    flagged = [e for e in per_target if e["flagged"]]
    return {
        "smiles": smiles,
        "num_atoms": mol.GetNumAtoms(),
        "flagged_targets": [e["target"] for e in flagged],
        "per_target": per_target,
    }


def explain(smiles: str, predictor, target_names=None, top_n: int = 6) -> dict:
    """End-to-end: run the predictor (with saliency) and build findings.
    `predictor` is a models.predictor.ToxicityPredictor (requires torch)."""
    probs, saliency = predictor.predict_with_saliency(
        smiles, target_names=target_names)
    return build_findings(smiles, probs, saliency, top_n=top_n)
