"""
Explain a single molecule end-to-end:
  predictions -> gradient x input saliency -> functional groups -> Ollama narrative
plus a saliency heatmap PNG and a functional-group highlight PNG.

Requires trained checkpoints in saved_models/ (see train.py). The LLM step uses
a local Ollama server if one is running, otherwise a templated fallback.

    python scripts/explain_molecule.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    python scripts/explain_molecule.py --name caffeine --model llama3.1
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from rdkit import Chem

from models.predictor import ToxicityPredictor
from explainability.saliency_report import explain
from explainability.visualize import saliency_heatmap, highlight_groups, save_png
from explainability.functional_groups import groups_for_atoms
from llm.report import generate_narrative


def resolve(name):
    import pubchempy as pcp
    return pcp.get_compounds(name, "name")[0].canonical_smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles"); ap.add_argument("--name")
    ap.add_argument("--models", default="saved_models/dmpnn_seed*.pt")
    ap.add_argument("--model", default=None, help="Ollama model name")
    ap.add_argument("--outdir", default="reports/explanations")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    smiles = args.smiles or resolve(args.name)
    predictor = ToxicityPredictor(model_glob=args.models)
    findings = explain(smiles, predictor)

    mol = Chem.MolFromSmiles(smiles)
    probs, saliency = predictor.predict_with_saliency(smiles)
    if saliency:
        first = next(iter(saliency.values()))
        save_png(saliency_heatmap(mol, first), os.path.join(args.outdir, "saliency.png"))
    save_png(highlight_groups(mol, groups_for_atoms(mol, list(range(mol.GetNumAtoms())))),
             os.path.join(args.outdir, "functional_groups.png"))

    narrative = generate_narrative(findings, model=args.model)
    findings["narrative"] = narrative
    json.dump(findings, open(os.path.join(args.outdir, "explanation.json"), "w"), indent=2)

    print(f"\nSMILES: {smiles}")
    print("Flagged:", ", ".join(findings["flagged_targets"]) or "none")
    print(f"\n[{narrative['source']}] narrative:\n{narrative['narrative']}")


if __name__ == "__main__":
    main()
