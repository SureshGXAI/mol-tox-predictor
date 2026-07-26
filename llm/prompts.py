"""Prompt construction for the cheminformatics narrative."""
import json

SYSTEM_PROMPT = (
    "You are a computational toxicologist writing concise, technically accurate "
    "explanations for a medicinal chemist. You interpret model predictions and "
    "atom-level saliency for the 12 Tox21 assays. Be specific about functional "
    "groups and their known toxicological mechanisms, and give actionable "
    "medicinal-chemistry suggestions (bioisosteres, scaffold edits). Do not "
    "invent assay values; use only what is provided. Keep it under ~250 words."
)


def build_prompt(findings: dict) -> str:
    """Turn a findings dict (from explainability.saliency_report) into a prompt."""
    lines = [f"Molecule SMILES: {findings['smiles']}",
             f"Heavy atoms: {findings['num_atoms']}",
             f"Flagged (toxic) targets: {', '.join(findings['flagged_targets']) or 'none'}",
             "", "Per-flagged-target saliency and matched functional groups:"]
    for e in findings["per_target"]:
        if not e.get("flagged"):
            continue
        lines.append(f"- {e['target']} ({e['description']}): p={e['probability']}")
        for g in e.get("functional_groups", [])[:4]:
            lines.append(f"    * group={g['group']} (atoms {g['atoms']}, "
                         f"{g['n_important']} high-saliency)")
    lines += [
        "",
        "Write a short report that: (1) identifies the key structural drivers of "
        "the flagged toxicity, (2) explains the plausible mechanism for each, and "
        "(3) proposes concrete structural modifications to reduce the predicted "
        "liability while preserving the likely pharmacophore.",
    ]
    return "\n".join(lines)
