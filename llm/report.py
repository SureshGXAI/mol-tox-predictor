"""
Turns explainability findings into a narrative. Prefers a local Ollama model;
if Ollama is unavailable it falls back to a deterministic templated summary so
the pipeline never hard-fails (and CI stays offline).
"""
from llm import ollama_client
from llm.prompts import SYSTEM_PROMPT, build_prompt


def _fallback_narrative(findings: dict) -> str:
    if not findings["flagged_targets"]:
        return ("No targets exceeded the toxicity threshold, so no structural "
                "alerts were prioritized for this molecule.")
    parts = ["[Templated summary — start an Ollama server for a full narrative.]", ""]
    for e in findings["per_target"]:
        if not e.get("flagged"):
            continue
        groups = ", ".join(sorted({g["group"] for g in e.get("functional_groups", [])}))
        parts.append(
            f"{e['target']} ({e['description']}) predicted toxic at "
            f"p={e['probability']}. High-saliency substructures: "
            f"{groups or 'no named functional group matched the top atoms'}.")
    return "\n".join(parts)


def generate_narrative(findings: dict, model: str = None, host: str = None) -> dict:
    """Returns {'narrative': str, 'source': 'ollama'|'fallback', 'model': str}."""
    if ollama_client.is_available(host):
        try:
            text = ollama_client.chat(
                build_prompt(findings), system=SYSTEM_PROMPT, model=model, host=host)
            return {"narrative": text, "source": "ollama",
                    "model": model or ollama_client.OLLAMA_MODEL}
        except RuntimeError:
            pass
    return {"narrative": _fallback_narrative(findings), "source": "fallback",
            "model": None}
