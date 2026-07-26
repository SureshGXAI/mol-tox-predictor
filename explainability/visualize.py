"""
Molecule rendering for explainability:

  * saliency_heatmap  — colors atoms by a per-atom importance vector (e.g. the
    D-MPNN's gradient x input scores) using RDKit's similarity-map machinery.
  * highlight_groups  — highlights the atoms of matched functional groups.

Both return PNG bytes (and a base64 helper) so they slot into the FastAPI
`/explain` response or get saved as figures.
"""
import base64
from io import BytesIO
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps


def _png_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def saliency_heatmap(mol, atom_scores, size=(500, 400)) -> bytes:
    """Render `mol` with atoms shaded by `atom_scores` (len == n_atoms).
    Higher score = warmer. Returns PNG bytes."""
    scores = np.asarray(atom_scores, dtype=float)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    weights = [float(s) for s in scores]

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights, draw2d=drawer, colorMap="coolwarm", contourLines=0)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def highlight_groups(mol, group_findings, size=(500, 400)) -> bytes:
    """Highlight atoms of the given functional-group findings, each group a
    distinct color. `group_findings` is the list from
    functional_groups.groups_for_atoms()."""
    palette = [(1.0, 0.70, 0.28), (0.40, 0.76, 0.94), (0.60, 0.85, 0.55),
               (0.95, 0.60, 0.72), (0.80, 0.70, 0.95), (0.98, 0.85, 0.40)]
    highlight_atoms, atom_colors, legend_bits = [], {}, []
    for i, f in enumerate(group_findings[:6]):
        color = palette[i % len(palette)]
        for a in f["atoms"]:
            highlight_atoms.append(a)
            atom_colors[a] = color
        legend_bits.append(f["group"])

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().legendFontSize = 16
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(set(highlight_atoms)),
        highlightAtomColors=atom_colors,
        legend=", ".join(legend_bits) if legend_bits else "")
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def save_png(png_bytes: bytes, path: str):
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path


def molecule_png(mol, size=(400, 300)) -> bytes:
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
