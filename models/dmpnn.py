"""
Multi-task Directed Message-Passing Neural Network (D-MPNN) for Tox21.

This replaces the GAT. The key differences that drive the accuracy gain:

  * Messages live on *directed bonds* and use bond features, so the model
    actually reads bond type / conjugation / ring / stereo (the old GATConv
    ignored bonds entirely).
  * A block of normalized RDKit descriptors is *fused* into the molecule
    representation before the prediction head (worth ~+0.05 mean AUC for the
    classical models in our experiments; it helps GNNs too).
  * Proper multi-task heads with masked, class-weighted BCE.

Pure PyTorch — no torch_geometric. Aggregation uses index_add_, which is in
base torch, so there are no torch-scatter / PyG install headaches.

Reference: Yang et al., "Analyzing Learned Molecular Representations for
Property Prediction" (J. Chem. Inf. Model. 2019) — the Chemprop D-MPNN.
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemistry.features import (
    ATOM_FDIM, BOND_FDIM, N_DESCRIPTORS, MolGraph,
)

TARGETS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
           "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


class BatchMolGraph:
    """Collate a list of MolGraph into padded/offset tensors for batched MP."""

    def __init__(self, mol_graphs: List[MolGraph], device="cpu"):
        f_atoms, f_bonds = [], []
        b2a_source, b2a_target, b2revb = [], [], []
        atom_scope = []          # molecule index per atom (for segment mean)
        atom_offset, bond_offset = 0, 0

        for mi, g in enumerate(mol_graphs):
            f_atoms.append(g.f_atoms)
            f_bonds.append(g.f_bonds)
            b2a_source += [x + atom_offset for x in g.b2a_source]
            b2a_target += [x + atom_offset for x in g.b2a_target]
            b2revb += [x + bond_offset for x in g.b2revb]
            atom_scope += [mi] * g.n_atoms
            atom_offset += g.n_atoms
            bond_offset += g.n_bonds

        self.n_mols = len(mol_graphs)
        self.n_atoms = atom_offset
        self.f_atoms = torch.tensor(np.concatenate(f_atoms, 0), dtype=torch.float, device=device)
        self.f_bonds = torch.tensor(
            np.concatenate(f_bonds, 0) if bond_offset else np.zeros((0, BOND_FDIM), np.float32),
            dtype=torch.float, device=device)
        self.b2a_source = torch.tensor(b2a_source, dtype=torch.long, device=device)
        self.b2a_target = torch.tensor(b2a_target, dtype=torch.long, device=device)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long, device=device)
        self.atom_scope = torch.tensor(atom_scope, dtype=torch.long, device=device)


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden=300, depth=4, dropout=0.15):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden, hidden)

    def forward(self, bmg: BatchMolGraph, f_atoms: torch.Tensor) -> torch.Tensor:
        """Returns a molecule embedding [n_mols, hidden].
        `f_atoms` is passed explicitly so callers can make it a grad leaf
        (used for saliency)."""
        n_atoms = f_atoms.shape[0]
        hidden = self.W_h.in_features

        if bmg.f_bonds.shape[0] == 0:
            # No bonds anywhere (e.g. a batch of single ions) — messages are zero.
            a_message = torch.zeros(n_atoms, hidden, device=f_atoms.device)
        else:
            # bond input = [source atom features || bond features]
            src_atom_feats = f_atoms[bmg.b2a_source]
            bond_input = torch.cat([src_atom_feats, bmg.f_bonds], dim=1)
            input = self.W_i(bond_input)                 # [n_bonds, hidden]
            message = self.act(input)

            for _ in range(self.depth - 1):
                a_msg = torch.zeros(n_atoms, hidden, device=f_atoms.device)
                a_msg.index_add_(0, bmg.b2a_target, message)   # incoming per atom
                m = a_msg[bmg.b2a_source] - message[bmg.b2revb]  # exclude reverse
                message = self.act(input + self.W_h(m))
                message = self.dropout(message)

            a_message = torch.zeros(n_atoms, hidden, device=f_atoms.device)
            a_message.index_add_(0, bmg.b2a_target, message)

        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_h = self.dropout(self.act(self.W_o(a_input)))    # [n_atoms, hidden]

        # mean-pool atoms within each molecule
        mol_vec = torch.zeros(bmg.n_mols, hidden, device=f_atoms.device)
        mol_vec.index_add_(0, bmg.atom_scope, atom_h)
        counts = torch.zeros(bmg.n_mols, device=f_atoms.device)
        counts.index_add_(0, bmg.atom_scope, torch.ones(n_atoms, device=f_atoms.device))
        mol_vec = mol_vec / counts.clamp(min=1).unsqueeze(1)
        return mol_vec


class Tox21DMPNN(nn.Module):
    """D-MPNN encoder + descriptor fusion + per-task heads."""

    def __init__(self, hidden=300, depth=4, dropout=0.15,
                 use_descriptors=True, num_tasks=12, ffn_hidden=300):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.encoder = DMPNNEncoder(hidden, depth, dropout)
        fused = hidden + (N_DESCRIPTORS if use_descriptors else 0)
        self.ffn = nn.Sequential(
            nn.Linear(fused, ffn_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.task_heads = nn.ModuleList([nn.Linear(ffn_hidden, 1) for _ in range(num_tasks)])
        # descriptor scaler stats saved with the checkpoint (see train.py)
        self.register_buffer("desc_mean", torch.zeros(N_DESCRIPTORS))
        self.register_buffer("desc_std", torch.ones(N_DESCRIPTORS))
        self.register_buffer("desc_median", torch.zeros(N_DESCRIPTORS))

    def _body(self, mol_vec, descriptors):
        if self.use_descriptors:
            mol_vec = torch.cat([mol_vec, descriptors], dim=1)
        z = self.ffn(mol_vec)
        return torch.cat([h(z) for h in self.task_heads], dim=1)

    def forward(self, bmg: BatchMolGraph, descriptors: torch.Tensor = None) -> torch.Tensor:
        mol_vec = self.encoder(bmg, bmg.f_atoms)
        return self._body(mol_vec, descriptors)

    # ------------------------------------------------------------------
    # Explainability: gradient x input saliency, drop-in compatible with the
    # existing /explain endpoint. Returns per-atom scores for a SINGLE molecule.
    # ------------------------------------------------------------------
    def compute_saliency(self, bmg: BatchMolGraph, descriptors: torch.Tensor,
                         target_indices: List[int]) -> Tuple[List[float], dict]:
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(bmg, descriptors))[0].tolist()

        saliency_maps = {}
        for t in target_indices:
            f_atoms = bmg.f_atoms.clone().detach().requires_grad_(True)
            mol_vec = self.encoder(bmg, f_atoms)
            logits = self._body(mol_vec, descriptors)
            score = torch.sigmoid(logits[0, t])
            self.zero_grad(set_to_none=True)
            score.backward()
            sal = (f_atoms.grad * f_atoms.detach()).abs().sum(dim=1)  # [n_atoms]
            saliency_maps[t] = sal.tolist()
        return probs, saliency_maps
