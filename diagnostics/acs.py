"""Average Causal Sensitivity (ACS).

Protocol:
  1. Sample a batch of latents z from the SCM prior.
  2. For each latent dimension i (in causal_parents):
     a. Sample two batches: z, z' where z' = do(z_i = z_i + delta).
     b. Render both to pixels: x = render(z), x' = render(z').
     c. Encode: h = f_encoder(x), h' = f_encoder(x').
     d. Compute sensitivity: s_i = ||h - h'||_2 (mean over samples).
  3. Hungarian-match latents to representation coordinates based on sensitivity
     matrix S where S[i,j] = sensitivity of coordinate j to intervention on latent i.
  4. Normalize: divide each matched sensitivity by the max over latents.
  5. Report concentration score: mean of normalized sensitivities.

Reference:
  - Lachapelle et al. (2022). "Disentanglement via Mechanism Sparsity
    Regularization." (ACS metric adapted for contrastive setting.)
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Callable, List


@torch.no_grad()
def average_causal_sensitivity(
    encoder: torch.nn.Module,
    scm: "data.synthetic_scm._BaseSCM",
    n_samples: int = 512,
    delta: float = 1.0,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Compute ACS.

    Args:
        encoder: Trained encoder module. Takes (B, 3, H, W) -> (B, D).
        scm: SCM generator instance.
        n_samples: Number of latent samples for sensitivity estimation.
        delta: Intervention strength.
        batch_size: Batch size for encoding.
        device: Torch device.

    Returns:
        dict with keys:
          - 'concentration': scalar ACS score (higher = more concentrated).
          - 'sensitivity_matrix': (n_latents, n_coordinates) numpy array.
          - 'matching': list of (latent_idx, coord_idx) Hungarian matches.
    """
    encoder.eval()
    encoder.to(device)

    causal_idxs = scm.get_causal_latent_indices()
    n_latents = len(causal_idxs)

    # Sample latents
    z_base = scm.sample_prior(n_samples)  # (N, L)
    rep_dim = None

    # Sensitivity matrix: (n_latents, rep_dim)
    sens_mat = []

    for li, latent_idx in enumerate(causal_idxs):
        # Intervene: shift by delta
        z_interv = scm.intervene(z_base, latent_idx, delta)

        all_sens = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            z_batch = z_base[start:end]
            z_interv_batch = z_interv[start:end]

            x_base = scm.render(z_batch.to(device))
            x_interv = scm.render(z_interv_batch.to(device))

            h_base, _ = encoder(x_base)
            h_interv, _ = encoder(x_interv)

            if rep_dim is None:
                rep_dim = h_base.shape[1]

            # Sensitivity per coordinate: |h_interv - h_base|
            sens = (h_interv - h_base).abs().mean(dim=0)  # (D,)
            all_sens.append(sens)

        sens_vec = torch.stack(all_sens, dim=0).mean(dim=0)  # (D,)
        sens_mat.append(sens_vec.cpu().numpy())

    sens_mat = np.array(sens_mat)  # (n_latents, rep_dim)

    # Hungarian matching: minimize distance -> maximize sensitivity
    # Use negative sensitivity as cost
    cost = -sens_mat
    latent_idx, coord_idx = linear_sum_assignment(cost)

    # Normalized concentration: each matched sensitivity / max over latents
    matched_sens = sens_mat[latent_idx, coord_idx]
    max_per_latent = sens_mat.max(axis=1, keepdims=True)
    max_per_latent = np.where(max_per_latent < 1e-8, 1.0, max_per_latent)
    normalized = matched_sens / max_per_latent[latent_idx]
    concentration = float(normalized.mean())

    return {
        "concentration": concentration,
        "sensitivity_matrix": sens_mat,
        "matching": list(zip(latent_idx.tolist(), coord_idx.tolist())),
    }
