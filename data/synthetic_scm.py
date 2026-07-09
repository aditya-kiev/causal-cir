"""Synthetic SCM benchmark suite for causal representation learning.

Three benchmarks:
  (A) IndependentSCM  — all latents mutually independent
  (B) CausalChainSCM  — causal chain: z1 -> z2 -> z3 -> ... -> zD
  (C) ConfoundedSCM   — confounder u influences two observed latents

Each generator:
  - samples from its prior p(z)
  - renders a pixel observation via a random nonlinear decoder
  - supports intervention: fix a chosen latent to a value, re-render

Causal parents S and nuisance latents are configurable.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class _BaseSCM:
    """Base class for synthetic SCM generators."""

    def __init__(
        self,
        causal_parents: List[int],
        nuisance_dims: int = 4,
        img_size: int = 32,
        latent_dim: int = 8,
        seed: int = 42,
    ):
        self.causal_parents = causal_parents
        self.nuisance_dims = nuisance_dims
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.seed = seed

        self._rng = np.random.RandomState(seed)
        self._render_weight = None  # lazy-init in _init_renderer

    def _init_renderer(self, n_samples: int = 1000):
        """Lazy-init a random nonlinear decoder: z -> pixel image."""
        gen = torch.Generator().manual_seed(self.seed)
        # 3-layer MLP decoder
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3 * self.img_size * self.img_size),
            torch.nn.Sigmoid(),
        )
        # warm up with a forward pass
        dummy = torch.randn(1, self.latent_dim, generator=gen)
        self._decoder(dummy)

    def sample_prior(self, n: int) -> torch.Tensor:
        """Sample z ~ p(z). Returns shape (n, latent_dim)."""
        raise NotImplementedError

    def render(self, z: torch.Tensor) -> torch.Tensor:
        """Render latents to pixel images. Returns (B, 3, H, W)."""
        if not hasattr(self, "_decoder"):
            self._init_renderer()
        raw = self._decoder(z)
        return raw.reshape(-1, 3, self.img_size, self.img_size)

    def intervene(self, z: torch.Tensor, latent_idx: int, value: float) -> torch.Tensor:
        """Hard intervention: do(z_latent_idx = value). Returns modified z."""
        z_interv = z.clone()
        z_interv[:, latent_idx] = value
        return z_interv

    def get_causal_latent_indices(self) -> List[int]:
        return self.causal_parents

    def get_nuisance_latent_indices(self) -> List[int]:
        all_idxs = list(range(self.latent_dim))
        return [i for i in all_idxs if i not in self.causal_parents]


class IndependentSCM(_BaseSCM):
    """(A) All latents are independent standard normals. No causal structure."""

    def sample_prior(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.latent_dim)


class CausalChainSCM(_BaseSCM):
    """(B) Linear causal chain: z1 ~ N(0,1), z2 = 0.8*z1 + eps2, z3 = 0.8*z2 + eps3, ...

    Causal parents S are the first len(causal_parents) latents in the chain.
    """

    def __init__(self, causal_parents: List[int], **kwargs):
        super().__init__(causal_parents, **kwargs)
        # transition strengths
        self._chain_coeffs = 0.8

    def sample_prior(self, n: int) -> torch.Tensor:
        z = torch.zeros(n, self.latent_dim)
        z[:, 0] = torch.randn(n)
        for i in range(1, self.latent_dim):
            z[:, i] = self._chain_coeffs * z[:, i - 1] + 0.6 * torch.randn(n)
        return z


class ConfoundedSCM(_BaseSCM):
    """(C) A confounder u influences two observed latents (z_j, z_k).

    z_j = u + eps_j
    z_k = u + eps_k
    All other latents are independent.
    """

    def __init__(
        self,
        causal_parents: List[int],
        confounded_pair: Tuple[int, int] = (0, 1),
        **kwargs,
    ):
        super().__init__(causal_parents, **kwargs)
        self._confounded_pair = confounded_pair

    def sample_prior(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim)
        u = torch.randn(n)
        j, k = self._confounded_pair
        z[:, j] = u + 0.3 * torch.randn(n)
        z[:, k] = u + 0.3 * torch.randn(n)
        return z


def get_scm(name: str, **kwargs) -> _BaseSCM:
    mapping = {
        "independent": IndependentSCM,
        "causal_chain": CausalChainSCM,
        "confounded": ConfoundedSCM,
    }
    if name not in mapping:
        raise ValueError(f"Unknown SCM '{name}'. Options: {list(mapping.keys())}")
    return mapping[name](**kwargs)
