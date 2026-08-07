# Ground-Truth Implementation Notes

Extracted from the actual code (not the paper draft) so the TMLR text can be
corrected to match what is implemented. Read-only investigation — no code was
changed to match the paper. Where the code and paper appear to disagree, that
is flagged explicitly.

References below are to `data/synthetic_scm.py`, `data/colored_mnist.py`,
`data/waterbirds.py`, `experiments/train.py`, `diagnostics/acs.py`,
`diagnostics/itg.py`, `losses/cir.py`, `losses/hsic.py`,
`experiments/run_table1.py`, `experiments/scaling_test.py`,
`experiments/significance_test.py`.

---

## (a) Appendix B.4 draft — Synthetic SCM Generation (as implemented)

Source: `data/synthetic_scm.py`.

### Shared configuration
- Latent dimensionality **K = 8** (`latent_dim: int = 8` in `_BaseSCM`).
- Observation dimensionality: rendered image is **(3, 32, 32)** (float in
  [0,1]); flattened that is **3072** dims (`img_size: int = 32`).
- Causal latent indices: `causal_parents` (TrainConfig default `[0, 1]`), i.e.
  **2 causal latents**.
- `nuisance_dims` (RESOLVED): the config/constructor previously accepted a
  `scm_nuisance_dims`/`nuisance_dims` parameter that was **stored and never
  used**. It has been removed from `TrainConfig`, `_BaseSCM`, and
  `build_dataloaders`. The SCM produces 8 latent dims total = 2 causal + 6
  non-causal (the "nuisance" split is implicit: every non-causal latent is a
  nuisance latent).
- All SCMs hard-code `seed = 42` for the renderer.

### Rendering function g(z)
`_init_renderer()` builds a random MLP decoder (lazy-initialized on first
`render` call):

```
g(z) = Sigmoid( Linear(128 -> 3*32*32) ( ReLU( Linear(64 -> 128) ( ReLU( Linear(8 -> 64) ( z ) ) ) ) ) )
```

- Input: z in R^8 → output reshaped to (B, 3, 32, 32), sigmoid → [0,1].
- Layer weights use PyTorch's default (global-RNG) initialization; the
  explicit `Generator(seed=42)` is only used for the dummy warm-up input.
  Consequence: g is deterministic only if `torch.manual_seed(42)` was set
  before the first render (the training script calls `set_seed`, so in
  practice it is reproducible within a run).

### Label function h(z)
**None.** SCM data has no class labels. `SCMDataset.__getitem__` returns
`(img, img, 0, 0)` — dummy label and dummy spurious value. ITG is never run
on SCM data (train.py gates ITG to `colored_mnist`/`waterbirds`); only ACS is
run, and ACS needs no labels.

### Dataset A — IndependentSCM
Structural equation:

```
z_i ~ N(0, 1),  i = 1..8, mutually independent
```

`sample_prior(n)` returns `torch.randn(n, 8)`. No causal structure; the
"causal parents" list is used only to pick which coordinates ACS intervenes on.

### Dataset B — CausalChainSCM
Linear AR(1) chain with coefficient 0.8 and noise std 0.6:

```
z_1 ~ N(0, 1)
z_i = 0.8 * z_{i-1} + 0.6 * eps_i,   eps_i ~ N(0, 1),  i = 2..8
```

The `causal_parents` argument does **not** change sampling — it only marks the
first `len(causal_parents)` chain variables as the ACS intervention targets.

### Dataset C — ConfoundedSCM
Confounder u drives two observed latents (default pair (0,1)):

```
u ~ N(0, 1)
z_j = u + 0.3 * eps_j,   j in confounded_pair
z_k = u + 0.3 * eps_k,   k in confounded_pair
z_i ~ N(0, 1) otherwise
```

`get_causal_latent_indices()` returns `causal_parents` (default `[0,1]`),
which by default coincides with the confounded pair.

### Intervention API (used by ACS, item (d))
`intervene(z, latent_idx, value)` **sets** `z[:, latent_idx] = value` — a hard
`do(z_i = value)` that replaces the column with a constant. It does **not**
add `value` to the existing value.

---

## (b) Colored MNIST and Waterbirds environment splits (as implemented)

Source: `data/colored_mnist.py`, `data/waterbirds.py`, `experiments/train.py`
(`build_dataloaders`).

### Colored MNIST
- **Train**: MNIST `train` (60,000 samples), `p_corr = cfg.colored_mnist_p_corr`
  (default **0.9**). Used both for contrastive training and for the ITG probe.
- **Correlated test env** (`corr_loader`): MNIST `test` (10,000), `p_corr = 0.9`.
- **Uncorrelated/anti-correlated test env** (`uncorr_loader`): MNIST `test`
  (10,000), `p_corr = 1.0 - 0.9 = 0.1`.
- **No held-out validation split** exists for Colored MNIST in this code.
  The two test sets double as the corr/uncorr evaluation environments.
- Tinting: each digit gets a fixed hue (`hue = digit/10` through a hue→RGB
  map). With probability `p_corr` the sample uses its class-default tint
  (`spurious_label = label`); otherwise `spurious_label = randint(0,10)`.
- RNG: train uses `seed=cfg.seed`; corr uses `cfg.seed + 1`; uncorr uses
  `cfg.seed + 2` (independent tint assignments).

### Waterbirds (WILDS)
- **Train**: WILDS `train` split (raw images + the contrastive augmentation
  pipeline; also used for the ITG probe with no augmentation).
- **Validation** (`corr_loader`): WILDS `val` split — this is the held-out
  validation environment.
- **Test** (`uncorr_loader`): WILDS `test` split.
- Spurious confounder: background type `metadata[0]` (0 = land, 1 = water);
  target `y` is the bird class. No group-balanced resampling is added on top
  of the official WILDS splits.

---

## (c) ITG vs the "ID Acc" / "OOD Acc" columns

Source: `diagnostics/itg.py`, `experiments/significance_test.py`.

They are **the same measurement pair**. There is a single computation:

1. Freeze the encoder; extract representations from the **train** split
   (`probe_train_loader`, no augmentation).
2. Fit a multiclass `LogisticRegression(C=1.0, max_iter=1000)` linear probe.
3. `acc_corr` = probe accuracy on the **correlated** env (`corr_loader`).
4. `acc_uncorr` = probe accuracy on the **uncorrelated/anti** env
   (`uncorr_loader`).
5. `accuracy_drop = (acc_corr - acc_uncorr) / acc_corr * 100`.

`significance_test.py` maps the reported column **"OOD Acc" → `acc_uncorr`**
and **"ITG Drop (%)" → `itg_drop`**. There is no separate "ID Acc" key in the
code; the ID side of the pair is `acc_corr` (accuracy on the correlated test
env). So if the paper reports "ID Acc"/"OOD Acc" side-by-side, the clarifying
sentence for §6.2 should say:

> "ID Acc and OOD Acc are the accuracies of the same frozen-encoder linear
> probe measured on the correlated and uncorrelated test environments
> respectively; ITG is their relative drop. The probe is always trained on
> the correlated train split."

**One caveat worth stating explicitly**: "ID Acc" is *not* train accuracy —
it is accuracy on a correlated *test* set that shares the train distribution's
spurious correlation.

---

## (d) ACS exact formula as implemented

Source: `diagnostics/acs.py`, `average_causal_sensitivity`.

Fixed parameters: `n_samples = 512`, `delta = 1.0`, `batch_size = 64`
(defaults; train.py uses `cfg.acs_n_samples=512`, `cfg.acs_delta=1.0`).

1. Sample latents once: `z_base = scm.sample_prior(512)`.
2. For each causal latent index `i in scm.get_causal_latent_indices()`
   (default `[0, 1]`, so 2 latents):
   - `z_interv = scm.intervene(z_base, i, delta)` — **hard do-intervention**:
     `z_interv[:, i] = delta` (sets to constant **1.0**; it is *not* an
     additive `z_i + delta` despite the docstring saying so).
   - Render both, encode both (`h_base`, `h_interv` are the backbone output,
     not the projection).
   - Per-coordinate sensitivity for latent `i`:

     ```
     E[Δr_j] = (1/512) * Σ_n | h_interv[n, j] - h_base[n, j] |
     ```

     (mean absolute difference per representation coordinate, averaged over
     the 512 samples in batches of 64).
   - Normalize by intervention magnitude: `sens_vec = E[Δr_j] / delta`
     (with `delta = 1.0` this is numerically a no-op but is the "E[Δz_i]"
     denominator the code comment references).
3. Build `S` of shape (n_causal_latents, rep_dim) = `sens_mat`.
4. **Hungarian matching** on `-S` (`scipy.optimize.linear_sum_assignment`) to
   get a one-to-one (latent → representation coordinate) matching.
5. Per-latent normalized concentration:

   ```
   c_i = S[i, matched_coord(i)] / Σ_j S[i, j]
   ```

   i.e. the fraction of latent `i`'s total sensitivity that lands on its
   matched coordinate.
6. **Single reported number**: `concentration = mean_i c_i`, returned as
   `acs_result["concentration"]` and logged under `history["acs"]`.

Note for the paper: the docstring says the intervention is `do(z_i = z_i +
delta)`; the code sets the latent to `delta` exactly. With `delta = 1.0` and
`E[z_i] = 0` this is approximately a +1 shift in expectation, but strictly it
is a set-to-1 do-intervention.

---

## (e) CIR penalty — confirm pairwise-coordinate HSIC

Source: `losses/cir.py`, `losses/hsic.py`.

**Yes — it matches Section 5.2.** The penalty is a sum over unordered
coordinate pairs (j, k), j ≠ k, of a *scalar biased HSIC* between the two
coordinate-columns of the stacked residuals. It is vectorized (einsum/trace
trick, no Python double loop), but mathematically it is exactly

```
R_CIR(R) = Σ_{j≠k} HSIC_b(R[:, j], R[:, k])
```

`losses/cir.py` (abridged, actual function):

```python
class CIRLoss(torch.nn.Module):
    def forward(self, h1, h2):
        h_bar = (h1 + h2) / 2.0
        r1 = h1 - h_bar
        r2 = h2 - h_bar
        R = torch.cat([r1, r2], dim=0)          # (2N, p)
        return self._hsic_fn(R)                  # PairwiseHSICExact or PairwiseHSICRFF
```

`losses/hsic.py` `pairwise_hsic_exact` (abridged):

```python
def pairwise_hsic_exact(R, sigma=None):
    n, p = R.shape
    K_all[j] = rbf_kernel_1d(R[:, j], sigma)     # per-column kernel (p, n, n)
    K_centered = K_all @ H
    T[j, k] = tr(K_j H K_k H)                    # einsum('jab,kba->jk', ...)
    total = T.sum() - T.trace()                  # drop diagonal (j == k)
    return total / (n ** 2)
```

The RFF estimator (`pairwise_hsic_rff`) approximates the same sum over pairs
and uses the same `1/n²` normalization. Use this to fix **Algorithm 1** in the
paper to say: compute per-pair scalar HSIC for every (j, k), j ≠ k, and sum.

---

## (f) Number of seeds per (method, dataset) cell

- `experiments/run_table1.py`: `SEEDS = [0, 1, 2]` → **3 seeds** per
  (method, dataset) cell. Methods: `simclr`, `simclr_marginal`,
  `barlow_twins`, `vicreg`, `cir`.
- `experiments/scaling_test.py`: `SEEDS = [0, 1, 2]` → **3 seeds** per
  (backbone, method) cell. Methods: `simclr`, `cir`; backbones: `resnet18`,
  `resnet50`.
- `experiments/significance_test.py`: default `[0, 1, 2]`, configurable via
  `--seeds`.

Consistent with the README's "every table entry is mean ± std over 3 seeds".

---

## Items to flag for the paper (no code changed)

1. **ACS docstring vs code**: docstring claims additive `do(z_i + delta)`;
   code sets `z_i = delta` (hard set-to-1.0). Decide which to describe.
2. **ACS vs §6.1 description**: code measures per-coordinate mean-abs change
   `E|Δr_j|`, not an L2 norm; the "E[Δz_i]" denominator is the constant
   `delta`.
3. **SCM latent structure**: `latent_dim = 8` fixed; the `nuisance_dims`
   config param was removed (never used), so the causal/nuisance split is
   2/6, not 2/4.
4. **Colored MNIST has no validation split** — only two test environments.
5. **ID/OOD Acc vs ITG** are the same probe measurement (item c).
6. **Waterbirds backbone/resolution (RESOLVED)**: Decision from Aditya — keep
   **resnet18** (paper-authoritative, keeps cross-method comparison fair; the
   resnet50/128 entry was a deliberate-but-flagged choice in commit eecab53,
   now reverted to resnet18). Keep **128x128 resolution** as a documented
   compute-constraint deviation from the paper's original 256x256, applied
   identically to all methods/baselines. **Appendix B.2 should state**: "128x128
   resolution (reduced from an original 256x256 design due to compute
   constraints; validation curves for both resolutions in Appendix X show
   consistent method ranking)."  `img_size` in the config is set to 128 to
   match the actual input (it is ViT-only; resnet ignores it).
7. **Optimizer (RESOLVED)**: Keep `torch.optim.Adam` as-is; the paper/config
   will be corrected to say Adam (not AdamW).
