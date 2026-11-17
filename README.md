# Incremental visual embedding recognition

Python 3.11 + NumPy/SciPy. Small fixed visual features — no image backbone and no
remote model. Row-vector convention throughout (`Z @ W`, each row one sample).

Environment: this workspace's system Python is externally managed (PEP 668), so
NumPy/SciPy are installed in a local venv. Use `.venv/bin/python` below; on a
machine with NumPy/SciPy already available, plain `python` works.

```bash
.venv/bin/python workflow.py fixtures/concepts.json     # run the full pipeline
.venv/bin/python tests.py                               # 77 verification checks
.venv/bin/python export_models.py fixtures/concepts.json artifacts  # save/load
```

## What was extended

`adapter.py` — constrained ridge representation adaptation.
`recognizer.py` — class-conditional nonconformity recognizer with abstention.
`workflow.py` — `run(data)` / `run(path)` orchestrating both, plus verification.
`export_models.py` — writes independently loadable `adapter.json` /
`recognizer.json` / `results.json`.

## 1. Representation adaptation

Learn a `d×d` matrix `W` minimising

```
J(W) = ||X W − T||_F² + λ ||W − I||_F²      subject to   A W = A
```

with `X, T` the paired supervised rows, `A` the old directions (rows), `λ > 0`
(hence a unique optimum on the feasible set). Only the **row space** of `A` is
protected: duplicate / correlated / zero rows are handled by a rank-revealing
SVD. `A` may have any number of rows (including none).

**Direct constrained solve (not post-hoc overwrite).** Let `Q` (`r×d`) be an
orthonormal basis of `rowspace(A)` and `P` (`(d−r)×d`) an orthonormal
complement, `P Qᵀ = 0`. Then `A W = A ⇔ Q W = Q`, and every feasible `W` is

```
W = Qᵀ Q + Pᵀ V ,   V free.
```

Substituting into the strictly convex quadratic gives the small SPD normal
equation

```
(P H Pᵀ) V = P H (W0 − Qᵀ Q),   H = XᵀX + λI,   W0 = H⁻¹(XᵀT + λI).
```

`W0` is the ordinary (unconstrained) ridge optimum; the equation above corrects
it only in the free subspace. Limiting cases: `rank(A)=0` → `W = W0`; `rank = d`
→ `W = I`; empty `X` → `W0 = I` is already feasible, so `W = I`.

`fit` returns `matrix`, a recomputable `objective` (`adapter.objective`), the
effective protected `rank`, and the constraint `residual` (`||AW−A||_F`).
`transform(Z, model)` actually computes `Z @ W`; it accepts any model dict or
raw square matrix.

### Two feasible-constraint strategies (tradeoff)

| | Null-space elimination (chosen) | Post-hoc projection / row overwrite |
|--|--|--|
| Optimality | Solves the KKT system exactly on the free block | Feasible but generally **not** optimal (higher `J`) |
| Rank handling | One SVD of `A`; rank-revealing, duplicates collapse | Needs the same basis, or risk degrading protected rows |
| Stability | Reduced system is SPD, conditioning bounded by `λ ≥ 1e-3` | No reduced solve, but silently changes directions |
| Cost | `O(n d² + d³)` | `O(n d² + d³)` similar; but wrong objective |

`tests.py::test_adapter_not_posthoc_overwrite` shows a feasible naive overwrite
of `W0` has strictly larger `J`, so the direct solve matters.

## 2. Recognition with abstention

Per class `c` the current support rows are transformed by the **current** `W`
into `z_i`. Statistics (population denominator, fixed `τ`, never `n−1`, never
per-class tuning):

```
μ_c = mean(z_i)
Σ_c = (Σ_i (z_i−μ_c)(z_i−μ_c)ᵀ)/n_c + τ I
s_c(z) = (z−μ_c)ᵀ Σ_c⁻¹ (z−μ_c)          (computed with a solve, no inverse)
```

The **same** representation and statistics score external queries and the
class's own calibration vectors. Calibration only calibrates rejection; it is
never part of representation training.

For a query `z`:

```
p_c = (1 + #{ j ∈ calibration_c : s_c(z_j) ≥ s_c(z) }) / (n_cal,c + 1)
accept = { c : p_c > α }        (ties counted via ≥; strict > in acceptance)
```

- empty accept set → `unknown`, `classes = []`
- exactly one → `unique`
- several → `ambiguous`, **all** candidates kept (no nearest-centre pruning)

`predict` returns for each query the sorted `classes` plus per-class `scores`,
integer `exceed_counts`, `denominators`, and `pvalues` (so `p` can be checked
against the integer counts). No coverage guarantee for arbitrary unknown
distributions is claimed.

### Two statistic-rebuild strategies (tradeoff)

| | Always recompute from current support/calibration (chosen) | Cache & incrementally update |
|--|--|--|
| Correctness after re-adaptation | Recomputed under the new `W`; no stale scale | Must invalidate on every `W` change or risk frozen old stats |
| Stability | Fresh centered sums; no error accumulation | Cheaper updates but drift/rounding accumulation |
| Update cost | `O(Σ n_c d²)` per build (tiny at this scale) | Lower amortised cost, higher bookkeeping/complexity |

At this scale recomputing is cheap and safest, so `build` always derives
everything from the current data. `recognizer.verify_model(model, support,
calibration)` re-derives statistics from the model's stored `W` and flags a new
`W` paired with frozen statistics.

## 3. Versioning / independent models

Models are plain JSON. The recognizer stores `W` (snapshot), `μ_c`, `Σ_c`,
calibration scores and counts, plus three SHA-256 fingerprints:
`adapter_fingerprint`, `statistics_fingerprint`, and a combined `fingerprint`
over all numeric content. `predict` re-validates the seal, so a raw "new W, old
statistics" swap is rejected; `verify_model` with the data additionally catches
a re-sealed but stale model. `export_models.py` reloads from disk and reproduces
predictions, confirming the saved model needs no training data.

## Verification highlights (`tests.py`)

- Hand-computed optimum `W=[[1,0],[12/13,29/65]]` and its exact objective.
- Independent null-space least squares and a separate KKT linear solve agree to
  `2e-8`; random trials include correlated/zero anchor rows.
- Empty and full protection spaces; empty `X`; `A` with no rows.
- `transform` uses the model's own `W` (not a hardcoded/snapped matrix).
- Population covariance `+τI`; score via solve equals explicit inverse.
- Repeated calibration vectors give exact ties, all counted (`≥`).
- Multiple accepted classes → `ambiguous` keeping both; α endpoints (0 → all,
  1 → none); strict `>` boundary; different within-class scales.
- Class add/remove recomputes the current set; no stale class survives.
- Re-adaptation changes the same query's class-conditional results and rebuilds
  statistics under the new `W`; a frozen-statistics model is detected.
- Invalid dims / non-finite / bad λ,τ,α / empty class / mismatched class sets are
  rejected.
