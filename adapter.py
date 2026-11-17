"""Constrained ridge representation adaptation (row-vector convention).

The adapter learns a square matrix ``W`` (``d x d``) that minimises

    J(W) = ||X W - T||_F^2 + lambda * ||W - I||_F^2
    subject to  A W = A

where ``X`` and ``T`` are the paired supervised matrices, ``A`` is a matrix
whose *rows* are old visual directions that must be preserved, and
``lambda > 0`` is the ridge strength.

Because ``A W = A`` is equivalent to ``Q W = Q`` for any orthonormal basis
``Q`` of ``rowspace(A)``, only the row space of ``A`` matters.  Duplicate,
correlated, and all-zero rows are therefore handled by a rank-revealing SVD.
The affine constraint is eliminated with an orthonormal complement ``P`` of
that row space:

    W = Q^T Q + P^T V,        V free (size (d-r) x d).

Substituting into the strictly convex quadratic gives the small normal
equation

    (P H P^T) V = P H (W0 - Q^T Q),    H = X^T X + lambda I,

with ``W0 = H^{-1}(X^T T + lambda I)`` the ordinary ridge solution.  This
solves the constrained problem directly; we never fit unconstrained and then
overwrite old directions row-by-row.

All numerics are float64.  The returned model is plain JSON-serialisable data
and can be loaded independently of the process that produced it.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_F64 = np.float64
_TINY = 1e-300

# Contract range from the task statement.
RIDGE_MIN = 1e-3
RIDGE_MAX = 2.0

ADAPTER_SCHEMA = "constrained_ridge_adapter/1"


class AdapterError(ValueError):
    """Raised for invalid adapter inputs or adapter models."""


# ---------------------------------------------------------------------------
# Small generic helpers (also used by recognizer.py)
# ---------------------------------------------------------------------------
def as_finite_array(value: Any, name: str) -> np.ndarray:
    """Coerce ``value`` to a finite float64 2-D array, rejecting bad input."""
    try:
        arr = np.asarray(value, dtype=_F64)
    except (TypeError, ValueError) as exc:  # ragged / non-numeric
        raise AdapterError(f"{name} must be a rectangular numeric array") from exc
    if arr.ndim != 2:
        raise AdapterError(f"{name} must be a 2-D matrix (rows are samples)")
    if arr.size and not np.all(np.isfinite(arr)):
        raise AdapterError(f"{name} contains non-finite values")
    return arr


def infer_dim(candidates: list[tuple[Any, str]]) -> int:
    """Infer the feature dimension ``d`` from the first 2-D candidate."""
    for value, name in candidates:
        try:
            arr = np.asarray(value, dtype=_F64)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"{name} must be a rectangular numeric array") from exc
        if arr.ndim == 2 and arr.shape[1] > 0:
            if arr.size and not np.all(np.isfinite(arr)):
                raise AdapterError(f"{name} contains non-finite values")
            return int(arr.shape[1])
    raise AdapterError(
        "cannot infer feature dimension d (supply a 2-D target/anchor/feature)")


def coerce_rows(value: Any, name: str, d: int) -> np.ndarray:
    """Coerce to a finite ``(n, d)`` float64 matrix, allowing empty inputs.

    An empty Python list / 0-D / 1-D array is interpreted as ``(0, d)`` so
    that "empty X is legal" works without an explicit dimension.
    """
    try:
        arr = np.asarray(value, dtype=_F64)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} must be a rectangular numeric array") from exc
    if arr.size == 0:
        # Never invent samples.  A genuinely empty X/T (no supervised rows) is
        # represented as (0, d); an explicit (0, d) collapses to the same.
        if arr.ndim > 2:
            raise AdapterError(f"{name} has an unsupported shape {arr.shape}")
        if arr.ndim == 2 and arr.shape[0] not in (0,) and arr.shape[1] == 0:
            raise AdapterError(f"{name} has rows but no features")
        if arr.ndim == 2 and arr.shape[1] not in (0, d):
            raise AdapterError(f"{name} has {arr.shape[1]} columns, expected {d}")
        return np.zeros((0, d), _F64)
    if arr.ndim != 2:
        raise AdapterError(f"{name} must be a 2-D matrix (rows are samples)")
    if arr.shape[1] != d:
        raise AdapterError(f"{name} has {arr.shape[1]} columns, expected {d}")
    if not np.all(np.isfinite(arr)):
        raise AdapterError(f"{name} contains non-finite values")
    return arr


def _float_array(value: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=_F64)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(arr)):
        raise AdapterError(f"{name} must be finite")
    return arr


def scalar_in_range(value: Any, name: str, lo: float, hi: float,
                    *, lo_open: bool = False) -> float:
    """Validate a scalar hyper-parameter against an inclusive (or open) range."""
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 0:
        raise AdapterError(f"{name} must be a scalar")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} must be a real number") from exc
    if not np.isfinite(x):
        raise AdapterError(f"{name} must be finite")
    below = x <= lo if lo_open else x < lo
    if below or x > hi:
        bracket = "(" if lo_open else "["
        raise AdapterError(f"{name}={x!r} outside allowed range {bracket}{lo}, {hi}]")
    return x


def canonical_digest(payload: Any) -> str:
    """Stable SHA-256 over canonical JSON (for model/version verification)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dims(value: Any, name: str, *, square: bool = False) -> np.ndarray:
    arr = as_finite_array(value, name)
    if square and arr.shape[0] != arr.shape[1]:
        raise AdapterError(f"{name} must be square, got {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Constraint basis
# ---------------------------------------------------------------------------
class ConstraintBasis:
    """Orthonormal row-space basis ``Q`` of ``A`` and a complement ``P``.

    ``Q`` is ``r x d`` with orthonormal rows; ``P`` is ``(d-r) x d`` with
    orthonormal rows and ``P Q^T = 0``.  ``rank`` is the numerically effective
    protection rank.  ``A W = A`` holds iff ``Q W = Q``; the free part of ``W``
    lives on ``P``.
    """

    __slots__ = ("Q", "P", "rank", "d")

    def __init__(self, Q: np.ndarray, P: np.ndarray, rank: int, d: int) -> None:
        self.Q, self.P, self.rank, self.d = Q, P, rank, d

    @classmethod
    def from_anchors(cls, anchors: Any) -> "ConstraintBasis":
        A = as_finite_array(anchors, "anchors")
        d = A.shape[1]
        if A.shape[0] == 0:
            return cls(np.zeros((0, d), _F64), np.eye(d), 0, d)
        # Rank-revealing SVD; effective rank uses a relative tolerance so
        # all-zero and correlated duplicate rows collapse cleanly.
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        smax = float(s[0]) if s.size else 0.0
        if smax <= _TINY:
            rank = 0
        else:
            rank = int(np.sum(s > max(A.shape) * np.finfo(_F64).eps * smax))
        rank = min(rank, d)
        Q = Vt[:rank, :].copy()
        if rank == 0:
            P = np.eye(d)
        else:
            _, _, VtP = np.linalg.svd(Q)
            P = VtP[rank:, :].copy()
        return cls(Q, P, rank, d)

    def constraint_residual(self, W: np.ndarray, A: np.ndarray) -> float:
        """Absolute Frobenius norm ``||A W - A||_F`` (keyed to atol=2e-8)."""
        if A.shape[0] == 0:
            return 0.0
        return float(np.linalg.norm(A @ W - A))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(X: Any, T: Any, W: Any, lam: float, *, strict: bool = True) -> float:
    """Recompute ``J(W)`` with the current data.

    ``J(W) = ||X W - T||_F^2 + lambda ||W - I||_F^2``.  ``X`` may be empty.
    Shapes are validated unless ``strict=False`` (internal, pre-checked data).
    """
    if strict:
        W = as_finite_array(W, "W")
        if W.shape[0] != W.shape[1]:
            raise AdapterError("W must be square")
        d = W.shape[1]
        X = coerce_rows(X, "X", d)
        T = coerce_rows(T, "T", d)
        lam = scalar_in_range(lam, "lambda", RIDGE_MIN, RIDGE_MAX)
    else:
        X = np.asarray(X, _F64)
        T = np.asarray(T, _F64)
        W = np.asarray(W, _F64)
        d = W.shape[1]
    if X.shape[0] != T.shape[0]:
        raise AdapterError("X and T must have the same number of rows")
    if X.shape[1] != d or T.shape[1] != d:
        raise AdapterError("X, T and W must share dimensionality d")
    eye = np.eye(d)
    return float(np.sum((X @ W - T) ** 2) + lam * np.sum((W - eye) ** 2))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _empty_X_fallback(T: np.ndarray, lam: float, constr: ConstraintBasis) -> np.ndarray:
    """Analytic minimiser when X is empty.

    With no data, H = lambda I and the unconstrained optimum is W0 = I.  That
    point is always feasible (``A I = A``), so it is also the constrained
    optimum: ``W = I`` for any anchor set.
    """
    return np.eye(constr.d, dtype=_F64)


def fit(features, targets, anchors, ridge=0.5):
    """Fit the constrained ridge adapter.

    Returns a plain-dict model with the matrix ``W``, a recomputable
    ``objective``, the effective protected ``rank`` ``A``, and the constraint
    ``residual``.  ``W`` is obtained by direct constrained optimisation, not by
    post-hoc row overwriting.
    """
    lam = scalar_in_range(ridge, "ridge/lambda", RIDGE_MIN, RIDGE_MAX)

    # Infer d from targets/anchors so that an empty X remains legal.
    d = infer_dim([(targets, "fit_targets"), (anchors, "anchors"),
                   (features, "fit_features")])
    X = coerce_rows(features, "fit_features", d)
    T = coerce_rows(targets, "fit_targets", d)
    A = coerce_rows(anchors, "anchors", d)

    if X.shape[0] != T.shape[0]:
        raise AdapterError("fit_features and fit_targets must have equal rows")

    constr = ConstraintBasis.from_anchors(A)
    H = X.T @ X + lam * np.eye(d)
    W0 = np.linalg.solve(H, X.T @ T + lam * np.eye(d))

    if constr.rank == 0:
        W = W0
    elif constr.rank == d:
        # Full protected space: the constraint pins the entire matrix (A W = A
        # with A full rank forces W = I on all of R^d).
        W = np.eye(d, dtype=_F64)
    else:
        Q, P = constr.Q, constr.P
        # Free block: (P H P^T) V = P H (W0 - Q^T Q).  H is SPD, so the
        # reduced matrix is SPD and well-conditioned for lambda >= 1e-3.
        PQP = P @ H @ P.T
        rhs = P @ H @ (W0 - Q.T @ Q)
        if PQP.size == 0:
            V = np.zeros((0, d), _F64)
        elif np.linalg.cond(PQP) > 1.0 / np.finfo(_F64).eps:
            V = np.linalg.lstsq(PQP, rhs, rcond=None)[0]
        else:
            V = np.linalg.solve(PQP, rhs)
        W = Q.T @ Q + P.T @ V

    obj = objective(X, T, W, lam, strict=False)
    residual = constr.constraint_residual(W, A)
    # Empirical lower bound on the optimum: the unconstrained ridge optimum.
    obj_lower = objective(X, T, W0, lam, strict=False)

    model = {
        "matrix": W.tolist(),
        "objective": float(obj),
        "rank": int(constr.rank),
        "constraint_residual": float(residual),
        "ridge": float(lam),
        "dim": int(d),
        "rows": int(X.shape[0]),
        "unconstrained_objective": float(obj_lower),
        "schema": ADAPTER_SCHEMA,
    }
    return model


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def _matrix_from(model: Any) -> np.ndarray:
    """Extract a square matrix from a model dict, a raw matrix, or a list."""
    if isinstance(model, Mapping):
        if "matrix" not in model:
            raise AdapterError("model must contain 'matrix'")
        W = np.asarray(model["matrix"], dtype=_F64)
    else:
        W = np.asarray(model, dtype=_F64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise AdapterError("adapter matrix must be square d x d")
    if not np.all(np.isfinite(W)):
        raise AdapterError("adapter matrix must be finite")
    return W


def transform(features, model):
    """Apply the given adapter's W: returns ``Z @ W``.

    ``model`` may be a model dict (with ``matrix``) or a raw square matrix.
    The passed ``features`` matrix must have the matrix dimension as columns.
    """
    W = _matrix_from(model)
    d = W.shape[0]
    Z = coerce_rows(features, "features", d)
    return Z @ W


# ---------------------------------------------------------------------------
# Save / load and independent verification
# ---------------------------------------------------------------------------
def save(model: Mapping, path: str | Path) -> Path:
    """Persist an adapter model as JSON."""
    validate_model(model)
    p = Path(path)
    p.write_text(json.dumps(dict(model), allow_nan=False, sort_keys=True, indent=2))
    return p


def load(path: str | Path) -> dict:
    """Load and structurally validate an adapter model from JSON."""
    p = Path(path)
    try:
        model = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{p} is not valid JSON") from exc
    validate_model(model)
    return model


def validate_model(model: Any) -> None:
    if not isinstance(model, Mapping):
        raise AdapterError("adapter model must be a mapping")
    for key in ("matrix", "objective", "rank", "constraint_residual", "ridge", "dim"):
        if key not in model:
            raise AdapterError(f"adapter model missing '{key}'")
    W = _dims(model["matrix"], "model['matrix']")
    if W.shape[0] != W.shape[1]:
        raise AdapterError("adapter matrix must be square")
    if int(model["dim"]) != W.shape[0]:
        raise AdapterError("adapter dim does not match matrix")
    scalar_in_range(model["ridge"], "ridge", RIDGE_MIN, RIDGE_MAX)
    if not (0 <= int(model["rank"]) <= W.shape[0]):
        raise AdapterError("adapter rank out of range")


def verify(model: Mapping, X, T, anchors, lam=None, *,
           atol: float = 2e-8, rtol: float = 2e-8) -> dict:
    """Independently recheck feasibility and (KKT) optimality.

    Parameters
    ----------
    X, T, anchors : the data used at fit time (needed to recompute J).
    lam : ridge value; defaults to ``model['ridge']``.

    Returns a dict with the recomputed objective, constraint residual, KKT
    residual, and booleans ``within_tolerance`` / ``is_stationary``.  The KKT
    check is done on the *free* block ``P`` directly: the minimiser satisfies

        P (H W - X^T T - lambda I) = 0,   H = X^T X + lambda I

    for every feasible W, so this is a necessary and sufficient test on the
    free variables.  If X is empty, ``W0 = I`` is also reported.
    """
    W = _dims(model["matrix"], "model['matrix']")
    d = W.shape[0]
    Xa = coerce_rows(X, "X", d)
    Ta = coerce_rows(T, "T", d)
    A = coerce_rows(anchors, "anchors", d)
    if lam is None:
        lam = float(model["ridge"])
    lam = scalar_in_range(lam, "lambda", RIDGE_MIN, RIDGE_MAX)
    if Xa.shape[0] != Ta.shape[0]:
        raise AdapterError("verify: X and T must have equal rows")

    constr = ConstraintBasis.from_anchors(A)
    obj = objective(Xa, Ta, W, lam, strict=False)
    residual = constr.constraint_residual(W, A)
    H = Xa.T @ Xa + lam * np.eye(d)
    grad = H @ W - (Xa.T @ Ta + lam * np.eye(d))
    kkt = float(np.linalg.norm(constr.P @ grad))
    kkt_unconstrained = float(np.linalg.norm(grad))
    allowed_obj = atol + rtol * max(abs(obj), 1.0)
    objective_agrees = abs(obj - float(model["objective"])) <= allowed_obj

    within = ((residual <= atol + rtol * max(float(np.linalg.norm(A)), 1.0))
              and obj <= model["objective"] + allowed_obj)
    stationary = kkt <= atol + rtol * max(float(np.linalg.norm(Xa.T @ Ta + lam * np.eye(d))), 1.0)

    out = {
        "objective": obj,
        "reported_objective": float(model["objective"]),
        "objective_agrees": bool(objective_agrees),
        "constraint_residual": residual,
        "kkt_residual": kkt,
        "kkt_residual_unconstrained": kkt_unconstrained,
        "rank": int(constr.rank),
        "within_tolerance": bool(within and objective_agrees),
        "is_stationary": bool(stationary),
        "atol": atol,
        "rtol": rtol,
    }
    if Xa.shape[0] == 0:
        W0 = _empty_X_fallback(Ta, lam, constr)
        out["empty_fallback_objective"] = objective(Xa, Ta, W0, lam, strict=False)
        out["matches_empty_fallback"] = bool(
            np.allclose(W, W0, atol=atol, rtol=rtol))
    return out


def fingerprint(model: Mapping) -> str:
    """Stable digest of the adapter's numeric content (matrix + ridge).

    Tolerates minimal externally supplied adapter models (only ``matrix``
    required); missing ``ridge``/``rank`` default to 0.0 / 0.
    """
    if not isinstance(model, Mapping) or "matrix" not in model:
        raise AdapterError("adapter model must contain 'matrix'")
    W = _dims(model["matrix"], "model['matrix']")
    if W.shape[0] != W.shape[1]:
        raise AdapterError("adapter matrix must be square")
    return canonical_digest({"w": W.tolist(),
                             "ridge": float(model.get("ridge", 0.0)),
                             "dim": int(model.get("dim", W.shape[0])),
                             "rank": int(model.get("rank", 0))})
