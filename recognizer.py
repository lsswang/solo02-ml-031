"""Class-conditional nonconformity recogniser with abstention.

For every class ``c`` the support vectors are pushed through the *current*
adapter matrix ``W`` to obtain ``z_i``.  The class model is

    mu_c      = mean(z_i)
    Sigma_c   = (sum_i (z_i - mu_c)(z_i - mu_c)^T) / n_c + tau * I

using the population (``n``) denominator and the supplied shrinkage ``tau``;
it is never changed to ``n - 1`` nor tuned per class.  The class-conditional
nonconformity score for an external query ``z`` is

    s_c(z) = (z - mu_c)^T Sigma_c^{-1} (z - mu_c)

and is computed with a linear solve (``np.linalg.solve``), never an explicit
inverse.  The *same* current representation and statistics score both external
queries and the class's own calibration vectors.

Calibration gives, for a query score ``s_c(z)`` and class ``c``,

    p_c = (1 + #{ j in calibration_c : s_c(z_j) >= s_c(z) }) / (n_cal_c + 1)

with ties counted (``>=``).  The decision set is ``{c : p_c > alpha}``:
empty -> ``unknown``, singleton -> ``unique``, several -> ``ambiguous`` with
every candidate kept.  No nearest-centre pruning is performed.  No coverage
guarantee for arbitrary unknown distributions is claimed.

All statistics are stored in the model so it can be saved, loaded and queried
without the training data.  A fingerprint ties the adapter ``W``, class means,
covariances and calibration scores to one version.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import adapter as _adapter

_F64 = np.float64
_TAU_MIN = 1e-3
_TAU_MAX = 2.0
_ALPHA_MIN = 0.0
_ALPHA_MAX = 1.0

RECOGNIZER_SCHEMA = "nonconformity_recognizer/1"


class RecognizerError(ValueError):
    """Raised for invalid recogniser inputs or models."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _labeled_rows(data: Any, name: str, d: int) -> dict[str, np.ndarray]:
    """Validate a class -> rows mapping with identical non-empty class sets."""
    if not isinstance(data, Mapping):
        raise RecognizerError(f"{name} must be a mapping class -> vectors")
    out: dict[str, np.ndarray] = {}
    for label, rows in data.items():
        if not isinstance(label, str) or not label:
            raise RecognizerError(f"{name} keys must be non-empty strings")
        arr = _adapter.coerce_rows(rows, f"{name}[{label!r}]", d)
        if arr.shape[0] == 0:
            raise RecognizerError(f"{name}[{label!r}] must not be empty")
        out[label] = arr
    if not out:
        raise RecognizerError(f"{name} must contain at least one class")
    return out


def _check_same_classes(support: Mapping, calibration: Mapping) -> list[str]:
    if set(support) != set(calibration):
        missing = set(support) ^ set(calibration)
        raise RecognizerError(
            f"support and calibration class sets differ: {sorted(missing)}")
    return sorted(support)


def _adapter_matrix(model: Any) -> np.ndarray:
    try:
        W = (np.asarray(model["matrix"], dtype=_F64)
             if isinstance(model, Mapping) else np.asarray(model, dtype=_F64))
    except (KeyError, TypeError, ValueError) as exc:
        raise RecognizerError(
            "adapter must be a model dict with 'matrix' or a square matrix") from exc
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise RecognizerError("adapter matrix must be square d x d")
    if not np.all(np.isfinite(W)):
        raise RecognizerError("adapter matrix must be finite")
    return W


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def _scores(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """s_c(z) = diff^T cov^{-1} diff via solve; z is (n, d) or (d,)."""
    diff = z - mean
    solved = np.linalg.solve(cov, np.atleast_2d(diff).T).T
    if diff.ndim == 1:
        solved = solved[0]
    return np.sum(diff * solved, axis=-1)


def _class_stats(z: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    n = z.shape[0]
    mean = z.mean(axis=0)
    centered = z - mean
    cov = (centered.T @ centered) / n + tau * np.eye(z.shape[1])
    return mean, cov


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(support, calibration, adapter, shrinkage=0.2):
    """Fit per-class visual distributions and their calibration scores.

    ``support`` and ``calibration`` are class -> row-vector mappings with
    identical, non-empty class sets.  ``adapter`` is an adapter model (a dict
    containing ``matrix``).  ``shrinkage`` is the positive covariance ridge
    ``tau``.  Both support and calibration are transformed by the *adapter
    model's own* ``W``; calibration is never used for representation training
    and is stored only for rejection calibration.
    """
    W = _adapter_matrix(adapter)
    d = W.shape[0]
    tau = _adapter.scalar_in_range(shrinkage, "shrinkage/tau", _TAU_MIN, _TAU_MAX)

    # Snapshot the adapter's numeric content so the recogniser model can be
    # loaded independently.  Accept any legal matrix, including a minimal
    # {"matrix": ...} dict; missing metadata is defaulted.
    adapter_snapshot = {
        "matrix": W.tolist(),
        "ridge": float(adapter.get("ridge", 0.0)) if isinstance(adapter, Mapping) else 0.0,
        "rank": int(adapter.get("rank", 0)) if isinstance(adapter, Mapping) else 0,
        "dim": int(d),
    }

    sup = _labeled_rows(support, "support", d)
    cal = _labeled_rows(calibration, "calibration", d)
    classes = _check_same_classes(sup, cal)

    means: dict[str, np.ndarray] = {}
    covs: dict[str, np.ndarray] = {}
    cal_scores: dict[str, np.ndarray] = {}
    n_cal: dict[str, int] = {}
    for c in classes:
        z_sup = _adapter.transform(sup[c], adapter)
        z_cal = _adapter.transform(cal[c], adapter)
        mean, cov = _class_stats(z_sup, tau)
        means[c] = mean
        covs[c] = cov
        # Same current representation & statistics score own-class calibration.
        cal_scores[c] = _scores(z_cal, mean, cov)
        n_cal[c] = int(z_cal.shape[0])

    model = {
        "schema": RECOGNIZER_SCHEMA,
        "classes": classes,
        "dimension": int(d),
        "shrinkage": float(tau),
        "means": {c: means[c].tolist() for c in classes},
        "covariances": {c: covs[c].tolist() for c in classes},
        "calibration_scores": {c: cal_scores[c].tolist() for c in classes},
        "calibration_counts": {c: n_cal[c] for c in classes},
        "adapter": adapter_snapshot,
    }
    _seal(model)
    return model


def _content_payload(model: Mapping) -> dict:
    """Numeric content that defines one model version (excludes digests)."""
    classes = list(model.get("classes", []))
    return {
        "schema": model.get("schema"),
        "classes": classes,
        "dimension": int(model.get("dimension", 0)),
        "shrinkage": float(model.get("shrinkage", 0.0)),
        "means": [[float(x) for x in model["means"][c]] for c in classes],
        "covariances": [[[float(x) for x in row] for row in model["covariances"][c]]
                        for c in classes],
        "calibration_scores": [[float(x) for x in model["calibration_scores"][c]]
                               for c in classes],
        "calibration_counts": [int(model["calibration_counts"][c]) for c in classes],
        "adapter": {"matrix": [[float(x) for x in row]
                              for row in model["adapter"]["matrix"]],
                    "ridge": float(model["adapter"].get("ridge", 0.0)),
                    "rank": int(model["adapter"].get("rank", 0))},
    }


def _seal(model: dict) -> None:
    payload = _content_payload(model)
    model["adapter_fingerprint"] = _adapter.fingerprint(model["adapter"])
    model["statistics_fingerprint"] = _adapter.canonical_digest(
        {k: v for k, v in payload.items() if k != "adapter"})
    model["fingerprint"] = _adapter.canonical_digest(payload)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(features, model, alpha=0.4):
    """Return one decision record per query row.

    Each record has ``decision`` in {unknown, unique, ambiguous}, the sorted
    accepted ``classes``, and per-class ``scores``, ``exceed_counts``,
    ``denominators`` and ``pvalues``.  A class is accepted iff ``p_c > alpha``.
    """
    validate_model(model)
    a = _adapter.scalar_in_range(alpha, "alpha", _ALPHA_MIN, _ALPHA_MAX)
    d = int(model["dimension"])
    Z = _adapter.coerce_rows(features, "queries", d)
    z = _adapter.transform(Z, model["adapter"])

    classes = list(model["classes"])
    means = {c: np.asarray(model["means"][c], _F64) for c in classes}
    covs = {c: np.asarray(model["covariances"][c], _F64) for c in classes}
    cal = {c: np.asarray(model["calibration_scores"][c], _F64) for c in classes}
    ncal = {c: int(model["calibration_counts"][c]) for c in classes}

    out = []
    for row in z:
        scores, exceeds, denoms, pvals = {}, {}, {}, {}
        accepted: list[str] = []
        for c in classes:
            s = float(_scores(row, means[c], covs[c]))
            # Exact float comparison, ties counted. No epsilon / tie band.
            count = int(np.count_nonzero(cal[c] >= s))
            denom = ncal[c] + 1
            p = (1 + count) / denom
            scores[c] = s
            exceeds[c] = count
            denoms[c] = denom
            pvals[c] = float(p)
            if p > a:
                accepted.append(c)
        accepted.sort()
        if not accepted:
            decision = "unknown"
        elif len(accepted) == 1:
            decision = "unique"
        else:
            decision = "ambiguous"
        out.append({
            "decision": decision,
            "classes": accepted,
            "scores": scores,
            "exceed_counts": exceeds,
            "denominators": denoms,
            "pvalues": pvals,
        })
    return out


def class_scores(features, model) -> dict[str, np.ndarray]:
    """Expose per-class conditional scores s_c(z) for inspection/tests."""
    validate_model(model)
    d = int(model["dimension"])
    Z = _adapter.transform(_adapter.coerce_rows(features, "features", d),
                           model["adapter"])
    out = {}
    for c in model["classes"]:
        mean = np.asarray(model["means"][c], _F64)
        cov = np.asarray(model["covariances"][c], _F64)
        out[c] = _scores(Z, mean, cov)
    return out


# ---------------------------------------------------------------------------
# Persistence and version verification
# ---------------------------------------------------------------------------
def save(model: Mapping, path: str | Path) -> Path:
    validate_model(model)
    p = Path(path)
    p.write_text(json.dumps(dict(model), allow_nan=False, sort_keys=True, indent=2))
    return p


def load(path: str | Path) -> dict:
    p = Path(path)
    try:
        model = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise RecognizerError(f"{p} is not valid JSON") from exc
    validate_model(model)
    return model


def validate_model(model: Any) -> None:
    if not isinstance(model, Mapping):
        raise RecognizerError("recogniser model must be a mapping")
    required = ("schema", "classes", "dimension", "shrinkage", "means",
                "covariances", "calibration_scores", "calibration_counts",
                "adapter", "fingerprint")
    for key in required:
        if key not in model:
            raise RecognizerError(f"recogniser model missing '{key}'")
    if model["schema"] != RECOGNIZER_SCHEMA:
        raise RecognizerError(f"unsupported recogniser schema {model['schema']!r}")
    classes = list(model["classes"])
    if classes != sorted(classes) or len(set(classes)) != len(classes):
        raise RecognizerError("classes must be sorted and unique")
    if not classes:
        raise RecognizerError("model must contain at least one class")
    d = int(model["dimension"])
    _adapter_matrix(model["adapter"])
    if np.asarray(model["adapter"]["matrix"]).shape != (d, d):
        raise RecognizerError("adapter dimension disagrees with model dimension")
    _adapter.scalar_in_range(model["shrinkage"], "tau", _TAU_MIN, _TAU_MAX)
    for c in classes:
        if not isinstance(c, str) or not c:
            raise RecognizerError("class names must be non-empty strings")
        mu = np.asarray(model["means"][c], _F64)
        cov = np.asarray(model["covariances"][c], _F64)
        cs = np.asarray(model["calibration_scores"][c], _F64)
        if mu.shape != (d,) or cov.shape != (d, d):
            raise RecognizerError(f"class {c!r} has wrong mean/covariance shape")
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov)) and np.all(np.isfinite(cs))):
            raise RecognizerError(f"class {c!r} has non-finite statistics")
        if int(model["calibration_counts"][c]) != cs.shape[0] or cs.shape[0] < 1:
            raise RecognizerError(f"class {c!r} calibration count mismatch")
    for key in ("means", "covariances", "calibration_scores", "calibration_counts"):
        if set(model[key]) != set(classes):
            raise RecognizerError(f"'{key}' keys must match classes")
    # Version seal: recomputed fingerprint must match the stored one.
    if _adapter.canonical_digest(_content_payload(model)) != model["fingerprint"]:
        raise RecognizerError("model fingerprint mismatch (corrupted/inconsistent)")


def verify_model(model: Mapping, support=None, calibration=None, *,
                 atol: float = 2e-8, rtol: float = 2e-8) -> dict:
    """Check that W, class statistics and calibration scores are one version.

    - Recomputes the seal over the stored content.
    - Confirms the stored adapter content matches the adapter fingerprint.
    - If ``support``/``calibration`` are supplied, re-derives means,
      covariances and calibration scores from *the model's own W* and compares
      them to the stored values (catches a new W paired with stale statistics,
      or stale statistics kept after re-adaptation).
    """
    validate_model(model)
    report: dict[str, Any] = {
        "seal_ok": _adapter.canonical_digest(_content_payload(model)) == model["fingerprint"],
        "adapter_fingerprint_ok": _adapter.fingerprint(model["adapter"])
        == model["adapter_fingerprint"],
    }
    if support is None and calibration is None:
        report["is_consistent"] = bool(report["seal_ok"] and
                                       report["adapter_fingerprint_ok"])
        return report
    if support is None or calibration is None:
        raise RecognizerError(
            "verify_model needs both support and calibration, or neither")

    d = int(model["dimension"])
    sup = _labeled_rows(support, "support", d)
    cal = _labeled_rows(calibration, "calibration", d)
    classes = _check_same_classes(sup, cal)
    report["classes_match"] = classes == list(model["classes"])
    if not report["classes_match"]:
        report["stale_classes"] = sorted(set(classes) ^ set(model["classes"]))
        return report

    tau = float(model["shrinkage"])
    max_err = 0.0
    stats_ok = True
    for c in classes:
        mean, cov = _class_stats(_adapter.transform(sup[c], model["adapter"]), tau)
        stored_mean = np.asarray(model["means"][c], _F64)
        stored_cov = np.asarray(model["covariances"][c], _F64)
        stored_cs = np.asarray(model["calibration_scores"][c], _F64)
        cs = _scores(_adapter.transform(cal[c], model["adapter"]), mean, cov)
        for a, b in ((mean, stored_mean), (cov, stored_cov), (cs, stored_cs)):
            err = float(np.max(np.abs(a - b))) if a.size else 0.0
            max_err = max(max_err, err)
            stats_ok = stats_ok and np.allclose(a, b, atol=atol, rtol=rtol)
        if int(model["calibration_counts"][c]) != cs.shape[0]:
            stats_ok = False
    report["max_abs_error"] = max_err
    report["statistics_match_current_data"] = bool(stats_ok)
    report["is_consistent"] = bool(report["seal_ok"] and
                                   report["adapter_fingerprint_ok"] and stats_ok)
    return report
