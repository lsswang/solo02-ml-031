"""End-to-end workflow: constrained adaptation + abstaining recognition.

Two equivalent entry points:

* ``run(data_dict)`` / ``run(path)`` -- the whole pipeline (``main``).
* the public functions ``adapter.fit`` / ``adapter.transform`` and
  ``recognizer.build`` / ``recognizer.predict`` -- external callers can drive
  the same semantics step by step and may pass any legal adapter matrix and any
  current support/calibration.  Because ``recognizer.build`` consumes the
  adapter *model* (``W`` included) and stores all class statistics, both paths
  agree and a re-adaptation always rebuilds the class statistics under the new
  ``W``.
"""
import json
from pathlib import Path

import adapter
import recognizer

DEFAULT_ALPHA = 0.4
DEFAULT_TAU = 0.2
DEFAULT_RIDGE = 0.5


def run(data):
    """Run the full pipeline on an in-memory fixture and return structured data."""
    data = dict(data)
    lam = data.get("ridge", DEFAULT_RIDGE)
    tau = data.get("shrinkage", DEFAULT_TAU)
    alpha = data.get("alpha", DEFAULT_ALPHA)

    a = adapter.fit(data.get("fit_features", []),
                    data["fit_targets"],
                    data.get("anchors", []),
                    lam)
    r = recognizer.build(data["support"], data["calibration"], a, tau)
    predictions = recognizer.predict(data.get("queries", []), r, alpha)

    verification = {
        "adapter": adapter.verify(
            a, data.get("fit_features", []), data["fit_targets"],
            data.get("anchors", [])),
        "recognizer": recognizer.verify_model(
            r, data["support"], data["calibration"]),
    }
    return {
        "method": _method_notes(),
        "adapter": a,
        "recognizer": {
            "classes": r["classes"],
            "shrinkage": r["shrinkage"],
            "means": r["means"],
            "covariances": r["covariances"],
            "calibration_scores": r["calibration_scores"],
            "calibration_counts": r["calibration_counts"],
            "fingerprint": r["fingerprint"],
        },
        "predictions": predictions,
        "verification": verification,
    }


def _method_notes():
    return {
        "adapter": (
            "Constrained ridge: J(W)=||XW-T||_F^2+lambda||W-I||_F^2 s.t. AW=A. "
            "AW=A is reduced to QW=Q for an orthonormal basis Q of rowspace(A); "
            "W=Q^T Q+P^T V with P an orthonormal complement, V solved from the "
            "small SPD normal equation (PHP^T)V=PH(W0-Q^T Q), "
            "W0=(X^T X+lambda I)^-1(X^T T+lambda I). Direct constrained solve, "
            "no post-hoc row overwrite. Rank via relative-tolerance SVD; empty "
            "X -> W=I (W0=I is feasible)."),
        "recognizer": (
            "Per class: mu_c and population covariance "
            "Sigma_c=(sum (z-mu)(z-mu)^T)/n + tau I under the current W. "
            "s_c(z)=(z-mu_c)^T Sigma_c^-1 (z-mu_c) via solve. "
            "p_c=(1+#{calibration s_c>=query s_c})/(n_cal+1), ties counted; "
            "accept {c: p_c>alpha}; empty=unknown, one=unique, many=ambiguous "
            "keeping all candidates. No coverage guarantee for arbitrary "
            "unknown distributions."),
        "versioning": (
            "Model stores W, means, population covariances and calibration "
            "scores plus SHA-256 fingerprints; verify_model re-derives all "
            "statistics from the stored W and the supplied data to detect a "
            "new W with stale statistics."),
    }


def run_path(path):
    return run(json.loads(Path(path).read_text()))


if __name__ == "__main__":
    import sys

    print(json.dumps(run_path(sys.argv[1]), ensure_ascii=False, indent=2))
