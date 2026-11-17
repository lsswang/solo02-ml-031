"""Verification suite for the constrained adapter and abstaining recognizer.

Run with:  python tests.py

Every check uses atol = rtol = 2e-8 for numeric agreement and exact integer
arithmetic for the p-value counts.  The last section writes a structured
``results.json`` summary.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

import adapter
import recognizer
import workflow

ATOL = RTOL = 2e-8
HERE = Path(__file__).resolve().parent
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail=None):
    RESULTS.append({"test": name, "pass": bool(ok),
                    "detail": _jsonable(detail)})
    if not ok:
        raise AssertionError(f"FAILED: {name} -> {detail}")
    print(f"PASS  {name}")


def close(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float),
                       atol=ATOL, rtol=RTOL)


def _jsonable(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def manual_adapter(W, ridge=0.5, rank=0):
    """A self-contained legal adapter model, independent of adapter.fit."""
    W = np.asarray(W, float)
    d = W.shape[0]
    return {"matrix": W.tolist(), "objective": 0.0, "rank": int(rank),
            "constraint_residual": 0.0, "ridge": float(ridge), "dim": int(d),
            "schema": "manual"}


def identity_adapter(d, ridge=0.5):
    """W = I via the empty-X path (legal: no supervised rows)."""
    empty = np.zeros((0, d))
    return adapter.fit(empty, empty, empty, ridge)


# ---------------------------------------------------------------------------
# Independent constrained least squares: W = I + N C, N = null(A), ridge in C.
# ---------------------------------------------------------------------------
def reference_constrained_ls(X, T, A, lam):
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    A = np.asarray(A, float)
    d = X.shape[1]
    N = null_space(A) if A.shape[0] else np.eye(d)
    if N.shape[1] == 0:
        return np.eye(d)
    U = X @ N
    E = X - T
    C = -np.linalg.solve(U.T @ U + lam * np.eye(N.shape[1]), U.T @ E)
    return np.eye(d) + N @ C


# ===========================================================================
# 1. Adapter: hand-computed optimum, feasibility, KKT
# ===========================================================================
def test_adapter_hand_example():
    X = [[0, 1], [0, 2], [1, 1]]
    T = [[1, 0.4], [2, 0.8], [2, 0.4]]
    A = [[1, 0], [2, 0], [0, 0]]
    lam = 0.5
    m = adapter.fit(X, T, A, lam)
    W = np.asarray(m["matrix"])
    p, q = 12 / 13, 29 / 65
    expected = np.array([[1.0, 0.0], [p, q]])
    check("adapter.hand.pq", close(W, expected), {"W": W, "expected": expected})
    # Hand objective: 6(p-1)^2 + 6(q-0.4)^2 + 0.5 p^2 + 0.5 (q-1)^2
    J = (6 * (p - 1) ** 2 + 6 * (q - 0.4) ** 2 + 0.5 * p ** 2
         + 0.5 * (q - 1) ** 2)
    check("adapter.hand.objective", abs(m["objective"] - J) <= ATOL,
          {"reported": m["objective"], "hand": J})
    check("adapter.hand.objective_recompute",
          abs(adapter.objective(X, T, W, lam) - m["objective"]) <= ATOL)
    check("adapter.hand.rank", m["rank"] == 1, m["rank"])
    check("adapter.hand.residual",
          m["constraint_residual"] <= ATOL, m["constraint_residual"])
    # Anchor rows (effective row space e1) preserved.
    Aw = np.asarray(A) @ W
    check("adapter.hand.AW=A", close(Aw, np.asarray(A)), Aw.tolist())
    v = adapter.verify(m, X, T, A)
    check("adapter.hand.stationary", v["is_stationary"]
          and v["within_tolerance"], v)


def test_adapter_random_vs_reference():
    rng = np.random.default_rng(7)
    worst = 0.0
    for trial in range(25):
        d = int(rng.integers(1, 9))
        n = int(rng.integers(0, 9))
        X = rng.uniform(-10, 10, size=(n, d))
        T = rng.uniform(-10, 10, size=(n, d))
        k = int(rng.integers(0, 6))
        A = rng.uniform(-10, 10, size=(k, d))
        # Inject correlated + zero rows sometimes.
        if k >= 2 and trial % 3 == 0:
            A[-1] = 2.0 * A[0]
        if k >= 1 and trial % 4 == 0:
            A[0] = 0.0
        lam = float(rng.uniform(1e-3, 2.0))
        m = adapter.fit(X, T, A, lam)
        ref = reference_constrained_ls(X, T, A, lam)
        err = float(np.max(np.abs(np.asarray(m["matrix"]) - ref)))
        worst = max(worst, err)
        v = adapter.verify(m, X, T, A)
        if not (close(m["matrix"], ref) and v["is_stationary"]
                and v["within_tolerance"] and m["constraint_residual"] <= ATOL):
            check(f"adapter.random[{trial}]", False,
                  {"err": err, "verify": v})
    check("adapter.random_vs_reference", worst <= ATOL, {"worst_err": worst})


def test_adapter_kkt_system_and_perturbation():
    """Independent KKT linear solve + feasible-perturbation optimality.

    The KKT system for  min J(W) s.t. AW=A  is
        H W + A^T Lambda = G,   A W = A,   H=X^T X+lambda I, G=X^T T+lambda I.
    Solving the full (W, Lambda) system is independent of the null-space
    elimination used by ``fit``.  We additionally confirm no feasible descent
    direction exists by probing random D with A D = 0.
    """
    rng = np.random.default_rng(11)
    d, n = 3, 6
    X = rng.uniform(-3, 3, size=(n, d))
    T = rng.uniform(-3, 3, size=(n, d))
    A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # full row rank
    lam = 0.7
    m = adapter.fit(X, T, A, lam)
    W = np.asarray(m["matrix"])

    k = A.shape[0]
    H = X.T @ X + lam * np.eye(d)
    G = X.T @ T + lam * np.eye(d)
    # Solve column-by-column. Each column w_j of W obeys
    #   H w_j + A^T lambda_j = g_j,   A w_j = a_j   (a_j = A[:, j]),
    # because A W = A decomposes per column. Build/solve the (d+k) KKT system.
    K = np.block([[H, A.T], [A, np.zeros((k, k))]])
    W_kkt = np.zeros((d, d))
    feasible_kkt = True
    for j in range(d):
        sol = np.linalg.solve(K, np.concatenate([G[:, j], A[:, j]]))
        W_kkt[:, j] = sol[:d]
        feasible_kkt &= np.allclose(A @ W_kkt[:, j], A[:, j], atol=1e-10, rtol=0)
    check("adapter.kkt_system_matches", close(W, W_kkt) and feasible_kkt,
          {"ours": W.tolist(), "kkt": W_kkt.tolist()})

    # Feasible perturbations: A D = 0, so A(W+tD)=A. No direction may
    # strictly reduce J; the directional derivative must vanish.
    N = null_space(A)
    base = adapter.objective(X, T, W, lam)
    monotone = True
    for _ in range(200):
        coef = rng.uniform(-1, 1, size=N.shape[1])
        D = np.outer(N @ coef, rng.uniform(-1, 1, size=d))
        A_D = np.max(np.abs(A @ D))
        if A_D > 1e-10:
            monotone = False
            break
        for t in (1e-6, -1e-6):
            if adapter.objective(X, T, W + t * D, lam) < base - 1e-12:
                monotone = False
    check("adapter.no_feasible_descent", monotone, {"base": base})


def test_adapter_constraint_cases():
    rng = np.random.default_rng(3)
    d = 3
    X = rng.uniform(-1, 1, size=(5, d))
    T = rng.uniform(-1, 1, size=(5, d))
    lam = 0.4

    # (a) empty protection (no anchor rows) == ordinary ridge.
    m_none = adapter.fit(X, T, [], lam)
    W0 = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ T + lam * np.eye(d))
    check("adapter.anchors.empty_is_ridge", close(m_none["matrix"], W0),
          {"rank": m_none["rank"]})
    check("adapter.anchors.empty_rank", m_none["rank"] == 0)

    # (b) all-zero anchors -> rank 0, same as ordinary ridge.
    m_zero = adapter.fit(X, T, np.zeros((4, d)), lam)
    check("adapter.anchors.zero_rank", m_zero["rank"] == 0)
    check("adapter.anchors.zero_is_ridge", close(m_zero["matrix"], W0))

    # (c) correlated / duplicate anchors reduce to the same row space.
    A_dup = np.array([[1.0, 0.0, 0.5], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])
    A_basis = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])
    m_dup = adapter.fit(X, T, A_dup, lam)
    m_basis = adapter.fit(X, T, A_basis, lam)
    check("adapter.anchors.correlated_rank", m_dup["rank"] == 2,
          m_dup["rank"])
    check("adapter.anchors.correlated_same_W",
          close(m_dup["matrix"], m_basis["matrix"]),
          {"dup": m_dup["matrix"], "basis": m_basis["matrix"]})
    check("adapter.anchors.correlated_feasible",
          np.allclose(A_dup @ np.asarray(m_dup["matrix"]), A_dup,
                      atol=ATOL, rtol=0))

    # (d) full protection: A = I forces W = I.
    m_full = adapter.fit(X, T, np.eye(d), lam)
    check("adapter.anchors.full_rank", m_full["rank"] == d)
    check("adapter.anchors.full_W_is_I", close(m_full["matrix"], np.eye(d)))

    # (e) empty X is legal -> W = I (W0 = I feasible), objective 0.
    #     X and T are paired, so both are empty; d comes from the anchors.
    m_empty = adapter.fit([], [], A_basis, lam)
    check("adapter.emptyX_W_is_I", close(m_empty["matrix"], np.eye(d)),
          m_empty["matrix"])
    check("adapter.emptyX_objective_zero", abs(m_empty["objective"]) <= ATOL,
          m_empty["objective"])
    ref = reference_constrained_ls(np.zeros((0, d)), np.zeros((0, d)),
                                   A_basis, lam)
    check("adapter.emptyX_matches_reference", close(m_empty["matrix"], ref))
    check("adapter.emptyX_rank", m_empty["rank"] == 2, m_empty["rank"])


def test_adapter_not_posthoc_overwrite():
    """Direct constrained solve differs from 'fit unconstrained then patch'."""
    rng = np.random.default_rng(31)
    d, n = 3, 7
    X = rng.uniform(-2, 2, size=(n, d))
    T = rng.uniform(-2, 2, size=(n, d))
    A = np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]])
    lam = 0.3
    m = adapter.fit(X, T, A, lam)
    W = np.asarray(m["matrix"])

    H = X.T @ X + lam * np.eye(d)
    W0 = np.linalg.solve(H, X.T @ T + lam * np.eye(d))
    constr = adapter.ConstraintBasis.from_anchors(A)
    Q = constr.Q
    # Naive post-hoc fix: keep the unconstrained free part, reset the
    # protected component so that Q W = Q.  Feasible but not optimal.
    W_naive = W0 + Q.T @ (Q - Q @ W0)
    check("adapter.naive_feasible",
          np.allclose(A @ W_naive, A, atol=1e-9, rtol=0))
    check("adapter.not_naive_overwrite", not close(W, W_naive),
          {"direct": W.tolist(), "naive": W_naive.tolist()})
    j_direct = adapter.objective(X, T, W, lam)
    j_naive = adapter.objective(X, T, W_naive, lam)
    check("adapter.direct_beats_naive", j_direct < j_naive - 1e-9,
          {"direct": j_direct, "naive": j_naive})


def test_adapter_transform_uses_own_W():
    d = 3
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(6, d))
    T = rng.uniform(-1, 1, size=(6, d))
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    m = adapter.fit(X, T, A, 0.6)
    W = np.asarray(m["matrix"])
    Z = rng.uniform(-2, 2, size=(4, d))
    out = adapter.transform(Z, m)
    check("adapter.transform.uses_W", np.array_equal(out, Z @ W),
          {"max_abs": float(np.max(np.abs(out - Z @ W)))})
    # A different matrix must give a different result (not snapped to W).
    m2 = manual_adapter(W + np.eye(d) * 0.1)
    out2 = adapter.transform(Z, m2)
    check("adapter.transform.not_hardcoded",
          not np.allclose(out, out2), {"delta": float(np.max(np.abs(out - out2)))})
    check("adapter.float64", np.asarray(m["matrix"]).dtype == np.float64)


def test_adapter_rejections():
    X = [[0.0, 1.0]]
    T = [[1.0, 0.0]]
    cases = [
        ("dim_mismatch", lambda: adapter.fit(X, [[1.0, 0.0, 0.0]], [], 0.5)),
        ("row_mismatch", lambda: adapter.fit(X, [[1.0, 0.0], [0.0, 1.0]], [], 0.5)),
        ("nonfinite", lambda: adapter.fit([[np.nan, 1.0]], T, [], 0.5)),
        ("lambda_zero", lambda: adapter.fit(X, T, [], 0.0)),
        ("lambda_high", lambda: adapter.fit(X, T, [], 3.0)),
        ("anchor_dim", lambda: adapter.fit(X, T, [[1.0, 0.0, 0.0]], 0.5)),
        ("ridge_nan", lambda: adapter.fit(X, T, [], np.nan)),
    ]
    rejected = {}
    for name, fn in cases:
        try:
            fn()
            rejected[name] = False
        except (adapter.AdapterError, ValueError):
            rejected[name] = True
    check("adapter.rejections", all(rejected.values()), rejected)


def test_adapter_persistence():
    rng = np.random.default_rng(9)
    d = 2
    X = rng.uniform(-1, 1, size=(4, d))
    T = rng.uniform(-1, 1, size=(4, d))
    A = np.array([[1.0, 0.0]])
    m = adapter.fit(X, T, A, 0.5)
    with tempfile.TemporaryDirectory() as td:
        p = adapter.save(m, Path(td) / "adapter.json")
        m2 = adapter.load(p)
    check("adapter.persist.matrix_equal",
          np.array_equal(np.asarray(m["matrix"]), np.asarray(m2["matrix"])))
    Z = rng.uniform(-1, 1, size=(3, d))
    check("adapter.persist.transform_equal",
          np.array_equal(adapter.transform(Z, m), adapter.transform(Z, m2)))
    check("adapter.persist.fingerprint",
          adapter.fingerprint(m) == adapter.fingerprint(m2))


# ===========================================================================
# 2. Recognizer statistics and scores
# ===========================================================================
def test_recognizer_population_covariance():
    d = 2
    a = identity_adapter(d, ridge=0.5)
    tau = 0.3
    support = {"x": [[0.0, 0.0], [2.0, 0.0]]}
    cal = {"x": [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]}
    m = recognizer.build(support, cal, a, tau)
    mu = np.array([1.0, 0.0])
    cov = np.array([[1.3, 0.0], [0.0, 0.3]])  # n denominator + tau I
    check("recognizer.mean", close(m["means"]["x"], mu), m["means"]["x"])
    check("recognizer.population_cov_plus_tau",
          close(m["covariances"]["x"], cov), m["covariances"]["x"])
    # Score via solve matches explicit inverse.
    z = np.array([0.0, 0.0])
    s = float(recognizer._scores(z, mu, cov))
    ref = float((z - mu) @ np.linalg.inv(cov) @ (z - mu))
    check("recognizer.score_vs_inverse", abs(s - ref) <= ATOL,
          {"solve": s, "inverse": ref})


def test_recognizer_pvalue_counts_and_ties():
    d = 2
    a = identity_adapter(d)
    # Two identical calibration vectors -> exact float tie included by '>='.
    support = {"a": [[0.0, 0.0], [1.0, 1.0]]}
    cal = {"a": [[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]}
    m = recognizer.build(support, cal, a, 0.2)
    cs = np.asarray(m["calibration_scores"]["a"])
    check("recognizer.tie.cal_scores_bitwise_equal",
          cs[0] == cs[1], cs.tolist())

    # Query duplicated in the input: same batching path, exact same score.
    preds = recognizer.predict([[0.0, 0.0], [0.0, 0.0]], m, alpha=0.4)
    q0, q1 = preds[0]["scores"]["a"], preds[1]["scores"]["a"]
    check("recognizer.tie.duplicate_queries_bitwise_equal", q0 == q1,
          {"q0": q0, "q1": q1})
    q = q0
    expected_count = int(np.count_nonzero(cs >= q))  # exact comparison
    check("recognizer.tie.count_is_exact_ge",
          preds[0]["exceed_counts"]["a"] == expected_count,
          {"count": preds[0]["exceed_counts"]["a"], "expected": expected_count})
    check("recognizer.tie.count_includes_equals",
          preds[0]["exceed_counts"]["a"] >= 2,  # both zero vectors tie
          preds[0]["exceed_counts"]["a"])
    check("recognizer.pvalue_formula",
          abs(preds[0]["pvalues"]["a"] - (1 + expected_count) / (len(cs) + 1)) <= 0,
          {"p": preds[0]["pvalues"]["a"], "denom": preds[0]["denominators"]["a"]})
    check("recognizer.denominator", preds[0]["denominators"]["a"] == len(cs) + 1)

    # Exact ties at the top: a duplicated *query* both score identically and
    # both see the same strict count (no approximate tie band).
    cal_tie = {"a": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]}
    m_tie = recognizer.build({"a": [[0.0, 0.0]]}, cal_tie, a, 0.2)
    p = recognizer.predict([[1.0, 1.0]], m_tie, 0.4)[0]
    check("recognizer.tie.all_calibration_counted",
          p["exceed_counts"]["a"] == 3 and p["pvalues"]["a"] == 1.0,
          {"count": p["exceed_counts"]["a"], "p": p["pvalues"]["a"]})


def test_recognizer_counts_recomputed_inverses():
    """Independently recompute every count/p-value from stored stats."""
    rng = np.random.default_rng(21)
    d = 3
    a = identity_adapter(d)
    support = {"a": rng.uniform(-2, 2, size=(4, d)),
               "b": rng.uniform(2, 4, size=(5, d))}
    cal = {"a": rng.uniform(-2, 2, size=(6, d)),
           "b": rng.uniform(2, 4, size=(7, d))}
    m = recognizer.build(support, cal, a, 0.15)
    queries = rng.uniform(-2, 4, size=(8, d))
    preds = recognizer.predict(queries, m, alpha=0.4)
    ok = True
    for z, pr in zip(queries, preds):
        for c in m["classes"]:
            mu = np.asarray(m["means"][c])
            cov = np.asarray(m["covariances"][c])
            s = float((z - mu) @ np.linalg.inv(cov) @ (z - mu))
            counts = int(np.count_nonzero(
                np.asarray(m["calibration_scores"][c]) >= s))
            p = (1 + counts) / (m["calibration_counts"][c] + 1)
            ok &= abs(s - pr["scores"][c]) <= ATOL
            ok &= counts == pr["exceed_counts"][c]
            ok &= abs(p - pr["pvalues"][c]) <= ATOL
            ok &= pr["denominators"][c] == m["calibration_counts"][c] + 1
    check("recognizer.counts.recomputed", ok)


def test_recognizer_decisions_and_alpha():
    d = 2
    a = identity_adapter(d)
    # Identical classes -> symmetric scores; query equals a calibration point.
    support = {"a": [[0.0, 0.0], [1.0, 0.0]],
               "b": [[0.0, 0.0], [1.0, 0.0]]}
    cal = {"a": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
           "b": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}
    m = recognizer.build(support, cal, a, 0.2)

    multi = recognizer.predict([[0.0, 0.0]], m, alpha=0.4)[0]
    check("recognizer.multiple_classes.ambiguous",
          multi["decision"] == "ambiguous" and multi["classes"] == ["a", "b"],
          multi)

    # Far-away query: p_min = 1/(n+1) = 0.25; alpha 0.4 -> unknown.
    unk = recognizer.predict([[100.0, 100.0]], m, alpha=0.4)[0]
    check("recognizer.unknown_when_p_below_alpha",
          unk["decision"] == "unknown" and unk["classes"] == [], unk)

    # alpha = 0 -> every class has p > 0 -> all candidates kept.
    lo = recognizer.predict([[100.0, 100.0]], m, alpha=0.0)[0]
    check("recognizer.alpha_zero_accepts_all",
          lo["decision"] == "ambiguous" and lo["classes"] == ["a", "b"], lo)

    # alpha = 1 -> p <= 1 never strictly greater -> always unknown.
    hi = recognizer.predict([[0.0, 0.0]], m, alpha=1.0)[0]
    check("recognizer.alpha_one_always_unknown",
          hi["decision"] == "unknown" and hi["classes"] == [], hi)

    # Strict '>' boundary: pick a query whose p is strictly between 0 and 1,
    # then alpha exactly at p must reject while alpha just below accepts.
    qmid = [1.5, 0.0]
    probe = recognizer.predict([qmid], m, alpha=0.0)[0]
    p_mid = probe["pvalues"]["a"]
    check("recognizer.alpha_mid_p_between", 0.0 < p_mid < 1.0, p_mid)
    at = recognizer.predict([qmid], m, alpha=p_mid)[0]
    check("recognizer.alpha_strict_boundary_excludes",
          "a" not in at["classes"], {"p": p_mid, "at": at})
    below = recognizer.predict([qmid], m, alpha=p_mid - 1e-6)[0]
    check("recognizer.alpha_just_below_accepts",
          "a" in below["classes"], {"p": p_mid, "below": below})

    # Unique: only class a calibrates this point, b's distribution is far.
    support2 = {"a": [[0.0, 0.0], [1.0, 0.0]],
                "b": [[50.0, 50.0], [51.0, 50.0]]}
    cal2 = {"a": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            "b": [[50.0, 50.0], [51.0, 50.0], [52.0, 50.0]]}
    m2 = recognizer.build(support2, cal2, a, 0.2)
    uni = recognizer.predict([[0.0, 0.0]], m2, alpha=0.4)[0]
    check("recognizer.unique_single_class",
          uni["decision"] == "unique" and uni["classes"] == ["a"], uni)


def test_recognizer_intra_class_scales():
    d = 2
    a = identity_adapter(d)
    # Same centre, different spreads. Same displacement -> tight class has a
    # larger Mahalanobis score; a global radius could not express this.
    support = {"tight": [[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0], [0.0, 0.2]],
               "wide": [[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]]}
    cal = {"tight": [[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]],
           "wide": [[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]}
    m = recognizer.build(support, cal, a, 0.05)
    z = np.array([[0.5, 0.0]])
    s = recognizer.class_scores(z, m)
    check("recognizer.intra_class_scale.tight_larger",
          s["tight"][0] > s["wide"][0],
          {"tight": float(s["tight"][0]), "wide": float(s["wide"][0])})
    # Covariance traces reflect the different scales, tau is identical.
    tr = {c: float(np.trace(np.asarray(m["covariances"][c]))) for c in m["classes"]}
    check("recognizer.intra_class_scale.cov_traces",
          tr["tight"] < tr["wide"], tr)


def test_recognizer_class_change():
    d = 2
    a = identity_adapter(d)
    sup1 = {"a": [[0.0, 0.0], [0.5, 0.0]],
            "b": [[5.0, 0.0], [5.5, 0.0]]}
    cal1 = {"a": [[0.0, 0.0], [0.5, 0.0]],
            "b": [[5.0, 0.0], [5.5, 0.0]]}
    m1 = recognizer.build(sup1, cal1, a, 0.2)
    check("recognizer.class_change.before_has_b", "b" in m1["classes"])
    # Remove b, add c.
    sup2 = {"a": sup1["a"], "c": [[-5.0, 0.0], [-5.5, 0.0]]}
    cal2 = {"a": cal1["a"], "c": [[-5.0, 0.0], [-5.5, 0.0]]}
    m2 = recognizer.build(sup2, cal2, a, 0.2)
    check("recognizer.class_change.removed_absent",
          "b" not in m2["classes"] and m2["classes"] == ["a", "c"],
          m2["classes"])
    check("recognizer.class_change.new_present",
          set(m2["means"]) == {"a", "c"} and
          set(m2["covariances"]) == {"a", "c"} and
          set(m2["calibration_scores"]) == {"a", "c"})
    pr = recognizer.predict([[0.0, 0.0]], m2, alpha=0.4)[0]
    check("recognizer.class_change.predict_uses_new_set",
          set(pr["scores"]) == {"a", "c"} and "b" not in pr["scores"], pr)
    # No stale class survives anywhere in the sealed model.
    check("recognizer.class_change.no_stale_b",
          all("b" not in m2[k] for k in
              ("means", "covariances", "calibration_scores",
               "calibration_counts")))


# ===========================================================================
# 3. Re-adaptation semantics, stale-statistics detection, persistence
# ===========================================================================
def _two_adapter_fixture():
    X = [[0, 1], [0, 2], [1, 1]]
    T = [[1, 0.4], [2, 0.8], [2, 0.4]]
    anchors = [[1, 0]]
    support = {"amber": [[0, 0], [0.3, 0.1], [0, 0.2]],
               "blue": [[2, 0], [2.2, 0.6], [1.8, -0.6]]}
    cal = {"amber": [[0.1, 0.1], [-0.2, 0.25], [0, -0.35]],
           "blue": [[2.1, 0.3], [1.7, 0.5], [2.3, -0.4]]}
    return X, T, anchors, support, cal


def test_readaptation_changes_same_query():
    X, T, anchors, support, cal = _two_adapter_fixture()
    a1 = adapter.fit(X, T, anchors, 0.5)
    # Legal update: different supervised targets, same protection.
    T2 = [[0.2, 0.9], [0.4, 1.8], [1.5, 0.9]]
    a2 = adapter.fit(X, T2, anchors, 0.5)
    check("readapt.W_differs",
          not close(a1["matrix"], a2["matrix"]),
          {"W1": a1["matrix"], "W2": a2["matrix"]})

    r1 = recognizer.build(support, cal, a1, 0.2)
    r2 = recognizer.build(support, cal, a2, 0.2)
    q = [[0.05, 0.05]]
    s1 = recognizer.class_scores(q, r1)
    s2 = recognizer.class_scores(q, r2)
    changed = any(abs(s1[c][0] - s2[c][0]) > 1e-6 for c in r1["classes"])
    check("readapt.same_query_scores_change", changed,
          {"s1": {c: float(s1[c][0]) for c in s1},
           "s2": {c: float(s2[c][0]) for c in s2}})
    # Both query transform AND class statistics moved to the new version.
    mu2 = np.asarray(support["amber"]) @ np.asarray(a2["matrix"])
    check("readapt.query_uses_new_W",
          close(adapter.transform(q, a2), np.asarray(q) @ np.asarray(a2["matrix"])))
    check("readapt.stats_use_new_W",
          close(r2["means"]["amber"], mu2.mean(axis=0)),
          {"stored": r2["means"]["amber"], "new_W_mean": mu2.mean(axis=0).tolist()})
    check("readapt.verify_consistent",
          recognizer.verify_model(r2, support, cal)["is_consistent"])


def test_stale_detection():
    X, T, anchors, support, cal = _two_adapter_fixture()
    a1 = adapter.fit(X, T, anchors, 0.5)
    a2 = adapter.fit(X, [[0.2, 0.9], [0.4, 1.8], [1.5, 0.9]], anchors, 0.5)
    r1 = recognizer.build(support, cal, a1, 0.2)
    r2 = recognizer.build(support, cal, a2, 0.2)

    # (i) Swap in a new W but keep old statistics: fingerprint catches it.
    corrupt = json.loads(json.dumps(r1))
    corrupt["adapter"] = json.loads(json.dumps(a2))
    caught = False
    try:
        recognizer.predict([[0.1, 0.1]], corrupt, 0.4)
    except recognizer.RecognizerError:
        caught = True
    check("stale.raw_swap_rejected", caught)

    # (ii) Even if the seal is recomputed, data-based re-derivation detects
    #      statistics computed under the old W (frozen class scale).
    resealed = json.loads(json.dumps(corrupt))
    recognizer._seal(resealed)
    report = recognizer.verify_model(resealed, support, cal)
    check("stale.resealed_detected", report["is_consistent"] is False
          and report.get("statistics_match_current_data") is False, report)

    # (iii) Only the query transform updated (new W) while old stats frozen
    #       is the same inconsistency; show the numeric consequence too.
    ok_report = recognizer.verify_model(r1, support, cal)
    check("stale.correct_model_verified", ok_report["is_consistent"] is True)

    # (iv) Demonstrate the actual behavior difference produced by the
    #      "new W + frozen old scale" bug versus proper rebuilding, so the
    #      frozen variant is observably wrong on the same query.
    frozen = json.loads(json.dumps(r1))
    frozen["adapter"] = json.loads(json.dumps(a2))          # swap in new W
    frozen["adapter_fingerprint"] = adapter.fingerprint(a2)
    recognizer._seal(frozen)                                # hide the mismatch
    q = [[0.05, 0.05]]
    p_frozen = recognizer.predict(q, frozen, 0.4)[0]
    p_fixed = recognizer.predict(q, r2, 0.4)[0]
    differs = (p_frozen["scores"] != p_fixed["scores"] or
               p_frozen["pvalues"] != p_fixed["pvalues"])
    check("stale.frozen_scores_differ_from_rebuilt", differs,
          {"frozen": p_frozen["scores"], "fixed": p_fixed["scores"]})
    check("stale.frozen_detected_by_data_verify",
          recognizer.verify_model(frozen, support, cal)["is_consistent"] is False)


def test_persistence_and_entry_points():
    data = json.loads((HERE / "fixtures" / "concepts.json").read_text())
    out = workflow.run(data)

    # Both entry points: workflow.run vs manual fit/build/predict.
    a = adapter.fit(data["fit_features"], data["fit_targets"],
                    data["anchors"], data["ridge"])
    r = recognizer.build(data["support"], data["calibration"], a,
                         data["shrinkage"])
    manual = recognizer.predict(data["queries"], r, data["alpha"])
    check("entrypoints.workflow_matches_manual",
          manual == out["predictions"],
          {"manual0": manual[0], "workflow0": out["predictions"][0]})

    # Save/load independently, no training data needed to predict.
    with tempfile.TemporaryDirectory() as td:
        ap = adapter.save(a, Path(td) / "a.json")
        rp = recognizer.save(r, Path(td) / "r.json")
        a_loaded = adapter.load(ap)
        r_loaded = recognizer.load(rp)
        pred_loaded = recognizer.predict(data["queries"], r_loaded,
                                         data["alpha"])
    check("persistence.loaded_predictions_equal", pred_loaded == manual)
    check("persistence.loaded_adapter_transform_equal",
          np.array_equal(adapter.transform(data["queries"], a),
                         adapter.transform(data["queries"], a_loaded)))
    # Verification without training data still checks the internal seal.
    check("persistence.seal_ok_without_data",
          recognizer.verify_model(r_loaded)["seal_ok"])


def test_recognizer_rejections():
    d = 2
    a = identity_adapter(d)
    good_sup = {"a": [[0.0, 0.0]]}
    good_cal = {"a": [[0.0, 0.0]]}
    cases = [
        ("empty_class", lambda: recognizer.build({"a": []}, {"a": [[0, 0]]}, a, 0.2)),
        ("class_set_mismatch",
         lambda: recognizer.build({"a": [[0, 0]], "b": [[1, 1]]},
                                  {"a": [[0, 0]]}, a, 0.2)),
        ("tau_zero", lambda: recognizer.build(good_sup, good_cal, a, 0.0)),
        ("tau_high", lambda: recognizer.build(good_sup, good_cal, a, 3.0)),
        ("bad_class_name", lambda: recognizer.build({"": [[0, 0]]},
                                                    {"": [[0, 0]]}, a, 0.2)),
        ("nonfinite_support",
         lambda: recognizer.build({"a": [[np.inf, 0]]}, {"a": [[0, 0]]}, a, 0.2)),
        ("empty_mapping", lambda: recognizer.build({}, {}, a, 0.2)),
        ("dim_mismatch",
         lambda: recognizer.build({"a": [[0, 0, 0]]}, {"a": [[0, 0]]}, a, 0.2)),
    ]
    rejected = {}
    for name, fn in cases:
        try:
            fn()
            rejected[name] = False
        except (recognizer.RecognizerError, ValueError):
            rejected[name] = True
    check("recognizer.build_rejections", all(rejected.values()), rejected)

    r = recognizer.build(good_sup, good_cal, a, 0.2)
    for name, alpha in [("alpha_low", -0.1), ("alpha_high", 1.1),
                        ("alpha_nan", float("nan"))]:
        try:
            recognizer.predict([[0, 0]], r, alpha)
            rejected[name] = False
        except (recognizer.RecognizerError, ValueError):
            rejected[name] = True
    check("recognizer.alpha_rejections", all(rejected.values()), rejected)


def test_arbitrary_adapter_matrix_consumed():
    """build/predict consume any legal adapter model, not only fit output."""
    W = np.array([[1.0, 0.0], [0.5, 2.0]])  # legal-ish arbitrary matrix
    a = manual_adapter(W)
    support = {"a": [[0.0, 0.0], [0.2, 0.1]], "b": [[2.0, 0.0], [2.2, 0.2]]}
    cal = {"a": [[0.0, 0.0], [0.2, 0.1]], "b": [[2.0, 0.0], [2.2, 0.2]]}
    m = recognizer.build(support, cal, a, 0.2)
    mu_a = (np.asarray(support["a"]) @ W).mean(axis=0)
    check("arbitrary_adapter.stats_use_given_W",
          close(m["means"]["a"], mu_a), {"stored": m["means"]["a"]})
    pred = recognizer.predict([[0.0, 0.0]], m, 0.4)[0]
    check("arbitrary_adapter.predict_runs", pred["decision"] in
          ("unique", "ambiguous", "unknown"), pred)


# ===========================================================================
# Run all
# ===========================================================================
def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    passed = sum(r["pass"] for r in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    Path(HERE / "results.json").write_text(json.dumps(
        {"checks": RESULTS, "passed": passed, "total": len(RESULTS)},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
