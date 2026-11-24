"""Ridge as a Metal float32 Gram plus a host float64 Cholesky.

Not an estimator and not part of the gated suite. `Ridge` is on the roadmap
(`README.md`, "planned, 0.2.x"), and scikit-learn's own dense default for it is the same
normal equations this measures: `resolve_solver_for_numpy` returns ``"cholesky"`` for
dense input with ``positive=False``, and `_solve_cholesky` is

    A = X'X ; Xy = X'y ; A.flat[::d+1] += alpha ; solve(A, Xy, assume_a="pos")

So the question is narrow and answerable: does building ``A`` in float32 on Metal and
factoring in float64 on the host beat scikit-learn doing both on the host, and over what
shapes?

Why this is worth measuring separately from `LinearRegression`
--------------------------------------------------------------
`LinearRegression` shipped at 0.1.7 with no Metal path at all — commit 344bfdf: "Metal has
no float64 and the float32 variants were inaccurate or no faster". That verdict was
reached for *unregularised* least squares, which must reproduce `lstsq`'s minimum-norm
solution plus `rank_` and `singular_`, and therefore needs an SVD rather than a Gram.

Ridge is the easier problem for a reason that is not obvious: ``alpha > 0`` makes
``X'X + alpha I`` positive definite by construction, bounding its condition number by
``(sigma_max^2 + alpha) / alpha``. The rank-deficiency machinery that forces
`LinearRegression` onto an SVD does not apply, so a Gram route is contract-correct here
where it is not there.

This module is self-contained on purpose. A sibling VAR(p) prototype uses the same three
Gram helpers, but Ridge should not depend on it: Ridge has a scikit-learn counterpart and
a roadmap entry, while VAR has neither and is blocked on a scope decision. If both ship,
the helpers belong in one shared module; until then the duplication is the cheaper error.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

__all__ = [
    "OBJECTIVE_RTOL",
    "BACKENDS",
    "gram_host",
    "gram_metal",
    "solve_normal_equations",
    "ridge_objective",
    "fit_ridge",
    "make_problem",
    "main",
]

#: Objective-value tolerance from ``docs/fp32_policy.md`` section 3. The coefficient
#: tolerances there (``atol=2e-3, rtol=5e-3``) are the weak assertion: on an
#: ill-conditioned design the coefficients are not determined by the data at all.
OBJECTIVE_RTOL = 1e-5

#: docs/fp32_policy.md section 3, the weak assertion. Recorded and gated alongside the
#: objective because the objective is nearly blind to coefficient error on an
#: ill-conditioned design: measured obj_rel 2.70e-06 (pass) with coef_rel 1.01e-01.
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3

_FELL_BACK = False


def _note_fallback() -> None:
    global _FELL_BACK
    _FELL_BACK = True

#: ``sklearn_bare`` is the fair baseline: scikit-learn's own `_solve_cholesky` with no
#: estimator around it. ``sklearn`` keeps the full `Ridge.fit` so the overhead is visible
#: as its own arm rather than folded into the Metal side's favour.
BACKENDS = ("metal_gram", "host_gram", "sklearn_bare", "sklearn")


def gram_host(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(X'X, X'y)`` in host float64."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    return X64.T @ X64, X64.T @ y64


def gram_metal(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(X'X, X'y)`` built in float32 on Metal, returned as float64.

    Uses ``mlxlearn_device._binary32.matmul``, the dispatching wrapper the library itself
    uses: the binary32 kernel where float32 matmul would round to TF32, ``mx.matmul``
    otherwise.
    """
    import mlx.core as mx

    from mlxlearn_device._binary32 import matmul

    Xf = mx.array(np.ascontiguousarray(X, dtype=np.float32))
    yf = mx.array(np.ascontiguousarray(y, dtype=np.float32))
    # `matmul` is the dispatching wrapper: it uses the binary32 kernel only on GPU
    # generations whose float32 matmul would round to TF32, and plain `mx.matmul`
    # otherwise. Calling `float32_matmul` directly measures a path the shipped
    # library would not take on this machine, and the slower one.
    G = matmul(Xf, Xf, transpose_a=True)
    C = matmul(Xf, yf, transpose_a=True)
    mx.eval(G, C)
    return np.array(G, dtype=np.float64), np.array(C, dtype=np.float64)


def solve_normal_equations(G: np.ndarray, C: np.ndarray, alpha: float, *,
                           X: np.ndarray | None = None, y: np.ndarray | None = None,
                           on_fallback=None) -> np.ndarray:
    """``(G + alpha I)^-1 C`` by float64 Cholesky, falling back as scikit-learn does.

    Does not mutate ``G``.
    """
    import scipy.linalg as sla

    G = np.array(G, dtype=np.float64, copy=True)
    if alpha:
        G[np.diag_indices_from(G)] += alpha
    C = np.asarray(C, dtype=np.float64)
    try:
        return sla.solve(G, C, assume_a="pos")
    except np.linalg.LinAlgError:
        if on_fallback is not None:
            on_fallback()
        if X is not None and y is not None:
            # What scikit-learn actually does: solver="svd" -> _solve_svd(X, y, alpha),
            # an SVD of X forming s/(s^2+alpha), bounded by 1/(2*sqrt(alpha)) in every
            # direction (sklearn/linear_model/_ridge.py:757-761, :299-308). lstsq on the
            # already-regularised Gram instead inverts the spurious negative eigenvalue,
            # so it is unbounded: measured 4.96e-02 relative error against _solve_svd's
            # 9.28e-09 on a near-collinear design.
            from sklearn.linear_model._ridge import _solve_svd

            y2 = np.asarray(y, dtype=np.float64)
            y2 = y2.reshape(-1, 1) if y2.ndim == 1 else y2
            return _solve_svd(np.asarray(X, dtype=np.float64), y2,
                              np.full(y2.shape[1], float(alpha))).T
        # Without X there is nothing better available; this path is unbounded and is
        # only reachable from a direct call.
        return np.linalg.lstsq(G, C, rcond=None)[0]


def make_problem(n: int, d: int, targets: int, seed: int, *, cond: float = 1.0,
                 snr: float = 10.0):
    """A dense regression problem, optionally ill-conditioned.

    ``cond`` sets the design's singular values geometrically from 1 down to ``1/cond``;
    ``snr`` fixes the ratio of signal RMS to noise scale, so conditioning is the only
    thing a ``--cond`` sweep varies.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Every cond goes through the same rebuild, including cond == 1.0. Taking the branch
    # only when cond > 1 left that row at the raw Gaussian scale while every other row was
    # normalised to sigma_max = 1, so a sweep varied the design scale as well as the
    # conditioning. Noise is then set from a fixed signal-to-noise ratio rather than a
    # literal 0.1, because a fixed absolute noise against a shrinking X made every
    # ill-conditioned row noise-dominated -- at cond=1e8, 97% of the objective was noise
    # the coefficients cannot touch, which is what made the objective gate uninformative.
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    spectrum = np.geomspace(1.0, 1.0 / cond, num=min(n, d))
    X = (U * spectrum) @ Vt
    beta = rng.normal(size=(d, targets))
    signal = X @ beta
    scale = float(np.sqrt((signal * signal).mean())) / snr
    y = signal + rng.normal(scale=scale, size=(n, targets))
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def ridge_objective(X: np.ndarray, y: np.ndarray, coef: np.ndarray, alpha: float) -> float:
    """``||y - X b||_F^2 + alpha ||b||_F^2`` in float64 on the true data."""
    R = np.asarray(y, dtype=np.float64) - np.asarray(X, dtype=np.float64) @ coef
    return float((R * R).sum()) + alpha * float((coef * coef).sum())


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float, *, backend: str) -> np.ndarray:
    """Coefficients of shape ``(d, targets)``. No intercept; callers centre if they want one."""
    y = np.asarray(y, dtype=np.float64)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if backend == "metal_gram":
        return solve_normal_equations(*gram_metal(X, y), alpha, X=X, y=y,
                                      on_fallback=_note_fallback)
    if backend == "host_gram":
        return solve_normal_equations(*gram_host(X, y), alpha, X=X, y=y,
                                      on_fallback=_note_fallback)
    if backend == "sklearn":
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=alpha, solver="cholesky", fit_intercept=False).fit(X, y)
        return np.atleast_2d(est.coef_).T
    if backend == "sklearn_bare":
        # scikit-learn's linear algebra with none of its estimator around it.
        # `Ridge.fit` runs `validate_data`/`check_array` on X and y, then
        # `_preprocess_data(copy=self.copy_X)` runs a SECOND `check_array` with
        # `copy=True` -- a full float64 copy of X. Timing that against a bare
        # `gram_metal` + solve, which validates nothing, is not like for like, and the
        # gap is largest at small d where the algebra is cheapest.
        from sklearn.linear_model._ridge import _solve_cholesky

        return _solve_cholesky(np.asarray(X, dtype=np.float64), y,
                               np.full(y.shape[1], float(alpha))).T
    raise ValueError(f"unknown backend {backend!r}")


def _run(n: int, d: int, targets: int, alpha_scale: float, cond: float,
         seed: int, iterations: int, records: list[dict], *,
         baseline: str = "sklearn_bare", alpha_abs: float | None = None,
         snr: float = 10.0) -> None:
    from benchmarks._harness import compare

    X, y = make_problem(n, d, targets, seed, cond=cond, snr=snr)
    G, _ = gram_host(X, y)
    # alpha_abs decouples the penalty from the design scale. With the trace rule,
    # rescaling the spectrum to set `cond` also collapses trace(X'X), so alpha moves
    # with it -- measured 3.27e-02 -> 2.00e-08 across cond 1..1e12, a 1.6-million-fold
    # change. A sweep using it varies two things at once and is not a conditioning sweep.
    alpha = alpha_abs if alpha_abs is not None else alpha_scale * float(np.trace(G)) / d

    cmp = compare(
        lambda: fit_ridge(X, y, alpha, backend="metal_gram"),
        lambda: fit_ridge(X, y, alpha, backend=baseline),
        label=f"ridge/n{n}/d{d}/t{targets}/cond{cond:g}",
        shape=(n, d),
        iterations=iterations,
    )
    ref = fit_ridge(X, y, alpha, backend=baseline)
    got = fit_ridge(X, y, alpha, backend="metal_gram")
    j_ref, j_got = ridge_objective(X, y, ref, alpha), ridge_objective(X, y, got, alpha)
    obj_rel = abs(j_got - j_ref) / abs(j_ref)
    denom = max(float(np.abs(ref).max()), 1e-300)
    coef_rel = float(np.abs(got - ref).max()) / denom

    global _FELL_BACK
    coef_ok = bool(np.allclose(got, ref, atol=COEF_ATOL, rtol=COEF_RTOL))
    records.append({**cmp.as_dict(), "n": n, "d": d, "targets": targets, "cond": cond,
                    "alpha": alpha, "objective_rel": obj_rel, "coef_rel": coef_rel,
                    "coef_ok": coef_ok, "fell_back": _FELL_BACK, "snr": snr,
                    "backend": "metal_gram", "baseline": baseline})
    _FELL_BACK = False
    gate = "pass" if obj_rel < OBJECTIVE_RTOL else "FAIL"
    sig = "" if cmp.significant else "  (ns)"
    print(f"  n={n:<7} d={d:<5} t={targets:<3} cond={cond:<8g} "
          f"{cmp.mlxlearn.median_s * 1000:7.1f} vs {cmp.sklearn.median_s * 1000:7.1f} ms   "
          f"{cmp.speedup:5.2f}x med {cmp.speedup_best:5.2f}x best{sig}   "
          f"obj {obj_rel:8.2e} [{gate}]  coef {coef_rel:8.2e} "
          f"[{'pass' if coef_ok else 'FAIL'}]"
          + ("  FELL BACK" if _FELL_BACK else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--d", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--cond", type=float, nargs="+", default=[1.0])
    parser.add_argument("--alpha-scale", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--baseline", default="sklearn_bare", choices=BACKENDS)
    parser.add_argument("--snr", type=float, default=10.0,
                        help="signal RMS over noise scale; held fixed across a cond sweep")
    parser.add_argument("--alpha-abs", type=float, default=None,
                        help="absolute penalty; decouples alpha from the design scale")
    parser.add_argument("--strict", action="store_true",
                        help="refuse to run on warnings too, not just blocking failures")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if any(c < 1.0 for c in args.cond):
        # Previously a value below 1 was silently ignored, because make_problem only
        # rescaled when cond > 1, and the fabricated value was then printed and recorded
        # as though it had been applied.
        parser.error("--cond must be >= 1.0; a condition number below 1 is not meaningful")

    from benchmarks.ridge.preflight import preflight

    checks = preflight(strict=args.strict)
    print()
    print(f"Ridge: Metal float32 Gram + host float64 Cholesky vs "
          f"baseline={args.baseline}, "
          f"alpha = {args.alpha_abs if args.alpha_abs is not None else str(args.alpha_scale) + ' * trace/d'}")
    print(f"objective gate: rtol < {OBJECTIVE_RTOL:g}  (docs/fp32_policy.md s3)")
    records: list[dict] = []
    for cond in args.cond:
        for n in args.n:
            for d in args.d:
                if d >= n:
                    continue
                _run(n, d, args.targets, args.alpha_scale, cond,
                     args.seed, args.iterations, records,
                     baseline=args.baseline, alpha_abs=args.alpha_abs, snr=args.snr)

    if args.output:
        from benchmarks._harness import environment

        with open(args.output, "w") as handle:
            json.dump({
                "driver": "benchmarks/ridge/ridge_prototype.py",
                "environment": environment(),
                "parameters": vars(args),
                "objective_rtol": OBJECTIVE_RTOL,
                "preflight": [
                    {"name": c.name, "ok": c.ok, "value": c.value, "blocking": c.blocking}
                    for c in checks
                ],
                "gate_passed": all(
                    r["objective_rel"] < OBJECTIVE_RTOL and r["coef_ok"] for r in records
                ),
                "results": records,
            }, handle, indent=2, sort_keys=True)
        print(f"\nwrote {len(records)} records to {args.output}")
    failed = [r for r in records if r["objective_rel"] >= OBJECTIVE_RTOL or not r["coef_ok"]]
    if failed:
        print(f"\n{len(failed)}/{len(records)} cells outside a tolerance in "
              f"docs/fp32_policy.md section 3")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
