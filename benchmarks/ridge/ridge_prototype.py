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


def gram_host(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(X'X, X'y)`` in host float64."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    return X64.T @ X64, X64.T @ y64


def gram_metal(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(X'X, X'y)`` built in float32 on Metal, returned as float64.

    Uses ``mlxlearn_device._binary32.float32_matmul``, which keeps binary32 products on
    GPUs whose float32 matmul would otherwise round to TF32.
    """
    import mlx.core as mx

    from mlxlearn_device._binary32 import float32_matmul

    Xf = mx.array(np.ascontiguousarray(X, dtype=np.float32))
    yf = mx.array(np.ascontiguousarray(y, dtype=np.float32))
    G = float32_matmul(Xf, Xf, transpose_a=True)
    C = float32_matmul(Xf, yf, transpose_a=True)
    mx.eval(G, C)
    return np.array(G, dtype=np.float64), np.array(C, dtype=np.float64)


def solve_normal_equations(G: np.ndarray, C: np.ndarray, alpha: float) -> np.ndarray:
    """``(G + alpha I)^-1 C`` by float64 Cholesky. Does not mutate ``G``."""
    import scipy.linalg as sla

    G = np.array(G, dtype=np.float64, copy=True)
    if alpha:
        G[np.diag_indices_from(G)] += alpha
    return sla.solve(G, np.asarray(C, dtype=np.float64), assume_a="pos")


def make_problem(n: int, d: int, targets: int, seed: int, *, cond: float = 1.0):
    """A dense regression problem, optionally ill-conditioned.

    ``cond`` scales the design's singular values geometrically from 1 down to ``1/cond``,
    which is how the sensitivity of the float32 Gram to conditioning gets exercised.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if cond > 1.0:
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        s = np.geomspace(1.0, 1.0 / cond, num=min(n, d))
        X = (U * s) @ Vt
    beta = rng.normal(size=(d, targets))
    y = X @ beta + rng.normal(scale=0.1, size=(n, targets))
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def ridge_objective(X: np.ndarray, y: np.ndarray, coef: np.ndarray, alpha: float) -> float:
    """``||y - X b||_F^2 + alpha ||b||_F^2`` in float64 on the true data."""
    R = np.asarray(y, dtype=np.float64) - np.asarray(X, dtype=np.float64) @ coef
    return float((R * R).sum()) + alpha * float((coef * coef).sum())


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float, *, backend: str) -> np.ndarray:
    """Coefficients of shape ``(d, targets)``. No intercept; callers centre if they want one."""
    if backend == "metal_gram":
        return solve_normal_equations(*gram_metal(X, y), alpha)
    if backend == "host_gram":
        return solve_normal_equations(*gram_host(X, y), alpha)
    if backend == "sklearn":
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=alpha, solver="cholesky", fit_intercept=False).fit(X, y)
        return np.atleast_2d(est.coef_).T
    raise ValueError(f"unknown backend {backend!r}")


def _run(n: int, d: int, targets: int, alpha_scale: float, cond: float,
         seed: int, iterations: int, records: list[dict]) -> None:
    from benchmarks._harness import compare

    X, y = make_problem(n, d, targets, seed, cond=cond)
    G, _ = gram_host(X, y)
    alpha = alpha_scale * float(np.trace(G)) / d

    cmp = compare(
        lambda: fit_ridge(X, y, alpha, backend="metal_gram"),
        lambda: fit_ridge(X, y, alpha, backend="sklearn"),
        label=f"ridge/n{n}/d{d}/t{targets}/cond{cond:g}",
        shape=(n, d),
        iterations=iterations,
    )
    ref = fit_ridge(X, y, alpha, backend="sklearn")
    got = fit_ridge(X, y, alpha, backend="metal_gram")
    j_ref, j_got = ridge_objective(X, y, ref, alpha), ridge_objective(X, y, got, alpha)
    obj_rel = abs(j_got - j_ref) / abs(j_ref)
    denom = max(float(np.abs(ref).max()), 1e-300)
    coef_rel = float(np.abs(got - ref).max()) / denom

    records.append({**cmp.as_dict(), "n": n, "d": d, "targets": targets, "cond": cond,
                    "alpha": alpha, "objective_rel": obj_rel, "coef_rel": coef_rel,
                    "backend": "metal_gram", "baseline": "sklearn_ridge"})
    gate = "pass" if obj_rel < OBJECTIVE_RTOL else "FAIL"
    sig = "" if cmp.significant else "  (ns)"
    print(f"  n={n:<7} d={d:<5} t={targets:<3} cond={cond:<8g} "
          f"{cmp.mlxlearn.median_s * 1000:7.1f} vs {cmp.sklearn.median_s * 1000:7.1f} ms   "
          f"{cmp.speedup:5.2f}x med {cmp.speedup_best:5.2f}x best{sig}   "
          f"obj {obj_rel:8.2e} [{gate}]  coef {coef_rel:8.2e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--d", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--targets", type=int, default=1)
    parser.add_argument("--cond", type=float, nargs="+", default=[1.0])
    parser.add_argument("--alpha-scale", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    print(f"Ridge: Metal float32 Gram + host float64 Cholesky vs "
          f"sklearn Ridge(solver='cholesky'), alpha = {args.alpha_scale:g} * trace(X'X)/d")
    print(f"objective gate: rtol < {OBJECTIVE_RTOL:g}  (docs/fp32_policy.md s3)")
    records: list[dict] = []
    for cond in args.cond:
        for n in args.n:
            for d in args.d:
                if d >= n:
                    continue
                _run(n, d, args.targets, args.alpha_scale, cond,
                     args.seed, args.iterations, records)

    if args.output:
        from benchmarks._harness import environment

        with open(args.output, "w") as handle:
            json.dump({
                "driver": "benchmarks/ridge/ridge_prototype.py",
                "environment": environment(),
                "parameters": vars(args),
                "objective_rtol": OBJECTIVE_RTOL,
                "results": records,
            }, handle, indent=2, sort_keys=True)
        print(f"\nwrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
