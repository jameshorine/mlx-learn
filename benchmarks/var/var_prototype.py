"""VAR(p) as multi-output ridge: which code path is fast, and where it stops being exact.

Not an estimator and not part of the gated suite. This answers one question before any
estimator is written: a vector autoregression reduces to multi-output least squares, so
*which* of the available solve paths wins, and over what range of lag orders does it stay
inside the tolerances in ``docs/fp32_policy.md`` section 3?

The reduction
-------------
A VAR(p) on ``k`` variables over ``n`` periods is::

    y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t

with ``y_t`` in R^k, each ``A_i`` in R^(k x k), and ``E[e_t e_t'] = Sigma``. Stack the
lags into one regressor vector and the coefficients into one matrix::

    z_t = [1, y_{t-1}', ..., y_{t-p}']'   in R^(kp+1)
    B   = [c, A_1, ..., A_p]              in R^(k x (kp+1))

so ``y_t = B z_t + e_t``, and stacking rows over ``t = p+1 ... n`` gives a plain
multi-output linear model::

    Y = Z B' + E        Z is (n-p) x (kp+1),  B' is (kp+1) x k,  Y is (n-p) x k
    B' = (Z'Z + lam I)^-1 Z'Y

The design width is ``d = kp + 1``. That is the whole reason the published
``LinearRegression`` crossover grid measures this computation directly.

The companion form is **not** an estimation route. A VAR(p) can be rewritten as a VAR(1)
on the stacked state ``Y_t = [y_t', ..., y_{t-p+1}']'`` with transition ``F``, but ``F``
has ``(kp)^2`` entries of which only the first block row is free -- the rest are fixed
identity and zero blocks. Fitting it unrestricted would estimate known constants.
``F`` is used here only for the stability check ``rho(F) < 1``.

Why the Gram is computed in float32 on the device
-------------------------------------------------
``_solvers/ols.py`` is certified float64, and float64 does not execute on Metal, so that
path can never use the GPU. Splitting the work the way ``_solvers/pca.py`` splits
``covariance_eigh`` -- build ``Z'Z`` and ``Z'Y`` in float32 on the device, add the ridge
diagonal and factor in float64 on the host -- puts the ``O(n d^2)`` term on the GPU and
leaves only the ``O(d^3)`` factorization, under 1% of the work at these shapes, on the
host.

Ridge, and why it is not optional
---------------------------------
At ``k=321, p=1`` the fit has 103,362 parameters. ``OLSConfig`` has no regularisation
hook, so for the ``solve_ols`` backend ridge is done by augmenting the design with
``sqrt(lam) * I_d`` rows and zero targets, which returns ``(Z'Z + lam I)^-1 Z'Y``
exactly. The augmented rows must not be centred, so that path centres explicitly, fits
without an intercept and restores it -- which is also the right statistics, since the
intercept should not be penalised. The Gram backends add ``lam`` to the diagonal directly.

Usage
-----
    python -m benchmarks.var.var_prototype --synthetic --k 321 --n 26304 --p 1
    python -m benchmarks.var.var_prototype --tsf electricity_hourly_dataset.tsf --p 1
    python -m benchmarks.var.var_prototype --tsf electricity_hourly_dataset.tsf --sweep 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

__all__ = [
    "lag_design",
    "ridge_augment",
    "gram_host",
    "gram_metal",
    "solve_normal_equations",
    "ridge_objective",
    "companion_matrix",
    "companion_spectral_radius",
    "default_rcond",
    "load_tsf",
    "fit_design",
    "fit_var",
    "BACKENDS",
    "UNREGULARISED",
]

#: Objective-value tolerance from ``docs/fp32_policy.md`` section 3. The coefficient
#: tolerances there (``atol=2e-3, rtol=5e-3``) are the weak assertion; on a design with
#: ``cond(Z'Z)`` near 1e14 the coefficients are not pinned down by the data at all.
OBJECTIVE_RTOL = 1e-5

BACKENDS = ("metal_gram", "host_gram", "solve_ols", "sklearn_ridge", "sklearn_ols", "lstsq")

#: Backends with no penalty term. With ``lam > 0`` they solve a different objective than
#: the rest, so their objective is reported as not applicable rather than as a failure.
UNREGULARISED = frozenset({"sklearn_ols"})


def default_rcond(shape: tuple[int, int], dtype: np.dtype) -> float:
    """scikit-learn 1.7 and 1.8's singular-value cutoff, ``max(n, d) * eps``.

    ``OLSConfig`` defaults to ``1e-6``, about six orders coarser. On a merely
    ill-conditioned design that truncates a direction stock keeps, which changes the
    minimum-norm solution rather than its last digits.
    """
    return max(shape) * float(np.finfo(dtype).eps)


def lag_design(Y: np.ndarray, p: int, *, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Block-Hankel design for a VAR(p).

    Parameters
    ----------
    Y : ndarray of shape ``(n, k)``
        Observations in time order, one row per period.
    p : int
        Lag order, ``>= 1``.
    intercept : bool, default=True
        Prepend a constant column, giving ``d = k*p + 1``.

    Returns
    -------
    Z : ndarray of shape ``(n - p, k*p + intercept)``
        Row ``t`` is ``[1, y_{t-1}', ..., y_{t-p}']`` with the most recent lag first,
        matching statsmodels' coefficient order.
    target : ndarray of shape ``(n - p, k)``
        Row ``t`` is ``y_t``.
    """
    Y = np.ascontiguousarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-d (n, k), got shape {Y.shape}")
    n, k = Y.shape
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if n <= p:
        raise ValueError(f"need n > p, got n={n}, p={p}")

    # windows[t] is Y[t : t + p]; row t needs y_{t-1} .. y_{t-p}, i.e. that block
    # reversed so the most recent lag leads.
    windows = np.lib.stride_tricks.sliding_window_view(Y, (p, k))[:-1, 0]
    lags = windows[:, ::-1, :].reshape(n - p, k * p)

    if intercept:
        Z = np.empty((n - p, k * p + 1), dtype=Y.dtype)
        Z[:, 0] = 1.0
        Z[:, 1:] = lags
    else:
        Z = np.ascontiguousarray(lags)
    return Z, Y[p:]


def ridge_augment(Z: np.ndarray, T: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows that turn an unregularised OLS solve into a ridge solve, exactly.

    Solving least squares on ``[Z ; sqrt(lam) I]`` against ``[T ; 0]`` returns
    ``(Z'Z + lam I)^-1 Z'T``.
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    d = Z.shape[1]
    Za = np.vstack([Z, np.sqrt(lam) * np.eye(d, dtype=Z.dtype)])
    Ta = np.vstack([T, np.zeros((d, T.shape[1]), dtype=T.dtype)])
    return Za, Ta


def gram_host(Z: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Z'Z, Z'T)`` in host float64."""
    Z64 = np.asarray(Z, dtype=np.float64)
    T64 = np.asarray(T, dtype=np.float64)
    return Z64.T @ Z64, Z64.T @ T64


def gram_metal(Z: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Z'Z, Z'T)`` built in float32 on Metal, returned as float64.

    Uses ``mlxlearn_device._binary32.float32_matmul``, which keeps binary32 products on
    GPUs whose float32 matmul would otherwise round to TF32.
    """
    import mlx.core as mx

    from mlxlearn_device._binary32 import float32_matmul

    Zf = mx.array(np.ascontiguousarray(Z, dtype=np.float32))
    Tf = mx.array(np.ascontiguousarray(T, dtype=np.float32))
    G = float32_matmul(Zf, Zf, transpose_a=True)
    C = float32_matmul(Zf, Tf, transpose_a=True)
    mx.eval(G, C)
    return np.array(G, dtype=np.float64), np.array(C, dtype=np.float64)


def solve_normal_equations(G: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    """``(G + lam I)^-1 C`` by float64 Cholesky; ``B'`` of shape ``(d, k)``."""
    import scipy.linalg as sla

    G = np.array(G, dtype=np.float64, copy=True)
    if lam:
        G[np.diag_indices_from(G)] += lam
    return sla.solve(G, np.asarray(C, dtype=np.float64), assume_a="pos")


def ridge_objective(Z: np.ndarray, T: np.ndarray, Bt: np.ndarray, lam: float) -> float:
    """``||T - Z B'||_F^2 + lam ||B'||_F^2``, evaluated in float64 on the true data.

    This is the quantity ``docs/fp32_policy.md`` calls the strong assertion: it is far
    better conditioned than its argument, so it catches a wrong solver where a
    coefficient comparison cannot.
    """
    Z64 = np.asarray(Z, dtype=np.float64)
    R = np.asarray(T, dtype=np.float64) - Z64 @ Bt
    return float((R * R).sum()) + lam * float((Bt * Bt).sum())


def companion_matrix(B: np.ndarray, k: int, p: int) -> np.ndarray:
    """The ``(kp) x (kp)`` companion matrix ``F`` of a VAR(p).

    ``B`` is ``(k, k*p)``, the coefficient blocks ``[A_1 ... A_p]`` without the intercept.
    ``F`` carries ``B`` on its first block row and identity blocks on the subdiagonal::

        F = [ A_1  A_2  ...  A_{p-1}  A_p ]
            [  I    0   ...     0      0  ]
            [  0    I   ...     0      0  ]
            [            ...               ]
            [  0    0   ...     I      0  ]
    """
    if B.shape != (k, k * p):
        raise ValueError(f"B must be (k, k*p) = {(k, k * p)}, got {B.shape}")
    F = np.zeros((k * p, k * p), dtype=np.float64)
    F[:k, :] = B
    if p > 1:
        F[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return F


def companion_spectral_radius(B: np.ndarray, k: int, p: int) -> float:
    """``rho(F)``; the process is stable iff this is strictly below 1."""
    return float(np.max(np.abs(np.linalg.eigvals(companion_matrix(B, k, p)))))


def fit_design(Z: np.ndarray, T: np.ndarray, *, lam: float = 0.0,
               backend: str = "metal_gram") -> dict:
    """Fit an already-built design through one backend.

    The design is taken as given so that callers can time the solve alone.
    ``lag_design`` copies ``(n-p) x d`` float64 -- 67.8 MB at ``k=321, p=1`` -- which is
    the same cost for every backend and would otherwise be added to all of them, flatten
    the ratios, and measure NumPy's copy rather than the solver.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    n, d = Z.shape

    start = time.perf_counter()
    if backend in ("metal_gram", "host_gram"):
        G, C = (gram_metal if backend == "metal_gram" else gram_host)(Z, T)
        Bt = solve_normal_equations(G, C, lam)
    elif backend == "solve_ols":
        from mlxlearn._solvers.ols import OLSConfig, solve_ols

        Za, Ta = ridge_augment(Z, T, lam) if lam else (Z, T)
        cfg = OLSConfig(fit_intercept=False, rcond=default_rcond(Za.shape, Za.dtype))
        Bt = solve_ols(Za, Ta, config=cfg).coef.T
    elif backend in ("sklearn_ridge", "sklearn_ols"):
        if backend == "sklearn_ridge":
            from sklearn.linear_model import Ridge as Est

            est = Est(alpha=lam, solver="cholesky", fit_intercept=False)
        else:
            from sklearn.linear_model import LinearRegression as Est

            est = Est(fit_intercept=False)
        Bt = np.atleast_2d(est.fit(Z, T).coef_).T
    else:  # lstsq -- what statsmodels' VAR calls
        Za, Ta = ridge_augment(Z, T, lam) if lam else (Z, T)
        Bt = np.linalg.lstsq(Za, Ta, rcond=1e-15)[0]
    elapsed = time.perf_counter() - start

    return {
        "backend": backend,
        "coef": Bt[1:].T,          # (k, k*p), intercept dropped
        "intercept": Bt[0],
        "objective": ridge_objective(Z, T, Bt, lam),
        "seconds": elapsed,
        "shape": (n, d),
    }


def fit_var(Y: np.ndarray, p: int, *, lam: float = 0.0, backend: str = "metal_gram") -> dict:
    """Build the VAR(p) design and fit it. Convenience wrapper over :func:`fit_design`."""
    Z, T = lag_design(Y, p, intercept=True)
    return fit_design(Z, T, lam=lam, backend=backend)


def load_tsf(path: str) -> np.ndarray:
    """Read a Monash ``.tsf`` file into an ``(n_timesteps, n_series)`` array.

    Only the equidistant, equal-length case is handled, which is what
    ``electricity_hourly`` is; a ragged file raises rather than being silently padded.
    """
    series: list[list[float]] = []
    with open(path, errors="replace") as handle:
        in_data = False
        for line in handle:
            if not in_data:
                in_data = line.strip().lower() == "@data"
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            series.append([float(v) for v in parts[2].split(",")])
    if not series:
        raise ValueError(f"no data rows found in {path}")
    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError(f"ragged series in {path}: lengths {sorted(lengths)[:5]}")
    return np.asarray(series, dtype=np.float64).T


def synthetic(n: int, k: int, p: int, seed: int) -> np.ndarray:
    """A stable VAR(p) path, so recovered coefficients can be checked."""
    rng = np.random.default_rng(seed)
    B = np.hstack([rng.normal(scale=1.0 / (k**0.5), size=(k, k)) for _ in range(p)])
    # rho(F) is not linear in the scale of B: scaling the blocks rescales only F's first
    # block row, while the subdiagonal identities are fixed. One division by rho is
    # therefore not enough for p > 1, so shrink until it is actually stable.
    for _ in range(64):
        rho = companion_spectral_radius(B, k, p)
        if rho < 0.95:
            break
        B = B * (0.9 / rho)
    else:
        raise RuntimeError(f"could not find a stable VAR({p}) at k={k}; rho={rho:.4f}")
    blocks = [B[:, i * k : (i + 1) * k] for i in range(p)]

    burn = 200
    Y = np.zeros((n + burn, k))
    for t in range(p, n + burn):
        acc = rng.normal(scale=0.1, size=k)
        for i, A in enumerate(blocks):
            acc += A @ Y[t - 1 - i]
        Y[t] = acc
    return Y[burn:]


def _bench(fn, repeats: int) -> float:
    fn()  # warm: MLX compiles per shape, and BLAS pools spin up on first call
    return min(_timed(fn) for _ in range(repeats))


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _report(Y: np.ndarray, p: int, lam_scale: float, backends: list[str], repeats: int) -> None:
    Z, T = lag_design(Y, p, intercept=True)
    n, d = Z.shape
    k = T.shape[1]
    G64, _ = gram_host(Z, T)
    lam = lam_scale * float(np.trace(G64)) / d
    print(f"\np={p}  k={k}  d=kp+1={d}  design {n} x {d}  lam={lam:.6g}")

    results = {}
    for backend in backends:
        try:
            # Z and T are built once, above; only the solve is timed.
            seconds = _bench(lambda b=backend: fit_design(Z, T, lam=lam, backend=b), repeats)
            out = fit_design(Z, T, lam=lam, backend=backend)
            out["seconds"] = seconds
            results[backend] = out
        except Exception as exc:  # noqa: BLE001 - a prototype reports and continues
            print(f"  {backend:14} FAILED: {type(exc).__name__}: {exc}")

    ref = results.get("host_gram")
    width = max(len(b) for b in results) if results else 0
    for backend, out in results.items():
        line = f"  {backend:{width}}  {out['seconds'] * 1000:8.1f} ms"
        if ref is not None:
            line += f"   {ref['seconds'] / out['seconds']:6.2f}x vs host_gram"
            if backend in UNREGULARISED and lam:
                # Solves min ||T - ZB'||^2 with no penalty, so its ridge objective is
                # legitimately larger. Comparing it to the ridge solution is meaningless.
                line += "   obj rel        n/a [ignores lam]"
            else:
                rel = abs(out["objective"] - ref["objective"]) / abs(ref["objective"])
                gate = "pass" if rel < OBJECTIVE_RTOL else "FAIL"
                line += f"   obj rel {rel:9.2e} [{gate}]"
        print(line)

    if "host_gram" in results:
        rho = companion_spectral_radius(results["host_gram"]["coef"], k, p)
        print(f"  companion rho = {rho:.4f} ({'stable' if rho < 1 else 'UNSTABLE'})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tsf", help="Monash .tsf file")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--sweep", type=int, nargs="+", help="lag orders to sweep instead of --p")
    parser.add_argument("--lam-scale", type=float, default=1e-6,
                        help="ridge lam as this multiple of trace(Z'Z)/d")
    parser.add_argument("--n", type=int, default=26304, help="synthetic only")
    parser.add_argument("--k", type=int, default=321, help="synthetic only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    args = parser.parse_args(argv)

    orders = args.sweep if args.sweep else [args.p]
    Y = load_tsf(args.tsf) if args.tsf else synthetic(args.n, args.k, max(orders), args.seed)
    print(f"data: n={Y.shape[0]} k={Y.shape[1]}  "
          f"source={'tsf' if args.tsf else 'synthetic'}  repeats={args.repeats}")
    print(f"objective gate: rtol < {OBJECTIVE_RTOL:g} vs host_gram (docs/fp32_policy.md s3)")
    for p in orders:
        _report(Y, p, args.lam_scale, list(args.backends), args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
