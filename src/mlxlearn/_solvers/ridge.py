"""Ridge regression on the host, in float64, by the routes scikit-learn's own default uses.

Same contract as every solver here: arrays and a frozen :class:`RidgeConfig` in, a frozen
:class:`RidgeSolution` out. No estimator, no scikit-learn import, no global configuration.

The problem is scikit-learn's, exactly
--------------------------------------
For a dense design with ``positive=False``, ``solver="auto"`` resolves to ``"cholesky"``
(``resolve_solver_for_numpy``), and ``_ridge_regression`` then branches on shape:

* ``n_features <= n_samples`` -- the **primal** route, ``_solve_cholesky``::

      A = X'X ;  Xy = X'y ;  A[::d+1] += alpha ;  solve(A, Xy, assume_a="pos")

* ``n_features > n_samples`` -- the **dual** route, ``_solve_cholesky_kernel``::

      K = X X' ;  K[::n+1] += alpha ;  dual = solve(K, y, assume_a="pos")
      coef = (X' dual)'

  with its own ``lstsq`` rescue, under a warning, when that solve raises.

* either route raising ``LinAlgError`` -- stock rebinds ``solver = "svd"`` and the
  ``svd`` branch runs ``s / (s**2 + alpha)`` on an SVD of ``X``.

This module reproduces all three. The dual route matters for more than speed: on a wide
design the two routes are different computations in finite precision, so serving the
primal everywhere would return answers close to stock's and not equal to them.

Why this is far smaller than :mod:`mlxlearn._solvers.ols`
---------------------------------------------------------
That module is dominated by rank certification, because ``LinearRegression`` must return
``lstsq``'s minimum-norm solution plus ``rank_`` and ``singular_``, and a Gram cannot
decide a rank whose conditioning it has squared. Ridge reports neither, and with
``alpha > 0`` the factored matrix is positive definite by construction, so there is no
rank to certify.

What ``alpha`` does and does not buy
------------------------------------
It bounds the condition number of the factored matrix by
``(sigma_max**2 + alpha) / alpha``. That is a statement about the *solve*, not about the
Gram: rounding in forming ``X'X`` is injected before ``alpha`` is added, and the penalty
only caps how far it is amplified. At a small ``alpha`` the cap is large -- measured at
``alpha = 1e-6 * trace(X'X)/d`` the penalty moved the condition number from 1.414 to
1.415. This module therefore works in float64 throughout, where the question does not
arise.

A fit is a pure function of its arguments. Nothing about a previous fit in the same
process can reach this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg

__all__ = ["RidgeConfig", "RidgeSolution", "solve_ridge"]

#: scikit-learn's singular-value cutoff in ``_solve_svd``: ``idx = s > 1e-15``, commented
#: "same default value as scipy.linalg.pinv". It is **absolute**, not relative to
#: ``s[0]``, so on a badly scaled design a relative cutoff keeps or drops different
#: directions and returns different coefficients.
_SVD_CUTOFF = 1e-15


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """Problem-independent settings for one ridge fit.

    Parameters
    ----------
    alpha : float, default=1.0
        The L2 penalty. Must be finite and non-negative. ``0.0`` is permitted, as stock
        permits it, and is ordinary least squares through the normal equations.
    """

    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not (self.alpha >= 0 and math.isfinite(self.alpha)):
            raise ValueError(f"alpha must be finite and non-negative, got {self.alpha}")


@dataclass(frozen=True, slots=True)
class RidgeSolution:
    """Fitted coefficients and which route produced them.

    Attributes
    ----------
    coef : ndarray of shape (n_targets, n_features), float64
        Always two-dimensional; the estimator ravels a single target.
    method : str
        ``"cholesky"`` when the primal or dual factorization succeeded, or ``"svd"`` when
        it did not and stock's fallback ran. The estimator reports this as ``solver_``,
        because stock rebinds its own ``solver`` on that fallback and returns the
        rebound value.
    """

    coef: np.ndarray
    method: str


def _solve_svd(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """scikit-learn's ``solver="svd"`` route, rebuilt from public API.

    Singular values at or below :data:`_SVD_CUTOFF` are dropped; the rest contribute
    ``s / (s**2 + alpha)``, a filter bounded above by ``1 / (2 * sqrt(alpha))``. That
    bound is what makes this safe where a least-squares solve of the regularised Gram --
    which inverts whatever spurious negative eigenvalue caused the failure -- is not.
    """
    U, s, Vt = linalg.svd(X, full_matrices=False)
    keep = s > _SVD_CUTOFF
    s_kept = s[keep][:, None]
    d = np.zeros((s.shape[0], 1), dtype=np.float64)
    d[keep] = s_kept / (s_kept**2 + alpha)
    return (Vt.T @ (d * (U.T @ y))).T


def _solve_primal(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """``_solve_cholesky``: factor the ``d x d`` Gram. Used when ``n_features <= n_samples``."""
    gram = X.T @ X
    rhs = X.T @ y
    if alpha:
        gram.flat[:: gram.shape[0] + 1] += alpha
    return linalg.solve(gram, rhs, assume_a="pos", overwrite_a=True).T


def _solve_dual(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """``_solve_cholesky_kernel``: factor the ``n x n`` kernel. Used when ``n_features > n_samples``.

    Stock rescues a singular kernel with ``lstsq`` here rather than falling through to
    the SVD route, so this does the same; the caller's ``LinAlgError`` handling is for
    the primal route.
    """
    kernel = X @ X.T
    if alpha:
        kernel.flat[:: kernel.shape[0] + 1] += alpha
    try:
        dual = linalg.solve(kernel, y, assume_a="pos", overwrite_a=False)
    except linalg.LinAlgError:
        dual = linalg.lstsq(kernel, y)[0]
    return (X.T @ dual).T


def solve_ridge(X: np.ndarray, y: np.ndarray, *, config: RidgeConfig) -> RidgeSolution:
    """Solve ``min ||y - X b||^2 + alpha ||b||^2`` in float64.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Already centred and row-scaled by the caller, as ``_preprocess_data`` leaves it.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Already centred to match.
    config : RidgeConfig

    Returns
    -------
    RidgeSolution
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if X64.ndim != 2:
        raise ValueError(f"X must be 2-d, got shape {X64.shape}")
    if y64.shape[0] != X64.shape[0]:
        raise ValueError(f"X has {X64.shape[0]} rows, y has {y64.shape[0]}")
    y2 = y64.reshape(-1, 1) if y64.ndim == 1 else y64

    n_samples, n_features = X64.shape
    alpha = float(config.alpha)
    method = "cholesky"
    try:
        if n_features > n_samples:
            coef = _solve_dual(X64, y2, alpha)
        else:
            coef = _solve_primal(X64, y2, alpha)
    except linalg.LinAlgError:
        # Stock rebinds solver = "svd" here and reports the rebound name.
        coef = _solve_svd(X64, y2, alpha)
        method = "svd"

    return RidgeSolution(coef=np.ascontiguousarray(coef, dtype=np.float64), method=method)
