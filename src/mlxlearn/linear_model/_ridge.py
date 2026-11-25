"""Ridge regression on the host, by the normal equations scikit-learn's default uses.

Subclassing :class:`sklearn.linear_model.Ridge` is what makes the drop-in promise real:
the constructor signature, ``get_params``/``set_params``, the parameter constraints and
the estimator tags are all inherited, so ``clone``, ``Pipeline``, ``GridSearchCV`` and
``isinstance`` keep working, and ``super().fit(...)`` *is* stock scikit-learn for the
patching layer to fall back to.

mlxlearn's path covers dense ridge regression with ``solver`` in ``{"auto", "cholesky"}``,
including multi-output targets and sample weights. For dense input with
``positive=False`` scikit-learn's own ``"auto"`` resolves to ``"cholesky"``
(``resolve_solver_for_numpy``), so the default path is the one served here. Every other
solver raises rather than being quietly redirected: ``svd``, ``lsqr``, ``sparse_cg``,
``sag``, ``saga`` and ``lbfgs`` minimise the same objective by different routes and stop
at different points, so serving them from here would return answers close enough to look
right and different enough to matter.

``positive=True`` is a capability mismatch — non-negative ridge is a different solver —
and sparse input is not implemented.

On the device
-------------
``device="gpu"`` is accepted and reported honestly. The factorization is float64 and
float64 does not execute on Metal, so the solve runs on the host and the recorded backend
is ``"cpu"`` with ``reason="host-solve"``, as :class:`~mlxlearn.linear_model.LinearRegression`
does. A device Gram would need a measured crossover first; ``execution_backend_`` says
which path actually ran either way.
"""

from __future__ import annotations

import numbers

import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import Ridge as _SkRidge
from sklearn.utils.validation import check_array

from .._common.base import BackendMixin, mlx_guard
from .._common.diagnostics import record_event
from .._common.validation import check_is_fitted, require_dense, validate_data
from .._solvers.ols import _check_process
from .._solvers.ridge import RidgeConfig, solve_ridge
from ..exceptions import UnsupportedParameterError

__all__ = ["Ridge"]

_SKLEARN = tuple(int(part) for part in sklearn.__version__.split(".")[:2])

#: The only solvers served. ``"auto"`` resolves to ``"cholesky"`` for dense input with
#: ``positive=False``, so it is the default path rather than a special case.
_SUPPORTED_SOLVERS = ("auto", "cholesky")


class Ridge(BackendMixin, _SkRidge):
    """Linear least squares with L2 regularisation, fitted by mlxlearn's host solver.

    A drop-in replacement for :class:`sklearn.linear_model.Ridge` on dense data with
    ``solver`` in ``{"auto", "cholesky"}``. Multi-output targets and ``sample_weight``
    are implemented; ``positive=True``, sparse input and every other solver are not.

    Parameters
    ----------
    alpha : float or array-like of shape (n_targets,), default=1.0
    fit_intercept : bool, default=True
    copy_X : bool, default=True
        Accepted. ``X`` is read but never written: the centred working design is always a
        separate array, so the caller's array is not modified even when ``copy_X=False``.
    max_iter : int, default=None
        Accepted and ignored, as scikit-learn ignores it for ``solver="cholesky"``.
    tol : float, default=1e-4
        Accepted and ignored, for the same reason: the direct solve has no tolerance.
    solver : str, default="auto"
        Only ``"auto"`` and ``"cholesky"`` are implemented. Everything else raises
        :class:`~mlxlearn.exceptions.UnsupportedParameterError` on direct use and falls
        back under ``patch_sklearn()``.
    positive : bool, default=False
        Only ``False`` is implemented.
    random_state : int, RandomState or None, default=None
        Accepted and ignored, as scikit-learn ignores it outside ``sag``/``saga``.

    Notes
    -----
    Not registered in ``patching/_registry.py``: ``docs/development.md`` admits an
    estimator there only after all six gates pass, and gate 6 wants a measured crossover
    wired into ``Tuning``. Until that measurement exists on fixed hardware there is no
    honest threshold to write, so this ships as a Layer 1 estimator reached by explicit
    import and ``patch_sklearn()`` leaves ``sklearn.linear_model.Ridge`` alone.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        float64.
    intercept_ : float or ndarray of shape (n_targets,)
        float64. ``0.0`` when ``fit_intercept=False``, matching scikit-learn.
    n_features_in_ : int
    n_iter_ : None
        ``None`` for a direct solve, as stock reports for ``"cholesky"``.
    solver_ : str
        The resolved solver, ``"cholesky"`` on this path.
    execution_backend_ : {"cpu", "sklearn"}
        ``"cpu"`` whenever mlxlearn fitted the model. With ``device="gpu"`` the solve
        still runs on the host and the diagnostics record ``reason="host-solve"``.
    """

    _mlxlearn_supports_multi_output = True

    def _check_ridge_capability(self, X, y=None) -> None:
        """Raise for anything mlxlearn's path cannot serve.

        Called through ``_select_backend(capability_check=...)`` so that
        ``_handle_capability_error`` can turn the refusal into a scikit-learn fallback
        for Layer 2. A check made later, inside the solve, raises past that machinery
        and breaks the patched contract.
        """
        name = type(self).__name__
        require_dense(X, estimator=name)
        if bool(self.positive):
            raise UnsupportedParameterError(
                f"{name} implements positive=False only; got positive={self.positive!r}. "
                "Non-negative ridge is a different solver.",
                parameter="positive",
                value=self.positive,
            )
        if self.solver not in _SUPPORTED_SOLVERS:
            raise UnsupportedParameterError(
                f"{name} implements solver in {_SUPPORTED_SOLVERS}; got "
                f"solver={self.solver!r}. The other solvers minimise the same objective "
                "by different routes and stop at different points.",
                parameter="solver",
                value=self.solver,
            )
        _alpha_as_stock(self.alpha, y)

    def fit(self, X, y, sample_weight=None):
        """Fit the ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        sample_weight : float or array-like of shape (n_samples,), default=None

        Returns
        -------
        self : Ridge
        """
        _check_process()
        if not self._mlxlearn_allow_fallback:
            # Layer 1 validates first, so its parameter errors are stock's. The patched
            # class leaves that to whichever fit runs.
            self._validate_params()

        self._clear_backend()
        shape = _safe_shape(X)
        if shape is None and not hasattr(X, "shape"):
            # Lists and other shape-less array-likes: convert once so the size crossover
            # applies. Anything that is not an ordinary numeric array (ragged rows, None,
            # strings, generators) goes on as the caller passed it, so the errors and
            # their order are stock's.
            try:
                converted = np.asarray(X)
            except (ValueError, TypeError):
                converted = None
            if converted is not None:
                shape = _safe_shape(converted)
                if converted.dtype.kind in "biuf" and converted.ndim >= 1:
                    X = converted
        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_ridge_capability(X, y),
            prefer_cpu=True,
            shape=shape,
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self
        if self._mlxlearn_allow_fallback:
            self._validate_params()

        # Stock's own validation, from Ridge.fit: dense, both float widths, multi-output.
        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            multi_output=True,
            y_numeric=True,
            ensure_min_samples=1,
            reset=True,
        )
        # _preprocess_data's own `check_array(y, dtype=X.dtype, ensure_2d=False)`.
        # Without it a narrowing cast (float32 X with out-of-range targets) fits instead
        # of raising, and a NaN introduced by the cast reaches the solver.
        y_checked = _targets_as_stock(y_checked, X_checked.dtype)
        y_is_1d = y_checked.ndim == 1
        weights = None
        if sample_weight is not None:
            weights = _sample_weight_as_stock(sample_weight, X_checked)
        if weights is not None and self.fit_intercept and float(np.sum(weights, dtype=np.float64)) == 0.0:
            # Where stock's weighted mean (np.average inside _preprocess_data) raises it.
            # On 1.9 the all-zero case is already rejected during validation; this covers
            # 1.7 and 1.8, and weights that cancel to zero on any version.
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        with mlx_guard(type(self).__name__, "fit"):
            centred_X, centred_y, x_offset, y_offset = _preprocess_as_stock(
                X_checked, y_checked, bool(self.fit_intercept), weights
            )
            solution = solve_ridge(
                centred_X,
                centred_y,
                config=RidgeConfig(alpha=_alpha_as_stock(self.alpha, y_checked)),
            )

        coef = solution.coef
        # Stock ravels on the target count, not on y's dimensionality: see
        # `if n_targets == 1: coef = _ravel(coef)` in `_ridge_regression`. So a
        # column-vector y of shape (n, 1) gives a 1-D coef_ exactly as y of shape (n,)
        # does, and `__sklearn_tags__` must advertise multi_output or sklearn's
        # check_supervised_y_2d demands a DataConversionWarning stock never emits.
        # Two different rules, both stock's. coef_ ravels on the *target count* --
        # `if n_targets == 1: coef = _ravel(coef)` in `_ridge_regression` -- so a
        # column-vector y gives a 1-D coef_ just as a 1-D y does. intercept_ instead
        # follows y's *dimensionality*: 1-D y gives a float64 scalar, a (n, 1) y gives a
        # one-element array, and fit_intercept=False gives a plain 0.0 in both cases.
        self.coef_ = np.ravel(coef) if coef.shape[0] == 1 else coef
        if self.fit_intercept:
            intercept = np.ascontiguousarray(y_offset - coef @ x_offset, dtype=np.float64)
            self.intercept_ = np.float64(np.ravel(intercept)[0]) if y_is_1d else intercept
        else:
            self.intercept_ = 0.0
        # Stock reports no iteration count for a direct solve, and names the solver it
        # resolved to rather than the one the user asked for.
        self.n_iter_ = None
        # Stock rebinds its own `solver` to "svd" when the Cholesky raises and reports
        # the rebound name, so this must follow the route taken rather than the request.
        self.solver_ = solution.method

        if backend == "mlx":
            # The factorization is float64 and float64 does not execute on Metal.
            backend = "cpu"
            record_event(
                type(self).__name__, "fit", backend, reason="host-solve",
                detail="the ridge factorization is float64, solved on the host",
                shape=X_checked.shape,
            )
        record_event(type(self).__name__, "fit", backend,
                     reason=f"ridge-{solution.method}", shape=X_checked.shape)
        self._set_backend(backend)
        return self

    def predict(self, X):
        """Predict using the ridge model.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_targets)
        """
        _check_process()
        check_is_fitted(self)
        if getattr(self, "_execution_backend_", None) is not None:
            # A model whose coef_/intercept_ were set by hand has no recorded backend;
            # stock predicts from it, and so does this.
            self._note_inference("predict", shape=_safe_shape(X))
        return super().predict(X)


def _alpha_as_stock(alpha, y) -> float:
    """One penalty for every target, as ``_solve_cholesky``'s ``one_alpha`` branch needs.

    scikit-learn accepts an array of per-target penalties. This path serves the uniform
    case; an array whose entries differ is a capability mismatch rather than something to
    average, so it raises and Layer 2 hands the fit to scikit-learn.

    The length rule is stock's, not ours: ``_ridge_regression`` raises
    ``ValueError("Number of targets and number of penalties do not correspond")``
    whenever ``alpha.size`` is neither 1 nor ``n_targets``, so a wrong-length alpha must
    raise that rather than being quietly accepted.
    """
    if isinstance(alpha, numbers.Number):
        return float(alpha)
    values = np.ravel(np.asarray(alpha, dtype=np.float64))
    if y is not None:
        shape = getattr(y, "shape", None)
        n_targets = shape[1] if shape is not None and len(shape) == 2 else 1
        if values.size not in (1, n_targets):
            raise ValueError(
                "Number of targets and number of penalties do not correspond: "
                f"{values.size} != {n_targets}"
            )
    if values.size and np.all(values == values[0]):
        return float(values[0])
    raise UnsupportedParameterError(
        "Ridge implements one penalty for every target; got a per-target alpha "
        f"{alpha!r}. Fitting targets under different penalties is a different problem.",
        parameter="alpha",
        value=alpha,
    )


def _preprocess_as_stock(X, y, fit_intercept, weights):
    """scikit-learn's ``_preprocess_data`` for the dense, unscaled case.

    Rebuilt from public API, since the project allows no private scikit-learn imports
    outside ``patching/``. With ``fit_intercept=True`` stock subtracts the (weighted)
    column means of ``X`` and mean of ``y``, then row-scales both by ``sqrt(w)``; with
    ``fit_intercept=False`` it only row-scales. ``X`` is never written in place, so the
    caller's array survives ``copy_X=False``.
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    y2 = y64.reshape(-1, 1) if y64.ndim == 1 else y64

    if fit_intercept:
        if weights is None:
            x_offset = X64.mean(axis=0)
            y_offset = y2.mean(axis=0)
        else:
            total = float(np.sum(weights, dtype=np.float64))
            x_offset = (weights @ X64) / total
            y_offset = (weights @ y2) / total
        X64 = X64 - x_offset
        y2 = y2 - y_offset
    else:
        x_offset = np.zeros(X64.shape[1], dtype=np.float64)
        y_offset = np.zeros(y2.shape[1], dtype=np.float64)

    if weights is not None:
        root = np.sqrt(weights)[:, None]
        X64 = X64 * root
        y2 = y2 * root
    return X64, y2, x_offset, np.ravel(y_offset)


def _targets_as_stock(y, dtype) -> np.ndarray:
    """``check_array(y, dtype=X.dtype, ensure_2d=False)`` as ``_preprocess_data`` runs it.

    ``validate_data`` has already checked a numeric y, so a clean one needs only the cast
    and one finiteness pass; sparse, string or object targets, and any non-finite value
    introduced by a narrowing cast or hidden in an object array, go through stock's own
    call for its exact error. Honours ``assume_finite`` as stock does.
    """
    if sp.issparse(y) or getattr(y, "dtype", None) is None or y.dtype.kind not in "biuf":
        return check_array(y, dtype=dtype, ensure_2d=False)
    y_arr = np.asarray(y, dtype=dtype)
    if not np.isfinite(y_arr).all():
        return check_array(y, dtype=dtype, ensure_2d=False)
    return y_arr


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def _sample_weight_as_stock(sample_weight, X) -> np.ndarray:
    """scikit-learn's ``_check_sample_weight(..., dtype=X.dtype)``.

    Rebuilt from public API with each version's rule: float64 weights unless X is
    float32 or float64; the short 1-D message on 1.7; the all-zero check from 1.9.
    Ridge, unlike ``LinearRegression``, does not reject negative weights.
    """
    n = X.shape[0]
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    if isinstance(sample_weight, numbers.Number):
        # Not an early return: stock's `_check_sample_weight` applies the all-zero guard
        # after both branches, so `sample_weight=0` must raise on 1.9 rather than fitting
        # zero coefficients.
        weights = np.full(n, sample_weight, dtype=dtype)
        if _SKLEARN >= (1, 9) and np.all(weights == 0):
            raise ValueError("Sample weights must contain at least one non-zero number.")
        return weights
    weights = check_array(sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
                          order="C", input_name="sample_weight")
    if weights.ndim != 1:
        if _SKLEARN >= (1, 8):
            raise ValueError(
                f"Sample weights must be 1D array or scalar, got {weights.ndim}D array. "
                f"Expected either a scalar value or a 1D array of length {n}."
            )
        raise ValueError("Sample weights must be 1D array or scalar")
    if weights.shape != (n,):
        raise ValueError(f"sample_weight.shape == {weights.shape}, expected {(n,)}!")
    if _SKLEARN >= (1, 9) and np.all(weights == 0):
        raise ValueError("Sample weights must contain at least one non-zero number.")
    return weights
