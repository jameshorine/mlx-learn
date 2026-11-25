"""Parity between :class:`mlxlearn.linear_model.Ridge` and scikit-learn's.

The tolerances are the ones in ``docs/fp32_policy.md`` section 3, and the objective is
asserted alongside the coefficients because it is the assertion that actually catches a
wrong solver: two correct solvers land in different places on an ill-conditioned problem,
so a coefficient-only test fails for a correct implementation and passes for a subtly
wrong one.

In practice this path is far tighter than those tolerances — it is the same normal
equations in the same precision — so the assertions below are mostly exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge as StockRidge

from mlxlearn.exceptions import UnsupportedParameterError
from mlxlearn.linear_model import Ridge

#: docs/fp32_policy.md section 3.
COEF_ATOL, COEF_RTOL = 2e-3, 5e-3
OBJECTIVE_RTOL = 1e-5

SHAPES = [(300, 12, 1), (300, 12, 4), (80, 150, 2), (1000, 20, 1)]


def _problem(n, d, targets, seed=0, cond=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if cond > 1.0:
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = (U * np.geomspace(1.0, 1.0 / cond, num=min(n, d))) @ Vt
    y = rng.normal(size=n) if targets == 1 else rng.normal(size=(n, targets))
    return X, y


def _objective(X, y, est, alpha):
    pred = est.predict(X)
    resid = np.asarray(y, dtype=np.float64) - pred
    coef = np.asarray(est.coef_, dtype=np.float64)
    return float((resid * resid).sum()) + alpha * float((coef * coef).sum())


def _stock(**kw):
    kw.setdefault("solver", "cholesky")
    return StockRidge(**kw)


# -------------------------------------------------------------------- coefficients


@pytest.mark.parametrize(("n", "d", "targets"), SHAPES)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_coefficients_match_stock(n, d, targets, fit_intercept):
    X, y = _problem(n, d, targets)
    alpha = 0.7
    mine = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    stock = _stock(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    np.testing.assert_allclose(mine.coef_, stock.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)
    np.testing.assert_allclose(mine.intercept_, stock.intercept_, atol=COEF_ATOL, rtol=COEF_RTOL)


@pytest.mark.parametrize(("n", "d", "targets"), SHAPES)
def test_sample_weight_matches_stock(n, d, targets):
    X, y = _problem(n, d, targets, seed=1)
    weights = np.random.default_rng(2).uniform(0.5, 2.0, size=n)
    alpha = 0.3
    mine = Ridge(alpha=alpha).fit(X, y, sample_weight=weights)
    stock = _stock(alpha=alpha).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(mine.coef_, stock.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)


def test_a_scalar_sample_weight_matches_stock():
    X, y = _problem(200, 10, 1, seed=3)
    mine = Ridge(alpha=0.5).fit(X, y, sample_weight=2.0)
    stock = _stock(alpha=0.5).fit(X, y, sample_weight=2.0)
    np.testing.assert_allclose(mine.coef_, stock.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)


# ----------------------------------------------------------------------- objective


@pytest.mark.parametrize("cond", [1.0, 1e4, 1e8])
def test_objective_matches_stock_even_where_coefficients_need_not(cond):
    # The strong assertion. At cond=1e8 the coefficients are not pinned down by the data
    # and two correct solvers may differ; the objective is far better conditioned and is
    # what distinguishes a different stopping point from a wrong answer.
    X, y = _problem(500, 40, 1, seed=4, cond=cond)
    alpha = 1.0
    mine = Ridge(alpha=alpha).fit(X, y)
    stock = _stock(alpha=alpha).fit(X, y)
    j_mine = _objective(X, y, mine, alpha)
    j_stock = _objective(X, y, stock, alpha)
    assert abs(j_mine - j_stock) / abs(j_stock) < OBJECTIVE_RTOL


@pytest.mark.parametrize("alpha", [1e-6, 1.0, 1e4])
def test_predictions_match_stock_across_penalties(alpha):
    X, y = _problem(400, 16, 3, seed=5)
    mine = Ridge(alpha=alpha).fit(X, y)
    stock = _stock(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(mine.predict(X), stock.predict(X), atol=1e-8, rtol=1e-8)


# -------------------------------------------------------------------------- shapes


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_coef_follows_target_count_and_intercept_follows_y_shape(fit_intercept):
    # Two different rules, both stock's, and easy to conflate. coef_ ravels on the
    # target count -- `if n_targets == 1: coef = _ravel(coef)` in _ridge_regression --
    # so a (n, 1) y gives a 1-D coef_ exactly as a 1-D y does. intercept_ instead
    # follows y's dimensionality, except that fit_intercept=False gives a plain 0.0.
    X, y = _problem(120, 5, 1, seed=6)
    for target in (y, y.reshape(-1, 1)):
        mine = Ridge(fit_intercept=fit_intercept).fit(X, target)
        stock = _stock(fit_intercept=fit_intercept).fit(X, target)
        assert mine.coef_.shape == stock.coef_.shape == (5,)
        assert np.ndim(mine.intercept_) == np.ndim(stock.intercept_)
        np.testing.assert_allclose(mine.intercept_, stock.intercept_, atol=COEF_ATOL)

    # And the rules really do differ: a (n, 1) y keeps a 1-D intercept_ while its
    # coef_ is ravelled, but only when an intercept was fitted.
    column = Ridge(fit_intercept=fit_intercept).fit(X, y.reshape(-1, 1))
    assert np.ndim(column.intercept_) == (1 if fit_intercept else 0)


def test_fitted_attributes_match_stocks_contract():
    X, y = _problem(200, 8, 1, seed=7)
    mine = Ridge().fit(X, y)
    stock = _stock().fit(X, y)
    assert mine.n_features_in_ == stock.n_features_in_ == 8
    assert mine.n_iter_ is stock.n_iter_ is None
    assert mine.solver_ == stock.solver_ == "cholesky"
    assert mine.coef_.dtype == np.float64


def test_auto_resolves_to_cholesky_as_stock_does():
    X, y = _problem(200, 8, 1, seed=8)
    assert Ridge(solver="auto").fit(X, y).solver_ == _stock(solver="auto").fit(X, y).solver_


# --------------------------------------------------------------------- capability


@pytest.mark.parametrize("solver", ["svd", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
def test_unserved_solvers_raise_rather_than_being_redirected(solver):
    X, y = _problem(100, 5, 1, seed=9)
    with pytest.raises(UnsupportedParameterError, match="solver"):
        Ridge(solver=solver).fit(X, y)


def test_positive_raises():
    X, y = _problem(100, 5, 1, seed=10)
    with pytest.raises(UnsupportedParameterError, match="positive"):
        Ridge(positive=True, solver="auto").fit(X, y)


def test_a_per_target_alpha_that_differs_raises():
    X, y = _problem(100, 5, 3, seed=11)
    with pytest.raises(UnsupportedParameterError, match="alpha"):
        Ridge(alpha=[0.1, 0.2, 0.3]).fit(X, y)


def test_a_uniform_per_target_alpha_is_accepted():
    X, y = _problem(200, 6, 3, seed=12)
    mine = Ridge(alpha=[0.5, 0.5, 0.5]).fit(X, y)
    stock = _stock(alpha=0.5).fit(X, y)
    np.testing.assert_allclose(mine.coef_, stock.coef_, atol=COEF_ATOL, rtol=COEF_RTOL)


def test_copy_x_false_does_not_write_to_the_callers_array():
    X, y = _problem(200, 6, 1, seed=13)
    before = X.copy()
    Ridge(copy_X=False).fit(X, y)
    np.testing.assert_array_equal(X, before)


# ------------------------------------------------------------------------ backend


def test_an_unregistered_layer1_estimator_is_not_patched():
    # Not in patching/_registry.py until gate 6 is met, so patch_sklearn() must leave
    # sklearn.linear_model.Ridge alone.
    import sklearn.linear_model

    import mlxlearn

    mlxlearn.patch_sklearn()
    try:
        assert sklearn.linear_model.Ridge is StockRidge
    finally:
        mlxlearn.unpatch_sklearn()


def test_the_backend_is_reported_and_sticky():
    X, y = _problem(200, 6, 1, seed=14)
    mine = Ridge().fit(X, y)
    assert mine.execution_backend_ == "cpu"
    mine.predict(X)
    assert mine.execution_backend_ == "cpu"


def test_a_gpu_request_is_honoured_as_a_host_solve_and_says_so():
    # The factorization is float64 and float64 does not execute on Metal, so the solve
    # runs on the host. The point is that it is recorded rather than silent.
    import mlxlearn

    X, y = _problem(200, 6, 1, seed=15)
    mlxlearn.clear_backend_diagnostics()
    with mlxlearn.config_context(device="gpu"):
        mine = Ridge().fit(X, y)
    assert mine.execution_backend_ == "cpu"
    reasons = {event.reason for event in mlxlearn.get_backend_diagnostics()}
    assert "host-solve" in reasons
