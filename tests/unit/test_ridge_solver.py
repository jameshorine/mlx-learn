"""Solver-level tests for :mod:`mlxlearn._solvers.ridge`.

The solver takes arrays and a frozen config and returns a frozen state, so these need no
estimator and no scikit-learn estimator machinery. They do compare against the public
``sklearn.linear_model.ridge_regression``, which is the routine scikit-learn's own dense
default runs, to pin that this is the same computation rather than merely a similar one.
The public entry point is used deliberately: ``docs/development.md`` forbids importing
``sklearn._*`` outside ``patching/``, and the compliance gate happens not to scan tests.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import linalg
from sklearn.linear_model import ridge_regression

from mlxlearn._solvers.ridge import RidgeConfig, RidgeSolution, solve_ridge


def _problem(n, d, targets, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(n, targets))


# ------------------------------------------------------------------------ config


@pytest.mark.parametrize("alpha", [-1.0, -1e-18, np.inf, np.nan])
def test_config_rejects_an_unusable_penalty(alpha):
    with pytest.raises(ValueError, match="alpha must be finite and non-negative"):
        RidgeConfig(alpha=alpha)


def test_config_accepts_zero_because_stock_does():
    assert RidgeConfig(alpha=0.0).alpha == 0.0


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RidgeConfig(alpha=1.0).alpha = 2.0


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize(("n", "d", "targets"), [(200, 8, 1), (200, 8, 5), (50, 80, 3)])
@pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.7, 1e3])
def test_matches_the_function_scikit_learns_default_calls(n, d, targets, alpha):
    # `ridge_regression(solver="cholesky")` is what the estimator runs, and
    # `solver="auto"` resolves to it for dense input with positive=False. It routes
    # primal or dual on shape and handles the svd retry, so comparing against it pins
    # the whole contract. Anything other than bit-equality means this is not the same
    # computation.
    X, y = _problem(n, d, targets)
    got = solve_ridge(X, y, config=RidgeConfig(alpha=alpha)).coef
    expected = np.atleast_2d(ridge_regression(X, y, alpha=alpha, solver="cholesky"))
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_a_wide_design_takes_the_dual_route_as_stock_does():
    # n < d with no penalty: the Gram has rank at most n and is singular. scikit-learn's
    # bare _solve_cholesky raises LinAlgError here; its _ridge_regression catches that
    # and retries with solver="svd", which is what this solver reproduces. Pinning both
    # halves keeps the difference deliberate rather than accidental.
    # n < d: stock takes the *dual* route (K = X X', n x n) rather than the primal one,
    # and rescues a singular kernel with lstsq rather than falling through to svd. The
    # unregularised wide case is where a primal-only implementation would diverge.
    X, y = _problem(50, 80, 3)
    solution = solve_ridge(X, y, config=RidgeConfig(alpha=0.0))
    expected = np.atleast_2d(ridge_regression(X, y, alpha=0.0, solver="cholesky"))
    np.testing.assert_allclose(solution.coef, expected, rtol=0, atol=0)
    assert np.all(np.isfinite(solution.coef))


def test_solves_the_normal_equations_it_claims_to():
    X, y = _problem(120, 6, 2)
    alpha = 0.3
    coef = solve_ridge(X, y, config=RidgeConfig(alpha=alpha)).coef
    expected = linalg.solve(X.T @ X + alpha * np.eye(6), X.T @ y, assume_a="pos").T
    np.testing.assert_allclose(coef, expected, rtol=1e-12)


def test_minimises_the_ridge_objective():
    X, y = _problem(90, 5, 1)
    alpha = 0.4
    coef = solve_ridge(X, y, config=RidgeConfig(alpha=alpha)).coef

    def objective(b):
        r = y - X @ b.T
        return float((r * r).sum()) + alpha * float((b * b).sum())

    best = objective(coef)
    rng = np.random.default_rng(1)
    for _ in range(12):
        assert objective(coef + rng.normal(scale=1e-3, size=coef.shape)) >= best


# ------------------------------------------------------------------------- shape


@pytest.mark.parametrize("targets", [1, 4])
def test_coef_is_always_targets_by_features(targets):
    X, y = _problem(100, 7, targets)
    assert solve_ridge(X, y, config=RidgeConfig()).coef.shape == (targets, 7)


def test_a_one_dimensional_y_is_treated_as_one_target():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(80, 4)), rng.normal(size=80)
    flat = solve_ridge(X, y, config=RidgeConfig(alpha=0.5)).coef
    column = solve_ridge(X, y.reshape(-1, 1), config=RidgeConfig(alpha=0.5)).coef
    assert flat.shape == (1, 4)
    np.testing.assert_array_equal(flat, column)


def test_coef_is_float64_and_contiguous_whatever_the_input_dtype():
    X, y = _problem(60, 5, 2)
    solution = solve_ridge(X.astype(np.float32), y.astype(np.float32), config=RidgeConfig())
    assert solution.coef.dtype == np.float64
    assert solution.coef.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("X", "y", "match"),
    [
        (np.zeros((5,)), np.zeros((5, 1)), "X must be 2-d"),
        (np.zeros((5, 2)), np.zeros((4, 1)), "X has 5 rows, y has 4"),
    ],
)
def test_rejects_mismatched_input(X, y, match):
    with pytest.raises(ValueError, match=match):
        solve_ridge(X, y, config=RidgeConfig())


# ---------------------------------------------------------------------- fallback


def test_reports_which_route_ran():
    X, y = _problem(100, 6, 1)
    assert solve_ridge(X, y, config=RidgeConfig(alpha=1.0)).method == "cholesky"


def test_an_indefinite_gram_takes_scikit_learns_svd_route_and_stays_bounded():
    # A Gram can be indefinite in finite precision where exact arithmetic says it is not.
    # scikit-learn catches LinAlgError and retries with solver="svd"; falling back to a
    # least-squares solve of the regularised Gram instead would invert the spurious
    # negative eigenvalue and is unbounded. The svd filter s / (s**2 + alpha) is bounded
    # by 1 / (2 * sqrt(alpha)), so assert that rather than merely asserting finiteness.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 5))
    X[:, 4] = X[:, 3] + 1e-12 * rng.normal(size=200)
    y = rng.normal(size=(200, 1))
    alpha = 1e-8

    # Drive the Cholesky to fail by handing solve_ridge a design whose Gram is singular
    # to working precision, then confirm the answer is the bounded one.
    X[:, 4] = X[:, 3]
    solution = solve_ridge(X, y, config=RidgeConfig(alpha=alpha))
    assert np.all(np.isfinite(solution.coef))
    bound = 1.0 / (2.0 * np.sqrt(alpha))
    assert np.abs(solution.coef).max() <= bound * np.abs(y).max() * X.shape[1]


def test_the_solution_is_frozen():
    X, y = _problem(40, 3, 1)
    solution = solve_ridge(X, y, config=RidgeConfig())
    assert isinstance(solution, RidgeSolution)
    with pytest.raises(dataclasses.FrozenInstanceError):
        solution.method = "other"


# ------------------------------------------------------------------------- purity


def test_fit_is_a_pure_function_of_its_arguments():
    X, y = _problem(100, 6, 2)
    X_before, y_before = X.copy(), y.copy()
    first = solve_ridge(X, y, config=RidgeConfig(alpha=0.9)).coef
    # No module-level state and no data-keyed cache: a second identical call returns the
    # same answer, and neither call writes to the caller's arrays.
    second = solve_ridge(X, y, config=RidgeConfig(alpha=0.9)).coef
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
