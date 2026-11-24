"""Contract tests for the Ridge prototype. CPU-only except two Metal-gated cases.

These check the algebra and the reporting, not the timings. The prototype's published
figures were wrong four times, and in every case a test of the kind below would have
caught it or bounded it:

* an unfair baseline -> `test_all_backends_agree_in_float64` pins the four arms to each
  other, so a baseline that computes something different shows up as disagreement;
* a silent transpose -> `test_every_backend_returns_d_by_targets` at `targets != 1`,
  where a transposed result is no longer square and cannot hide;
* a crash where scikit-learn degrades -> `test_indefinite_gram_falls_back`;
* a conditioning sweep that moved `alpha` with `cond` ->
  `test_alpha_abs_is_independent_of_the_design_scale`.

Note: CI runs ``python -m pytest -q benchmarks/test_knn_*.py``, so this file is not picked
up by that glob. Run it directly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.ridge.preflight import Check, run_checks
from benchmarks.ridge.render_report import render
from benchmarks.ridge.ridge_prototype import (
    OBJECTIVE_RTOL,
    fit_ridge,
    gram_host,
    make_problem,
    ridge_objective,
    solve_normal_equations,
)

HOST_BACKENDS = ("host_gram", "sklearn_bare", "sklearn")


def _metal() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(getattr(mx, "metal", None) and mx.metal.is_available())


requires_metal = pytest.mark.skipif(not _metal(), reason="requires an MLX GPU")


# ------------------------------------------------------------------- make_problem


@pytest.mark.parametrize("cond", [1e2, 1e4, 1e8])
def test_make_problem_produces_the_requested_condition_number(cond):
    X, _ = make_problem(400, 20, 1, seed=0, cond=cond)
    s = np.linalg.svd(X, compute_uv=False)
    assert s[0] / s[-1] == pytest.approx(cond, rel=1e-6)


def test_make_problem_leaves_the_natural_scale_when_cond_is_one():
    # The `cond > 1.0` branch is skipped, so this row keeps the Gaussian scale while
    # every other row is normalised to sigma_max = 1. A sweep over cond therefore varies
    # the data scale too, which is why conditioning claims from it were withdrawn.
    X1, _ = make_problem(400, 20, 1, seed=0, cond=1.0)
    X2, _ = make_problem(400, 20, 1, seed=0, cond=1e4)
    assert np.linalg.svd(X1, compute_uv=False)[0] > 10.0
    assert np.linalg.svd(X2, compute_uv=False)[0] == pytest.approx(1.0)


def test_make_problem_is_deterministic_in_its_seed():
    a, ya = make_problem(100, 8, 2, seed=3)
    b, yb = make_problem(100, 8, 2, seed=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ya, yb)
    assert not np.array_equal(a, make_problem(100, 8, 2, seed=4)[0])


def test_make_problem_returns_float64_so_conversions_are_not_hidden():
    X, y = make_problem(50, 4, 1, seed=0)
    assert X.dtype == np.float64 and y.dtype == np.float64


# ------------------------------------------------------------------- orientation


@pytest.mark.parametrize("targets", [1, 8])
@pytest.mark.parametrize("backend", HOST_BACKENDS)
def test_every_backend_returns_d_by_targets(backend, targets):
    # At targets=1 a transposed (d, 1) vs (1, d) error is easy to miss; at targets=8 it
    # cannot hide, which is why both are checked.
    n, d = 200, 12
    X, y = make_problem(n, d, targets, seed=1)
    coef = fit_ridge(X, y, 0.5, backend=backend)
    assert coef.shape == (d, targets), f"{backend} returned {coef.shape}"


@requires_metal
@pytest.mark.parametrize("targets", [1, 8])
def test_metal_backend_returns_d_by_targets(targets):
    X, y = make_problem(200, 12, targets, seed=1)
    assert fit_ridge(X, y, 0.5, backend="metal_gram").shape == (12, targets)


# ------------------------------------------------------------------- agreement


@pytest.mark.parametrize("targets", [1, 4])
def test_all_backends_agree_in_float64(targets):
    # host_gram, sklearn_bare and sklearn solve the same ridge problem by the same
    # normal equations. Disagreement here means one arm is not solving what the others
    # are, which is the defect that made the first published speedups meaningless.
    X, y = make_problem(300, 16, targets, seed=2)
    alpha = 0.7
    ref = fit_ridge(X, y, alpha, backend="host_gram")
    for backend in ("sklearn_bare", "sklearn"):
        got = fit_ridge(X, y, alpha, backend=backend)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11,
                                   err_msg=f"{backend} disagrees with host_gram")


@requires_metal
def test_metal_gram_clears_the_objective_gate_against_float64():
    X, y = make_problem(2000, 32, 1, seed=4)
    alpha = 1e-3 * float(np.trace(gram_host(X, y)[0])) / X.shape[1]
    host = fit_ridge(X, y, alpha, backend="host_gram")
    metal = fit_ridge(X, y, alpha, backend="metal_gram")
    j_host = ridge_objective(X, y, host, alpha)
    j_metal = ridge_objective(X, y, metal, alpha)
    # The strong assertion from docs/fp32_policy.md section 3. Coefficients are the weak
    # one and are deliberately not compared.
    assert abs(j_metal - j_host) / abs(j_host) < OBJECTIVE_RTOL


# ------------------------------------------------------------------- the solve


def test_solve_normal_equations_matches_the_explicit_inverse():
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(80, 6)), rng.normal(size=(80, 2))
    G, C = gram_host(X, y)
    alpha = 0.25
    np.testing.assert_allclose(
        solve_normal_equations(G, C, alpha),
        np.linalg.inv(G + alpha * np.eye(6)) @ C,
        rtol=1e-9,
    )


def test_solve_normal_equations_does_not_mutate_its_gram():
    G, before = np.eye(4) * 3.0, np.eye(4) * 3.0
    solve_normal_equations(G, np.ones((4, 1)), 9.0)
    np.testing.assert_array_equal(G, before)


def test_indefinite_gram_falls_back_rather_than_raising():
    # A float32 Gram can be indefinite where the float64 one is not -- measured
    # lambda_min = -1.315e-08 at n=32768, d=512, cond=1e12. scikit-learn catches the same
    # failure and retries with solver="svd" (sklearn/linear_model/_ridge.py:748-752), so a
    # prototype that raises is not comparing like for like.
    G = np.diag([1.0, 1.0, -1e-9])          # not positive definite
    C = np.ones((3, 1))
    out = solve_normal_equations(G, C, 0.0)  # must not raise
    assert out.shape == (3, 1)
    assert np.all(np.isfinite(out))


def test_zero_alpha_is_accepted_and_is_plain_least_squares():
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(60, 5)), rng.normal(size=(60, 1))
    G, C = gram_host(X, y)
    np.testing.assert_allclose(
        solve_normal_equations(G, C, 0.0),
        np.linalg.lstsq(X, y, rcond=None)[0],
        rtol=1e-7,
    )


# ------------------------------------------------------------------- objective


def test_ridge_objective_matches_the_definition():
    rng = np.random.default_rng(7)
    X, y, coef = rng.normal(size=(40, 5)), rng.normal(size=(40, 2)), rng.normal(size=(5, 2))
    alpha = 0.6
    R = y - X @ coef
    assert ridge_objective(X, y, coef, alpha) == pytest.approx(
        float((R * R).sum()) + alpha * float((coef * coef).sum())
    )


def test_ridge_objective_is_minimised_by_the_ridge_solution():
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(90, 7)), rng.normal(size=(90, 2))
    alpha = 0.4
    best_coef = solve_normal_equations(*gram_host(X, y), alpha)
    best = ridge_objective(X, y, best_coef, alpha)
    for _ in range(12):
        nudged = best_coef + rng.normal(scale=1e-3, size=best_coef.shape)
        assert ridge_objective(X, y, nudged, alpha) >= best


def test_alpha_abs_is_independent_of_the_design_scale():
    # The withdrawn conditioning sweep used alpha = scale * trace(X'X)/d, and setting
    # `cond` rescales the spectrum, so alpha moved 1.6-million-fold across it. An absolute
    # alpha must not move with the design.
    for cond in (1.0, 1e6):
        X, y = make_problem(300, 16, 1, seed=9, cond=cond)
        coef = solve_normal_equations(*gram_host(X, y), 1.0)
        assert np.all(np.isfinite(coef))
    trace_based = [
        float(np.trace(gram_host(*make_problem(300, 16, 1, seed=9, cond=c))[0]))
        for c in (1.0, 1e6)
    ]
    assert trace_based[0] / trace_based[1] > 1e3, "the trace rule really does move with cond"


# ------------------------------------------------------------------- preflight


def test_preflight_checks_are_well_formed():
    checks = run_checks()
    names = {c.name for c in checks}
    assert {"blas", "metal", "source", "load", "thermal"} <= names
    for c in checks:
        assert isinstance(c, Check)
        assert isinstance(c.ok, bool) and isinstance(c.blocking, bool)
        assert c.value, f"{c.name} reported an empty value"


def test_blas_and_metal_are_the_blocking_checks():
    blocking = {c.name for c in run_checks() if c.blocking}
    assert blocking == {"blas", "metal"}


# ------------------------------------------------------------------- rendering


def _record(**kw):
    base = {
        "label": "x", "shape": [1, 1], "n": 8192, "d": 64, "cond": 1.0, "alpha": 1.0,
        "objective_rel": 1e-9, "coef_rel": 1e-7, "speedup": 2.0, "speedup_best": 2.0,
        "significant": True, "baseline": "sklearn_bare",
        "mlxlearn": {"median_s": 0.001, "min_s": 0.001, "max_s": 0.001, "iqr_s": 0.0, "iterations": 7},
        "sklearn": {"median_s": 0.002, "min_s": 0.002, "max_s": 0.002, "iqr_s": 0.0, "iterations": 7},
    }
    base.update(kw)
    return base


def _write(tmp_path, records, **extra):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"environment": {}, "results": records, **extra}))
    return p


def test_render_reports_a_crossover_when_losses_and_wins_separate(tmp_path):
    recs = [_record(d=64, speedup=0.2), _record(d=128, speedup=0.5),
            _record(d=512, speedup=2.0), _record(d=1024, speedup=3.0)]
    text = "\n".join(render(_write(tmp_path, recs)))
    assert "crossover is between d=128 and d=256" not in text  # 256 is not in this grid
    assert "crossover is between d=128 and d=512" in text


def test_render_does_not_quote_a_range_spanning_losses_and_wins(tmp_path):
    recs = [_record(d=64, speedup=0.2), _record(d=1024, speedup=3.0)]
    text = "\n".join(render(_write(tmp_path, recs)))
    assert "0.20x-3.00x" not in text and "0.20x–3.00x" not in text


def test_render_reports_a_plain_range_when_every_cell_wins(tmp_path):
    recs = [_record(d=512, speedup=1.5), _record(d=1024, speedup=3.0)]
    text = "\n".join(render(_write(tmp_path, recs)))
    assert "1.50x-3.00x" in text


def test_render_refuses_to_claim_anything_when_nothing_is_significant(tmp_path):
    recs = [_record(d=512, speedup=2.0, significant=False)]
    text = "\n".join(render(_write(tmp_path, recs)))
    assert "No cell is statistically significant" in text


def test_render_marks_a_non_accelerate_blas_as_untrustworthy(tmp_path):
    from benchmarks.ridge.render_report import _blas

    assert "accelerate" in _blas({}, [{"name": "blas", "value": "accelerate", "ok": True}])
    bad = _blas({}, [{"name": "blas", "value": "openblas", "ok": False}])
    assert "not trustworthy" in bad


def test_render_says_so_when_an_artifact_records_no_blas(tmp_path):
    from benchmarks.ridge.render_report import _blas

    assert "not recorded" in _blas({}, None)
