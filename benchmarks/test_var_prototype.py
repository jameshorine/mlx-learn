"""Contract tests for the VAR(p) prototype. CPU-only except one Metal-gated case.

These check the algebra the benchmark depends on, not its timings: that the block-Hankel
design is the one the VAR(p) definition implies, that ridge-by-augmentation is exact,
that the companion matrix has the structure the stability check assumes, and that the
synthetic generator produces a stable process.

Note: CI runs ``python -m pytest -q benchmarks/test_knn_*.py``, so this file is not
picked up by that glob. Run it directly, or widen the glob if the VAR work lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.var.var_prototype import (
    OBJECTIVE_RTOL,
    companion_matrix,
    companion_spectral_radius,
    default_rcond,
    gram_host,
    lag_design,
    load_tsf,
    ridge_augment,
    ridge_objective,
    solve_normal_equations,
    synthetic,
)


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(getattr(mx, "metal", None) and mx.metal.is_available())


requires_metal = pytest.mark.skipif(not _metal_available(), reason="requires an MLX GPU")


# --------------------------------------------------------------------------- lag_design


def test_lag_design_shapes_follow_d_equals_kp_plus_one():
    Y = np.arange(40.0).reshape(10, 4)
    for p in (1, 2, 3):
        Z, T = lag_design(Y, p)
        assert Z.shape == (10 - p, 4 * p + 1)
        assert T.shape == (10 - p, 4)


def test_lag_design_places_most_recent_lag_first():
    # Distinct rows make the ordering unambiguous.
    Y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    Z, T = lag_design(Y, 2)
    # Row 0 predicts y_2 from [1, y_1, y_0].
    np.testing.assert_allclose(T[0], Y[2])
    np.testing.assert_allclose(Z[0], [1.0, *Y[1], *Y[0]])
    np.testing.assert_allclose(Z[1], [1.0, *Y[2], *Y[1]])


def test_lag_design_without_intercept_drops_the_constant_column():
    Y = np.arange(24.0).reshape(6, 4)
    Z, _ = lag_design(Y, 2, intercept=False)
    assert Z.shape == (4, 8)
    assert not np.all(Z[:, 0] == 1.0)


def test_lag_design_recovers_known_coefficients_from_a_noiseless_path():
    rng = np.random.default_rng(0)
    k, p, n = 3, 2, 400
    A1 = np.array([[0.4, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.2]])
    A2 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    c = np.array([0.5, -0.25, 1.0])
    Y = np.zeros((n, k))
    Y[:p] = rng.normal(size=(p, k))
    for t in range(p, n):
        Y[t] = c + A1 @ Y[t - 1] + A2 @ Y[t - 2]

    Z, T = lag_design(Y, p)
    G, C = gram_host(Z, T)
    Bt = solve_normal_equations(G, C, 0.0)
    # Recovery is exact in exact arithmetic, so this checks the design, not the solver.
    # The tolerance is 1e-5 rather than machine epsilon because the normal equations
    # square the condition number: forming Z'Z costs about half the available digits
    # even in float64. Measured residual here is ~1.7e-6, and a tighter bound would be
    # asserting that a Gram route is something it structurally is not.
    np.testing.assert_allclose(Bt[0], c, atol=1e-5)
    np.testing.assert_allclose(Bt[1:].T, np.hstack([A1, A2]), atol=1e-5)


@pytest.mark.parametrize(
    ("Y", "p", "match"),
    [
        (np.zeros((5,)), 1, "must be 2-d"),
        (np.zeros((5, 2)), 0, "p must be >= 1"),
        (np.zeros((3, 2)), 3, "need n > p"),
    ],
)
def test_lag_design_rejects_bad_input(Y, p, match):
    with pytest.raises(ValueError, match=match):
        lag_design(Y, p)


# ------------------------------------------------------------------------ ridge augment


def test_ridge_augmentation_equals_the_explicit_ridge_solution():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(120, 7))
    T = rng.normal(size=(120, 3))
    lam = 0.37

    Za, Ta = ridge_augment(Z, T, lam)
    assert Za.shape == (120 + 7, 7)
    assert Ta.shape == (120 + 7, 3)
    augmented = np.linalg.lstsq(Za, Ta, rcond=None)[0]

    G, C = gram_host(Z, T)
    explicit = solve_normal_equations(G, C, lam)
    np.testing.assert_allclose(augmented, explicit, rtol=1e-10, atol=1e-12)


def test_ridge_augmentation_rejects_a_negative_penalty():
    with pytest.raises(ValueError, match="non-negative"):
        ridge_augment(np.zeros((4, 2)), np.zeros((4, 1)), -1.0)


# ---------------------------------------------------------------------------- companion


def test_companion_matrix_has_coefficients_on_top_and_identities_below():
    k, p = 2, 3
    B = np.arange(float(k * k * p)).reshape(k, k * p)
    F = companion_matrix(B, k, p)
    assert F.shape == (k * p, k * p)
    np.testing.assert_allclose(F[:k, :], B)
    np.testing.assert_allclose(F[k:, : k * (p - 1)], np.eye(k * (p - 1)))
    np.testing.assert_allclose(F[k:, k * (p - 1) :], 0.0)


def test_companion_matrix_at_p_equals_one_is_the_coefficient_block():
    B = np.array([[0.5, 0.1], [0.0, 0.2]])
    np.testing.assert_allclose(companion_matrix(B, 2, 1), B)


def test_companion_matrix_rejects_a_mis_shaped_block():
    with pytest.raises(ValueError, match=r"must be \(k, k\*p\)"):
        companion_matrix(np.zeros((2, 5)), 2, 3)


def test_companion_spectral_radius_separates_stable_from_explosive():
    stable = np.array([[0.5, 0.0], [0.0, 0.25]])
    assert companion_spectral_radius(stable, 2, 1) == pytest.approx(0.5)

    explosive = np.array([[1.5, 0.0], [0.0, 0.25]])
    assert companion_spectral_radius(explosive, 2, 1) > 1.0

    # A scalar AR(2) y_t = 0.5 y_{t-1} + 0.2 y_{t-2}: roots of z^2 - 0.5z - 0.2.
    ar2 = np.array([[0.5, 0.2]])
    expected = max(abs(np.roots([1.0, -0.5, -0.2])))
    assert companion_spectral_radius(ar2, 1, 2) == pytest.approx(expected)


# ---------------------------------------------------------------------------- numerics


def test_solve_normal_equations_matches_the_explicit_inverse():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(60, 5))
    T = rng.normal(size=(60, 2))
    G, C = gram_host(Z, T)
    lam = 0.1
    expected = np.linalg.inv(G + lam * np.eye(5)) @ C
    np.testing.assert_allclose(solve_normal_equations(G, C, lam), expected, rtol=1e-9)


def test_solve_normal_equations_does_not_mutate_its_gram():
    G = np.eye(3) * 2.0
    before = G.copy()
    solve_normal_equations(G, np.ones((3, 1)), 5.0)
    np.testing.assert_allclose(G, before)


def test_ridge_objective_matches_the_definition():
    rng = np.random.default_rng(3)
    Z, T = rng.normal(size=(30, 4)), rng.normal(size=(30, 2))
    Bt, lam = rng.normal(size=(4, 2)), 0.75
    R = T - Z @ Bt
    expected = float((R * R).sum()) + lam * float((Bt * Bt).sum())
    assert ridge_objective(Z, T, Bt, lam) == pytest.approx(expected)


def test_ridge_objective_is_minimised_by_the_ridge_solution():
    rng = np.random.default_rng(4)
    Z, T = rng.normal(size=(80, 6)), rng.normal(size=(80, 2))
    lam = 0.5
    G, C = gram_host(Z, T)
    Bt = solve_normal_equations(G, C, lam)
    best = ridge_objective(Z, T, Bt, lam)
    for _ in range(10):
        perturbed = Bt + rng.normal(scale=1e-3, size=Bt.shape)
        assert ridge_objective(Z, T, perturbed, lam) >= best


def test_default_rcond_is_scikit_learns_cutoff_not_the_olsconfig_default():
    rcond = default_rcond((26303, 322), np.dtype(np.float64))
    assert rcond == pytest.approx(26303 * np.finfo(np.float64).eps)
    assert rcond < 1e-6  # the OLSConfig default, which is far coarser


# ---------------------------------------------------------------------------- synthetic


@pytest.mark.parametrize("p", [1, 2, 3])
def test_synthetic_is_always_stable(p):
    # Regression test: rho(F) is not linear in the scale of B, so a single division by
    # rho does not stabilise p > 1 and produced an explosive path that overflowed float32.
    Y = synthetic(n=300, k=5, p=p, seed=7)
    assert Y.shape == (300, 5)
    assert np.all(np.isfinite(Y))
    Z, T = lag_design(Y, p)
    G, C = gram_host(Z, T)
    B = solve_normal_equations(G, C, 0.0)[1:].T
    assert companion_spectral_radius(B, 5, p) < 1.0


def test_synthetic_is_deterministic_in_its_seed():
    a = synthetic(n=50, k=3, p=1, seed=11)
    b = synthetic(n=50, k=3, p=1, seed=11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, synthetic(n=50, k=3, p=1, seed=12))


# ---------------------------------------------------------------------------- load_tsf


def _write_tsf(path, rows):
    path.write_text("@relation x\n@data\n" + "".join(rows))


def test_load_tsf_reads_series_as_columns(tmp_path):
    f = tmp_path / "ok.tsf"
    _write_tsf(f, ["s1:2012-01-01 00-00-00:1,2,3\n", "s2:2012-01-01 00-00-00:4,5,6\n"])
    Y = load_tsf(str(f))
    assert Y.shape == (3, 2)
    np.testing.assert_allclose(Y[:, 0], [1, 2, 3])
    np.testing.assert_allclose(Y[:, 1], [4, 5, 6])


def test_load_tsf_rejects_ragged_series_rather_than_padding(tmp_path):
    f = tmp_path / "ragged.tsf"
    _write_tsf(f, ["s1:t:1,2,3\n", "s2:t:4,5\n"])
    with pytest.raises(ValueError, match="ragged"):
        load_tsf(str(f))


def test_load_tsf_rejects_a_file_with_no_data_rows(tmp_path):
    f = tmp_path / "empty.tsf"
    f.write_text("@relation x\n@data\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_tsf(str(f))


# ------------------------------------------------------------------------------- Metal


@requires_metal
def test_metal_gram_agrees_with_the_host_gram_on_the_objective():
    from benchmarks.var.var_prototype import gram_metal

    rng = np.random.default_rng(5)
    Y = rng.normal(size=(4000, 24))
    Z, T = lag_design(Y, 2)
    lam = 1e-6 * float(np.trace(gram_host(Z, T)[0])) / Z.shape[1]

    G64, C64 = gram_host(Z, T)
    G32, C32 = gram_metal(Z, T)
    assert G32.shape == G64.shape and C32.shape == C64.shape

    host = solve_normal_equations(G64, C64, lam)
    metal = solve_normal_equations(G32, C32, lam)
    j_host = ridge_objective(Z, T, host, lam)
    j_metal = ridge_objective(Z, T, metal, lam)
    # The strong assertion from docs/fp32_policy.md section 3; the coefficients
    # themselves are the weak one and are not compared here.
    assert abs(j_metal - j_host) / abs(j_host) < OBJECTIVE_RTOL
