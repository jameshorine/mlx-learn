# VAR(p) benchmarks

**A VAR(1) on 321 electricity series over 26,304 hours fits in 80.6 ms against
scikit-learn's 140.8 ms — 1.75x — solving the identical ridge normal equations, with the
objective agreeing to 8.7e-08 against a 1e-5 gate. At p=2 it is 2.05x. From p=3 the
float32 Gram stops clearing the gate, so the usable envelope on this data is p <= 2.**

A prototype, not part of the gated suite. It exists to answer one question before any
estimator is written: a vector autoregression reduces to multi-output least squares, so
*which* solve path wins, and over what range of lag orders does it stay inside the
tolerances in [`docs/fp32_policy.md`](../../docs/fp32_policy.md) section 3?

| File | Purpose |
|---|---|
| `var_prototype.py` | Design construction, six solve backends, objective-gated comparison |
| [`../test_var_prototype.py`](../test_var_prototype.py) | Contract tests for the algebra (26 cases) |

CI runs `python -m pytest -q benchmarks/test_knn_*.py`, so the tests here are not picked
up by that glob. Run them directly.

## The reduction

A VAR(p) on ``k`` variables over ``n`` periods is

```
y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t
```

with ``y_t`` in ``R^k``, each ``A_i`` in ``R^(k x k)``, and ``E[e_t e_t'] = Sigma``.
Stacking the lags into one regressor vector and the coefficients into one matrix,

```
z_t = [1, y_{t-1}', ..., y_{t-p}']'   in R^(kp+1)
B   = [c, A_1, ..., A_p]              in R^(k x (kp+1))
```

gives ``y_t = B z_t + e_t``, and stacking rows over ``t = p+1 ... n`` gives a plain
multi-output linear model:

```
Y = Z B' + E          Z is (n-p) x (kp+1),  B' is (kp+1) x k,  Y is (n-p) x k
B' = (Z'Z + lam I)^-1 Z'Y
```

The design width is ``d = kp + 1``. That is the whole reason the published
`LinearRegression` crossover grid
([`docs/linear-regression-2026-11-22.md`](../../docs/linear-regression-2026-11-22.md))
measures this computation directly, and why AR(p) — the ``k = 1`` case, ``d = p + 1`` —
sits in the columns that never win.

**The companion form is not an estimation route.** A VAR(p) can be rewritten as a VAR(1)
on the stacked state ``Y_t = [y_t', ..., y_{t-p+1}']'`` with transition

```
F = [ A_1  A_2  ...  A_{p-1}  A_p ]
    [  I    0   ...     0      0  ]
    [  0    I   ...     0      0  ]
    [             ...             ]
    [  0    0   ...     I      0  ]        F is (kp) x (kp)
```

but only ``F``'s first block row is free: the rest are fixed identity and zero blocks, so
an unrestricted VAR(1) fit on the stacked state would estimate ``(kp)^2`` parameters where
the model has ``k^2 p``. ``F`` is used here only for the stability check ``rho(F) < 1``.

## Why the Gram is built in float32 on the device

`_solvers/ols.py` is certified float64, and float64 does not execute on Metal
(`float64 is not supported on the GPU`), so that path can never use the device. Splitting
the work as `_solvers/pca.py` splits `covariance_eigh` — build ``Z'Z`` and ``Z'Y`` in
float32 on the device, add the ridge diagonal and factor in float64 on the host — puts the
``O(n d^2)`` term on the GPU and leaves only the ``O(d^3)`` factorization on the host. That
factorization is under 1% of the flops at every shape measured here.

## Backends

| Backend | What runs |
|---|---|
| `metal_gram` | ``Z'Z``, ``Z'Y`` via `_binary32.float32_matmul`, then host float64 Cholesky |
| `host_gram` | the same normal equations entirely in host float64 |
| `solve_ols` | `_solvers.ols.solve_ols`, with ridge by design augmentation |
| `sklearn_ridge` | `sklearn.linear_model.Ridge(solver="cholesky")` — the fair baseline |
| `sklearn_ols` | `LinearRegression`; ignores ``lam``, so its objective is reported `n/a` |
| `lstsq` | `numpy.linalg.lstsq(rcond=1e-15)` — what statsmodels' `VAR` calls |

Ridge matters: at ``k=321, p=1`` the fit has 103,362 parameters. `OLSConfig` has no
penalty hook, so `solve_ols` gets ridge by augmenting with ``sqrt(lam) * I_d`` rows and
zero targets, which returns ``(Z'Z + lam I)^-1 Z'Y`` exactly. The Gram backends add
``lam`` to the diagonal directly.

`rcond` matters too. `OLSConfig` defaults to ``1e-6``; scikit-learn 1.7 and 1.8 cut at
``max(n, d) * eps``, about ``5.8e-12`` here. On this design, whose ``cond(Z) = 1.18e7``
and smallest singular ratio is ``8.5e-08``, the coarser cutoff truncates a direction stock
keeps, reports rank 321 of 322, and shifts the minimum-norm solution by about 10%.

## Quick start

```bash
python -m benchmarks.var.var_prototype --synthetic --k 321 --n 26304 --p 1
python -m benchmarks.var.var_prototype --tsf electricity_hourly_dataset.tsf --p 1
python -m benchmarks.var.var_prototype --tsf electricity_hourly_dataset.tsf --sweep 1 2 3 5
```

## What has been measured

Electricity-hourly (321 series, 26,304 hours; Monash `electricity_hourly`), ridge at
``lam = 1e-6 * trace(Z'Z) / d``, on an **M1 Pro (8P+2E, 32 GiB)**, MLX 0.32.4,
scikit-learn 1.7.2, SciPy 1.16.3, NumPy 2.3.5. Min of 3 with warmup.

**Timed region**: the solve only. ``lag_design`` is built once outside it — it copies
67.8 MB at ``p=1`` and is identical for every backend, so timing it would flatten the
ratios and measure NumPy's copy. The host-to-device upload **is** inside `metal_gram`'s
interval.

| p | d | `metal_gram` | `host_gram` | `sklearn_ridge` | `solve_ols` | vs sklearn | objective rel | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 322 | **80.6 ms** | 124.1 | 140.8 | 1625.6 | **1.75x** | 8.71e-08 | pass |
| 2 | 643 | **128.8 ms** | 274.6 | 264.4 | 2642.9 | **2.05x** | 4.03e-06 | pass |
| 3 | 964 | 187.0 ms | 417.6 | 425.7 | 4046.6 | 2.28x | 1.59e-05 | **FAIL** |
| 5 | 1606 | 290.2 ms | 765.1 | 942.1 | 9923.8 | 3.25x | 5.27e-04 | **FAIL** |

Three things to read off it.

**Speed improves with p; accuracy does not.** `metal_gram` gains on both host paths as the
design widens — 1.54x to 2.64x over `host_gram` — because the Gram grows as ``O(n d^2)``
while the host Cholesky it leaves behind grows only as ``O(d^3)``, still under 1% of the
work at ``d = 1606``. But the objective error grows roughly sixfold per lag and crosses the
``1e-5`` gate at ``p = 3``. Consecutive lags of the same series are nearly collinear, so
each lag block worsens ``cond(Z'Z)`` while adding more float32 accumulation. The usable
envelope here is ``p <= 2``.

**`solve_ols` is the wrong path for this problem**, at 0.08x–0.10x. It is certified
float64, so it cannot reach the GPU, and the ridge augmentation adds ``d`` rows to a
solver that then certifies the rank of all of them.

**The accuracy failure is in the objective, which is the assertion that matters.** The
coefficients fail their own (weaker) tolerance earlier still — at ``cond(Z'Z) = 1.38e14``
they are not pinned down by the data at all. `fp32_policy.md` section 3 states the
principle directly: "Coefficients are the weak assertion; objective value is the strong
one... two solvers can reach very different coefficients that are equally good."

## What is not established

- One dataset, one shape, one machine, one seed. Nothing here is a crossover.
- Timings are min-of-3 with warmup but **not interleaved** and with no thermal gating, so
  they do not meet the bar in [`docs/development.md`](../../docs/development.md). Observed
  run-to-run spread on the scikit-learn side is around 25%.
- An earlier revision of this file reported 8.0x at ``p=1``. That number excluded the
  host-to-device upload (9.3 ms of the 15.4 ms Gram cost) and compared against a bare
  normal-equations solve rather than `Ridge`. With the upload inside the timed region and
  `Ridge` as the baseline it is 1.75x. Treat any figure here as provisional until it is
  produced by the interleaved harness.
- Whether ``Sigma`` and the ``T - Kp - 1`` degrees-of-freedom correction survive a float32
  Gram is untested. Both inherit coefficient error, which fails above, rather than
  objective error, which passes.
- Whether a compensated or mixed-precision Gram (`mlxlearn_device/_compensated.py`) would
  push the envelope past ``p = 2``.
- Whether any of this reproduces on the Ultra hardware the published grid used, where
  `ols.py` already wins and the host path may simply be the better answer.

## Scope

Vector autoregression has no scikit-learn counterpart, so none of this can be registered
in `patching/_registry.py`, and gate 6 has no stock estimator to pair against. See issue
[#4](https://github.com/jameshorine/mlx-learn/issues/4) — the scope question is the
blocker, not the performance case.
