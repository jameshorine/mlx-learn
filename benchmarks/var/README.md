# VAR(p) benchmarks

**A VAR(1) on 321 electricity series over 26,304 hours fits in 92.9 ms against
scikit-learn's 162.5 ms — 1.75x, significant — solving the identical ridge normal
equations, with the objective agreeing to 8.7e-08 against a 1e-5 gate. At p=2 it is 2.29x.
From p=3 the float32 Gram stops clearing the gate, so the usable envelope on this data is
p <= 2.** Separately, fitting one independent AR(p) per series as a single batched Gram
runs 1.6x-2.1x a host loop; that is the only AR shape that wins at all.

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
``lam = 1e-6 * trace(Z'Z) / d``, on an **Apple M1 Pro, 10 cores, 32 GiB**, MLX 0.32.4 on
GPU, scikit-learn 1.7.2, SciPy 1.16.3, NumPy 2.3.5.

Timing goes through [`benchmarks/_harness.py`](../_harness.py)'s `compare`: two warmup
pairs, then seven rounds alternating the two sides inside one gc-disabled window, reported
as medians with a spread and a significance test on the median's own uncertainty. Raw
records, including the environment block:

- [`benchmarks/results/var-ecl-2026-11-24.json`](../results/var-ecl-2026-11-24.json)
- [`benchmarks/results/batched-ar-ecl-2026-11-24.json`](../results/batched-ar-ecl-2026-11-24.json)

Reproduce with:

```bash
python -m benchmarks.var.var_prototype --tsf electricity_hourly_dataset.tsf \
    --sweep 1 2 3 --backends metal_gram host_gram sklearn_ridge \
    --output benchmarks/results/var-ecl-<date>.json
```

**Timed region**: the solve only. `lag_design` is built once outside it — it copies
67.8 MB at ``p=1`` and is identical for every backend, so timing it would flatten the
ratios. The host-to-device upload **is** inside the Metal interval.

### VAR(p), against `sklearn.Ridge(solver="cholesky")`

| p | d | `metal_gram` median | `sklearn_ridge` median | median | best | significant | objective rel | gate |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 1 | 322 | 92.9 ms | 162.5 ms | **1.75x** | 1.94x | yes | 8.71e-08 | pass |
| 2 | 643 | 149.3 ms | 341.3 ms | **2.29x** | 1.98x | yes | 4.03e-06 | pass |
| 3 | 964 | 217.2 ms | 504.7 ms | 2.32x | 2.44x | yes | 1.59e-05 | **FAIL** |

`host_gram` — the same normal equations entirely in host float64 — is **not significant**
against scikit-learn at any of the three (1.20x-1.29x, inside the noise). The win is
specifically the Metal Gram, not normal-equations-versus-`Ridge`.

### Batched AR(p): one independent model per series, against a host loop

| p | d | Metal batched median | host loop median | median | best | significant | coef rel |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 2 | 52.6 ms | 87.7 ms | **1.67x** | 1.85x | yes | 1.59e-04 |
| 5 | 6 | 75.0 ms | 123.1 ms | **1.64x** | 1.66x | yes | 2.66e-04 |
| 10 | 11 | 99.8 ms | 207.4 ms | **2.08x** | 2.12x | yes | 4.91e-04 |
| 20 | 21 | 164.0 ms | 319.5 ms | **1.95x** | 2.00x | yes | 6.08e-03 |

A single AR(p) fit does not win and cannot: at ``k=1`` the design is ``d = p+1``, the whole
problem is 0.2-23 Mflop — three to four orders below where PCA's crossover sits — and Metal
measures 0.24x at p=1 rising only to 0.96x at p=20, never crossing 1. Batching 321 of them
into one Gram is what makes the work GPU-sized. Note the coefficient error at ``p=20``
(6.08e-03) exceeds the ``atol=2e-3, rtol=5e-3`` tolerance in `docs/fp32_policy.md`.

### Reading these

`solve_ols` is the wrong path for this problem, at 0.08x-0.10x: it is certified float64,
float64 does not execute on Metal, so it cannot reach the device, and ridge by design
augmentation adds ``d`` rows whose rank it then certifies.

Speed improves with ``p`` while accuracy degrades. The Gram grows ``O(n d^2)`` while the
host Cholesky it leaves behind grows ``O(d^3)`` and stays under 1% of the flops, but
consecutive lags of the same series are nearly collinear, so each lag block worsens
``cond(Z'Z)`` while adding more float32 accumulation.

The objective is the assertion that matters. At ``cond(Z'Z) = 1.38e14`` the coefficients
are not pinned down by the data at all — `fp32_policy.md` section 3: "Coefficients are the
weak assertion; objective value is the strong one... two solvers can reach very different
coefficients that are equally good."

## What is not established

- One dataset, one shape, one machine, one seed. Nothing here is a crossover.
- **Run-to-run variance is large even interleaved.** Two consecutive runs of the same
  sweep gave 1.81x and 1.75x at ``p=1``, and 2.62x and 2.29x at ``p=2`` — roughly 10-15%
  on the ratio. Treat the third digit as noise.
- Earlier revisions of this file reported min-of-3 timings taken without interleaving.
  On this machine that underestimated the median by 16-30%, and unevenly between the two
  sides, which biases the ratio rather than only the absolutes. One figure was badly
  wrong: batched AR was reported at 3.7x-11.5x against a host loop and 6.6x-22x against
  scikit-learn, the latter comparing min-of-3 against a **single** scikit-learn run. The
  interleaved numbers above are 1.6x-2.1x. Every figure in this file now comes from the
  committed JSON rather than from a transcription.
- Whether ``Sigma`` and the ``T - Kp - 1`` degrees-of-freedom correction survive a float32
  Gram is untested. Both inherit coefficient error, which already fails its tolerance at
  batched ``p=20``, rather than objective error, which passes.
- Whether a compensated or mixed-precision Gram (`mlxlearn_device/_compensated.py`) would
  push the VAR envelope past ``p = 2``.
- Whether any of this reproduces on the Ultra hardware the published grid used, where
  `ols.py` already wins and the host path may simply be the better answer.

## Scope

Vector autoregression has no scikit-learn counterpart, so none of this can be registered
in `patching/_registry.py`, and gate 6 has no stock estimator to pair against. See the
VAR scope issue (#4): the scope question is the blocker, not the performance case.
