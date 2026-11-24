# Ridge benchmarks

**`Ridge(solver="cholesky")` with the Gram built in float32 on Metal and factored in
float64 on the host runs 1.6x-5.3x stock scikit-learn across `d` in 64..1024 at
`n` in 8192..32768, with the objective agreeing to 4e-11..6e-8 against a 1e-5 gate.
Speed is independent of conditioning; accuracy is not — the gate holds comfortably to
`cond(X) <= 1e4` and goes marginal from `cond(X) >= 1e8`.**

A prototype, not part of the gated suite.

| File | Purpose |
|---|---|
| `ridge_prototype.py` | Shape and conditioning sweeps against `sklearn.linear_model.Ridge` |

Self-contained on purpose. A sibling VAR(p) prototype uses the same three Gram helpers,
but Ridge should not depend on it: Ridge has a scikit-learn counterpart and a roadmap
entry, while VAR has neither and is blocked on a scope decision. If both ship, the helpers
belong in one shared module.

## Why Ridge and not LinearRegression

`LinearRegression` shipped at 0.1.7 with **no Metal path** — `344bfdf`: "Metal has no
float64 and the float32 variants were inaccurate or no faster." That verdict was reached
for *unregularised* least squares, which must reproduce `lstsq`'s minimum-norm solution
plus `rank_` and `singular_`, and therefore needs an SVD rather than a Gram.

Ridge is a different problem for a reason that is easy to miss: `alpha > 0` makes
`X'X + alpha I` positive definite by construction, bounding its condition number by
`(sigma_max^2 + alpha) / alpha`. The rank-deficiency machinery that forces
`LinearRegression` onto an SVD does not apply, so a Gram route is contract-correct here
where it is not there.

scikit-learn agrees about the algorithm: `resolve_solver_for_numpy` returns `"cholesky"`
for dense input with `positive=False`, and `_solve_cholesky` is

```
A = X'X ; Xy = X'y ; A.flat[::d+1] += alpha ; solve(A, Xy, assume_a="pos")
```

So this measures the same computation on both sides; only where the Gram is formed differs.

## Quick start

```bash
python -m benchmarks.ridge.ridge_prototype --n 8192 32768 --d 64 128 256 512 1024 \
    --output benchmarks/results/ridge-grid-<date>.json
python -m benchmarks.ridge.ridge_prototype --n 32768 --d 256 --cond 1 1e4 1e8 1e12
```

## What has been measured

Apple M1 Pro, 10 cores, 32 GiB, MLX 0.32.4 on GPU, scikit-learn 1.7.2, SciPy 1.16.3,
NumPy 2.3.5. Timing through [`../_harness.py`](../_harness.py)'s `compare`: two warmup
pairs then seven interleaved rounds, medians with spread and a significance test.
Raw records: [`../results/ridge-grid-2026-11-24.json`](../results/ridge-grid-2026-11-24.json),
[`../results/ridge-cond-2026-11-24.json`](../results/ridge-cond-2026-11-24.json).

### Shape grid, `alpha = 1e-6 * trace(X'X)/d`, well-conditioned, single target

| n | d | Metal | sklearn | median | best | significant | objective rel | coef rel |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8,192 | 64 | 2.1 ms | 6.1 | **2.87x** | 1.69x | yes | 1.03e-10 | 3.51e-07 |
| 8,192 | 128 | 5.8 | 9.4 | 1.61x | 1.95x | no | 2.77e-10 | 2.10e-07 |
| 8,192 | 256 | 9.6 | 17.3 | 1.79x | 2.09x | no | 7.37e-10 | 4.28e-07 |
| 8,192 | 512 | 17.2 | 46.6 | **2.71x** | 2.78x | yes | 2.36e-09 | 2.55e-07 |
| 8,192 | 1024 | 47.5 | 130.6 | **2.75x** | 2.73x | yes | 2.31e-08 | 6.31e-07 |
| 32,768 | 64 | 4.8 | 14.7 | 3.07x | 2.23x | no | 4.35e-11 | 1.35e-07 |
| 32,768 | 128 | 5.8 | 30.5 | **5.28x** | 4.96x | yes | 1.01e-10 | 1.05e-07 |
| 32,768 | 256 | 16.0 | 59.2 | **3.69x** | 2.97x | yes | 2.54e-10 | 1.86e-07 |
| 32,768 | 512 | 44.0 | 176.9 | **4.02x** | 3.82x | yes | 2.01e-09 | 3.37e-07 |
| 32,768 | 1024 | 102.3 | 420.8 | **4.11x** | 5.14x | yes | 5.86e-08 | 1.08e-06 |

Every cell wins. Coefficients agree to 1e-7..1e-6, far inside the `atol=2e-3, rtol=5e-3`
in [`docs/fp32_policy.md`](../../docs/fp32_policy.md) section 3 — so here, even the weak
assertion holds.

### Conditioning, n=32,768, d=256

| cond(X) | median | objective rel | gate | coef rel |
|---:|---:|---:|---|---:|
| 1 | 4.54x | 2.54e-10 | pass | 1.86e-07 |
| 1e4 | 4.23x | 9.99e-07 | pass | 3.26e-02 |
| 1e8 | 5.34x | 1.47e-05 | **FAIL** | 3.20e-01 |
| 1e12 | 5.13x | 7.18e-06 | pass | 4.20e-01 |

**Speed does not depend on conditioning** (4.2x-5.3x throughout) — it is the same GEMM
either way. **Accuracy does.** The objective clears the gate by four orders at `cond = 1`,
is marginal around it from `cond >= 1e8`, and the 1e8/1e12 ordering is non-monotonic, which
says those two sit inside the noise of the gate rather than on a clean trend.

Coefficients degrade from 1.9e-07 to 4.2e-01. That is the data, not the arithmetic: at
`cond = 1e12` the coefficients are not determined, which is exactly the case
`fp32_policy.md` has in mind when it calls them the weak assertion.

Multi-target behaves the same: at `t=64`, `cond=1` gives 3.81x and `2.37e-10`; `cond=1e8`
gives 4.58x and `9.60e-06`, just inside the gate.

## What this implies for an estimator

Unlike VAR, Ridge needs no scope decision. It is on the roadmap (`README.md`, "planned,
0.2.x"), `sklearn.linear_model.Ridge` exists, so it can be registered in
`patching/_registry.py`, and `benchmarks/run.py::_runner` has a real `sk_fn` for gate 6.

The capability boundary the data suggests: serve `solver="cholesky"` (and `"auto"`, which
resolves to it for dense `positive=False`) on well-to-moderately conditioned designs, and
route the rest to the host or to stock. A dispatch rule would need a cheap conditioning
estimate rather than the shape alone, which is a difference from the existing
`_below_crossover` methods and is the main open design question.

## What is not established

- One machine, one seed, synthetic Gaussian designs only. No real dataset has been run.
- Timings are interleaved medians but have no thermal gating, and run-to-run variance
  elsewhere in this tree has been 10-15% on the ratio. Three of ten grid cells are not
  significant.
- `fit_intercept=True` is not measured; the prototype fits without an intercept.
- `sample_weight`, sparse input, and the other `solver` values are not implemented.
- Where exactly the accuracy boundary sits between `cond = 1e4` and `cond = 1e8`, and
  whether a compensated Gram (`mlxlearn_device/_compensated.py`) moves it.
- Whether a cheap conditioning estimate exists that is reliable enough to dispatch on.
