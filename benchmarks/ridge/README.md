# Ridge benchmarks

> **The numbers previously published here are withdrawn.** An adversarial review found
> the comparison was not like-for-like and the accuracy claim was attributed to the wrong
> variable. The driver has been fixed; the grid has **not** been re-run on a corrected
> environment. Treat this directory as work in progress, not as a result.

A prototype, not part of the gated suite. It asks one question: does building Ridge's Gram
in float32 on Metal and factoring in float64 on the host beat scikit-learn doing both on
the host, and over what shapes?

| File | Purpose |
|---|---|
| `ridge_prototype.py` | Shape and conditioning sweeps against scikit-learn |

Self-contained on purpose. A sibling VAR(p) prototype uses the same three Gram helpers, but
Ridge should not depend on it: Ridge has a scikit-learn counterpart and a roadmap entry,
while VAR has neither and is blocked on a scope decision.

## Why Ridge and not LinearRegression

`LinearRegression` shipped at 0.1.7 with **no Metal path** — `344bfdf`: "Metal has no
float64 and the float32 variants were inaccurate or no faster." That verdict was reached
for *unregularised* least squares, which must reproduce `lstsq`'s minimum-norm solution
plus `rank_` and `singular_`, and so needs an SVD rather than a Gram.

Ridge has no such obligation: `alpha > 0` makes `X'X + alpha I` positive definite by
construction, bounding `cond` by `(sigma_max^2 + alpha)/alpha`, so a Gram route is
contract-correct here where it is not there.

**That argument is sound as a contract argument and was vacuous as an accuracy argument in
the withdrawn measurements.** At the alpha actually used, `alpha/lambda_min(X'X)` was
`1.20e-06` and the penalty moved `cond(X'X + alpha I)` from 1.414 to 1.415. The
regularisation the whole distinction rests on was numerically inert in every committed
record. A re-run must use an alpha where the penalty does something.

scikit-learn agrees about the algorithm — `resolve_solver_for_numpy` returns `"cholesky"`
for dense input with `positive=False`, and `_solve_cholesky` is

```
A = X'X ; Xy = X'y ; A.flat[::d+1] += alpha ; solve(A, Xy, assume_a="pos")
```

so both sides run the same computation; only where the Gram is formed differs.

## What the review found

Four defects, all now fixed in the driver, none yet re-measured.

| # | Defect | Status |
|---|---|---|
| 1 | **Baseline was not like-for-like.** `Ridge.fit` runs `check_array` on X and y, then `_preprocess_data(copy=copy_X)` runs a *second* `check_array` with `copy=True` — a full float64 copy — all inside the timed region, against a bare `gram_metal` + solve that validates nothing. | fixed: `sklearn_bare` arm calls `_solve_cholesky` directly and is the default baseline |
| 2 | **Wrong Metal path.** `gram_metal` called `float32_matmul` directly. That kernel defeats TF32 rounding on GPU generation >= 17; this machine is `applegpu_g13s`, so `matmul_may_round_float32()` is False and the shipped wrapper dispatches to `mx.matmul`. The benchmark measured a path the library would not take, and the slower one. | fixed: calls `_binary32.matmul` |
| 3 | **Crashes where scikit-learn degrades.** A float32 Gram can be indefinite where the float64 one is not — measured `lambda_min = -1.315e-08` at `n=32768, d=512, cond=1e12` against `+1.905e-08`. `assume_a="pos"` raises; stock catches it and retries with `solver="svd"` (`sklearn/linear_model/_ridge.py:748-752`). | fixed: falls back |
| 4 | **The conditioning sweep varied two things.** `alpha = alpha_scale * trace(X'X)/d`, and `make_problem` rescales the spectrum to set `cond`, collapsing the trace: alpha moved `3.27e-02 -> 2.00e-08` across `cond` 1..1e12, a 1.6-million-fold change. `cond(X'X + alpha I)` — the matrix actually factored — moved only ~3 orders. | fixed: `--alpha-abs` decouples them |

Two further findings that change the *framing* rather than the code:

**Accuracy depends on shape, not conditioning.** The withdrawn README made conditioning the
whole story. At `cond(X) = 1` — the most benign design possible — the objective gate
**fails** at `n=65536, d=2048`, one doubling outside the measured box, because the Gram
error is set by `_split_span`'s performance heuristic, which stops splitting K at
`d >= 2048`. Objective error grows as roughly `d^2.60` at fixed conditioning.

**"That is the data, not the arithmetic" was wrong.** The withdrawn text explained
coefficient degradation as the data being ill-conditioned. Running the identical problems
through `gram_host` (float64, same solve) gives `coef_rel = 0.000e+00` against
scikit-learn at every conditioning level tested. The degradation is the float32 Gram.

And one environment defect that invalidated the baseline independently of all of the above:
the numpy backing the comparison was conda's OpenBLAS built for `neoversen1`, an ARM
**server** core, which never touches the AMX coprocessor. See
`benchmarks/_harness.py::environment()`, which records no BLAS field.

## To do

**Before any number is published again**, in order. Each step is worthless if the one
above it has not been done.

- [ ] **Build the environment the nightly uses.** `python3 -m venv .venv` then
      `.venv/bin/pip install -e ".[dev]"`, matching
      `.github/workflows/nightly-benchmarks.yml`. pip wheels link Accelerate; conda links
      OpenBLAS.
- [ ] **Verify the BLAS before trusting anything.**
      `threadpoolctl.threadpool_info()` must report `accelerate`, not
      `openblas` / `neoversen1`. If it does not, stop.
- [ ] **Record the environment**: `python -m benchmarks.run --environment-only --output
      bench/environment.json`.
- [ ] **Re-run the shape grid** against `sklearn_bare`, with `d` extended to 2048 where the
      gate is known to fail, writing JSON.
- [ ] **Re-run the conditioning sweep with `--alpha-abs`** so it varies conditioning alone,
      and at an alpha where `alpha/lambda_min` is not ~1e-6.
- [ ] **Rewrite this file and the tracking issue from the JSON**, not from transcription.

Then, only if a win survives:

- [ ] **Separate the Metal arm's costs.** At `n=32768, d=1024` the arm measured as
      narrowing 27.7 ms, upload 5.2 ms, GPU GEMM 29.1 ms, readback 3.4 ms, host Cholesky
      35.3 ms. The GPU does under a third of the work; the float64->float32 narrowing on
      the host nearly equals the GEMM. If the narrowing can be avoided or overlapped, that
      is a larger lever than the GEMM itself.
- [ ] **Decide the dispatch rule.** Accuracy depends on `d` *and* on `alpha` relative to
      the spectrum, not on shape alone as every existing `_below_crossover` does. One
      candidate is validated: raising `alpha_scale` from 1e-6 to 1e-4 at `cond = 1e8` moved
      the objective error from `1.47e-05` (fail) to `6.14e-10`.
- [ ] **Test on a real dataset.** Everything so far is synthetic Gaussian designs.
- [ ] **Add thermal gating**, as `benchmarks/experiments/kmeans-plusplus-2026-11-19/` does —
      wait for macOS thermal pressure level 0 before each timed call and record it.
- [ ] **Measure `fit_intercept=True`**, `sample_weight`, and the refused `solver` values.

Out of scope here but worth filing separately:

- [ ] `benchmarks/_harness.py::environment()` records no BLAS backend, which is how the
      OpenBLAS problem went unnoticed. `threadpoolctl` is already a hard requirement of
      scikit-learn, so recording it costs no new dependency.
- [ ] `benchmarks/_harness.py` names a field `iqr_s` that is the full range at the default
      `iterations=7`: `quartile = max(1, 7//4) = 1`, so it is `samples[-1] - samples[0]`.
      Every committed record satisfies `iqr_s == max_s - min_s`. The docstring reasons
      explicitly about `IQR / sqrt(n)`.

## Running it

```bash
python -m benchmarks.ridge.ridge_prototype --n 8192 32768 --d 64 128 256 512 1024 2048 \
    --output benchmarks/results/ridge-grid-<date>.json
python -m benchmarks.ridge.ridge_prototype --n 32768 --d 256 \
    --cond 1 1e2 1e3 1e4 1e6 1e8 --alpha-abs 1.0
```

| Backend | What runs |
|---|---|
| `metal_gram` | `X'X`, `X'y` via `_binary32.matmul`, then host float64 Cholesky |
| `host_gram` | the same normal equations entirely in host float64 |
| `sklearn_bare` | `sklearn.linear_model._ridge._solve_cholesky`, no estimator — the fair baseline |
| `sklearn` | the full `Ridge.fit`, kept so its validation overhead is visible as its own arm |

## What is not established

Everything quantitative. The driver is fixed; nothing has been re-measured on a corrected
environment. The prior from the one smoke run after the fixes — `0.34x` at `n=2000, d=64`
against `sklearn_bare`, still on the handicapped OpenBLAS — is that Ridge is at parity or
worse across much of the grid, and that Accelerate will move it further that way.
