# Benchmarks

The standard runner compares patched mlxlearn estimators with scikit-learn on
synthetic data. It measures latency, not numerical correctness. Focused
experiments use separate designs; see their reports before comparing results.

## Benchmark design

The full grid combines every listed row count (`n`) and feature count (`d`).
KNN also combines each shape with every query count (`q`).

| Model of Interest | Operation and configuration | Full grid |
|---|---|---|
| KNN | Fit and `kneighbors`; 10 neighbors; sklearn brute force | `n`: 500–512,000 (6 sizes); `d`: 8, 64, 768; `q`: 1, 256, 4,096 |
| Logistic regression | Fit; `max_iter=200` | `n`: 1,000, 10,000, 100,000, 500,000; `d`: 16, 128, 1,024 |
| Linear regression | Not benchmarked; experimental design not yet specified | Not specified |
| PCA | Fit and transform; up to 32 components | Fit `n`: 1,000, 10,000, 50,000, 100,000; transform `n`: 10,000, 50,000, 100,000; `d`: 64, 256, 1,024 |
| SVC | RBF fit and prediction | Fit `n`: 500, 2,000, 8,000, 32,000; prediction `n`: 2,000, 8,000; `d`: 16, 128 |
| VAR(p) | Not benchmarked; experimental design not yet specified | Not specified |

The existing grid and smaller smoke profile are in [`run.py`](run.py).
Inputs are normally distributed; classification labels come from the first
feature plus noise. The default seed is 0.

| Design | Standard runner |
|---|---|
| Comparison | Patched mlxlearn vs. stock scikit-learn; `--layer direct` measures a different path. |
| Timing | Two warmups, then seven rounds by default. Each round measures mlxlearn first and scikit-learn second; order is not reversed. MLX is synchronized. |
| Boundaries | Fit includes estimator construction. Other operations use prefitted models; PCA transforms the training data and SVC predicts its first 1,000 rows. Cold start is excluded. |

## Metrics and tests

| Measure or test | What it establishes | Limitations |
|---|---|---|
| Latency | Median, minimum, maximum, and IQR for each operation. | JSON retains summaries, not individual samples. |
| Speedup | sklearn time divided by mlxlearn time, for medians and minima. | Above 1 is faster; median and minimum can tell different stories. |
| Crossover | Smallest tested `n` per operation with a significant median win and a minimum-time win. | Pools different widths and query counts; it is not a guarantee for every shape. “Significant” is a heuristic based on combined IQR divided by √rounds, not a formal statistical test. |
| Performance gate | Flags significant cases with a median speedup below 0.95 at or above the run's crossover, or when no crossover exists. | Crossovers come from the same run; below-crossover cases and documented accepted regressions are exempt. See [`gate.py`](gate.py). |
| Correctness | Separate tests check estimator behavior and benchmark tooling. | The standard runner does not check accuracy or numerical parity. Focused experiments may perform their own audits. |

## Run

On Apple silicon, from the repository root:

```bash
pip install -e ".[bench]"
python -m benchmarks.run --profile smoke --output bench/results.json --report bench/report.md
python -m benchmarks.gate --results bench/results.json
```

Replace `smoke` with `full` for the larger grid, which can take hours. The
nightly performance gate runs on fixed self-hosted hardware; shared CI runs
correctness tests, including `python -m pytest -q benchmarks/test_knn_*.py`.

## Example findings

- **O-means vs. e-means:** On M4 Max, the float32 geometric-mean speedup to
  matched held-out error was 4.13× across 18 datasets and three seeds; the
  time-to-error estimate uses interpolation. [Report](../docs/o-means-benchmarks-m4-max-2026-11-19.md)
- **SVC row cache:** On M4 Max, standard MNIST fit time at a 200 MiB cache
  fell from 9.09 s to 3.59 s versus the previous mlxlearn version.
  [Report](../docs/svc-row-cache-2026-11-17.md)

These are focused experiments, not results from the standard grid. See
[`docs/benchmarks.md`](../docs/benchmarks.md) for historical results and the
[`e_means`](e_means/README.md) and [`radius`](radius/README.md) guides for
other protocols.
