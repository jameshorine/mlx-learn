"""Render the Ridge README's tables from a results JSON.

Every wrong number published in this directory was hand-transcribed from terminal
output. This exists so the write-up is generated from the artifact instead: run it,
paste the output, and the prose cannot disagree with the file it cites.

    python -m benchmarks.ridge.render_report benchmarks/results/ridge-grid-<date>.json
    python -m benchmarks.ridge.render_report benchmarks/results/*.json

It prints an environment block, one table per input, and a summary of what cleared the
objective gate. Cells that are not statistically significant are marked, because
``benchmarks/gate.py`` treats those as "within noise" rather than as wins and a range
quoted across them overstates the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OBJECTIVE_RTOL = 1e-5


def _blas(env: dict, preflight: list | None = None) -> str:
    """The BLAS line, or a warning that the artifact does not record one.

    The first Ridge campaign was invalidated by numpy linking OpenBLAS built for
    ``neoversen1``, an ARM server core that never reaches the AMX coprocessor, and
    nothing in the recorded output said so.
    """
    # The driver's own preflight records it; upstream `environment()` does not yet.
    for c in preflight or []:
        if c.get("name") == "blas":
            v = c.get("value", "?")
            return v if c.get("ok") else f"**{v} — not Accelerate; numbers are not trustworthy**"
    pools = env.get("threadpools")
    if not pools or isinstance(pools, str):
        return "**BLAS not recorded in this artifact** — see the to-do in README.md"
    parts = [
        f"{p.get('internal_api')} ({p.get('architecture')}, {p.get('num_threads')} threads)"
        for p in pools
    ]
    line = ", ".join(parts)
    if not any("accelerate" in p.lower() for p in parts):
        line += "  **← not Accelerate; numbers are not trustworthy**"
    return line


def _env_block(env: dict, preflight: list | None = None) -> list[str]:
    cpu = env.get("cpu", "?")
    cores = env.get("cores", "?")
    mem = env.get("memory_bytes")
    gib = f"{mem / 2**30:.0f} GiB" if isinstance(mem, int) else "?"
    return [
        f"{cpu}, {cores} cores, {gib}. MLX {env.get('mlx','?')}, "
        f"scikit-learn {env.get('sklearn','?')}, SciPy {env.get('scipy','?')}, "
        f"NumPy {env.get('numpy','?')}.",
        "",
        f"BLAS: {_blas(env, preflight)}",
    ]


def _fmt(x: float, places: int = 1) -> str:
    return f"{x * 1000:.{places}f}"


def _rows_vary(records: list[dict], key: str) -> bool:
    return len({r.get(key) for r in records}) > 1


def render(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    records = payload.get("results", [])
    rtol = payload.get("objective_rtol", OBJECTIVE_RTOL)
    if not records:
        return [f"*(no records in {path.name})*"]

    by_cond = _rows_vary(records, "cond")
    baseline = records[0].get("baseline", "baseline")
    out = [
        f"### {'Conditioning sweep' if by_cond else 'Shape grid'} — `{path.name}`",
        "",
    ]
    if by_cond:
        head = "| cond(X) | alpha | Metal | " + baseline + " | median | best | significant | objective rel | gate | coef rel |"
        sep = "|---:|---:|---:|---:|---:|---:|---|---:|---|---:|"
    else:
        head = "| n | d | Metal | " + baseline + " | median | best | significant | objective rel | gate | coef rel |"
        sep = "|---:|---:|---:|---:|---:|---:|---|---:|---|---:|"
    out += [head, sep]

    wins = sig = passes = 0
    for r in sorted(records, key=lambda r: (r.get("cond", 0), r.get("n", 0), r.get("d", 0))):
        obj = r.get("objective_rel", float("nan"))
        gate = "pass" if obj < rtol else "**FAIL**"
        passes += obj < rtol
        significant = r.get("significant", False)
        sig += bool(significant)
        wins += r.get("speedup", 0) > 1.0
        spd = f"**{r['speedup']:.2f}x**" if significant and r["speedup"] > 1 else f"{r['speedup']:.2f}x"
        lead = (f"| {r.get('cond'):g} | {r.get('alpha', float('nan')):.3g} "
                if by_cond else f"| {r.get('n')} | {r.get('d')} ")
        out.append(
            lead
            + f"| {_fmt(r['mlxlearn']['median_s'])} ms | {_fmt(r['sklearn']['median_s'])} ms "
            + f"| {spd} | {r['speedup_best']:.2f}x | {'yes' if significant else 'no'} "
            + f"| {obj:.2e} | {gate} | {r.get('coef_rel', float('nan')):.2e}"
            + ("" if r.get("coef_ok", True) else " **x**") + " |"
        )

    n = len(records)
    out += [
        "",
        f"{wins}/{n} cells above 1.0x, {sig}/{n} statistically significant, "
        f"{passes}/{n} inside the {rtol:g} objective gate.",
    ]
    sig_recs = [r for r in records if r.get("significant")]
    if not sig_recs:
        out.append("**No cell is statistically significant.** No speedup claim is supportable.")
        return out

    # Per n, not pooled. Pooling takes max(loss d) from one n and min(win d) from another
    # and prints a bracket supported at neither: this grid gives (128, 512) at n=8192 and
    # (64, 256) at n=32768, which pooled reads as (128, 256).
    brackets = []
    for n in sorted({r.get("n") for r in sig_recs}):
        rows = [r for r in sig_recs if r.get("n") == n]
        losses = [r for r in rows if r["speedup"] < 1.0]
        wins = [r for r in rows if r["speedup"] > 1.0]
        if not (losses and wins):
            continue
        lo_d = max(r.get("d", 0) for r in losses)
        hi_d = min(r.get("d", 0) for r in wins)
        if lo_d >= hi_d:
            # Wins and losses interleave in d, so no single threshold separates them.
            brackets.append((n, None, None))
        else:
            brackets.append((n, lo_d, hi_d))

    if not brackets:
        sp = [r["speedup"] for r in sig_recs]
        out.append(f"All significant cells on one side: {min(sp):.2f}x-{max(sp):.2f}x.")
        return out

    for n, lo_d, hi_d in brackets:
        if lo_d is None:
            out.append(f"n={n}: significant wins and losses **overlap in d**; no single "
                       f"threshold is supported.")
        else:
            out.append(f"n={n}: significant losses up to d={lo_d}, significant wins from "
                       f"d={hi_d} — **crossover in ({lo_d}, {hi_d})**.")
    spans = [(lo, hi) for _, lo, hi in brackets if lo is not None]
    if len(spans) > 1 and len(set(spans)) > 1:
        out += ["", "The brackets differ by n, so the crossover is not a single d. Any "
                    "dispatch rule needs both, as `Tuning`'s linreg fields already do."]
    return out

    losses = [r for r in sig_recs if r["speedup"] < 1.0]
    wins = [r for r in sig_recs if r["speedup"] > 1.0]
    if losses and wins:
        # A single range spanning both sides says nothing useful. The boundary does.
        worst = min(losses, key=lambda r: r["speedup"])
        best = max(wins, key=lambda r: r["speedup"])
        lo_d = max(r.get("d", 0) for r in losses)
        hi_d = min(r.get("d", 0) for r in wins)
        out += [
            f"Significant **losses** up to d={lo_d} (worst {worst['speedup']:.2f}x at "
            f"n={worst.get('n')}, d={worst.get('d')}); significant **wins** from d={hi_d} "
            f"(best {best['speedup']:.2f}x at n={best.get('n')}, d={best.get('d')}).",
            "",
            f"**The crossover is between d={lo_d} and d={hi_d}.** Quoting one range across "
            f"both sides would be meaningless: this is a dispatch threshold, not a speedup.",
        ]
    else:
        sp = [r["speedup"] for r in sig_recs]
        out.append(f"Range across significant cells: {min(sp):.2f}x-{max(sp):.2f}x.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)

    first = json.loads(args.results[0].read_text())
    lines = _env_block(first.get("environment", {}), first.get("preflight")) + [""]
    for path in args.results:
        lines += render(path) + [""]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
