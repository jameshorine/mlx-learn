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


def _blas(env: dict) -> str:
    """The BLAS line, or a warning that the artifact does not record one.

    The first Ridge campaign was invalidated by numpy linking OpenBLAS built for
    ``neoversen1``, an ARM server core that never reaches the AMX coprocessor, and
    nothing in the recorded output said so.
    """
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


def _env_block(env: dict) -> list[str]:
    cpu = env.get("cpu", "?")
    cores = env.get("cores", "?")
    mem = env.get("memory_bytes")
    gib = f"{mem / 2**30:.0f} GiB" if isinstance(mem, int) else "?"
    return [
        f"{cpu}, {cores} cores, {gib}. MLX {env.get('mlx','?')}, "
        f"scikit-learn {env.get('sklearn','?')}, SciPy {env.get('scipy','?')}, "
        f"NumPy {env.get('numpy','?')}.",
        "",
        f"BLAS: {_blas(env)}",
    ]


def _fmt(x: float, places: int = 1) -> str:
    return f"{x * 1000:.{places}f}"


def _rows_vary(records: list[dict], key: str) -> bool:
    return len({r.get(key) for r in records}) > 1


def render(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    records = payload.get("results", [])
    if not records:
        return [f"*(no records in {path.name})*"]

    by_cond = _rows_vary(records, "cond")
    baseline = records[0].get("baseline", "baseline")
    out = [
        f"### {'Conditioning sweep' if by_cond else 'Shape grid'} — `{path.name}`",
        "",
    ]
    if by_cond:
        head = "| cond(X) | alpha | Metal | " + baseline + " | median | best | significant | objective rel | gate |"
        sep = "|---:|---:|---:|---:|---:|---:|---|---:|---|"
    else:
        head = "| n | d | Metal | " + baseline + " | median | best | significant | objective rel | gate |"
        sep = "|---:|---:|---:|---:|---:|---:|---|---:|---|"
    out += [head, sep]

    wins = sig = passes = 0
    for r in sorted(records, key=lambda r: (r.get("cond", 0), r.get("n", 0), r.get("d", 0))):
        obj = r.get("objective_rel", float("nan"))
        gate = "pass" if obj < OBJECTIVE_RTOL else "**FAIL**"
        passes += obj < OBJECTIVE_RTOL
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
            + f"| {obj:.2e} | {gate} |"
        )

    n = len(records)
    out += [
        "",
        f"{wins}/{n} cells above 1.0x, {sig}/{n} statistically significant, "
        f"{passes}/{n} inside the {OBJECTIVE_RTOL:g} objective gate.",
    ]
    if sig:
        sig_speeds = [r["speedup"] for r in records if r.get("significant")]
        out.append(
            f"Range across significant cells only: {min(sig_speeds):.2f}x–{max(sig_speeds):.2f}x. "
            f"Quoting a range that includes non-significant cells overstates the result, since "
            f"`benchmarks/gate.py` classifies those as within noise."
        )
    else:
        out.append("**No cell is statistically significant.** No speedup claim is supportable.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)

    first = json.loads(args.results[0].read_text())
    lines = _env_block(first.get("environment", {})) + [""]
    for path in args.results:
        lines += render(path) + [""]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
