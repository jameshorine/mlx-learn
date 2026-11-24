"""Refuse to benchmark in an environment that cannot produce a valid number.

Every wrong figure published from this directory came from a measurement that ran
happily and reported something plausible. The environment was wrong, or the machine was
busy, or the comparison was unfair — and nothing stopped it or recorded why. Discipline
did not work; this converts the checks into a gate.

The repository already argues for exactly this, in ``benchmarks/gate.py``:

    The design goal is "never slower than scikit-learn". The *gate* is narrower and
    measurable, because a goal you cannot fail is not a gate.

A benchmark you cannot fail to run is not a benchmark either.

    from benchmarks.ridge.preflight import preflight
    preflight()                 # raises on a blocking problem, warns on the rest
    preflight(strict=True)      # raises on warnings too
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

__all__ = ["Check", "run_checks", "preflight", "PreflightError"]


class PreflightError(RuntimeError):
    """The environment cannot produce a number worth recording."""


class Check:
    """One environment fact, its value, and whether it blocks."""

    def __init__(self, name: str, ok: bool, value: str, why: str, blocking: bool) -> None:
        self.name, self.ok, self.value, self.why, self.blocking = name, ok, value, why, blocking

    def __str__(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.blocking else "warn")
        return f"  [{mark}] {self.name}: {self.value}" + ("" if self.ok else f"\n         {self.why}")


def _blas() -> Check:
    """numpy's BLAS. The one that silently invalidated a whole campaign.

    conda ships OpenBLAS built for ``neoversen1``, an ARM *server* core, which never
    reaches the AMX coprocessor that makes Apple silicon fast at GEMM. Timing Metal
    against it flatters Metal by a wide margin, and nothing in the output says so.
    """
    try:
        import numpy as np

        name = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception as exc:  # noqa: BLE001 - a missing/odd numpy is itself the finding
        return Check("blas", False, f"undetermined ({exc})", "cannot verify the baseline", True)

    arm_mac = sys.platform == "darwin" and platform.machine() == "arm64"
    ok = name == "accelerate" or not arm_mac
    return Check(
        "blas", ok, name,
        "on arm64 macOS the baseline must link Accelerate, or the CPU side is handicapped. "
        "Build the environment the nightly uses: python3 -m venv .venv && "
        '.venv/bin/pip install -e ".[dev]"',
        True,
    )


def _load() -> Check:
    """Other work on the machine. Interleaving defends against drift, not contention."""
    try:
        one = os.getloadavg()[0]
    except OSError:  # pragma: no cover
        return Check("load", True, "unavailable", "", False)
    cores = os.cpu_count() or 1
    ratio = one / cores
    return Check(
        "load", ratio < 0.4, f"{one:.1f} over {cores} cores",
        "something else is using this machine. Interleaved A/B survives a slow drift but "
        "not a competing process that lands on one arm; measured spreads reached 130% "
        "under load here.",
        False,
    )


def _thermal() -> Check:
    """macOS thermal pressure. Sustained GPU work throttles a laptop.

    ``benchmarks/experiments/kmeans-plusplus-2026-11-19/`` waits for level 0 before every
    timed call and records the wait; this only reports it.
    """
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except Exception:  # noqa: BLE001 - absent on non-Darwin, not a failure
        return Check("thermal", True, "unavailable", "", False)
    for line in out.splitlines():
        if "CPU_Scheduler_Limit" in line and "100" not in line:
            return Check("thermal", False, line.strip(), "the machine is throttling", False)
    return Check("thermal", True, "nominal", "", False)


def _gpu() -> Check:
    try:
        import mlx.core as mx

        avail = bool(getattr(mx, "metal", None) and mx.metal.is_available())
        return Check("metal", avail, str(mx.device_info().get("architecture", "?")),
                     "no Metal device; the GPU arm cannot run", True)
    except Exception as exc:  # noqa: BLE001
        return Check("metal", False, f"unavailable ({exc})", "mlx is not importable", True)


def _editable() -> Check:
    """Which tree is actually imported. A venv's editable install can point elsewhere."""
    try:
        import mlxlearn

        path = os.path.dirname(os.path.dirname(os.path.dirname(mlxlearn.__file__)))
        here = os.getcwd()
        ok = os.path.realpath(path).startswith(os.path.realpath(here))
        return Check("source", ok, mlxlearn.__file__,
                     "the imported mlxlearn is outside this checkout, so you are "
                     "benchmarking a different tree. Run from the repository root with "
                     "PYTHONPATH=src.", False)
    except Exception as exc:  # noqa: BLE001
        return Check("source", False, f"unavailable ({exc})", "mlxlearn is not importable", True)


def run_checks() -> list[Check]:
    return [_blas(), _gpu(), _editable(), _load(), _thermal()]


def preflight(*, strict: bool = False, quiet: bool = False) -> list[Check]:
    """Run the checks, print them, and raise if the environment is unusable."""
    checks = run_checks()
    if not quiet:
        print("preflight:")
        for c in checks:
            print(c)
    bad = [c for c in checks if not c.ok and (c.blocking or strict)]
    if bad:
        raise PreflightError(
            "refusing to benchmark: " + "; ".join(f"{c.name}={c.value}" for c in bad)
        )
    return checks


if __name__ == "__main__":
    try:
        preflight(strict="--strict" in sys.argv)
    except PreflightError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print("\nenvironment is usable")
