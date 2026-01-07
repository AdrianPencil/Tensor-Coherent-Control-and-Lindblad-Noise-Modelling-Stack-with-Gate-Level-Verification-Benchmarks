"""
qip.cli

Command-line entry point.

Design:
- keep CLI thin and stable
- delegate real work to case_studies (or workflows)
"""

from __future__ import print_function

import argparse
from dataclasses import dataclass

from qip.case_studies.sc_qubit_end_to_end import run_sc_qubit_end_to_end
from qip.case_studies.parameters import ScQubitCaseStudyParams

__all__ = [
    "main",
]


@dataclass(frozen=True, slots=True)
class _CLIArgs:
    """
    Parsed CLI arguments mapped into a structured object.

    Keeping an internal dataclass makes it explicit what the CLI actually needs,
    and keeps `main()` free of ad-hoc dictionaries.
    """

    data_dir: str
    n_steps: int
    dt: float
    quiet: bool


def _parse_args(argv: list[str] | None) -> _CLIArgs:
    """
    Parse command-line options.

    The CLI intentionally exposes only a few controls:
    - data directory (calibration + PSD)
    - time step configuration (n_steps, dt)
    - verbosity toggle
    """
    p = argparse.ArgumentParser(prog="qip", description="qip-stack CLI")
    p.add_argument("--data-dir", type=str, default="src/qip/data/superconducting_qubit")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=1.0e-10)
    p.add_argument("--quiet", action="store_true")
    ns = p.parse_args(argv)
    return _CLIArgs(data_dir=str(ns.data_dir), n_steps=int(ns.n_steps), dt=float(ns.dt), quiet=bool(ns.quiet))


def main(argv: list[str] | None = None) -> int:
    """
    Entry point used by `qip` console_script.

    Current default behavior: run the superconducting-qubit end-to-end case study.
    """
    args = _parse_args(argv)

    params = ScQubitCaseStudyParams.default(
        data_dir=args.data_dir,
        n_steps=args.n_steps,
        dt=args.dt,
    )
    result = run_sc_qubit_end_to_end(params)

    if not args.quiet:
        for k, v in result.summary.items():
            print(f"{k}: {v}")

    return 0
