#!/usr/bin/env python3
"""CLI runner for Belle Zb core + threshold dressing structure test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structure_tests.zb_core_threshold import render_report, result_to_dict, run_structure_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Belle Zb core + threshold dressing structure test",
    )
    parser.add_argument("--mode", choices=["auto", "spectrum", "ratio"], default="auto")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-starts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("structure_tests/out/zb_core_threshold"))
    parser.add_argument("--fast", action="store_true", help="Use small bootstrap/settings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_boot = 20 if args.fast else args.n_boot
    n_starts = 10 if args.fast else args.n_starts

    result = run_structure_test(
        mode=args.mode,
        n_boot=n_boot,
        n_starts=n_starts,
        seed=args.seed,
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    report_path = outdir / "STRUCTURE_REPORT.md"
    json_path = outdir / "STRUCTURE_RESULT.json"

    report_path.write_text(render_report(result), encoding="utf-8")
    json_path.write_text(
        json.dumps(result_to_dict(result), indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
