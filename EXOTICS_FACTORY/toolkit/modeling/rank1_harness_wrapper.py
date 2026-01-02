"""Wrapper interface to invoke cms_rank1_test.py harness (no execution)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HarnessSpec:
    harness_path: str
    config_path: str
    output_dir: str
    extra_args: list[str] | None = None


def build_harness_command(spec: HarnessSpec) -> list[str]:
    """Build the command list to run the harness."""
    cmd = [
        "python",
        spec.harness_path,
        "--config",
        spec.config_path,
        "--output",
        spec.output_dir,
    ]
    if spec.extra_args:
        cmd.extend(spec.extra_args)
    return cmd
