"""CMSSW backend templates (no execution)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CMSSWRunnerSpec:
    """Template for CMSSW docker runner scripts."""

    cmssw_release: str
    scram_arch: str
    cfg_path: str
    output_dir: str
    use_cvmfs: bool = True


def build_docker_runner(spec: CMSSWRunnerSpec) -> str:
    """Return a shell script template for running CMSSW in Docker."""
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Placeholder runner script (no execution performed by factory)
export SCRAM_ARCH={spec.scram_arch}
CMSSW_RELEASE={spec.cmssw_release}
CFG_PATH={spec.cfg_path}
OUTPUT_DIR={spec.output_dir}

# Example steps (commented out):
# source /cvmfs/cms.cern.ch/cmsset_default.sh
# cmsrel $CMSSW_RELEASE
# cd $CMSSW_RELEASE/src
# cmsenv
# cmsRun $CFG_PATH
"""


def build_gen_only_hook() -> str:
    """Return placeholder text for GEN-only validation hook."""
    return "# GEN-only validation placeholder"


def build_real_data_hook() -> str:
    """Return placeholder text for real data reproduction hook."""
    return "# DAS query placeholder (no execution)"


def build_nanoaod_hook() -> str:
    """Return placeholder text for NanoAOD/BPHNano workflows."""
    return "# NanoAOD/BPHNano placeholder"
