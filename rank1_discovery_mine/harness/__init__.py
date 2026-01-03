"""
Rank-1 test harness module.

Provides generic rank-1 factorization test for exotic hadron spectroscopy.
"""

from .rank1_core import (
    load_hepdata_csv,
    breit_wigner,
    fit_single_spectrum,
    fit_joint_constrained,
    fit_joint_unconstrained,
    run_pair_test,
)
from .state_configs import get_state_config, STATE_CONFIGS
from .run_test import run_rank1_test

__all__ = [
    'load_hepdata_csv',
    'breit_wigner',
    'fit_single_spectrum',
    'fit_joint_constrained',
    'fit_joint_unconstrained',
    'run_pair_test',
    'run_rank1_test',
    'get_state_config',
    'STATE_CONFIGS',
]
