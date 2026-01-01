#!/usr/bin/env python3
"""
sim_sweep.py - Publication-grade power & calibration sweep

Runs comprehensive MC sweep for rank-1 bottleneck tests:
- Type I error under M0 (rank-1 true): 300 trials per test/stats
- Power under M1 (rank-1 false): 300 trials per test/stats
- Power under M4 (rank-2): 200 trials per grid point
- M4 grid sweep: dr x dphi x stats combinations

Features:
- Parallelization with multiprocessing
- Checkpointing for long-running jobs
- Fit-health gates and identifiability tracking
- Quick fits for trials, full bootstrap for final results

Usage:
    nohup python3 -u sim_sweep.py --config configs/tests_top3.json > sweep.log 2>&1 &
"""

import numpy as np
import json
import os
import sys
import pickle
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, asdict, field
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_generate import generate_dataset
from sim_fit import run_quick_fit, run_full_fit, DOF_DIFF


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrialResult:
    """Result of a single MC trial"""
    test_name: str
    mechanism: str
    stats_level: float
    trial_idx: int

    # Fit result
    converged: bool
    Lambda: float
    p_wilks: float
    gates: str

    # Health metrics
    chi2_dof_a: float
    chi2_dof_b: float
    deviance_dof_a: float
    deviance_dof_b: float

    # Identifiability
    identifiable: bool
    ident_reason: str

    # Fitted parameters
    r_shared: float = np.nan
    phi_shared: float = np.nan
    r_a: float = np.nan
    phi_a: float = np.nan
    r_b: float = np.nan
    phi_b: float = np.nan

    # M4-specific
    dr: float = 0.0
    dphi: float = 0.0


@dataclass
class AggregatedResult:
    """Aggregated results for a test/mechanism/stats combination"""
    test_name: str
    mechanism: str
    stats_level: float
    data_type: str
    nbins_a: int
    nbins_b: int

    n_trials: int
    n_converged: int
    n_passed: int  # gates == PASS
    n_identifiable: int

    # Type I / Power
    rejection_rate: float
    rejection_rate_se: float

    # Metrics
    median_lambda: float
    mean_chi2_dof: float
    pass_rate: float
    identifiability_rate: float

    # M4-specific
    dr: float = 0.0
    dphi: float = 0.0


@dataclass
class M4GridResult:
    """Result for a single M4 grid point"""
    test_name: str
    dr: float
    dphi: float
    stats_level: float
    power: float
    power_se: float
    pass_rate: float
    n_trials: int


@dataclass
class SweepState:
    """State for checkpointing"""
    completed_trials: List[TrialResult] = field(default_factory=list)
    completed_aggregates: List[AggregatedResult] = field(default_factory=list)
    m4_grid_results: List[M4GridResult] = field(default_factory=list)
    current_test: str = ""
    current_mechanism: str = ""
    current_stats: float = 0.0
    total_trials_done: int = 0
    start_time: str = ""


# ============================================================================
# Trial Functions
# ============================================================================

def run_single_trial(args: Tuple) -> Dict:
    """
    Run a single MC trial. Used by multiprocessing pool.
    """
    test_config, mechanism, scale_factor, trial_idx, dr, dphi = args

    seed = trial_idx * 1000 + int(scale_factor * 100) + hash(test_config['name']) % 10000

    try:
        # Generate data
        if mechanism == 'M4':
            dataset = generate_dataset(test_config, mechanism, scale_factor,
                                       seed=seed, dr=dr, dphi_deg=dphi)
        else:
            dataset = generate_dataset(test_config, mechanism, scale_factor, seed=seed)

        # Run quick fit (no bootstrap for speed)
        result = run_quick_fit(dataset, n_restarts=30)

        return {
            'success': True,
            'test_name': test_config['name'],
            'mechanism': mechanism,
            'stats_level': scale_factor,
            'trial_idx': trial_idx,
            'converged': result['converged'],
            'Lambda': result['Lambda'],
            'p_wilks': result['p_wilks'],
            'gates': result['gates'],
            'chi2_dof_a': result.get('chi2_dof_a', np.nan),
            'chi2_dof_b': result.get('chi2_dof_b', np.nan),
            'deviance_dof_a': result.get('deviance_dof_a', np.nan),
            'deviance_dof_b': result.get('deviance_dof_b', np.nan),
            'identifiable': result.get('identifiable', True),
            'ident_reason': result.get('ident_reason', 'ok'),
            'r_shared': result.get('r_shared', np.nan),
            'phi_shared': result.get('phi_shared', np.nan),
            'r_a': result.get('r_a', np.nan),
            'phi_a': result.get('phi_a', np.nan),
            'r_b': result.get('r_b', np.nan),
            'phi_b': result.get('phi_b', np.nan),
            'dr': dr,
            'dphi': dphi
        }
    except Exception as e:
        return {
            'success': False,
            'test_name': test_config['name'],
            'mechanism': mechanism,
            'stats_level': scale_factor,
            'trial_idx': trial_idx,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'dr': dr,
            'dphi': dphi
        }


def result_to_trial(r: Dict) -> Optional[TrialResult]:
    """Convert raw result dict to TrialResult."""
    if not r.get('success', False):
        return None

    return TrialResult(
        test_name=r['test_name'],
        mechanism=r['mechanism'],
        stats_level=r['stats_level'],
        trial_idx=r['trial_idx'],
        converged=r['converged'],
        Lambda=r['Lambda'],
        p_wilks=r['p_wilks'],
        gates=r['gates'],
        chi2_dof_a=r.get('chi2_dof_a', np.nan),
        chi2_dof_b=r.get('chi2_dof_b', np.nan),
        deviance_dof_a=r.get('deviance_dof_a', np.nan),
        deviance_dof_b=r.get('deviance_dof_b', np.nan),
        identifiable=r.get('identifiable', True),
        ident_reason=r.get('ident_reason', ''),
        r_shared=r.get('r_shared', np.nan),
        phi_shared=r.get('phi_shared', np.nan),
        r_a=r.get('r_a', np.nan),
        phi_a=r.get('phi_a', np.nan),
        r_b=r.get('r_b', np.nan),
        phi_b=r.get('phi_b', np.nan),
        dr=r.get('dr', 0.0),
        dphi=r.get('dphi', 0.0)
    )


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_trials(trials: List[TrialResult], test_config: Dict) -> AggregatedResult:
    """Aggregate trial results into summary statistics."""
    if not trials:
        return None

    first = trials[0]
    test_name = first.test_name
    mechanism = first.mechanism
    stats_level = first.stats_level

    # Data type info
    data_type = test_config['channelA']['type']
    nbins_a = test_config['channelA']['nbins']
    nbins_b = test_config['channelB']['nbins']

    # Filter valid trials
    converged = [t for t in trials if t.converged]
    passed = [t for t in converged if t.gates == 'PASS']
    identifiable = [t for t in passed if t.identifiable]

    n_trials = len(trials)
    n_converged = len(converged)
    n_passed = len(passed)
    n_identifiable = len(identifiable)

    # Rejection rate (using Wilks p-value for speed, on passed trials)
    if len(passed) > 0:
        # chi2(2) at 95% is 5.99, so Lambda > 5.99 means rejection
        chi2_threshold = 5.991  # chi2(2, 0.05)
        rejections = sum(1 for t in passed if t.Lambda > chi2_threshold)
        rejection_rate = rejections / len(passed)
        rejection_rate_se = np.sqrt(rejection_rate * (1 - rejection_rate) / len(passed))
    else:
        rejection_rate = np.nan
        rejection_rate_se = np.nan

    # Lambda statistics
    lambdas = [t.Lambda for t in passed if not np.isnan(t.Lambda)]
    median_lambda = np.median(lambdas) if lambdas else np.nan

    # Chi2 statistics
    chi2s = []
    for t in converged:
        if not np.isnan(t.chi2_dof_a):
            chi2s.append(t.chi2_dof_a)
        if not np.isnan(t.chi2_dof_b):
            chi2s.append(t.chi2_dof_b)
    mean_chi2_dof = np.mean(chi2s) if chi2s else np.nan

    # Rates
    pass_rate = n_passed / n_converged if n_converged > 0 else 0.0
    identifiability_rate = n_identifiable / n_passed if n_passed > 0 else 0.0

    return AggregatedResult(
        test_name=test_name,
        mechanism=mechanism,
        stats_level=stats_level,
        data_type=data_type,
        nbins_a=nbins_a,
        nbins_b=nbins_b,
        n_trials=n_trials,
        n_converged=n_converged,
        n_passed=n_passed,
        n_identifiable=n_identifiable,
        rejection_rate=rejection_rate,
        rejection_rate_se=rejection_rate_se,
        median_lambda=median_lambda,
        mean_chi2_dof=mean_chi2_dof,
        pass_rate=pass_rate,
        identifiability_rate=identifiability_rate,
        dr=first.dr,
        dphi=first.dphi
    )


# ============================================================================
# Main Sweep Functions
# ============================================================================

def run_mechanism_sweep(test_config: Dict, mechanism: str, stats_level: float,
                        n_trials: int, n_workers: int,
                        dr: float = 0.0, dphi: float = 0.0) -> List[TrialResult]:
    """
    Run trials for a single mechanism/stats combination.
    """
    args_list = [
        (test_config, mechanism, stats_level, i, dr, dphi)
        for i in range(n_trials)
    ]

    with Pool(n_workers) as pool:
        raw_results = pool.map(run_single_trial, args_list)

    # Convert to TrialResult
    trials = []
    for r in raw_results:
        trial = result_to_trial(r)
        if trial is not None:
            trials.append(trial)

    return trials


def run_full_sweep(config: Dict, output_dir: str,
                   n_trials_m0: int = 300,
                   n_trials_m1: int = 300,
                   n_trials_m4: int = 200,
                   n_workers: Optional[int] = None,
                   checkpoint_file: Optional[str] = None) -> SweepState:
    """
    Run full power & calibration sweep.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    tests = config['tests']
    stats_multipliers = config['tests'][0].get('stats_multipliers', [0.5, 1.0, 2.0, 4.0])
    m4_grid = config.get('M4_grid', {})
    dr_values = m4_grid.get('dr_values', [0.05, 0.10, 0.20])
    dphi_values = m4_grid.get('dphi_values', [10, 20, 40])
    m4_stats_levels = m4_grid.get('stats_levels', [1.0, 2.0])

    # Load or create state
    state = SweepState()
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        print(f"  Resuming from trial {state.total_trials_done}")
    else:
        state.start_time = datetime.now().isoformat()

    # Calculate total work
    total_work = (len(tests) * len(stats_multipliers) * 2 +  # M0 + M1
                  len(tests) * len(dr_values) * len(dphi_values) * len(m4_stats_levels))
    work_done = 0

    print("\n" + "="*70)
    print("RANK-1 BOTTLENECK POWER & CALIBRATION SWEEP")
    print("="*70)
    print(f"Tests: {[t['name'] for t in tests]}")
    print(f"Stats levels: {stats_multipliers}")
    print(f"M0 trials: {n_trials_m0}, M1 trials: {n_trials_m1}, M4 trials: {n_trials_m4}")
    print(f"M4 grid: dr={dr_values}, dphi={dphi_values}")
    print(f"Workers: {n_workers}")
    print("="*70)

    # Track completed combinations
    completed = set()
    for agg in state.completed_aggregates:
        key = (agg.test_name, agg.mechanism, agg.stats_level, agg.dr, agg.dphi)
        completed.add(key)

    # M0 and M1 sweeps
    for test in tests:
        test_name = test['name']

        for stats_level in stats_multipliers:
            # M0 (Type I error)
            key = (test_name, 'M0', stats_level, 0.0, 0.0)
            if key not in completed:
                print(f"\n[{test_name}] M0 @ {stats_level}x stats...")
                state.current_test = test_name
                state.current_mechanism = 'M0'
                state.current_stats = stats_level

                trials = run_mechanism_sweep(test, 'M0', stats_level,
                                            n_trials_m0, n_workers)
                agg = aggregate_trials(trials, test)

                state.completed_trials.extend(trials)
                if agg:
                    state.completed_aggregates.append(agg)
                    print(f"  Type I (Wilks): {agg.rejection_rate:.3f} +/- {agg.rejection_rate_se:.3f}")
                    print(f"  Pass rate: {agg.pass_rate:.2f}, Identifiable: {agg.identifiability_rate:.2f}")

                # Checkpoint
                if checkpoint_file:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(state, f)

                work_done += 1
                state.total_trials_done += len(trials)

            # M1 (Power)
            key = (test_name, 'M1', stats_level, 0.0, 0.0)
            if key not in completed:
                print(f"\n[{test_name}] M1 @ {stats_level}x stats...")
                state.current_test = test_name
                state.current_mechanism = 'M1'
                state.current_stats = stats_level

                trials = run_mechanism_sweep(test, 'M1', stats_level,
                                            n_trials_m1, n_workers)
                agg = aggregate_trials(trials, test)

                state.completed_trials.extend(trials)
                if agg:
                    state.completed_aggregates.append(agg)
                    print(f"  Power (Wilks): {agg.rejection_rate:.3f} +/- {agg.rejection_rate_se:.3f}")
                    print(f"  Pass rate: {agg.pass_rate:.2f}, Identifiable: {agg.identifiability_rate:.2f}")

                # Checkpoint
                if checkpoint_file:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(state, f)

                work_done += 1
                state.total_trials_done += len(trials)

    # M4 grid sweep
    print("\n" + "-"*70)
    print("M4 GRID SWEEP (Rank-2 Detectability)")
    print("-"*70)

    for test in tests:
        test_name = test['name']

        for dr in dr_values:
            for dphi in dphi_values:
                for stats_level in m4_stats_levels:
                    key = (test_name, 'M4', stats_level, dr, dphi)
                    if key not in completed:
                        print(f"\n[{test_name}] M4 dr={dr:.2f}, dphi={dphi:.0f}deg @ {stats_level}x...")

                        trials = run_mechanism_sweep(test, 'M4', stats_level,
                                                    n_trials_m4, n_workers,
                                                    dr=dr, dphi=dphi)
                        agg = aggregate_trials(trials, test)

                        state.completed_trials.extend(trials)
                        if agg:
                            state.completed_aggregates.append(agg)

                            # Store grid result
                            grid_result = M4GridResult(
                                test_name=test_name,
                                dr=dr,
                                dphi=dphi,
                                stats_level=stats_level,
                                power=agg.rejection_rate,
                                power_se=agg.rejection_rate_se,
                                pass_rate=agg.pass_rate,
                                n_trials=agg.n_trials
                            )
                            state.m4_grid_results.append(grid_result)

                            print(f"  Power: {agg.rejection_rate:.3f} +/- {agg.rejection_rate_se:.3f}")

                        # Checkpoint
                        if checkpoint_file:
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump(state, f)

                        work_done += 1
                        state.total_trials_done += len(trials)

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE: {state.total_trials_done} total trials")
    print(f"{'='*70}")

    return state


# ============================================================================
# Output Functions
# ============================================================================

def save_results(state: SweepState, config: Dict, output_dir: str):
    """Save all results to output files."""
    os.makedirs(output_dir, exist_ok=True)

    tests = config['tests']
    test_names = [t['name'] for t in tests]
    stats_multipliers = tests[0].get('stats_multipliers', [0.5, 1.0, 2.0, 4.0])

    # 1. POWER_TABLE_TOP3.csv
    csv_path = os.path.join(output_dir, 'POWER_TABLE_TOP3.csv')
    with open(csv_path, 'w') as f:
        header = ['test_name', 'data_type', 'nbins_a', 'nbins_b', 'stats_level',
                  'typeI_M0', 'typeI_M0_se', 'power_M1', 'power_M1_se',
                  'pass_rate_M0', 'identifiable_rate_M0', 'median_lambda_M0',
                  'pass_rate_M1', 'identifiable_rate_M1', 'median_lambda_M1']
        f.write(','.join(header) + '\n')

        for test_name in test_names:
            for stats in stats_multipliers:
                # Find M0 and M1 results
                m0 = next((a for a in state.completed_aggregates
                          if a.test_name == test_name and a.mechanism == 'M0'
                          and a.stats_level == stats), None)
                m1 = next((a for a in state.completed_aggregates
                          if a.test_name == test_name and a.mechanism == 'M1'
                          and a.stats_level == stats), None)

                if m0 or m1:
                    agg = m0 or m1
                    row = [
                        test_name,
                        agg.data_type,
                        str(agg.nbins_a),
                        str(agg.nbins_b),
                        f"{stats:.1f}",
                        f"{m0.rejection_rate:.4f}" if m0 else "nan",
                        f"{m0.rejection_rate_se:.4f}" if m0 else "nan",
                        f"{m1.rejection_rate:.4f}" if m1 else "nan",
                        f"{m1.rejection_rate_se:.4f}" if m1 else "nan",
                        f"{m0.pass_rate:.3f}" if m0 else "nan",
                        f"{m0.identifiability_rate:.3f}" if m0 else "nan",
                        f"{m0.median_lambda:.3f}" if m0 and not np.isnan(m0.median_lambda) else "nan",
                        f"{m1.pass_rate:.3f}" if m1 else "nan",
                        f"{m1.identifiability_rate:.3f}" if m1 else "nan",
                        f"{m1.median_lambda:.3f}" if m1 and not np.isnan(m1.median_lambda) else "nan"
                    ]
                    f.write(','.join(row) + '\n')

    print(f"Saved: {csv_path}")

    # 2. POWER_TABLE_TOP3.md
    md_path = os.path.join(output_dir, 'POWER_TABLE_TOP3.md')
    with open(md_path, 'w') as f:
        f.write("# Power & Calibration Results - Top 3 Decisive Tests\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Type I Error (M0) and Power (M1) by Stats Level\n\n")
        f.write("| Test | Stats | Type I (M0) | Power (M1) | Pass Rate | Identifiable |\n")
        f.write("|------|-------|-------------|------------|-----------|-------------|\n")

        for test_name in test_names:
            for stats in stats_multipliers:
                m0 = next((a for a in state.completed_aggregates
                          if a.test_name == test_name and a.mechanism == 'M0'
                          and a.stats_level == stats), None)
                m1 = next((a for a in state.completed_aggregates
                          if a.test_name == test_name and a.mechanism == 'M1'
                          and a.stats_level == stats), None)

                typeI = f"{m0.rejection_rate:.3f}+-{m0.rejection_rate_se:.3f}" if m0 else "-"
                power = f"{m1.rejection_rate:.3f}+-{m1.rejection_rate_se:.3f}" if m1 else "-"
                pass_r = f"{m0.pass_rate:.2f}" if m0 else "-"
                ident = f"{m0.identifiability_rate:.2f}" if m0 else "-"

                f.write(f"| {test_name} | {stats:.1f}x | {typeI} | {power} | {pass_r} | {ident} |\n")

        # Calibration check
        f.write("\n## Calibration Check\n\n")
        f.write("Expected Type I error: 0.05 (5%)\n\n")

        calibrated = []
        not_calibrated = []
        for test_name in test_names:
            m0_1x = next((a for a in state.completed_aggregates
                         if a.test_name == test_name and a.mechanism == 'M0'
                         and a.stats_level == 1.0), None)
            if m0_1x:
                if 0.02 < m0_1x.rejection_rate < 0.10:  # Within [2%, 10%]
                    calibrated.append(f"- {test_name}: {m0_1x.rejection_rate:.3f}")
                else:
                    not_calibrated.append(f"- {test_name}: {m0_1x.rejection_rate:.3f}")

        f.write("**Calibrated tests** (Type I in [2%, 10%]):\n")
        f.write('\n'.join(calibrated) if calibrated else "- None\n")
        f.write("\n\n**Potentially miscalibrated**:\n")
        f.write('\n'.join(not_calibrated) if not_calibrated else "- None\n")

    print(f"Saved: {md_path}")

    # 3. M4_DETECTABILITY_MAPS.md
    m4_path = os.path.join(output_dir, 'M4_DETECTABILITY_MAPS.md')
    with open(m4_path, 'w') as f:
        f.write("# M4 (Rank-2) Detectability Maps\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Power to detect rank-2 deviation as a function of (dr, dphi)\n\n")

        m4_grid = config.get('M4_grid', {})
        dr_values = m4_grid.get('dr_values', [0.05, 0.10, 0.20])
        dphi_values = m4_grid.get('dphi_values', [10, 20, 40])
        m4_stats_levels = m4_grid.get('stats_levels', [1.0, 2.0])

        for test_name in test_names:
            f.write(f"## {test_name}\n\n")

            for stats in m4_stats_levels:
                f.write(f"### Stats level: {stats}x\n\n")

                # Header
                header = ["dr\\dphi"] + [f"{d}deg" for d in dphi_values]
                f.write("| " + " | ".join(header) + " |\n")
                f.write("|" + "|".join(["---"] * len(header)) + "|\n")

                for dr in dr_values:
                    row = [f"{dr:.2f}"]
                    for dphi in dphi_values:
                        result = next((r for r in state.m4_grid_results
                                      if r.test_name == test_name
                                      and abs(r.dr - dr) < 0.001
                                      and abs(r.dphi - dphi) < 0.1
                                      and abs(r.stats_level - stats) < 0.1), None)
                        if result:
                            row.append(f"{result.power:.2f}")
                        else:
                            row.append("-")
                    f.write("| " + " | ".join(row) + " |\n")

                f.write("\n")

    print(f"Saved: {m4_path}")

    # 4. FINAL_RECOMMENDATIONS.md
    rec_path = os.path.join(output_dir, 'FINAL_RECOMMENDATIONS.md')
    with open(rec_path, 'w') as f:
        f.write("# Final Recommendations for Rank-1 Tests\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for test in tests:
            test_name = test['name']
            f.write(f"## {test_name}\n\n")
            f.write(f"**Description**: {test.get('description', 'N/A')}\n\n")

            # Find best stats level with >80% power
            best_stats = None
            for stats in sorted(stats_multipliers):
                m1 = next((a for a in state.completed_aggregates
                          if a.test_name == test_name and a.mechanism == 'M1'
                          and a.stats_level == stats), None)
                if m1 and not np.isnan(m1.rejection_rate) and m1.rejection_rate >= 0.8:
                    best_stats = stats
                    break

            # Get 1x results for baseline
            m0_1x = next((a for a in state.completed_aggregates
                         if a.test_name == test_name and a.mechanism == 'M0'
                         and a.stats_level == 1.0), None)
            m1_1x = next((a for a in state.completed_aggregates
                         if a.test_name == test_name and a.mechanism == 'M1'
                         and a.stats_level == 1.0), None)

            if m0_1x and m1_1x:
                f.write(f"### At 1x Statistics\n\n")
                f.write(f"- Type I error: {m0_1x.rejection_rate:.3f}\n")
                f.write(f"- Power vs M1: {m1_1x.rejection_rate:.3f}\n")
                f.write(f"- Pass rate: {m0_1x.pass_rate:.2f}\n")
                f.write(f"- Identifiability: {m0_1x.identifiability_rate:.2f}\n\n")

            if best_stats:
                f.write(f"### Recommended Minimum Statistics: **{best_stats}x**\n\n")
                f.write("At this level, the test achieves >=80% power to detect M1 deviations.\n\n")
            else:
                f.write("### Recommendation: **Insufficient power at tested levels**\n\n")
                f.write("Consider higher statistics or different analysis strategy.\n\n")

            # Identifiability concern?
            if m0_1x and m0_1x.identifiability_rate < 0.8:
                f.write("**Warning**: Low identifiability rate suggests potential phase ambiguity.\n\n")

            # Type I calibration concern?
            if m0_1x and (m0_1x.rejection_rate < 0.02 or m0_1x.rejection_rate > 0.10):
                f.write(f"**Warning**: Type I error ({m0_1x.rejection_rate:.3f}) deviates from expected 5%.\n\n")

        # Overall ranking
        f.write("## Overall Ranking\n\n")
        f.write("Tests ranked by power at 2x statistics:\n\n")

        rankings = []
        for test_name in test_names:
            m1_2x = next((a for a in state.completed_aggregates
                         if a.test_name == test_name and a.mechanism == 'M1'
                         and a.stats_level == 2.0), None)
            if m1_2x:
                rankings.append((test_name, m1_2x.rejection_rate, m1_2x.pass_rate))

        rankings.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)

        for i, (name, power, pass_rate) in enumerate(rankings, 1):
            f.write(f"{i}. **{name}**: Power = {power:.3f}, Pass rate = {pass_rate:.2f}\n")

    print(f"Saved: {rec_path}")

    # 5. REPRODUCIBILITY.md
    repro_path = os.path.join(output_dir, 'REPRODUCIBILITY.md')
    with open(repro_path, 'w') as f:
        f.write("# Reproducibility Information\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n")

        f.write("## Sweep Statistics\n\n")
        f.write(f"- Total trials: {state.total_trials_done}\n")
        f.write(f"- Start time: {state.start_time}\n")
        f.write(f"- Tests: {[t['name'] for t in tests]}\n")
        f.write(f"- Stats levels: {stats_multipliers}\n")
        f.write(f"- M4 grid: dr={config.get('M4_grid', {}).get('dr_values', [])}, "
               f"dphi={config.get('M4_grid', {}).get('dphi_values', [])}\n\n")

        f.write("## Commands to Reproduce\n\n")
        f.write("```bash\n")
        f.write("cd sim_rank_sweep_v2/code/src\n")
        f.write("nohup python3 -u sim_sweep.py --config ../configs/tests_top3.json \\\n")
        f.write("  --trials-m0 300 --trials-m1 300 --trials-m4 200 \\\n")
        f.write("  --output ../../out > ../../logs/sweep.log 2>&1 &\n")
        f.write("```\n")

    print(f"Saved: {repro_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run power & calibration sweep')
    parser.add_argument('--config', default='../configs/tests_top3.json',
                       help='Path to config file')
    parser.add_argument('--output', default='../../out',
                       help='Output directory')
    parser.add_argument('--trials-m0', type=int, default=300,
                       help='M0 trials per condition')
    parser.add_argument('--trials-m1', type=int, default=300,
                       help='M1 trials per condition')
    parser.add_argument('--trials-m4', type=int, default=200,
                       help='M4 trials per grid point')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--checkpoint', default=None,
                       help='Checkpoint file for resume')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode for testing')

    args = parser.parse_args()

    if args.quick:
        args.trials_m0 = 30
        args.trials_m1 = 30
        args.trials_m4 = 20

    # Resolve paths
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    output_dir = script_dir / args.output

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Run sweep
    checkpoint_file = args.checkpoint
    if checkpoint_file is None:
        checkpoint_file = str(output_dir / 'checkpoint.pkl')

    os.makedirs(output_dir, exist_ok=True)

    state = run_full_sweep(
        config=config,
        output_dir=str(output_dir),
        n_trials_m0=args.trials_m0,
        n_trials_m1=args.trials_m1,
        n_trials_m4=args.trials_m4,
        n_workers=args.workers,
        checkpoint_file=checkpoint_file
    )

    # Save results
    save_results(state, config, str(output_dir))

    print("\n" + "="*70)
    print("ALL OUTPUTS SAVED")
    print("="*70)


if __name__ == "__main__":
    main()
