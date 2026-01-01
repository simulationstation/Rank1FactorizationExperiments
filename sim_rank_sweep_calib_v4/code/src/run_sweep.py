#!/usr/bin/env python3
"""
run_sweep.py - Main runner script for the power study

This is the entry point for running the full simulation sweep.
Designed to be run with nohup for long-running analysis.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure we're in the right directory
script_dir = Path(__file__).parent.parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir / 'src'))

from sim_sweep import run_full_sweep, save_results, generate_summary


def main():
    """Run the full power study."""
    print("="*70)
    print("RANK-1 BOTTLENECK POWER STUDY - FULL SWEEP")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print("="*70)

    # Load config
    config_path = 'configs/tests.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    settings = config.get('global_settings', {})

    # Run parameters
    n_trials = settings.get('n_mc_trials', 100)
    n_bootstrap = settings.get('n_bootstrap', 200)
    stats_levels = settings.get('stats_levels', [0.5, 1.0, 2.0, 4.0, 8.0])

    print(f"Tests: {len(config['tests'])}")
    print(f"Trials per condition: {n_trials}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Stats levels: {stats_levels}")
    print("="*70)

    # Run sweep
    try:
        results = run_full_sweep(
            config_path=config_path,
            output_dir='out',
            n_trials=n_trials,
            n_bootstrap=n_bootstrap,
            stats_levels=stats_levels
        )

        # Save results
        save_results(results, 'out')
        generate_summary(results, 'out')

        print("\n" + "="*70)
        print("SWEEP COMPLETED SUCCESSFULLY")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Log completion
        with open('logs/COMMANDS.txt', 'a') as f:
            f.write(f"{datetime.now()}: Power study completed - {len(results)} results\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        with open('logs/COMMANDS.txt', 'a') as f:
            f.write(f"{datetime.now()}: Power study FAILED - {e}\n")

        sys.exit(1)


if __name__ == "__main__":
    main()
