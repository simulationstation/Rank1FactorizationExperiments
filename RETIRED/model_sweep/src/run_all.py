#!/usr/bin/env python3
"""
Model Sweep Orchestrator
========================

Runs rank-1 bottleneck tests for all models in configs/models.yaml.
Parallelizes model runs across available CPU cores.

Usage:
    python run_all.py
    python run_all.py --parallel   # Run models in parallel
    python run_all.py --only testable  # Only run testable models
"""

import yaml
import json
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime
import sys
import argparse
import traceback

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_model import run_model

# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
RUNS_DIR = BASE_DIR / "runs"
OUT_DIR = BASE_DIR / "out"
LOGS_DIR = BASE_DIR / "logs"

for d in [RUNS_DIR, OUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# WORKER
# ==============================================================================

def run_model_safe(config):
    """
    Run model with exception handling.
    """
    model_name = config.get('name', 'unknown')

    try:
        result = run_model(config)
        return result
    except Exception as e:
        print(f"ERROR running {model_name}: {e}")
        traceback.print_exc()
        return {
            'model': model_name,
            'verdict': 'ERROR',
            'reason': str(e)
        }


# ==============================================================================
# SUMMARY GENERATION
# ==============================================================================

def generate_summary(results):
    """
    Generate master summary table from all results.
    """
    rows = []

    for r in results:
        row = {
            'model': r.get('model', ''),
            'display_name': r.get('display_name', ''),
            'paper_ref': r.get('paper_ref', ''),
            'verdict': r.get('verdict', 'ERROR'),
            'R_A': r.get('channel_A', {}).get('r', ''),
            'R_B': r.get('channel_B', {}).get('r', ''),
            'R_shared': r.get('shared', {}).get('r', ''),
            'phi_A': r.get('channel_A', {}).get('phi_deg', ''),
            'phi_B': r.get('channel_B', {}).get('phi_deg', ''),
            'phi_shared': r.get('shared', {}).get('phi_deg', ''),
            'Lambda': r.get('likelihood_ratio', {}).get('Lambda', ''),
            'p_boot': r.get('bootstrap', {}).get('p_value', ''),
            'chi2_A': r.get('channel_A', {}).get('chi2_dof', ''),
            'chi2_B': r.get('channel_B', {}).get('chi2_dof', ''),
            'health_A': r.get('channel_A', {}).get('health', ''),
            'health_B': r.get('channel_B', {}).get('health', ''),
            'phase_ident': r.get('optimizer_stability', {}).get('phase_identifiable', ''),
            'r_stable': r.get('optimizer_stability', {}).get('r_stable', ''),
            'in_both_95': r.get('contour_check', {}).get('shared_in_both_95', ''),
            'notes': r.get('reason', '')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_markdown_summary(df, results):
    """
    Generate markdown summary report.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    n_total = len(df)
    n_supported = len(df[df['verdict'] == 'SUPPORTED'])
    n_disfavored = len(df[df['verdict'] == 'DISFAVORED'])
    n_inconclusive = len(df[df['verdict'] == 'INCONCLUSIVE'])
    n_mismatch = len(df[df['verdict'] == 'MODEL MISMATCH'])
    n_nodata = len(df[df['verdict'] == 'NO DATA'])
    n_error = len(df[df['verdict'].isin(['ERROR', 'OPTIMIZER FAILURE'])])

    md = f"""# Rank-1 Bottleneck Test Model Sweep Summary

Generated: {timestamp}

## Overview

| Verdict | Count |
|---------|-------|
| SUPPORTED | {n_supported} |
| DISFAVORED | {n_disfavored} |
| INCONCLUSIVE | {n_inconclusive} |
| MODEL MISMATCH | {n_mismatch} |
| NO DATA | {n_nodata} |
| ERROR/FAILURE | {n_error} |
| **Total** | **{n_total}** |

## Testable Models (Numeric Data Available)

| Model | Verdict | r_A | r_B | r_shared | Lambda | p_boot | Notes |
|-------|---------|-----|-----|----------|--------|--------|-------|
"""

    for _, row in df.iterrows():
        if row['verdict'] not in ['NO DATA', 'ERROR', 'needs_verification']:
            r_A = f"{row['R_A']:.3f}" if isinstance(row['R_A'], float) else str(row['R_A'])
            r_B = f"{row['R_B']:.3f}" if isinstance(row['R_B'], float) else str(row['R_B'])
            r_sh = f"{row['R_shared']:.3f}" if isinstance(row['R_shared'], float) else str(row['R_shared'])
            lam = f"{row['Lambda']:.3f}" if isinstance(row['Lambda'], float) else str(row['Lambda'])
            p = f"{row['p_boot']:.4f}" if isinstance(row['p_boot'], float) else str(row['p_boot'])

            md += f"| {row['display_name']} | **{row['verdict']}** | {r_A} | {r_B} | {r_sh} | {lam} | {p} | {row['notes'][:50]}... |\n"

    md += """

## Models Without Numeric Data

| Model | Status | Notes |
|-------|--------|-------|
"""

    for _, row in df.iterrows():
        if row['verdict'] in ['NO DATA', 'needs_verification']:
            md += f"| {row['display_name']} | {row['verdict']} | {row['notes']} |\n"

    md += """

## Strongest Candidates for Paper Inclusion

Models with SUPPORTED verdict and high statistical significance:

"""

    supported = df[df['verdict'] == 'SUPPORTED']
    if len(supported) > 0:
        for _, row in supported.iterrows():
            if isinstance(row['p_boot'], float) and row['p_boot'] > 0.1:
                md += f"- **{row['display_name']}**: p_boot = {row['p_boot']:.3f}, Lambda = {row['Lambda']:.3f}\n"

        if len(supported[supported['p_boot'].apply(lambda x: isinstance(x, float) and x > 0.1)]) == 0:
            for _, row in supported.iterrows():
                md += f"- **{row['display_name']}**: {row['notes']}\n"
    else:
        md += "_No models with SUPPORTED verdict._\n"

    md += """

## Email Requests Generated

"""

    requests_dir = OUT_DIR / "requests"
    if requests_dir.exists():
        email_files = list(requests_dir.glob("*.txt"))
        if email_files:
            for ef in email_files:
                md += f"- `{ef.name}`\n"
        else:
            md += "_No email requests generated._\n"
    else:
        md += "_No email requests generated._\n"

    md += """

---
*Generated by model_sweep rank-1 bottleneck test framework*
"""

    return md


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run rank-1 bottleneck sweep for all models')
    parser.add_argument('--parallel', action='store_true', help='Run models in parallel')
    parser.add_argument('--only', type=str, choices=['testable', 'all'],
                       default='testable', help='Which models to run')
    parser.add_argument('--models', type=str, nargs='+', help='Specific model names to run')

    args = parser.parse_args()

    print("=" * 70)
    print("RANK-1 BOTTLENECK TEST MODEL SWEEP")
    print("=" * 70)
    print()

    # Load configs
    with open(CONFIGS_DIR / 'models.yaml', 'r') as f:
        all_configs = yaml.safe_load(f)

    models = all_configs['models']

    # Filter models
    if args.models:
        models = [m for m in models if m['name'] in args.models]
    elif args.only == 'testable':
        models = [m for m in models if m.get('status') == 'testable']

    print(f"Running {len(models)} models...")
    print()

    # Run models
    results = []

    if args.parallel and len(models) > 1:
        n_workers = min(len(models), max(1, cpu_count() - 1))
        print(f"Running in parallel with {n_workers} workers...")

        with Pool(n_workers) as pool:
            results = pool.map(run_model_safe, models)
    else:
        for config in models:
            result = run_model_safe(config)
            results.append(result)

    # Generate summary
    print()
    print("=" * 70)
    print("GENERATING SUMMARY")
    print("=" * 70)

    df = generate_summary(results)

    # Save CSV
    csv_path = OUT_DIR / 'MASTER_SUMMARY.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Save markdown
    md_content = generate_markdown_summary(df, results)
    md_path = OUT_DIR / 'MASTER_SUMMARY.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  Saved MD: {md_path}")

    # Save all results as JSON
    json_path = OUT_DIR / 'all_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # Print summary
    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print()

    print("Verdicts:")
    for verdict in ['SUPPORTED', 'DISFAVORED', 'INCONCLUSIVE', 'MODEL MISMATCH', 'NO DATA', 'ERROR']:
        count = len(df[df['verdict'] == verdict])
        if count > 0:
            print(f"  {verdict}: {count}")

    print()
    print("Testable models:")
    for _, row in df.iterrows():
        if row['verdict'] not in ['NO DATA', 'ERROR', 'needs_verification']:
            print(f"  {row['model']}: {row['verdict']}")

    print()
    print("Models requiring data requests:")
    for _, row in df.iterrows():
        if row['verdict'] == 'NO DATA':
            print(f"  {row['model']}")

    # Return exit code based on results
    n_errors = len(df[df['verdict'].isin(['ERROR', 'OPTIMIZER FAILURE'])])
    return 0 if n_errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
