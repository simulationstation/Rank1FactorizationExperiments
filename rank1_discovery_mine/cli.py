"""
Command-line interface for Rank1 Discovery Mine.

Usage:
    python -m rank1_discovery_mine plan          # Build plan + scaffold directories
    python -m rank1_discovery_mine validate      # Fast schema checks
    python -m rank1_discovery_mine status        # Show current progress
    python -m rank1_discovery_mine run --resume  # Resume processing (requires --execute)
    python -m rank1_discovery_mine run --one <slug>  # Process single candidate
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .registry import Registry
from .candidates import CandidateLoader, validate_candidates, get_default_config_path
from .pipeline import Pipeline, PipelineStep
from .reporting.write_master_table import initialize_master_table


def setup_logging(verbose: bool = False):
    """Configure logging.

    IMPORTANT: Only enable DEBUG for our own loggers.
    Third-party libraries (especially pdfminer) generate millions of DEBUG
    log lines when processing PDFs, causing memory exhaustion.
    """
    # Silence noisy third-party loggers that cause memory issues
    noisy_loggers = [
        'pdfminer',
        'pdfminer.pdfparser',
        'pdfminer.pdfdocument',
        'pdfminer.pdfpage',
        'pdfminer.pdfinterp',
        'pdfminer.psparser',
        'pdfplumber',
        'urllib3',
        'requests',
        'numba',
        'PIL',
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Ensure our package loggers get the requested level
    logging.getLogger("rank1_discovery_mine").setLevel(level)


def get_paths(args) -> tuple:
    """Get paths from args or defaults."""
    # Repository root (parent of rank1_discovery_mine package)
    repo_root = Path(__file__).parent.parent

    config_path = Path(args.config) if args.config else get_default_config_path()
    discoveries_path = Path(args.discoveries) if args.discoveries else repo_root / "discoveries"

    return config_path, discoveries_path


def cmd_plan(args):
    """
    Build plan and scaffold directories for all candidates.

    Does NOT execute downloads or tests.
    """
    config_path, discoveries_path = get_paths(args)

    print(f"Config: {config_path}")
    print(f"Discoveries: {discoveries_path}")
    print()

    # Load and validate candidates
    loader = CandidateLoader(config_path)

    print("Validating candidates...")
    is_valid, errors = loader.validate(strict=not args.lenient)

    if not is_valid:
        print("Validation errors:")
        for err in errors:
            print(f"  - {err}")
        if not args.lenient:
            print("\nUse --lenient to continue despite errors.")
            return 1

    candidates = loader.load()
    print(f"Loaded {len(candidates)} candidates")
    print()

    # Initialize registry
    registry = Registry(discoveries_path)
    registry.ensure_directories()

    # Initialize master table
    initialize_master_table(registry.registry_path)

    # Create pipeline (dry-run mode)
    pipeline = Pipeline(
        registry=registry,
        candidate_loader=loader,
        execute=False,  # Never execute in plan mode
    )

    # Scaffold all candidates
    print("Scaffolding candidates...")
    count = pipeline.plan_all()

    print()
    print(f"Scaffolded {count} candidates")
    print(f"Discoveries directory: {discoveries_path}")
    print(f"Registry: {registry.registry_path}")
    print()
    print("Next steps:")
    print("  python -m rank1_discovery_mine status    # View status")
    print("  python -m rank1_discovery_mine validate  # Validate setup")
    print("  python -m rank1_discovery_mine run --resume --execute  # Run with downloads")

    return 0


def cmd_validate(args):
    """
    Validate configuration and setup.

    Fast, offline checks only. v2.0: Includes testability checks.
    """
    config_path, discoveries_path = get_paths(args)

    print("=" * 60)
    print("Rank1 Discovery Mine - Validation (v2.0)")
    print("=" * 60)
    print()

    errors = []
    warnings = []

    # 1. Validate config file exists
    print(f"Config file: {config_path}")
    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}")
        print("  [FAIL] File not found")
    else:
        print("  [OK] File exists")

        # 2. Validate YAML schema
        print()
        print("Validating YAML schema...")
        loader = CandidateLoader(config_path)
        is_valid, schema_errors = loader.validate(strict=True)

        if is_valid:
            print("  [OK] Schema valid")
            candidates = loader.load()
            print(f"  [OK] Loaded {len(candidates)} candidates")

            # 3. v2.0: Testability analysis
            print()
            print("Testability analysis:")
            testable = []
            not_testable = []
            disabled = []
            completed_elsewhere = []
            pinned_sources = []

            for slug, candidate in candidates.items():
                if not candidate.enabled:
                    disabled.append(slug)
                elif candidate.completed_elsewhere:
                    completed_elsewhere.append((slug, candidate.completed_elsewhere))
                elif candidate.not_testable_reason:
                    not_testable.append((slug, candidate.not_testable_reason))
                else:
                    testable.append(slug)

                if candidate.pinned_hepdata_record:
                    pinned_sources.append((slug, candidate.pinned_hepdata_record))

            print(f"  [OK] Testable candidates: {len(testable)}")

            if not_testable:
                print(f"  [WARN] NOT_TESTABLE candidates: {len(not_testable)}")
                for slug, reason in not_testable[:5]:
                    print(f"    - {slug}: {reason}")
                if len(not_testable) > 5:
                    print(f"    ... and {len(not_testable) - 5} more")
                warnings.extend([f"{s}: {r}" for s, r in not_testable])

            if disabled:
                print(f"  [INFO] Disabled candidates: {len(disabled)}")
                for slug in disabled[:5]:
                    print(f"    - {slug}")

            if completed_elsewhere:
                print(f"  [INFO] Completed elsewhere: {len(completed_elsewhere)}")
                for slug, path in completed_elsewhere[:5]:
                    print(f"    - {slug} -> {path}")

            if pinned_sources:
                print(f"  [OK] Pinned HEPData sources: {len(pinned_sources)}")
                for slug, record in pinned_sources[:5]:
                    print(f"    - {slug}: {record}")

        else:
            print("  [FAIL] Schema errors:")
            for err in schema_errors:
                print(f"    - {err}")
                errors.append(err)

    # 4. Validate discoveries directory
    print()
    print(f"Discoveries directory: {discoveries_path}")
    if discoveries_path.exists():
        print("  [OK] Directory exists")

        # Check registry
        registry = Registry(discoveries_path)
        state = registry.load_global_state()
        if state:
            print(f"  [OK] Registry initialized ({len(state.candidate_slugs)} candidates)")

            # Check for source audits
            audit_count = 0
            for slug in state.candidate_slugs:
                audit_file = discoveries_path / slug / "out" / "source_audit.json"
                if audit_file.exists():
                    audit_count += 1
            if audit_count > 0:
                print(f"  [OK] Source audits available: {audit_count}")
        else:
            print("  [INFO] Registry not initialized (run 'plan' first)")
    else:
        print("  [INFO] Directory does not exist (will be created by 'plan')")

    # 5. Check for required dependencies
    print()
    print("Checking dependencies...")
    deps = [
        ("yaml", "pyyaml"),
        ("pandas", "pandas"),
        ("pdfplumber", "pdfplumber"),
        ("requests", "requests"),
    ]

    for module, package in deps:
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [WARN] {package} not installed (some features may not work)")

    # Summary
    print()
    print("=" * 60)
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s)")
        return 1
    elif warnings:
        print(f"Validation PASSED with {len(warnings)} warning(s)")
        return 0
    else:
        print("Validation PASSED")
        return 0


def cmd_status(args):
    """
    Show current progress and status.
    """
    config_path, discoveries_path = get_paths(args)

    print("=" * 60)
    print("Rank1 Discovery Mine - Status")
    print("=" * 60)
    print()

    registry = Registry(discoveries_path)
    summary = registry.get_status_summary()

    if not summary.get("initialized"):
        print("Registry not initialized.")
        print("Run 'python -m rank1_discovery_mine plan' first.")
        return 0

    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Last updated: {summary['last_update']}")
    print()

    # Status breakdown
    print("Status breakdown:")
    for status, count in sorted(summary['status_counts'].items()):
        pct = 100 * count / summary['total_candidates']
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    # Progress
    print(f"Processed: {summary['total_processed']}")
    print(f"  - Success: {summary['total_success']}")
    print(f"  - No data: {summary['total_no_data']}")
    print(f"  - Blocked: {summary['total_blocked']}")
    print(f"  - Error: {summary['total_error']}")
    print()

    # Next candidate
    if summary['next_candidate']:
        print(f"Next candidate: {summary['next_candidate']}")
    else:
        print("All candidates processed!")

    # Show detailed status if requested
    if args.detailed:
        print()
        print("=" * 60)
        print("Detailed Status")
        print("=" * 60)

        state = registry.load_global_state()
        for slug in state.candidate_slugs:
            status = state.candidate_statuses.get(slug, "PENDING")
            print(f"  {slug}: {status}")

    return 0


def cmd_run(args):
    """
    Run the discovery pipeline.

    By default, runs in dry-run mode. Use --execute to actually download/run.
    """
    config_path, discoveries_path = get_paths(args)

    if not args.execute:
        print("=" * 60)
        print("DRY-RUN MODE (use --execute to actually download/run)")
        print("=" * 60)
        print()

    # Load candidates
    loader = CandidateLoader(config_path)
    candidates = loader.load()

    # Initialize registry
    registry = Registry(discoveries_path)

    # Check if initialized
    state = registry.load_global_state()
    if state is None:
        print("Registry not initialized. Run 'plan' first.")
        return 1

    # Create pipeline
    pipeline = Pipeline(
        registry=registry,
        candidate_loader=loader,
        execute=args.execute,
    )

    if args.one:
        # Process single candidate
        slug = args.one
        if slug not in candidates:
            print(f"Unknown candidate: {slug}")
            print(f"Available: {', '.join(list(candidates.keys())[:10])}...")
            return 1

        print(f"Processing: {slug}")
        result = pipeline.run_candidate(slug)
        print(f"Result: {result.value}")

    elif args.resume:
        # Resume from where we left off
        print("Resuming pipeline...")
        results = pipeline.run_all(resume=True)

        print()
        print("Results:")
        for slug, status in results.items():
            print(f"  {slug}: {status.value}")

    else:
        print("Specify --resume or --one <slug>")
        return 1

    return 0


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="rank1_discovery_mine",
        description="Automated exotic hadron data discovery and rank-1 testing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to discovery_candidates.yaml",
    )
    parser.add_argument(
        "-d", "--discoveries",
        type=str,
        help="Path to discoveries directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # plan command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Build plan and scaffold directories",
    )
    plan_parser.add_argument(
        "--lenient",
        action="store_true",
        help="Continue despite validation errors",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and setup",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show current progress",
    )
    status_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-candidate status",
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the discovery pipeline",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete candidate",
    )
    run_parser.add_argument(
        "--one",
        type=str,
        metavar="SLUG",
        help="Process only the specified candidate",
    )
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute downloads and tests (default: dry-run)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)

    # Dispatch to command handler
    handlers = {
        "plan": cmd_plan,
        "validate": cmd_validate,
        "status": cmd_status,
        "run": cmd_run,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
