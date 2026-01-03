"""
Pipeline state machine for processing candidates.

Implements the step-by-step workflow:
    STEP 0: scaffold       - Create directory structure
    STEP 1: locate_data    - Search for data sources (dry-run unless --execute)
    STEP 2: acquire        - Download data (requires --execute)
    STEP 3: extract_numeric - Extract tables from raw data
    STEP 4: run_rank1      - Execute rank-1 test harness
    STEP 5: summarize      - Update master table and finalize
"""

import gc
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .registry import Registry, CandidateState
from .candidates import Candidate, CandidateLoader


class PipelineStep(Enum):
    """Pipeline steps in order."""
    SCAFFOLD = "scaffold"
    LOCATE_DATA = "locate_data"
    ACQUIRE = "acquire"
    EXTRACT_NUMERIC = "extract_numeric"
    RUN_RANK1 = "run_rank1"
    SUMMARIZE = "summarize"

    @classmethod
    def ordered(cls) -> List["PipelineStep"]:
        return [
            cls.SCAFFOLD,
            cls.LOCATE_DATA,
            cls.ACQUIRE,
            cls.EXTRACT_NUMERIC,
            cls.RUN_RANK1,
            cls.SUMMARIZE,
        ]

    def next_step(self) -> Optional["PipelineStep"]:
        """Get the next step in the pipeline."""
        steps = self.ordered()
        try:
            idx = steps.index(self)
            if idx + 1 < len(steps):
                return steps[idx + 1]
        except ValueError:
            pass
        return None


class CandidateStatus(Enum):
    """Terminal and non-terminal statuses for candidates."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PLANNED = "PLANNED"  # Dry-run completed, ready for execution
    DONE = "DONE"
    NO_DATA = "NO_DATA"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"

    def is_terminal(self) -> bool:
        return self in {
            CandidateStatus.DONE,
            CandidateStatus.NO_DATA,
            CandidateStatus.BLOCKED,
            CandidateStatus.ERROR,
        }


@dataclass
class StepResult:
    """Result of executing a pipeline step."""
    success: bool
    status: CandidateStatus
    message: str
    data: Dict[str, Any] = None
    blocked_reason: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class Pipeline:
    """
    Manages the execution of pipeline steps for candidates.

    Supports:
    - Dry-run mode (--execute=False): Plans but doesn't download/run
    - Resume mode: Picks up from last incomplete candidate
    - Single candidate mode: Process just one candidate
    """

    def __init__(
        self,
        registry: Registry,
        candidate_loader: CandidateLoader,
        execute: bool = False,
        harness_path: Optional[Path] = None,
    ):
        self.registry = registry
        self.candidate_loader = candidate_loader
        self.execute = execute
        self.harness_path = harness_path
        self.logger = logging.getLogger("rank1_discovery_mine.pipeline")

        # Step handlers will be registered here
        self._step_handlers: Dict[PipelineStep, Callable] = {}
        self._register_handlers()

    def _register_handlers(self):
        """Register step handler functions."""
        self._step_handlers = {
            PipelineStep.SCAFFOLD: self._step_scaffold,
            PipelineStep.LOCATE_DATA: self._step_locate_data,
            PipelineStep.ACQUIRE: self._step_acquire,
            PipelineStep.EXTRACT_NUMERIC: self._step_extract_numeric,
            PipelineStep.RUN_RANK1: self._step_run_rank1,
            PipelineStep.SUMMARIZE: self._step_summarize,
        }

    def _get_candidate(self, slug: str) -> Candidate:
        """Get candidate config."""
        candidate = self.candidate_loader.get_candidate(slug)
        if candidate is None:
            raise ValueError(f"Unknown candidate: {slug}")
        return candidate

    def _write_log(self, slug: str, step: str, message: str):
        """Write to candidate's log file."""
        log_dir = self.registry.get_candidate_dir(slug) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{step}.log"
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _update_meta(self, slug: str, updates: Dict[str, Any]):
        """Update meta.json with new data."""
        meta_file = self.registry.get_candidate_dir(slug) / "meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}
        meta.update(updates)
        meta["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

    # -------------------------------------------------------------------------
    # Step Handlers
    # -------------------------------------------------------------------------

    def _step_scaffold(self, slug: str, candidate: Candidate) -> StepResult:
        """STEP 0: Create directory structure and meta.json."""
        self._write_log(slug, "scaffold", f"Starting scaffold for {slug}")

        try:
            self.registry.scaffold_candidate(slug, candidate.to_meta_dict())
            self._write_log(slug, "scaffold", "Directory structure created")

            return StepResult(
                success=True,
                status=CandidateStatus.IN_PROGRESS,
                message=f"Scaffolded {slug}",
            )
        except Exception as e:
            self._write_log(slug, "scaffold", f"ERROR: {e}")
            return StepResult(
                success=False,
                status=CandidateStatus.ERROR,
                message=f"Scaffold failed: {e}",
            )

    def _step_locate_data(self, slug: str, candidate: Candidate) -> StepResult:
        """
        STEP 1: Search for data sources.

        In dry-run mode (execute=False):
            - Plans queries but doesn't execute them
            - Writes planned queries to logs
            - Returns PLANNED status

        In execute mode:
            - Actually queries HEPData, arXiv, etc.
            - Saves discovered URLs to meta.json
        """
        from .sources import hepdata, arxiv, cern_opendata, github_search

        self._write_log(slug, "locate_data", f"Starting locate_data for {slug}")
        self._write_log(slug, "locate_data", f"Execute mode: {self.execute}")

        planned_queries = []
        discovered_urls = []

        # Plan queries based on preferred_sources order
        for source in candidate.preferred_sources:
            if source == "hepdata":
                queries = hepdata.plan_queries(candidate)
                planned_queries.extend([("hepdata", q) for q in queries])

                if self.execute:
                    urls = hepdata.execute_search(candidate)
                    discovered_urls.extend(urls)

            elif source == "arxiv_pdf":
                queries = arxiv.plan_queries(candidate)
                planned_queries.extend([("arxiv", q) for q in queries])

                if self.execute:
                    urls = arxiv.execute_search(candidate)
                    discovered_urls.extend(urls)

            elif source == "cern_opendata":
                queries = cern_opendata.plan_queries(candidate)
                planned_queries.extend([("cern_opendata", q) for q in queries])

                if self.execute:
                    urls = cern_opendata.execute_search(candidate)
                    discovered_urls.extend(urls)

            elif source == "github":
                queries = github_search.plan_queries(candidate)
                planned_queries.extend([("github", q) for q in queries])

                if self.execute:
                    urls = github_search.execute_search(candidate)
                    discovered_urls.extend(urls)

        # Log planned queries
        self._write_log(slug, "locate_data", "Planned queries:")
        for source, query in planned_queries:
            self._write_log(slug, "locate_data", f"  [{source}] {query}")

        # Update meta with planned queries
        self._update_meta(slug, {
            "planned_queries": [{"source": s, "query": q} for s, q in planned_queries],
            "discovered_urls": discovered_urls,
        })

        if not self.execute:
            self._write_log(slug, "locate_data", "DRY-RUN: Queries planned but not executed")
            return StepResult(
                success=True,
                status=CandidateStatus.PLANNED,
                message="Data location planned (dry-run)",
                data={"planned_queries": planned_queries},
            )

        if discovered_urls:
            self._write_log(slug, "locate_data", f"Found {len(discovered_urls)} URLs")
            return StepResult(
                success=True,
                status=CandidateStatus.IN_PROGRESS,
                message=f"Found {len(discovered_urls)} data sources",
                data={"discovered_urls": discovered_urls},
            )
        else:
            self._write_log(slug, "locate_data", "No data sources found")
            return StepResult(
                success=False,
                status=CandidateStatus.NO_DATA,
                message="No public data sources found",
                blocked_reason="NO_DATA_PUBLIC",
            )

    def _step_acquire(self, slug: str, candidate: Candidate) -> StepResult:
        """
        STEP 2: Download data files.

        Requires --execute flag.
        """
        from .sources import hepdata, arxiv

        self._write_log(slug, "acquire", f"Starting acquire for {slug}")

        if not self.execute:
            self._write_log(slug, "acquire", "DRY-RUN: Skipping downloads")
            return StepResult(
                success=True,
                status=CandidateStatus.PLANNED,
                message="Acquisition planned (dry-run)",
            )

        # Load discovered URLs from meta
        meta_file = self.registry.get_candidate_dir(slug) / "meta.json"
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        discovered_urls = meta.get("discovered_urls", [])
        if not discovered_urls:
            return StepResult(
                success=False,
                status=CandidateStatus.NO_DATA,
                message="No URLs to download",
                blocked_reason="NO_DATA_PUBLIC",
            )

        raw_dir = self.registry.get_candidate_dir(slug) / "raw"
        downloaded_files = []

        for url_info in discovered_urls:
            try:
                url = url_info if isinstance(url_info, str) else url_info.get("url", "")
                source_type = url_info.get("type", "unknown") if isinstance(url_info, dict) else "unknown"

                if "hepdata" in url.lower():
                    files = hepdata.download(url, raw_dir)
                    downloaded_files.extend(files)
                elif "arxiv" in url.lower():
                    files = arxiv.download(url, raw_dir)
                    downloaded_files.extend(files)
                else:
                    self._write_log(slug, "acquire", f"Unknown URL type: {url}")

            except Exception as e:
                self._write_log(slug, "acquire", f"Download error: {e}")

        self._update_meta(slug, {"downloaded_files": downloaded_files})

        if downloaded_files:
            return StepResult(
                success=True,
                status=CandidateStatus.IN_PROGRESS,
                message=f"Downloaded {len(downloaded_files)} files",
                data={"downloaded_files": downloaded_files},
            )
        else:
            return StepResult(
                success=False,
                status=CandidateStatus.BLOCKED,
                message="Downloads failed",
                blocked_reason="DOWNLOAD_FAILED",
            )

    def _step_extract_numeric(self, slug: str, candidate: Candidate) -> StepResult:
        """
        STEP 3: Extract numeric tables from raw data.
        """
        from .extract import pdf_tables
        from .adapters import to_rank1_inputs

        self._write_log(slug, "extract_numeric", f"Starting extraction for {slug}")

        if not self.execute:
            self._write_log(slug, "extract_numeric", "DRY-RUN: Skipping extraction")
            return StepResult(
                success=True,
                status=CandidateStatus.PLANNED,
                message="Extraction planned (dry-run)",
            )

        raw_dir = self.registry.get_candidate_dir(slug) / "raw"
        extracted_dir = self.registry.get_candidate_dir(slug) / "extracted"

        extracted_files = []

        # Try HEPData JSON/CSV first
        for f in raw_dir.glob("*.json"):
            try:
                csv_files = to_rank1_inputs.convert_hepdata_json(f, extracted_dir)
                extracted_files.extend(csv_files)
            except Exception as e:
                self._write_log(slug, "extract_numeric", f"HEPData conversion error: {e}")

        # Try PDF table extraction
        pdf_files = list(raw_dir.glob("*.pdf"))
        if pdf_files:
            self._write_log(slug, "extract_numeric", f"Processing {len(pdf_files)} PDF files")
        for f in pdf_files:
            try:
                tables = pdf_tables.extract_tables(f)
                for i, table in enumerate(tables):
                    csv_path = extracted_dir / f"pdf_table_{i}.csv"
                    table.to_csv(csv_path, index=False)
                    extracted_files.append(str(csv_path))
            except Exception as e:
                self._write_log(slug, "extract_numeric", f"PDF extraction error: {e}")
            finally:
                # Force cleanup after each PDF to prevent memory accumulation
                gc.collect()

        # Write README
        readme_path = extracted_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# Extracted Data for {slug}\n\n")
            f.write(f"Extraction timestamp: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("## Files\n\n")
            for ef in extracted_files:
                f.write(f"- {Path(ef).name}\n")

        if extracted_files:
            return StepResult(
                success=True,
                status=CandidateStatus.IN_PROGRESS,
                message=f"Extracted {len(extracted_files)} tables",
                data={"extracted_files": extracted_files},
            )
        else:
            return StepResult(
                success=False,
                status=CandidateStatus.BLOCKED,
                message="No tables extracted",
                blocked_reason="PDF_ONLY_NEEDS_EXTRACTION",
            )

    def _step_run_rank1(self, slug: str, candidate: Candidate) -> StepResult:
        """
        STEP 4: Run the rank-1 test harness.

        Requires --execute flag.
        """
        self._write_log(slug, "run_rank1", f"Starting rank-1 test for {slug}")

        candidate_dir = self.registry.get_candidate_dir(slug)
        out_dir = candidate_dir / "out"

        if not self.execute:
            self._write_log(slug, "run_rank1", "DRY-RUN: Skipping rank-1 test")

            return StepResult(
                success=True,
                status=CandidateStatus.PLANNED,
                message="Rank-1 test planned (dry-run)",
                data={"candidate_dir": str(candidate_dir)},
            )

        # Check if we have state configuration
        from .harness.state_configs import get_state_config
        config = get_state_config(slug)

        if config is None:
            self._write_log(slug, "run_rank1", f"No state configuration for {slug}")
            return StepResult(
                success=False,
                status=CandidateStatus.BLOCKED,
                message=f"No state configuration for {slug}",
                blocked_reason="NO_STATE_CONFIG",
            )

        # Check if we have raw data
        raw_dir = candidate_dir / "raw"
        csv_files = list(raw_dir.glob("hepdata_*.csv"))

        if not csv_files:
            self._write_log(slug, "run_rank1", "No HEPData CSV files found")
            return StepResult(
                success=False,
                status=CandidateStatus.NO_DATA,
                message="No HEPData CSV files available for testing",
                blocked_reason="NO_DATA",
            )

        # Run the rank-1 test
        self._write_log(slug, "run_rank1", f"Running rank-1 test on {len(csv_files)} tables")

        try:
            from .harness.run_test import run_rank1_test
            results = run_rank1_test(candidate_dir, slug, n_boot=100)

            status = results.get("status", "ERROR")
            verdict = results.get("overall_verdict", "UNKNOWN")

            self._write_log(slug, "run_rank1", f"Test completed: {status}, verdict: {verdict}")

            if status == "COMPLETED":
                return StepResult(
                    success=True,
                    status=CandidateStatus.DONE,
                    message=f"Rank-1 test completed: {verdict}",
                    data={"verdict": verdict, "pairs_tested": len(results.get("pairs", []))},
                )
            elif status == "NO_SUITABLE_DATA":
                return StepResult(
                    success=False,
                    status=CandidateStatus.BLOCKED,
                    message="No suitable table pairs found for testing",
                    blocked_reason="NO_SUITABLE_PAIRS",
                )
            else:
                return StepResult(
                    success=False,
                    status=CandidateStatus.ERROR,
                    message=f"Rank-1 test failed: {results.get('error', 'Unknown error')}",
                )

        except Exception as e:
            self.logger.exception(f"Rank-1 test error for {slug}")
            self._write_log(slug, "run_rank1", f"ERROR: {e}")
            return StepResult(
                success=False,
                status=CandidateStatus.ERROR,
                message=f"Rank-1 test error: {e}",
            )

    def _step_summarize(self, slug: str, candidate: Candidate) -> StepResult:
        """
        STEP 5: Update master table and finalize.
        """
        self._write_log(slug, "summarize", f"Starting summarize for {slug}")

        # Load current state
        state = self.registry.load_candidate_state(slug)
        meta_file = self.registry.get_candidate_dir(slug) / "meta.json"

        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Determine final status
        result_file = self.registry.get_candidate_dir(slug) / "out" / "result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
            verdict = result.get("verdict", "UNKNOWN")
            p_value = result.get("p_boot", "N/A")
            final_status = CandidateStatus.DONE
        else:
            verdict = "N/A"
            p_value = "N/A"
            if state and state.blocked_reason:
                final_status = CandidateStatus.BLOCKED
            else:
                final_status = CandidateStatus.PLANNED if not self.execute else CandidateStatus.BLOCKED

        # Append to master table
        self.registry.append_master_table_row({
            "slug": slug,
            "title": candidate.title,
            "status": final_status.value,
            "data_sources_found": len(meta.get("discovered_urls", [])),
            "channels_extracted": len(list(
                (self.registry.get_candidate_dir(slug) / "extracted").glob("channel*.csv")
            )),
            "p_value": p_value,
            "verdict": verdict,
            "blocker_reason": state.blocked_reason if state else "",
        })

        # Update global registry
        self.registry.update_candidate_status(slug, final_status.value)

        return StepResult(
            success=True,
            status=final_status,
            message=f"Summarized {slug}: {final_status.value}",
        )

    # -------------------------------------------------------------------------
    # Main Execution Methods
    # -------------------------------------------------------------------------

    def run_step(self, slug: str, step: PipelineStep) -> StepResult:
        """Execute a single pipeline step."""
        candidate = self._get_candidate(slug)
        handler = self._step_handlers.get(step)

        if handler is None:
            return StepResult(
                success=False,
                status=CandidateStatus.ERROR,
                message=f"No handler for step: {step}",
            )

        try:
            result = handler(slug, candidate)

            # Update candidate state
            state = self.registry.load_candidate_state(slug)
            if state is None:
                state = CandidateState(
                    slug=slug,
                    current_step=step.value,
                    completed_steps=[],
                    last_update_utc=datetime.now(timezone.utc).isoformat(),
                )

            if result.success:
                if step.value not in state.completed_steps:
                    state.completed_steps.append(step.value)
                next_step = step.next_step()
                state.current_step = next_step.value if next_step else "DONE"
            else:
                state.blocked_reason = result.blocked_reason

            self.registry.save_candidate_state(state)
            self.registry.update_candidate_status(slug, result.status.value)

            return result

        except Exception as e:
            self.logger.exception(f"Error in step {step} for {slug}")
            self.registry.append_error(slug, step.value, str(e))
            return StepResult(
                success=False,
                status=CandidateStatus.ERROR,
                message=f"Step {step} failed: {e}",
            )

    def run_candidate(self, slug: str, start_step: Optional[PipelineStep] = None) -> CandidateStatus:
        """
        Run all pipeline steps for a candidate.

        Returns final status.
        """
        steps = PipelineStep.ordered()

        # Find starting point
        start_idx = 0
        if start_step:
            try:
                start_idx = steps.index(start_step)
            except ValueError:
                pass
        else:
            # Check existing state for resume
            state = self.registry.load_candidate_state(slug)
            if state and state.completed_steps:
                # If in execute mode, check if locate_data was only dry-run
                # (i.e., discovered_urls is empty). If so, re-run from locate_data.
                if self.execute and "locate_data" in state.completed_steps:
                    meta_file = self.registry.get_candidate_dir(slug) / "meta.json"
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                        if not meta.get("discovered_urls"):
                            # Re-run locate_data since it was only planned
                            start_idx = steps.index(PipelineStep.LOCATE_DATA)
                            state.completed_steps.remove("locate_data")
                            self.registry.save_candidate_state(state)
                            self.logger.info(f"Re-running locate_data for {slug} (was dry-run)")

                # Start from first incomplete step
                if start_idx == 0:
                    for i, step in enumerate(steps):
                        if step.value not in state.completed_steps:
                            start_idx = i
                            break

        # Run steps
        final_status = CandidateStatus.IN_PROGRESS
        for step in steps[start_idx:]:
            self.logger.info(f"Running {step.value} for {slug}")
            result = self.run_step(slug, step)

            if not result.success or result.status.is_terminal():
                final_status = result.status
                break

            final_status = result.status

        return final_status

    def run_all(self, resume: bool = True) -> Dict[str, CandidateStatus]:
        """
        Run pipeline for all candidates.

        If resume=True, skips candidates in terminal states.
        """
        results = {}
        slugs = self.candidate_loader.get_slugs()

        for slug in slugs:
            if resume:
                state = self.registry.load_global_state()
                if state:
                    status = state.candidate_statuses.get(slug, "PENDING")
                    if status in Registry.TERMINAL_STATES:
                        self.logger.info(f"Skipping {slug} (status: {status})")
                        results[slug] = CandidateStatus(status)
                        continue

            self.logger.info(f"Processing {slug}")
            results[slug] = self.run_candidate(slug)

            # Force garbage collection after each candidate to prevent memory accumulation
            # This is important when processing many candidates with large PDFs/data
            gc.collect()

        return results

    def plan_all(self) -> int:
        """
        Scaffold all candidates without executing downloads.

        Returns number of candidates scaffolded.
        """
        slugs = self.candidate_loader.get_slugs()

        # Initialize global state
        self.registry.initialize_global_state(slugs)

        count = 0
        for slug in slugs:
            candidate = self._get_candidate(slug)

            # Scaffold
            result = self.run_step(slug, PipelineStep.SCAFFOLD)
            if result.success:
                count += 1

            # Plan data location (dry-run)
            self.run_step(slug, PipelineStep.LOCATE_DATA)

            # Cleanup between candidates
            gc.collect()

        return count
