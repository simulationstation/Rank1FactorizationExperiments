"""
Registry module for managing global and per-candidate state.

Provides atomic file writes and resumable state tracking.
"""

import json
import os
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import csv


@dataclass
class CandidateState:
    """State for a single candidate."""
    slug: str
    current_step: str
    completed_steps: List[str]
    last_update_utc: str
    blocked_reason: Optional[str] = None
    error_message: Optional[str] = None
    discovered_urls: List[str] = None

    def __post_init__(self):
        if self.discovered_urls is None:
            self.discovered_urls = []

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CandidateState":
        return cls(**data)


@dataclass
class GlobalState:
    """Global registry state."""
    version: str
    candidate_slugs: List[str]
    candidate_statuses: Dict[str, str]  # slug -> terminal status or current step
    next_candidate_index: int
    last_update_utc: str
    total_processed: int = 0
    total_success: int = 0
    total_no_data: int = 0
    total_blocked: int = 0
    total_error: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "GlobalState":
        return cls(**data)


class Registry:
    """
    Manages the discoveries registry with atomic writes.

    Registry structure:
        discoveries/_registry/
            state.json          - Global progress state
            MASTER_TABLE.csv    - One row per candidate attempt
            MASTER_TABLE.md     - Markdown version of master table
            errors.jsonl        - Append-only error log
            README.md           - Documentation
    """

    TERMINAL_STATES = {"DONE", "NO_DATA", "BLOCKED", "ERROR"}

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.registry_path = self.base_path / "_registry"
        self.state_file = self.registry_path / "state.json"
        self.master_csv = self.registry_path / "MASTER_TABLE.csv"
        self.master_md = self.registry_path / "MASTER_TABLE.md"
        self.errors_file = self.registry_path / "errors.jsonl"

    def ensure_directories(self):
        """Create registry directories if they don't exist."""
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, path: Path, data: Dict):
        """Write JSON atomically using tmp file + rename."""
        self.ensure_directories()
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="tmp_",
            dir=self.registry_path
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _atomic_write_text(self, path: Path, content: str):
        """Write text atomically using tmp file + rename."""
        self.ensure_directories()
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="tmp_",
            dir=self.registry_path
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                f.write(content)
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load_global_state(self) -> Optional[GlobalState]:
        """Load global state from state.json."""
        if not self.state_file.exists():
            return None
        with open(self.state_file, 'r') as f:
            data = json.load(f)
        return GlobalState.from_dict(data)

    def save_global_state(self, state: GlobalState):
        """Save global state atomically."""
        state.last_update_utc = datetime.now(timezone.utc).isoformat()
        self._atomic_write_json(self.state_file, state.to_dict())

    def initialize_global_state(self, candidate_slugs: List[str]) -> GlobalState:
        """Initialize a new global state."""
        state = GlobalState(
            version="1.0",
            candidate_slugs=candidate_slugs,
            candidate_statuses={slug: "PENDING" for slug in candidate_slugs},
            next_candidate_index=0,
            last_update_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.save_global_state(state)
        return state

    def get_candidate_dir(self, slug: str) -> Path:
        """Get the directory path for a candidate."""
        return self.base_path / slug

    def load_candidate_state(self, slug: str) -> Optional[CandidateState]:
        """Load per-candidate status.json."""
        status_file = self.get_candidate_dir(slug) / "status.json"
        if not status_file.exists():
            return None
        with open(status_file, 'r') as f:
            data = json.load(f)
        return CandidateState.from_dict(data)

    def save_candidate_state(self, state: CandidateState):
        """Save per-candidate status.json atomically."""
        candidate_dir = self.get_candidate_dir(state.slug)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        status_file = candidate_dir / "status.json"
        state.last_update_utc = datetime.now(timezone.utc).isoformat()

        # Use atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="tmp_",
            dir=candidate_dir
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            shutil.move(tmp_path, status_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def get_next_candidate(self) -> Optional[str]:
        """Find the first candidate not in a terminal state."""
        state = self.load_global_state()
        if state is None:
            return None

        for slug in state.candidate_slugs:
            status = state.candidate_statuses.get(slug, "PENDING")
            if status not in self.TERMINAL_STATES:
                return slug
        return None

    def update_candidate_status(self, slug: str, status: str):
        """Update a candidate's status in global state."""
        state = self.load_global_state()
        if state is None:
            raise ValueError("Global state not initialized")

        old_status = state.candidate_statuses.get(slug)
        state.candidate_statuses[slug] = status

        # Update counters if transitioning to terminal
        if status in self.TERMINAL_STATES and old_status not in self.TERMINAL_STATES:
            state.total_processed += 1
            if status == "DONE":
                state.total_success += 1
            elif status == "NO_DATA":
                state.total_no_data += 1
            elif status == "BLOCKED":
                state.total_blocked += 1
            elif status == "ERROR":
                state.total_error += 1

        # Update next candidate index
        for i, s in enumerate(state.candidate_slugs):
            if state.candidate_statuses.get(s, "PENDING") not in self.TERMINAL_STATES:
                state.next_candidate_index = i
                break
        else:
            state.next_candidate_index = len(state.candidate_slugs)

        self.save_global_state(state)

    def append_error(self, slug: str, step: str, error: str, details: Optional[Dict] = None):
        """Append an error to the errors.jsonl file."""
        self.ensure_directories()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "slug": slug,
            "step": step,
            "error": error,
            "details": details or {}
        }
        with open(self.errors_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def append_master_table_row(self, row: Dict[str, Any]):
        """Append a row to MASTER_TABLE.csv and regenerate .md."""
        self.ensure_directories()

        fieldnames = [
            "slug", "title", "status", "data_sources_found",
            "channels_extracted", "p_value", "verdict",
            "blocker_reason", "timestamp"
        ]

        # Ensure all fields present
        for field in fieldnames:
            if field not in row:
                row[field] = ""

        row["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Check if file exists to determine if we need header
        write_header = not self.master_csv.exists()

        with open(self.master_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Regenerate markdown table
        self._regenerate_master_md()

    def _regenerate_master_md(self):
        """Regenerate MASTER_TABLE.md from CSV."""
        if not self.master_csv.exists():
            return

        rows = []
        with open(self.master_csv, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        if not rows:
            return

        # Build markdown table
        lines = ["# Rank1 Discovery Mine - Master Results Table\n"]
        lines.append(f"Last updated: {datetime.now(timezone.utc).isoformat()}\n")
        lines.append("")

        # Header
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")

        # Rows
        for row in rows:
            values = [str(row.get(f, "")) for f in fieldnames]
            lines.append("| " + " | ".join(values) + " |")

        lines.append("")
        self._atomic_write_text(self.master_md, "\n".join(lines))

    def scaffold_candidate(self, slug: str, meta: Dict) -> Path:
        """
        Create directory structure for a candidate.

        Creates:
            discoveries/<slug>/
                raw/
                extracted/
                out/
                logs/
                meta.json
                status.json
        """
        candidate_dir = self.get_candidate_dir(slug)

        # Create subdirectories
        (candidate_dir / "raw").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "out").mkdir(parents=True, exist_ok=True)
        (candidate_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Write meta.json
        meta["slug"] = slug
        meta["created_utc"] = datetime.now(timezone.utc).isoformat()
        meta_file = candidate_dir / "meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        # Initialize status.json
        state = CandidateState(
            slug=slug,
            current_step="scaffold",
            completed_steps=["scaffold"],
            last_update_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.save_candidate_state(state)

        return candidate_dir

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current registry status."""
        state = self.load_global_state()
        if state is None:
            return {
                "initialized": False,
                "message": "Registry not initialized. Run 'plan' first."
            }

        # Count statuses
        status_counts = {}
        for status in state.candidate_statuses.values():
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "initialized": True,
            "total_candidates": len(state.candidate_slugs),
            "next_candidate_index": state.next_candidate_index,
            "next_candidate": (
                state.candidate_slugs[state.next_candidate_index]
                if state.next_candidate_index < len(state.candidate_slugs)
                else None
            ),
            "status_counts": status_counts,
            "total_processed": state.total_processed,
            "total_success": state.total_success,
            "total_no_data": state.total_no_data,
            "total_blocked": state.total_blocked,
            "total_error": state.total_error,
            "last_update": state.last_update_utc,
        }
