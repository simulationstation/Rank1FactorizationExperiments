"""
Candidate loading and validation module.

Handles:
- Loading candidate configurations from YAML
- Schema validation
- Slug computation and uniqueness checking
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import yaml


@dataclass
class Channel:
    """A decay/production channel for a candidate."""
    id: str
    label: str
    final_state: str
    notes: Optional[str] = None


@dataclass
class Candidate:
    """A candidate exotic hadron family for rank-1 testing."""
    slug: str
    title: str
    states: List[str]
    channels: List[Channel]
    preferred_sources: List[str]
    search_terms: List[str]
    notes: str
    expected_data_kind: str
    rank1_mode: str
    arxiv_ids: List[str] = field(default_factory=list)
    hepdata_ids: List[str] = field(default_factory=list)
    collaboration: Optional[str] = None
    category: str = "exotic"

    def to_meta_dict(self) -> Dict[str, Any]:
        """Convert to meta.json format."""
        return {
            "slug": self.slug,
            "title": self.title,
            "states": self.states,
            "channels": [
                {"id": c.id, "label": c.label, "final_state": c.final_state, "notes": c.notes}
                for c in self.channels
            ],
            "preferred_sources": self.preferred_sources,
            "search_terms": self.search_terms,
            "notes": self.notes,
            "expected_data_kind": self.expected_data_kind,
            "rank1_mode": self.rank1_mode,
            "arxiv_ids": self.arxiv_ids,
            "hepdata_ids": self.hepdata_ids,
            "collaboration": self.collaboration,
            "category": self.category,
        }


class ValidationError(Exception):
    """Raised when candidate validation fails."""
    pass


VALID_DATA_KINDS = {
    "binned_spectrum",
    "fit_table",
    "cross_section_scan",
    "efficiency_corrected_spectrum",
    "amplitude_fit_fractions",
}

VALID_RANK1_MODES = {
    "two_state_ratio",
    "multi_state_pairwise",
}

VALID_SOURCES = {
    "hepdata",
    "cds",
    "arxiv_pdf",
    "github",
    "cern_opendata",
    "inspire",
    "supplementary",
}

REQUIRED_KEYS = {
    "title",
    "states",
    "channels",
    "preferred_sources",
    "search_terms",
    "notes",
    "expected_data_kind",
    "rank1_mode",
}


def compute_slug(title: str) -> str:
    """
    Compute a URL-safe slug from title.

    Example: "LHCb J/psi-phi X(4140) Family" -> "lhcb_jpsiphi_x4140_family"
    """
    # Lowercase
    slug = title.lower()
    # Replace common physics notation
    slug = slug.replace("j/psi", "jpsi")
    slug = slug.replace("j/ψ", "jpsi")
    slug = slug.replace("ψ", "psi")
    slug = slug.replace("φ", "phi")
    slug = slug.replace("λ", "lambda")
    slug = slug.replace("ξ", "xi")
    slug = slug.replace("π", "pi")
    slug = slug.replace("(", "_")
    slug = slug.replace(")", "_")
    slug = slug.replace("-", "_")
    slug = slug.replace("/", "_")
    slug = slug.replace(" ", "_")
    slug = slug.replace("'", "")
    slug = slug.replace("+", "plus")
    slug = slug.replace("*", "star")
    # Remove multiple underscores
    slug = re.sub(r'_+', '_', slug)
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    return slug


def validate_candidate_dict(slug: str, data: Dict, strict: bool = True) -> List[str]:
    """
    Validate a single candidate dictionary.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"[{slug}] Missing required key: {key}")

    # Validate title
    if "title" in data and not isinstance(data["title"], str):
        errors.append(f"[{slug}] 'title' must be a string")

    # Validate states (must be list with >= 2 items for two_state_ratio)
    if "states" in data:
        if not isinstance(data["states"], list):
            errors.append(f"[{slug}] 'states' must be a list")
        elif len(data["states"]) < 1:
            errors.append(f"[{slug}] 'states' must have at least 1 entry")
        elif data.get("rank1_mode") == "two_state_ratio" and len(data["states"]) < 2:
            errors.append(f"[{slug}] 'states' must have >= 2 entries for two_state_ratio mode")

    # Validate channels (must be list with >= 2 items)
    if "channels" in data:
        if not isinstance(data["channels"], list):
            errors.append(f"[{slug}] 'channels' must be a list")
        elif len(data["channels"]) < 2:
            errors.append(f"[{slug}] 'channels' must have >= 2 entries for rank-1 test")
        else:
            for i, ch in enumerate(data["channels"]):
                if not isinstance(ch, dict):
                    errors.append(f"[{slug}] channels[{i}] must be a dict")
                elif "id" not in ch or "label" not in ch:
                    errors.append(f"[{slug}] channels[{i}] missing 'id' or 'label'")

    # Validate preferred_sources
    if "preferred_sources" in data:
        if not isinstance(data["preferred_sources"], list):
            errors.append(f"[{slug}] 'preferred_sources' must be a list")
        elif strict:
            for src in data["preferred_sources"]:
                if src not in VALID_SOURCES:
                    errors.append(f"[{slug}] Invalid source '{src}'. Valid: {VALID_SOURCES}")

    # Validate search_terms
    if "search_terms" in data:
        if not isinstance(data["search_terms"], list):
            errors.append(f"[{slug}] 'search_terms' must be a list")
        elif len(data["search_terms"]) < 1:
            errors.append(f"[{slug}] 'search_terms' must not be empty")

    # Validate expected_data_kind
    if "expected_data_kind" in data:
        if strict and data["expected_data_kind"] not in VALID_DATA_KINDS:
            errors.append(
                f"[{slug}] Invalid expected_data_kind '{data['expected_data_kind']}'. "
                f"Valid: {VALID_DATA_KINDS}"
            )

    # Validate rank1_mode
    if "rank1_mode" in data:
        if data["rank1_mode"] not in VALID_RANK1_MODES:
            errors.append(
                f"[{slug}] Invalid rank1_mode '{data['rank1_mode']}'. "
                f"Valid: {VALID_RANK1_MODES}"
            )

    return errors


def validate_candidates(candidates: Dict[str, Dict], strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate all candidates.

    Returns (is_valid, list_of_errors).
    """
    all_errors = []

    # Check for unique slugs
    slugs = list(candidates.keys())
    if len(slugs) != len(set(slugs)):
        seen = set()
        for slug in slugs:
            if slug in seen:
                all_errors.append(f"Duplicate slug: {slug}")
            seen.add(slug)

    # Validate each candidate
    for slug, data in candidates.items():
        errors = validate_candidate_dict(slug, data, strict=strict)
        all_errors.extend(errors)

    return len(all_errors) == 0, all_errors


class CandidateLoader:
    """Loads and parses candidate configurations from YAML."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._candidates: Optional[Dict[str, Candidate]] = None
        self._raw_data: Optional[Dict] = None

    def load(self) -> Dict[str, Candidate]:
        """Load candidates from YAML file."""
        if self._candidates is not None:
            return self._candidates

        with open(self.config_path, 'r') as f:
            self._raw_data = yaml.safe_load(f)

        candidates_data = self._raw_data.get("candidates", {})
        self._candidates = {}

        for slug, data in candidates_data.items():
            # Parse channels
            channels = []
            for ch_data in data.get("channels", []):
                channels.append(Channel(
                    id=ch_data.get("id", ""),
                    label=ch_data.get("label", ""),
                    final_state=ch_data.get("final_state", ""),
                    notes=ch_data.get("notes"),
                ))

            candidate = Candidate(
                slug=slug,
                title=data.get("title", slug),
                states=data.get("states", []),
                channels=channels,
                preferred_sources=data.get("preferred_sources", []),
                search_terms=data.get("search_terms", []),
                notes=data.get("notes", ""),
                expected_data_kind=data.get("expected_data_kind", "binned_spectrum"),
                rank1_mode=data.get("rank1_mode", "two_state_ratio"),
                arxiv_ids=data.get("arxiv_ids", []),
                hepdata_ids=data.get("hepdata_ids", []),
                collaboration=data.get("collaboration"),
                category=data.get("category", "exotic"),
            )
            self._candidates[slug] = candidate

        return self._candidates

    def get_raw_data(self) -> Dict:
        """Get raw YAML data for validation."""
        if self._raw_data is None:
            with open(self.config_path, 'r') as f:
                self._raw_data = yaml.safe_load(f)
        return self._raw_data

    def get_slugs(self) -> List[str]:
        """Get ordered list of candidate slugs."""
        candidates = self.load()
        return list(candidates.keys())

    def get_candidate(self, slug: str) -> Optional[Candidate]:
        """Get a single candidate by slug."""
        candidates = self.load()
        return candidates.get(slug)

    def validate(self, strict: bool = True) -> Tuple[bool, List[str]]:
        """Validate all candidates."""
        raw = self.get_raw_data()
        candidates_data = raw.get("candidates", {})
        return validate_candidates(candidates_data, strict=strict)


def get_default_config_path() -> Path:
    """Get the default config path."""
    # Relative to this file
    package_dir = Path(__file__).parent
    repo_root = package_dir.parent
    return repo_root / "configs" / "discovery_candidates.yaml"
