"""
HEPData source module with scoring-based record selection.

HEPData (https://www.hepdata.net) is the primary source for published
high-energy physics data tables.

v2.0: Added record scoring to eliminate false-positive matches.

Scoring criteria:
- Exact INSPIRE ID match (if candidate specifies it): +1000
- Collaboration match: +100
- State token match (e.g., X(6900), Pc(4440)): +50 per match
- Channel token match (e.g., J/psi J/psi, D0 D0 pi): +30 per match
- Title keyword match: +10 per match
- Penalty for wrong collaboration: -200
- Penalty for unrelated physics: -500
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HEPDATA_BASE = "https://www.hepdata.net"
HEPDATA_SEARCH_URL = "https://www.hepdata.net/search"

# Minimum score to accept a HEPData record
ACCEPTANCE_THRESHOLD = 50

# Number of top candidates to consider
TOP_K_CANDIDATES = 15


@dataclass
class HEPDataScore:
    """Scoring result for a HEPData record."""
    record_id: int
    title: str
    collaboration: str
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "collaboration": self.collaboration,
            "score": self.score,
            "breakdown": self.breakdown,
            "reasons": self.reasons,
        }


@dataclass
class TableValidation:
    """Result of validating a HEPData table."""
    table_name: str
    valid: bool
    x_column_found: bool = False
    y_column_found: bool = False
    n_points: int = 0
    x_range: Tuple[float, float] = (0.0, 0.0)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "valid": self.valid,
            "x_column_found": self.x_column_found,
            "y_column_found": self.y_column_found,
            "n_points": self.n_points,
            "x_range": list(self.x_range),
            "reasons": self.reasons,
        }


@dataclass
class SourceAudit:
    """Full audit of HEPData source selection."""
    query: str
    top_k_hits: List[HEPDataScore]
    selected_record: Optional[int]
    selection_reason: str
    table_validations: List[TableValidation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k_hits": [h.to_dict() for h in self.top_k_hits],
            "selected_record": self.selected_record,
            "selection_reason": self.selection_reason,
            "table_validations": [t.to_dict() for t in self.table_validations],
        }


def _normalize_token(token: str) -> str:
    """Normalize a token for matching."""
    t = token.lower()
    # Normalize physics notation
    t = t.replace("j/psi", "jpsi").replace("j/ψ", "jpsi")
    t = t.replace("ψ", "psi").replace("φ", "phi")
    t = t.replace("π", "pi").replace("γ", "gamma")
    t = t.replace("(", "").replace(")", "")
    t = t.replace("+", "plus").replace("-", "minus")
    t = t.replace("*", "star").replace("'", "prime")
    t = re.sub(r'\s+', '', t)
    return t


def _extract_state_tokens(text: str) -> List[str]:
    """Extract state tokens like X(6900), Pc(4440), Zb(10610) from text."""
    # Pattern for exotic states: Letter(s) + optional subscript + (mass)
    pattern = r'[XYZPT]c?s?b?_?[a-z]*\s*\(\s*\d{3,5}\s*\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [_normalize_token(m) for m in matches]


def _extract_channel_tokens(text: str) -> List[str]:
    """Extract channel tokens like J/psi p, D0 D0 pi from text."""
    tokens = []
    # Common channel patterns
    patterns = [
        r'J/ψ\s*[pK]', r'J/psi\s*[pK]',
        r'J/ψ\s*J/ψ', r'J/psi\s*J/psi', r'JψJψ',
        r'D[0\+\-]\s*D[0\+\-]', r'D\*?\s*D\*?',
        r'π\+?\-?\s*π\+?\-?', r'pi\+?\-?\s*pi',
        r'Υ\s*π', r'Upsilon\s*pi',
        r'h[bc]\s*π', r'hb\s*pi', r'hc\s*pi',
        r'ψ\s*\(2S\)', r'psi\s*2S',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            tokens.append(_normalize_token(re.search(p, text, re.IGNORECASE).group()))
    return tokens


def score_hepdata_record(candidate: "Candidate", record: Dict[str, Any]) -> HEPDataScore:
    """
    Score a HEPData record for relevance to a candidate.

    Higher score = better match.
    """
    score = 0.0
    breakdown = {}
    reasons = []

    record_id = record.get("recid", record.get("record_id", 0))
    title = record.get("title", "")
    collab = record.get("collaboration", "").upper()
    abstract = record.get("abstract", "")
    full_text = f"{title} {abstract}".lower()

    # 1. Exact INSPIRE ID match (highest priority)
    inspire_id = record.get("inspire_id", "")
    if inspire_id and candidate.hepdata_ids:
        for hep_id in candidate.hepdata_ids:
            if str(hep_id) == str(inspire_id) or f"ins{hep_id}" == str(inspire_id):
                score += 1000
                breakdown["inspire_id_match"] = 1000
                reasons.append(f"Exact INSPIRE ID match: {inspire_id}")
                break

    # 2. Collaboration match
    if candidate.collaboration:
        cand_collab = candidate.collaboration.upper()
        if cand_collab in collab or collab in cand_collab:
            score += 100
            breakdown["collaboration_match"] = 100
            reasons.append(f"Collaboration match: {collab}")
        elif collab and cand_collab and collab != cand_collab:
            # Wrong collaboration penalty
            score -= 200
            breakdown["wrong_collaboration"] = -200
            reasons.append(f"Wrong collaboration: expected {cand_collab}, got {collab}")

    # 3. State token matching
    title_states = _extract_state_tokens(title + " " + abstract)
    candidate_states = [_normalize_token(s) for s in candidate.states]

    state_matches = 0
    for cs in candidate_states:
        for ts in title_states:
            if cs in ts or ts in cs:
                state_matches += 1
                break

    if state_matches > 0:
        pts = state_matches * 50
        score += pts
        breakdown["state_matches"] = pts
        reasons.append(f"State matches: {state_matches}")

    # 4. Channel token matching
    title_channels = _extract_channel_tokens(title + " " + abstract)
    candidate_channels = []
    for ch in candidate.channels:
        candidate_channels.extend(_extract_channel_tokens(ch.final_state))
        candidate_channels.extend(_extract_channel_tokens(ch.label))

    channel_matches = 0
    for cc in candidate_channels:
        for tc in title_channels:
            if cc in tc or tc in cc:
                channel_matches += 1
                break

    if channel_matches > 0:
        pts = channel_matches * 30
        score += pts
        breakdown["channel_matches"] = pts
        reasons.append(f"Channel matches: {channel_matches}")

    # 5. Search term matching
    term_matches = 0
    for term in candidate.search_terms:
        if term.lower() in full_text:
            term_matches += 1

    if term_matches > 0:
        pts = term_matches * 10
        score += pts
        breakdown["term_matches"] = pts
        reasons.append(f"Search term matches: {term_matches}")

    # 6. Penalty for clearly unrelated physics
    unrelated_keywords = [
        "higgs", "top quark", "w boson", "z boson", "supersymmetry", "susy",
        "dark matter", "neutrino oscillation", "cms detector", "atlas detector",
        "trigger", "calibration", "luminosity", "minimum bias",
    ]
    for kw in unrelated_keywords:
        if kw in full_text and not any(kw in t.lower() for t in candidate.search_terms):
            score -= 50
            breakdown.setdefault("unrelated_penalty", 0)
            breakdown["unrelated_penalty"] -= 50
            reasons.append(f"Unrelated keyword: {kw}")

    return HEPDataScore(
        record_id=record_id,
        title=title,
        collaboration=collab,
        score=score,
        breakdown=breakdown,
        reasons=reasons,
    )


def validate_table_content(
    csv_content: str,
    expected_x_keywords: Optional[List[str]] = None,
    expected_y_keywords: Optional[List[str]] = None,
    expected_x_range: Optional[Tuple[float, float]] = None,
    min_points: int = 10,
    table_name: str = "unknown",
) -> TableValidation:
    """
    Validate that a HEPData table CSV contains expected data.

    Args:
        csv_content: Raw CSV content
        expected_x_keywords: Keywords to find in x-axis column headers
        expected_y_keywords: Keywords to find in y-axis column headers
        expected_x_range: Expected (min, max) range for x values
        min_points: Minimum number of data points required
        table_name: Name for logging
    """
    reasons = []
    x_found = False
    y_found = False
    n_points = 0
    x_range = (0.0, 0.0)

    lines = csv_content.strip().split('\n')

    # Skip comment lines (start with #)
    data_lines = [l for l in lines if not l.startswith('#')]

    if not data_lines:
        reasons.append("No data lines found")
        return TableValidation(table_name, False, reasons=reasons)

    # Parse header
    header = data_lines[0].lower() if data_lines else ""

    # Check x-axis keywords
    if expected_x_keywords:
        for kw in expected_x_keywords:
            if kw.lower() in header:
                x_found = True
                break
        if not x_found:
            reasons.append(f"X-axis keywords not found in header: {expected_x_keywords}")
    else:
        # Default: look for mass, energy, invariant mass
        default_x = ['mass', 'gev', 'mev', 'sqrt', 'energy', 'm(', 'm_']
        x_found = any(kw in header for kw in default_x)

    # Check y-axis keywords
    if expected_y_keywords:
        for kw in expected_y_keywords:
            if kw.lower() in header:
                y_found = True
                break
        if not y_found:
            reasons.append(f"Y-axis keywords not found in header: {expected_y_keywords}")
    else:
        # Default: look for cross section, events, yield
        default_y = ['sigma', 'cross', 'events', 'yield', 'dn/', 'n/', 'counts']
        y_found = any(kw in header for kw in default_y)

    # Count data points and extract x range
    n_points = len(data_lines) - 1  # Subtract header

    if n_points > 0:
        try:
            # Try to extract first column as x values
            x_values = []
            for line in data_lines[1:]:
                parts = line.split(',')
                if parts:
                    try:
                        x_values.append(float(parts[0]))
                    except ValueError:
                        pass
            if x_values:
                x_range = (min(x_values), max(x_values))
        except Exception:
            pass

    # Check minimum points
    if n_points < min_points:
        reasons.append(f"Too few points: {n_points} < {min_points}")

    # Check x range if specified
    if expected_x_range and x_range[1] > 0:
        exp_min, exp_max = expected_x_range
        if x_range[1] < exp_min or x_range[0] > exp_max:
            reasons.append(f"X range {x_range} outside expected {expected_x_range}")

    # Determine validity
    valid = (
        n_points >= min_points and
        (x_found or not expected_x_keywords) and
        (y_found or not expected_y_keywords) and
        not any("outside expected" in r for r in reasons)
    )

    return TableValidation(
        table_name=table_name,
        valid=valid,
        x_column_found=x_found,
        y_column_found=y_found,
        n_points=n_points,
        x_range=x_range,
        reasons=reasons,
    )


def plan_queries(candidate: "Candidate") -> List[str]:
    """
    Plan HEPData search queries for a candidate.

    Returns list of query URLs that would be executed.
    """
    queries = []

    # Query by search terms
    for term in candidate.search_terms:
        encoded = quote_plus(term)
        queries.append(f"{HEPDATA_SEARCH_URL}?q={encoded}&format=json")

    # Query by known HEPData IDs (INSPIRE IDs)
    for hep_id in candidate.hepdata_ids:
        queries.append(f"{HEPDATA_BASE}/record/ins{hep_id}?format=json")

    # Query by collaboration + states
    if candidate.collaboration:
        for state in candidate.states:
            term = f"{candidate.collaboration} {state}"
            encoded = quote_plus(term)
            queries.append(f"{HEPDATA_SEARCH_URL}?q={encoded}&format=json")

    return queries


def execute_search_with_scoring(
    candidate: "Candidate",
    validation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], SourceAudit]:
    """
    Execute HEPData searches with scoring-based selection.

    Returns:
        Tuple of (discovered_urls, audit)
    """
    import requests

    discovered = []
    all_records = []
    queries = plan_queries(candidate)

    # Check for pinned source
    pinned_record = getattr(candidate, 'pinned_hepdata_record', None)
    if pinned_record:
        logger.info(f"Using pinned HEPData record: {pinned_record}")
        queries = [f"{HEPDATA_BASE}/record/{pinned_record}?format=json"]

    for query_url in queries:
        try:
            logger.info(f"HEPData query: {query_url}")
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Handle search results
            if "results" in data:
                for result in data["results"][:TOP_K_CANDIDATES]:
                    record_id = result.get("recid")
                    if record_id:
                        all_records.append({
                            "recid": record_id,
                            "record_id": record_id,
                            "title": result.get("title", ""),
                            "collaboration": result.get("collaboration", ""),
                            "abstract": result.get("abstract", ""),
                            "inspire_id": result.get("inspire_id", ""),
                        })

            # Handle direct record lookup
            elif "record" in data:
                record = data["record"]
                all_records.append({
                    "recid": record.get("recid"),
                    "record_id": record.get("recid"),
                    "title": record.get("title", ""),
                    "collaboration": ",".join(record.get("collaborations", [])),
                    "abstract": record.get("abstract", ""),
                    "inspire_id": record.get("inspire_id", ""),
                })

        except Exception as e:
            logger.warning(f"HEPData query failed: {e}")

    # Deduplicate by record_id
    seen_ids = set()
    unique_records = []
    for rec in all_records:
        rid = rec.get("recid")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            unique_records.append(rec)

    # Score all records
    scored = [score_hepdata_record(candidate, rec) for rec in unique_records]
    scored.sort(key=lambda x: x.score, reverse=True)

    # Select best record if above threshold
    selected_record = None
    selection_reason = ""

    if scored:
        best = scored[0]
        if pinned_record:
            selected_record = int(pinned_record.replace("ins", ""))
            selection_reason = f"Pinned record: {pinned_record}"
        elif best.score >= ACCEPTANCE_THRESHOLD:
            selected_record = best.record_id
            selection_reason = f"Score {best.score:.1f} >= threshold {ACCEPTANCE_THRESHOLD}"
        else:
            selection_reason = f"Best score {best.score:.1f} < threshold {ACCEPTANCE_THRESHOLD}"
            logger.warning(f"No HEPData record met threshold for {candidate.slug}")
    else:
        selection_reason = "No records found"

    # Build discovered URLs for selected record
    if selected_record:
        discovered.append({
            "url": f"{HEPDATA_BASE}/record/{selected_record}?format=json",
            "type": "hepdata_record",
            "record_id": selected_record,
            "title": scored[0].title if scored else "",
            "collaboration": scored[0].collaboration if scored else "",
            "score": scored[0].score if scored else 0,
        })

    audit = SourceAudit(
        query="; ".join(queries[:3]),  # First 3 queries for brevity
        top_k_hits=scored[:TOP_K_CANDIDATES],
        selected_record=selected_record,
        selection_reason=selection_reason,
    )

    return discovered, audit


def execute_search(candidate: "Candidate") -> List[Dict[str, Any]]:
    """
    Execute HEPData searches and return discovered URLs.

    NOTE: This is the legacy interface. Use execute_search_with_scoring for new code.
    """
    discovered, _ = execute_search_with_scoring(candidate)
    return discovered


def download(url: str, dest_dir: Path, timeout: int = 60) -> List[str]:
    """
    Download HEPData table(s) to destination directory.

    Returns list of downloaded file paths.
    """
    import requests
    from urllib.parse import quote

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        # Determine if this is a table download or record URL
        if "/download/table/" in url:
            # Direct table CSV download
            logger.debug(f"Downloading table CSV: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            path_part = url.split("?")[0].rstrip("/csv")
            parts = path_part.split("/")
            table_name = parts[-2] if len(parts) > 2 else parts[-1]
            inspire_id = parts[-3] if len(parts) > 3 else "unknown"
            safe_name = table_name.replace(" ", "_").replace("/", "_")
            filename = f"hepdata_{inspire_id}_{safe_name}.csv"
            filepath = dest_dir / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)
            downloaded.append(str(filepath))
            logger.info(f"Downloaded: {filename}")

        elif "/record/" in url:
            # Full record - fetch JSON first
            if "format=json" not in url:
                url = url + ("&" if "?" in url else "?") + "format=json"

            logger.debug(f"Fetching record: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            path_part = url.split("?")[0]
            record_id = path_part.split("/")[-1]
            filename = f"hepdata_{record_id}.json"
            filepath = dest_dir / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)
            downloaded.append(str(filepath))
            logger.info(f"Downloaded: {filename}")

            data = response.json()
            inspire_id = data.get("record", {}).get("inspire_id")
            version = data.get("version", 1)

            tables = data.get("data_tables", [])
            if tables:
                logger.info(f"Record has {len(tables)} tables")
                for table in tables:
                    table_name = table.get("name")
                    if table_name and inspire_id:
                        try:
                            encoded_name = quote(table_name, safe='')
                            table_url = f"{HEPDATA_BASE}/download/table/ins{inspire_id}/{encoded_name}/{version}/csv"
                            table_files = download(table_url, dest_dir, timeout)
                            downloaded.extend(table_files)
                        except Exception as te:
                            logger.warning(f"Failed to download table '{table_name}': {te}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HEPData HTTP error ({e.response.status_code}): {url}")
    except Exception as e:
        logger.error(f"HEPData download failed: {e}")

    return downloaded


def get_record_metadata(record_id: str) -> Optional[Dict]:
    """
    Get metadata for a HEPData record.
    """
    import requests

    try:
        if str(record_id).startswith("ins"):
            url = f"{HEPDATA_BASE}/record/{record_id}?format=json"
        else:
            url = f"{HEPDATA_BASE}/record/{record_id}?format=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get HEPData metadata: {e}")
        return None
