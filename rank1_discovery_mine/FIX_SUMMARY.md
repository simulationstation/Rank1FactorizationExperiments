# HEPData False-Positive Elimination - FIX_SUMMARY

**Date**: 2026-01-02
**Version**: v2.0

## Problem Statement

The HEPData auto-search was returning false positives by keyword match, causing wrong records/tables to be downloaded:
- `cms_x6900_x7100_jpsijpsi`: Downloaded Higgs anomalous couplings data instead of di-J/psi spectrum
- `belle_zb_hb_pipi`: Downloaded B+ rare decay data instead of Zb spectra
- `lhcb_tcc_d0d0pi`: Single state (Tcc+) incorrectly marked as testable

These silent failures can generate incorrect "results" from unrelated physics data.

---

## What Changed

### 1. HEPData Record Scoring System (`sources/hepdata.py`)

**NEW**: Scoring-based record selection instead of naive keyword-first.

```python
ACCEPTANCE_THRESHOLD = 50  # Minimum score to accept

def score_hepdata_record(candidate, record) -> HEPDataScore:
    # Scoring criteria:
    # +1000: Exact INSPIRE ID match
    # +100:  Collaboration match (CMS/LHCb/Belle/BESIII)
    # +50:   State token match (X(6900), Pc(4440), etc.)
    # +30:   Channel token match (J/psi J/psi, D0 D0 pi, etc.)
    # +10:   Search term match
    # -200:  Wrong collaboration penalty
    # -50:   Unrelated physics keyword penalty
```

Records below threshold are rejected with `BLOCKED_WRONG_HEPDATA` status.

### 2. Source Audit Artifacts

Every HEPData search now writes `out/source_audit.json`:
```json
{
  "query": "...",
  "top_k_hits": [
    {"record_id": 12345, "title": "...", "score": 150, "breakdown": {...}},
    ...
  ],
  "selected_record": 12345,
  "selection_reason": "Score 150.0 >= threshold 50"
}
```

### 3. Per-Candidate Source Locks (`candidates.py`, `discovery_candidates.yaml`)

New YAML schema fields:
```yaml
candidate_slug:
  enabled: true/false          # Skip if false
  completed_elsewhere: "path/" # Already done in another directory
  sources:
    hepdata_record: "ins1728691"  # Pinned INSPIRE ID
    hepdata_tables: ["Table 1", "Table 2"]  # Specific tables
```

### 4. Auto-Detection of NOT_TESTABLE Candidates

New `Candidate` properties:
```python
@property
def is_testable(self) -> bool:
    """Check if candidate has >= 2 states for rank-1."""

@property
def not_testable_reason(self) -> Optional[str]:
    """Returns SINGLE_STATE, DISABLED, or COMPLETED_ELSEWHERE."""
```

Pipeline now skips NOT_TESTABLE candidates with appropriate status.

### 5. New CandidateStatus Values

```python
class CandidateStatus(Enum):
    NOT_TESTABLE = "NOT_TESTABLE"              # Single state
    DISABLED = "DISABLED"                       # Explicitly disabled
    BLOCKED_WRONG_HEPDATA = "BLOCKED_WRONG_HEPDATA"  # No record met threshold
    COMPLETED_ELSEWHERE = "COMPLETED_ELSEWHERE"  # Done in another directory
```

### 6. Candidate Configuration Updates

| Candidate | Change | Reason |
|-----------|--------|--------|
| `lhcb_pc_4312_extensions` | Pinned `ins1728691` | Verified correct record |
| `cms_x6900_x7100_jpsijpsi` | `enabled: false` | Completed in `cms_x6900_rank1_v4/` |
| `belle_zb_hb_pipi` | `enabled: false` | Completed in `belle_zb_rank1/` |
| `lhcb_tcc_d0d0pi` | `enabled: false` | Single state - not testable |

---

## Files Modified

```
rank1_discovery_mine/sources/hepdata.py      # Scoring system, table validation
rank1_discovery_mine/candidates.py           # Source locks, testability checks
rank1_discovery_mine/pipeline.py             # NOT_TESTABLE handling, audit saving
rank1_discovery_mine/cli.py                  # Enhanced validate command
configs/discovery_candidates.yaml            # Pinned sources, disabled candidates
```

---

## Verification

Run validation (no network):
```bash
python3 -m rank1_discovery_mine validate
```

Expected output includes:
- Testable candidates count
- NOT_TESTABLE warnings
- Disabled/completed-elsewhere info
- Pinned HEPData sources

---

## Interpreting Status Values

| Status | Meaning | Action |
|--------|---------|--------|
| `BLOCKED_WRONG_HEPDATA` | No record met scoring threshold | Check `source_audit.json`, manually pin correct record |
| `NOT_TESTABLE` | Single state - can't do rank-1 test | Expected for Tcc+, X(3872) as single state |
| `DISABLED` | Explicitly disabled in config | Intentional - already done elsewhere |
| `COMPLETED_ELSEWHERE` | Analysis exists in another directory | Reference the existing analysis |

---

## Philosophy

**Prefer BLOCKED + audit over continuing with uncertain sources.**

A false negative (missing data) is far better than a false positive (wrong data producing garbage results that look legitimate).

---

*Generated during rank1_discovery_mine v2.0 patch session*
