# Rank1 Discovery Mine

Automated system for discovering publicly available exotic hadron data and running rank-1 factorization tests.

**Version**: 2.0 (2026-01-02)

## Quick Start

```bash
# Validate configuration (no network)
python -m rank1_discovery_mine validate

# View current status
python -m rank1_discovery_mine status

# Plan and scaffold (dry-run)
python -m rank1_discovery_mine plan

# Run with data acquisition
python -m rank1_discovery_mine run --resume --execute
```

## v2.0 Features

### HEPData Scoring System

Replaces naive keyword matching with scoring-based record selection:

| Criterion | Score |
|-----------|-------|
| Exact INSPIRE ID match | +1000 |
| Collaboration match | +100 |
| State token match (X(6900), Pc(4440)) | +50 each |
| Channel token match (J/psi J/psi) | +30 each |
| Search term match | +10 each |
| Wrong collaboration | -200 |
| Unrelated physics keyword | -50 each |

**Acceptance threshold**: 50 points minimum

Records below threshold are rejected with `BLOCKED_WRONG_HEPDATA` status.

### Source Pinning

Pin specific HEPData records in `discovery_candidates.yaml`:

```yaml
my_candidate:
  sources:
    hepdata_record: "ins1728691"  # Use this specific record
    hepdata_tables: ["Table 1", "Table 3"]  # Optional: specific tables
```

### Candidate Controls

```yaml
my_candidate:
  enabled: false               # Skip this candidate
  completed_elsewhere: "path/" # Already analyzed in another directory
```

### Testability Detection

Candidates with < 2 states are automatically marked `NOT_TESTABLE`:
- Tcc+ (single doubly-charmed tetraquark)
- Single-state control samples

### Source Audit

Every HEPData search writes `out/source_audit.json`:
```json
{
  "query": "...",
  "top_k_hits": [{"record_id": 123, "score": 150, ...}],
  "selected_record": 123,
  "selection_reason": "Score 150.0 >= threshold 50"
}
```

## Status Values

| Status | Meaning |
|--------|---------|
| `DONE` | Successfully completed |
| `IN_PROGRESS` | Currently processing |
| `PLANNED` | Dry-run completed |
| `BLOCKED` | Generic block |
| `BLOCKED_WRONG_HEPDATA` | No record met scoring threshold |
| `NOT_TESTABLE` | Single state (can't do rank-1) |
| `DISABLED` | Explicitly disabled in config |
| `COMPLETED_ELSEWHERE` | Already done in another directory |
| `NO_DATA` | No public data found |
| `ERROR` | Processing error |

## Pipeline Steps

1. **scaffold** - Create directory structure
2. **locate_data** - Search HEPData with scoring
3. **acquire** - Download selected records
4. **extract_numeric** - Extract tables from CSVs/PDFs
5. **run_rank1** - Execute rank-1 test harness
6. **summarize** - Update master table

## File Structure

```
discoveries/
  {candidate_slug}/
    meta.json           # Candidate metadata
    status.json         # Processing state
    raw/                # Downloaded files
    extracted/          # Processed tables
    out/
      source_audit.json # HEPData selection audit
      rank1_result.json # Test results
      RANK1_REPORT.md   # Human-readable report
    logs/               # Step logs
```

## Troubleshooting

### BLOCKED_WRONG_HEPDATA

1. Check `out/source_audit.json` for scored candidates
2. Find correct HEPData record manually
3. Add to `discovery_candidates.yaml`:
   ```yaml
   sources:
     hepdata_record: "ins{correct_id}"
   ```
4. Re-run pipeline

### NOT_TESTABLE

This is expected for single-state candidates. The rank-1 test requires two states to compare coupling ratios.

### Lambda=0 / p=1

Check `rank1_result.json` for:
- `nll_con` and `nll_unc` values
- `Lambda_raw` (before clamping)
- `invariant_holds` flag

If optima are identical, it indicates strong rank-1 compatibility.

## See Also

- `FIX_SUMMARY.md` - Detailed changelog for v2.0
- `../FIX_REPORT.md` - Harness fix documentation
- `../configs/discovery_candidates.yaml` - Candidate definitions
