# Rank1 Discovery Mine - Registry

This directory contains the global state and tracking files for the Rank1 Discovery Mine.

## Files

### `state.json`

Global progress state including:
- `version`: Schema version
- `candidate_slugs`: Ordered list of all candidate slugs
- `candidate_statuses`: Map of slug -> current status
- `next_candidate_index`: Index of next candidate to process
- `last_update_utc`: Timestamp of last update
- `total_processed`, `total_success`, `total_no_data`, `total_blocked`, `total_error`: Counters

### `MASTER_TABLE.csv`

CSV file with one row per candidate attempt:
- `slug`: Candidate identifier
- `title`: Human-readable title
- `status`: Final status (DONE, NO_DATA, BLOCKED, ERROR)
- `data_sources_found`: Number of data sources discovered
- `channels_extracted`: Number of channels extracted
- `p_value`: Bootstrap p-value (if test ran)
- `verdict`: Rank-1 test verdict
- `blocker_reason`: Reason for blocking (if applicable)
- `timestamp`: When the row was added

### `MASTER_TABLE.md`

Markdown rendering of the master table for human viewing.

### `errors.jsonl`

Append-only log of errors encountered during processing. Each line is a JSON object:
```json
{
  "timestamp": "2026-01-02T12:00:00Z",
  "slug": "candidate_slug",
  "step": "acquire",
  "error": "Error message",
  "details": {}
}
```

## Candidate Statuses

| Status | Terminal? | Meaning |
|--------|-----------|---------|
| PENDING | No | Not yet started |
| IN_PROGRESS | No | Currently being processed |
| PLANNED | No | Dry-run completed, ready for execution |
| DONE | Yes | Successfully completed rank-1 test |
| NO_DATA | Yes | No public data available |
| BLOCKED | Yes | Data found but extraction/test failed |
| ERROR | Yes | Processing error occurred |

## Resumption Logic

When `run --resume` is executed:

1. Load `state.json`
2. Find first candidate where `candidate_statuses[slug]` is not terminal
3. Load that candidate's `status.json` from `discoveries/<slug>/`
4. Resume from `current_step`

## Atomic Writes

All state files are written atomically:
1. Write to temporary file in same directory
2. Rename temporary file to final name

This ensures no partial/corrupt state files on crash.

## Per-Candidate State

Each candidate has its own `status.json` in `discoveries/<slug>/`:

```json
{
  "slug": "candidate_slug",
  "current_step": "locate_data",
  "completed_steps": ["scaffold"],
  "last_update_utc": "2026-01-02T12:00:00Z",
  "blocked_reason": null,
  "error_message": null,
  "discovered_urls": []
}
```

## Commands

```bash
# View current status
python -m rank1_discovery_mine status

# View detailed per-candidate status
python -m rank1_discovery_mine status --detailed

# Initialize registry (creates state.json)
python -m rank1_discovery_mine plan

# Resume processing
python -m rank1_discovery_mine run --resume --execute
```
