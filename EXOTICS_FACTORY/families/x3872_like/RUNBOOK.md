# Runbook: X(3872)-like channel sharing

## Purpose
This runbook outlines how to execute the pipeline later. The factory only scaffolds files and does not run any downloads, extraction, or fits.

## Dry-run (recommended first)
1. Ensure Python environment is available.
2. Run the launcher with DRY_RUN (default) to print planned steps.

## Full run (later, not in factory)
1. Acquire all required sources listed in spec.yaml (tables, PDFs, or workspaces).
2. Populate local data paths in the spec or runtime config.
3. Run the launcher with the --run flag to execute the pipeline.

## Notes
- Do not download data during the build stage.
- Ensure overlay proofs are generated for PDF extraction workflows.
