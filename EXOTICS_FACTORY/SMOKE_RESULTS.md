# EXOTICS_FACTORY Smoke Test Results

**Date**: 2026-01-01
**Status**: ALL PASS

## Summary

- **Families tested**: 9/9
- **Syntax errors**: 0
- **Schema errors**: 0
- **Launcher failures**: 0
- **Network steps skipped**: All (no downloads attempted)

## Test Results

| Family | Schema | Launcher dry-run | Harness hook | PDF backend | Notes |
|--------|--------|------------------|--------------|-------------|-------|
| belle_zb | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| besiii_belle_isr_y | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| besiii_y_pipijpsi_hc | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| cms_atlas_dicharmonium_other | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| control_babar_omega | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| control_babar_phi | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| lhcb_pc_doublet | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| strange_pcs | PASS | PASS | PASS | SKIPPED (needs network) | No issues |
| x3872_like | PASS | PASS | PASS | SKIPPED (needs network) | No issues |

## Component Tests

| Component | Test | Result | Notes |
|-----------|------|--------|-------|
| Python syntax | `py_compile` all .py | PASS | 0 errors |
| Schema validator | All spec.yaml files | PASS | 9/9 valid |
| Launcher | Dry-run mode | PASS | All families print plan and exit 0 |
| Harness wrapper | Command builder | PASS | Correctly builds command list |
| PDF backend | Graceful failure | PASS | Returns empty list on missing file |

## Blockers for Full Runs

1. **Network access required** - HEPData downloads and PDF fetches need internet
2. **PDF files needed** - Real figure PDFs required for extraction
3. **Calibration runs** - Bootstrap calibration needed before production fits

## Next Steps

1. Download required PDFs and HEPData files
2. Run extraction on real figures
3. Calibrate bootstrap p-values
4. Execute full rank-1 tests
