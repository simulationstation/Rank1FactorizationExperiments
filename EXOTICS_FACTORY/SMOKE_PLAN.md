# EXOTICS_FACTORY Smoke Test Plan

Do not execute these tests in the factory. This plan describes the minimum checks to run later.

## 1. Schema validation
- Validate every `spec.yaml` with `toolkit/common/schema_validate.py`.
- Expected: all specs pass without errors.

## 2. Import/py_compile
- Run `python -m py_compile` on `launcher.py`, each `pipeline.py`, and toolkit modules.
- Expected: no syntax errors.

## 3. Dry-run plan print
- Run `EXOTICS_FACTORY/launcher.py --family <id>` with no `--run` flag.
- Expected: prints planned steps without executing.

## 4. Synthetic dataset pass-through
- Provide a tiny synthetic dataset to the rank-1 harness wrapper (no bootstrap, minimal data).
- Expected: dry-run configuration only; no fit execution in this factory.

## 5. PDF extractor self-test
- If a tiny local PDF is available, run vector extraction and overlay generation on it.
- If not available, skip with a clear message.
