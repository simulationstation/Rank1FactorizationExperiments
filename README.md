# Rank-1 Factorization Experiments (Exotic Hadron Families)

**Repository:** [github.com/simulationstation/Rank1FactorizationExperiments](https://github.com/simulationstation/Rank1FactorizationExperiments)

---

## What This Is (Plain-English)

Many "exotic" hadrons show up as *two nearby peaks* (a "doublet") in different decay channels.
This project tests a simple organizing idea:

> If those two peaks come from **one underlying production bottleneck**, then the *way the two peaks mix* should be the **same across channels**.

Concretely: if state2 vs state1 appears in channel A with some relative magnitude and phase, that same complex ratio should appear in channel B--up to well-understood effects (thresholds, spin flips, etc.). When that holds, it's evidence the doublet behaves like a *single coherent two-state subsystem* rather than two unrelated mechanisms.

This repo implements a strict, reproducible hypothesis-test protocol and runs it on **public** experimental data products (HEPData tables, arXiv PDF tables/figures, CDS figures, etc.).

---

## Scientific Statement (Minimal Math)

For a two-state family {X1, X2} observed in two channels {A, B}, define the channel-specific complex ratio:

```
R_A = g_A(X2) / g_A(X1)
R_B = g_B(X2) / g_B(X1)
```

**Rank-1 / bottleneck hypothesis:** `R_A = R_B` (same complex mixture ratio across channels).

We test this using a **constrained vs unconstrained** likelihood-ratio test:

- **Constrained:** `R_A = R_B`
- **Unconstrained:** `R_A` and `R_B` independent

Test statistic:

```
Lambda = 2 * (NLL_con - NLL_unc)
```

with **dof_diff = 2** (magnitude + phase of a complex ratio). Primary p-values are **parametric bootstrap** (Wilks used only as a reference sanity check).

---

## What "NOT_REJECTED" Means (Important)

**NOT_REJECTED** does not equal "proved true."

It means: under our model + data + fit-health requirements, we do **not** have statistical evidence at the chosen alpha to say "the channels require different mixtures."

We explicitly gate out misleading situations:

| Situation | Gate | Verdict |
|-----------|------|---------|
| Underconstrained fits (too-good-to-be-true chi2/dof) | chi2/dof < 0.5 | `INCONCLUSIVE` |
| Model mismatch (chi2/dof too large) | chi2/dof > 3.0 | `MODEL_MISMATCH` |
| Optimizer issues (nested-model invariant violated) | NLL_unc > NLL_con | `OPTIMIZER_FAILURE` / retried / audited |

---

## Headline Real-Data Results (Public Sources)

### Summary Table (completed, fit-healthy, publishable under protocol)

| Family / System | Channels Compared | Lambda | p-value | Verdict | Notes |
|-----------------|-------------------|-------:|--------:|---------|-------|
| **CMS X(6900)/X(7100)** | di-J/psi (HEPData) vs J/psi psi(2S)->4mu (public figure reconstruction) | 0.50 | p_boot~0.40 | **NOT_REJECTED** | Both channels fit-healthy |
| **BESIII Y(4220)/Y(4320)** | pi+pi-J/psi vs pi+pi-h_c | ~3.00 | p_boot~0.40 | **NOT_REJECTED** | Required upgraded shared-subspace model |
| **Belle Zb(10610)/Zb(10650) (hidden-bottom)** | Upsilon(1S,2S,3S)pi and h_b(1P,2P)pi | chi2 consistency | p~0.41-0.55 | **NOT_REJECTED** | Includes 180 deg spin-flip correction for h_b |
| **LHCb Pc(4440)/Pc(4457)** | multiple 1D projections / cuts of m(J/psi p) | 5.68 / 1.93 | p_boot=0.050 / 0.440 | **NOT_REJECTED** | Pair 1 borderline; projection-based limitation |
| **Belle Zb open-bottom vs hidden-bottom** | BB*pi vs Upsilon pi-family | chi2 consistency | p<<0.05 | **DISFAVORED** | Interpretable as threshold enhancement (expected) |

---

## Key Interpretation (Plain-English)

- When the test is **NOT_REJECTED** across channels, it suggests the doublet behaves like a **single coherent two-state object** whose internal mixture is stable across different ways of observing it.

- When the test is **DISFAVORED** *across channel families*, it can still be physically meaningful (e.g., **threshold effects** can genuinely change apparent couplings in open-flavor decays).

---

## What's In This Repo

### 1) The Rank-1 Test Harness (Core)

The harness implements:

- Constrained vs unconstrained fits
- Parametric bootstrap p-values (primary)
- Fit-health gates (prevents false passes)
- Multi-start optimization + stability checks
- Nested-model invariant enforcement: **NLL_unc <= NLL_con**
- Standardized outputs: `REPORT.md`, `RANK1_RESULT.md`, `result.json`, plots

Primary harness location:
```
docker_cmssw_rank1/configs/cms_rank1_test.py
```

### 1b) Pre-registered structure tests

- **Belle Zb core + threshold dressing test:** `python3 structure_tests/run_zb_core_threshold.py`
  (docs: `docs/STRUCTURE_ZB_CORE_THRESHOLD.md`)

### 2) "Discovery Mine" (Automated Public-Data Scouting)

**Rank1 Discovery Mine** is an automated pipeline that:

1. Enumerates candidate exotic families (YAML registry)
2. Searches public sources (HEPData, arXiv, CERN Open Data, INSPIRE, GitHub)
3. Downloads what it can (optional; requires `--execute`)
4. Extracts numeric tables (PDF/CSV)
5. Adapts them into the harness input format
6. Runs the rank-1 test
7. Writes per-candidate reports + a global master table

Package locations:
```
rank1_discovery_mine/
discoveries/
```

---

## Quick Start

### Install Dependencies (Python-only workflows)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy matplotlib pyyaml requests pandas
# optional PDF extraction:
pip install pymupdf pdfplumber
```

### Run the Discovery Mine (recommended entry point)

**Dry-run (no downloads):**
```bash
python3 -m rank1_discovery_mine validate
python3 -m rank1_discovery_mine plan
python3 -m rank1_discovery_mine status
python3 -m rank1_discovery_mine run --resume
```

**Execute (downloads + extraction + tests):**
```bash
python3 -m rank1_discovery_mine run --resume --execute
```

**Single candidate:**
```bash
python3 -m rank1_discovery_mine run --one <candidate_slug> --execute
```

**Monitor:**
```bash
tail -f discoveries/pipeline_run.log
python3 -m rank1_discovery_mine status
```

---

## Directory Layout (High-Level)

```
.
|-- cms_x6900_rank1_v4/            # CMS X(6900)/X(7100) analysis (public data)
|-- besiii_y_rank1/                # BESIII Y(4220)/Y(4320) analysis (public data)
|-- belle_zb_rank1/                # Belle Zb hidden-bottom analysis (public data)
|-- belle_zb_openbottom_rank1/     # Belle Zb open-bottom / threshold analysis
|-- lhcb_pc_rank1_v5/              # LHCb Pc(4440)/Pc(4457) projection-based tests
|
|-- docker_cmssw_rank1/            # Test harness + (optional) CMSSW/docker utilities
|   +-- configs/
|       |-- cms_rank1_test.py
|       |-- rank1_injection.py
|       +-- RANK1_HARNESS_README.md
|
|-- rank1_discovery_mine/          # Automated public-data discovery + testing
+-- discoveries/                   # Auto-generated per-candidate results
    |-- _registry/
    |   |-- state.json
    |   |-- MASTER_TABLE.csv
    |   |-- MASTER_TABLE.md
    |   +-- errors.jsonl
    +-- <candidate_slug>/
        |-- raw/
        |-- extracted/
        |-- out/
        |-- logs/
        |-- meta.json
        +-- status.json
```

---

## Repro Notes and Caveats (Read This Before Interpreting)

### Projection-based tests are weaker than full amplitude fits
Some public datasets are 1D projections of a multidimensional amplitude analysis. Passing rank-1 there is meaningful, but not as definitive as using the collaboration workspace.

### Tables derived from the same dataset can be correlated
When comparing multiple "cuts" from the same experiment, the tests are not independent replications.

### Model mismatch is informative
If chi2/dof is bad, the correct conclusion is usually "your model is missing structure," not "rank-1 fails."

### We separate "instrument validation" from "real-data claims"
Any synthetic studies (calibration, power) are clearly labeled as such and are not treated as physics evidence.

---

## Contributing / Extending

If you want to add a new exotic family:

1. Add a candidate entry to `configs/discovery_candidates.yaml`
2. Implement/extend a source module (HEPData/arXiv/etc) if needed
3. Implement extraction (CSV/PDF) -> harness CSV adapter
4. Run `rank1_discovery_mine validate` and a minimal smoke run (dry-run, then `--execute`)

---

## License

Code is provided for research use. Data products remain under their respective experimental and publication licenses.
