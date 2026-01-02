# EXOTICS_FACTORY

The **Exotics Rank-1 Test Factory** provides scaffolding for exotic-family rank-1 tests across multiple channels. It generates specs, placeholders, and runbooks but **does not** download data, extract plots, or run fits.

## Contents

- `registry/`: registry of all pipelines and schema documentation.
- `toolkit/`: shared utilities for validation, backends, modeling, and reporting.
- `families/`: one folder per exotic/control family with specs and pipeline stubs.
- `launcher.py`: master launcher (dry-run default).
- `SMOKE_PLAN.md`: minimal smoke test plan (not executed here).

## Families included

### Exotic families
- `belle_zb`: Zb(10610)/Zb(10650) across Upsilon and hb channels (PDF/HEPData).
- `lhcb_pc_doublet`: Pc(4440)/Pc(4457) shared-subspace proxy with workspace placeholder.
- `strange_pcs`: Pcs(4459) shared-shape proxy with HEPData/PDF placeholders.
- `besiii_y_pipijpsi_hc`: BESIII pi pi J/psi vs pi pi hc shared-subspace with 3 vs 2 resonance allowance.
- `besiii_belle_isr_y`: ISR pi pi J/psi vs pi pi psi(2S) (underconstrained risk flagged).
- `cms_atlas_dicharmonium_other`: Other di-charmonium structures (skeleton only).
- `x3872_like`: X(3872) shared-shape with workspace preferred.

### Control families (negative controls)
- `control_babar_phi`: phi(1680)/phi(2170) across phi f0 vs phi pi pi.
- `control_babar_omega`: omega(1420)/omega(1650) across omega pi pi vs omega f0.

## Usage (dry-run)

```bash
python EXOTICS_FACTORY/launcher.py --family belle_zb
```

Use `--run` only after supplying real data sources and verifying the pipeline.
