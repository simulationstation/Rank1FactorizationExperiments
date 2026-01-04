# Belle Zb “core + threshold dressing” structure test

This pipeline implements the pre-registered nested test for the Belle Zb(10610)/Zb(10650)
**core + threshold dressing** structure hypothesis.

## Hypotheses
- **H0 (constrained):** `R_hid == R_open == R_core`
- **H1 (unconstrained):** `R_hid` and `R_open` independent

Test statistic:
```
Lambda = 2 * (NLL_con - NLL_unc)
```
with `dof_diff = 2`.

Primary p-values are **parametric bootstrap under H0**. Wilks p-values are reported for
reference only.

## Data sources
- **Hidden-bottom ratios:** published Table I ratios from Belle arXiv:1110.2251 for
  Υ(2S)π, Υ(3S)π, hb(1P)π, hb(2P)π, with the 180° spin-flip applied in the model.
- **Open-bottom inputs:**
  - **Spectrum mode:** `belle_zb_openbottom_rank1/extracted/bb_star_pi.csv`
  - **Ratio mode (fallback):** `belle_zb_openbottom_rank1/out/result_table.json`

If no open-bottom inputs are found, the pipeline reports **SKIPPED**.

## Open-bottom model
For spectrum mode, the intensity is:
```
I(m) = norm * | A1(m) + R_open * A2(m) |^2 + background
```
with dressed widths:
```
Gamma_i(m) = Gamma0_i * (1 + kappa_i * rho_i(m))
ho_i(m) = sqrt(max(0, 1 - mthr_i^2 / m^2))
```
Thresholds:
- `mthr1 = mB + mB*`
- `mthr2 = 2*mB*`

Dressing parameters `kappa1,kappa2 >= 0` are only active in spectrum mode.

## Health gating
The decision is gated on **unconstrained fit health** using `chi2/dof`:
- `< 0.5` → `INCONCLUSIVE`
- `> 3.0` → `MODEL_MISMATCH`

A nested invariant is enforced:
```
NLL_unc <= NLL_con + 1e-4
```
If violated, the run returns `OPTIMIZER_FAILURE`.

## CLI
```
python3 structure_tests/run_zb_core_threshold.py \
  --mode auto \
  --n-boot 200 \
  --n-starts 40 \
  --seed 42 \
  --outdir structure_tests/out/zb_core_threshold
```

Use `--fast` for a quick sanity run (small bootstrap, fewer starts).

## Outputs
The run writes:
- `STRUCTURE_REPORT.md`
- `STRUCTURE_RESULT.json`

Both are placed in the specified `--outdir`.
