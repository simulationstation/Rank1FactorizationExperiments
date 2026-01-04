# Belle Zb core + threshold dressing structure test

**Status:** DISFAVORED
**Health Label:** HEALTHY
**Reason:** p_boot=0.000 < 0.05
**Mode:** spectrum

## Fit summary
- R_core = 0.875 @ -20.2°
- R_hid = 0.935 @ -8.9°
- R_open = 0.569 @ -177.2°
- kappa1 = 0.000
- kappa2 = 19.987

## Effect size
- |R_open - R_hid| = 1.497
- Re(delta_R) = -1.492
- Im(delta_R) = 0.116

## Fit health (DOF-aware)
- Hidden: chi2/dof = 2.95/6 = 0.49, p_low = 0.1845, p_high = 0.8155
- Open: chi2/dof = 34.58/18 = 1.92, p_low = 0.9893, p_high = 0.0107

Health thresholds: p_low < 0.001 => UNDERCONSTRAINED, chi2/dof > 3.0 with p_high < 0.001 => MODEL_MISMATCH

## Boundary hits
- norm = 0.0021 at lower bound (0.001)
- kappa1 = 0.0000 at lower bound (0.0)
- kappa2 = 19.9871 at upper bound (20.0)

**IDENTIFIABILITY_WARNING:** kappa parameter(s) at bound even after expansion

## Test statistic
- Lambda = 18.332
- p_boot = 0.0000
- p_wilks = 0.0001

## Bootstrap details
- n_boot = 300
- n_valid = 300
- n_failed = 0
- fail_frac = 0.000
