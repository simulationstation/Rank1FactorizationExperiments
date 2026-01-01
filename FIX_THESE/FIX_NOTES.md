# Root Cause Verification

## Generator/Fitter Mismatch CONFIRMED

### Generator (sim_generate.py lines 44-56, 94-99):
```python
# intensity_model() returns signal + background combined:
signal = np.abs(bw1 + R * bw2)**2
background = b0 + b1 * (x - np.mean(x))
return signal + np.maximum(background, 0)

# Then NORMALIZES THE WHOLE SUM:
intensity = intensity_model(x, R, bw_params, b0, b1)
intensity = intensity / np.max(intensity) * mean_scale  # <-- WRONG
```

### Fitter (sim_fit_v3.py lines 160-170):
```python
def model(params):
    signal = intensity_shape(x, R, bw_params)
    signal = signal / np.max(signal)  # Normalize SIGNAL ONLY
    bg = b0 + b1 * (x - np.mean(x))
    return scale * signal + np.maximum(bg, 0)  # Add bg AFTER scaling
```

## The Mismatch:
- Generator: normalizes (signal + background) together
- Fitter: normalizes signal only, then adds background separately

This causes model mismatch → inflated Lambda_obs under M0 → broken calibration.

## Additional Issues Found:
1. compute_deviance() uses y_obs_safe = max(y, 0.5) - changes the statistic
2. generate_bootstrap_data_poisson() uses y_pred_safe = max(y_pred, 0.1) - should be tiny epsilon
