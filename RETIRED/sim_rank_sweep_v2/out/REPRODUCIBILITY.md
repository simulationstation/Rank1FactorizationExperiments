# Reproducibility Information

Generated: 2025-12-31 12:43:22

## Configuration

```json
{
  "description": "Top 3 decisive rank-1 tests for power & calibration sweep",
  "version": "2.0",
  "tests": [
    {
      "name": "Y-states",
      "description": "Y(4260)/Y(4360) in pipi J/psi vs pipi h_c",
      "type": "gaussian",
      "channelA": {
        "name": "pipi_Jpsi",
        "type": "gaussian",
        "nbins": 45,
        "x_range": [
          4.01,
          4.6
        ],
        "error_frac_min": 0.03,
        "error_frac_max": 0.06,
        "correlated_syst": true,
        "syst_scale": 0.05,
        "syst_tilt": 0.02,
        "mean_count_scale": 100,
        "bg_level": 0.4
      },
      "channelB": {
        "name": "pipi_hc",
        "type": "gaussian",
        "nbins": 45,
        "x_range": [
          4.01,
          4.6
        ],
        "error_frac_min": 0.04,
        "error_frac_max": 0.08,
        "correlated_syst": true,
        "syst_scale": 0.05,
        "syst_tilt": 0.02,
        "mean_count_scale": 60,
        "bg_level": 0.5
      },
      "bw_params": {
        "m1": 4.23,
        "m2": 4.36,
        "gamma1": 0.08,
        "gamma2": 0.1
      },
      "R_true": {
        "r": 0.65,
        "phi_deg": -45
      },
      "deltaR_M1": {
        "dr": 0.25,
        "dphi_deg": 35
      },
      "deltaR_M4": {
        "dr": 0.08,
        "dphi_deg": 12
      },
      "stats_multipliers": [
        0.5,
        1.0,
        2.0,
        4.0
      ]
    },
    {
      "name": "Zc-like",
      "description": "Zc(3900) in pi J/psi vs D D*",
      "type": "poisson",
      "channelA": {
        "name": "pi_Jpsi",
        "type": "poisson",
        "nbins": 60,
        "x_range": [
          3.82,
          4.02
        ],
        "mean_count_scale": 350,
        "bg_level": 0.35
      },
      "channelB": {
        "name": "DD_star",
        "type": "poisson",
        "nbins": 60,
        "x_range": [
          3.82,
          4.02
        ],
        "mean_count_scale": 200,
        "bg_level": 0.4
      },
      "bw_params": {
        "m1": 3.887,
        "m2": 3.92,
        "gamma1": 0.028,
        "gamma2": 0.035
      },
      "R_true": {
        "r": 0.55,
        "phi_deg": -60
      },
      "deltaR_M1": {
        "dr": 0.2,
        "dphi_deg": 40
      },
      "deltaR_M4": {
        "dr": 0.06,
        "dphi_deg": 10
      },
      "stats_multipliers": [
        0.5,
        1.0,
        2.0,
        4.0
      ]
    },
    {
      "name": "Di-charmonium",
      "description": "X(6900)/X(7100) in J/psi J/psi vs J/psi psi(2S)",
      "type": "poisson",
      "channelA": {
        "name": "Jpsi_Jpsi",
        "type": "poisson",
        "nbins": 90,
        "x_range": [
          6.2,
          7.4
        ],
        "mean_count_scale": 150,
        "bg_level": 0.45
      },
      "channelB": {
        "name": "Jpsi_psi2S",
        "type": "poisson",
        "nbins": 90,
        "x_range": [
          6.2,
          7.4
        ],
        "mean_count_scale": 80,
        "bg_level": 0.5
      },
      "bw_params": {
        "m1": 6.9,
        "m2": 7.1,
        "gamma1": 0.1,
        "gamma2": 0.12
      },
      "R_true": {
        "r": 0.7,
        "phi_deg": -50
      },
      "deltaR_M1": {
        "dr": 0.22,
        "dphi_deg": 38
      },
      "deltaR_M4": {
        "dr": 0.07,
        "dphi_deg": 11
      },
      "stats_multipliers": [
        0.5,
        1.0,
        2.0,
        4.0
      ]
    }
  ],
  "M4_grid": {
    "dr_values": [
      0.05,
      0.1,
      0.2
    ],
    "dphi_values": [
      10,
      20,
      40
    ],
    "stats_levels": [
      1.0,
      2.0
    ]
  },
  "trials": {
    "M0": 300,
    "M1": 300,
    "M4": 200
  },
  "bootstrap_replicates": 200,
  "optimizer_starts": 80,
  "fit_health": {
    "chi2_dof_min": 0.5,
    "chi2_dof_max": 3.0,
    "deviance_max": 3.0
  },
  "identifiability": {
    "nll_threshold": 2.0,
    "r_spread_max": 0.3,
    "phi_spread_max": 90
  }
}
```

## Sweep Statistics

- Total trials: 18000
- Start time: 2025-12-31T12:00:16.354189
- Tests: ['Y-states', 'Zc-like', 'Di-charmonium']
- Stats levels: [0.5, 1.0, 2.0, 4.0]
- M4 grid: dr=[0.05, 0.1, 0.2], dphi=[10, 20, 40]

## Commands to Reproduce

```bash
cd sim_rank_sweep_v2/code/src
nohup python3 -u sim_sweep.py --config ../configs/tests_top3.json \
  --trials-m0 300 --trials-m1 300 --trials-m4 200 \
  --output ../../out > ../../logs/sweep.log 2>&1 &
```
