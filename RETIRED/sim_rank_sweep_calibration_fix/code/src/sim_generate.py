#!/usr/bin/env python3
"""
sim_generate.py - Generate synthetic datasets for M0/M1/M4 mechanisms

Mechanisms:
- M0 (Rank-1): Same shared complex ratio R across channels
- M1 (Unconstrained coherent): Different ratios per channel
- M4 (Rank-2): Partial alignment with variable deltaR

Supports:
- Poisson counting statistics
- Gaussian with correlated systematics (scale + tilt nuisance)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import json
from dataclasses import dataclass


@dataclass
class BWParams:
    """Breit-Wigner parameters"""
    m1: float = 0.0
    m2: float = 1.0
    gamma1: float = 0.10
    gamma2: float = 0.08


def breit_wigner(x: np.ndarray, m: float, gamma: float) -> np.ndarray:
    """
    Simple Breit-Wigner amplitude (complex).
    BW(x) = 1 / ((x - m) - i * Gamma/2)
    """
    return 1.0 / ((x - m) - 1j * gamma / 2)


def complex_from_polar(r: float, phi_deg: float) -> complex:
    """Convert polar (r, phi_deg) to complex number."""
    phi_rad = np.deg2rad(phi_deg)
    return r * np.exp(1j * phi_rad)


def intensity_model(x: np.ndarray, R: complex, bw_params: BWParams,
                    b0: float, b1: float) -> np.ndarray:
    """
    Intensity model for a channel:
    I(x) = |BW1(x) + R * BW2(x)|^2 + B(x)

    Where B(x) = b0 + b1 * x is the background.
    """
    bw1 = breit_wigner(x, bw_params.m1, bw_params.gamma1)
    bw2 = breit_wigner(x, bw_params.m2, bw_params.gamma2)
    signal = np.abs(bw1 + R * bw2)**2
    background = b0 + b1 * (x - np.mean(x))  # Center for stability
    return signal + np.maximum(background, 0)


def generate_x_bins(nbins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bin centers and edges for x variable."""
    edges = np.linspace(x_range[0], x_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges


def generate_channel_data(x: np.ndarray, R: complex, bw_params: BWParams,
                          channel_config: Dict, scale_factor: float = 1.0,
                          rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Generate synthetic data for a single channel.

    Args:
        x: Bin centers
        R: Complex amplitude ratio
        bw_params: Breit-Wigner parameters
        channel_config: Channel configuration dict
        scale_factor: Multiplier for statistics level
        rng: Random number generator

    Returns:
        Dict with 'y' (observed), 'y_true' (expected), 'sigma' (errors), 'type'
    """
    if rng is None:
        rng = np.random.default_rng()

    data_type = channel_config.get('type', 'poisson')
    mean_scale = channel_config.get('mean_count_scale', 50) * scale_factor
    bg_level = channel_config.get('bg_level', 0.3)

    # Background parameters scaled to signal
    b0 = bg_level * mean_scale * 0.5
    b1 = bg_level * mean_scale * 0.02  # Gentler slope

    # True intensity
    intensity = intensity_model(x, R, bw_params, b0, b1)

    # Normalize and scale
    intensity = intensity / np.max(intensity) * mean_scale
    intensity = np.maximum(intensity, 0.1)  # Ensure positive

    if data_type == 'poisson':
        # Poisson counts
        y_true = intensity
        y_obs = rng.poisson(y_true)
        sigma = np.sqrt(np.maximum(y_obs, 1))  # Poisson errors

        # Compute deviance for health check
        deviance = 2 * np.sum(y_obs * np.log(np.maximum(y_obs, 0.5) / np.maximum(y_true, 0.5))
                              - (y_obs - y_true))

        return {
            'y': y_obs,
            'y_true': y_true,
            'sigma': sigma,
            'type': data_type,
            'deviance': deviance
        }

    elif data_type == 'gaussian':
        # Gaussian with optional correlated systematics
        error_frac_min = channel_config.get('error_frac_min', 0.03)
        error_frac_max = channel_config.get('error_frac_max', 0.06)
        syst_scale = channel_config.get('syst_scale', 0.0)
        syst_tilt = channel_config.get('syst_tilt', 0.0)
        correlated_syst = channel_config.get('correlated_syst', False)

        y_true = intensity

        # Randomly vary fractional error per point
        frac_errors = rng.uniform(error_frac_min, error_frac_max, size=len(x))
        stat_error = frac_errors * np.sqrt(y_true + 1)  # sqrt(N) scaled

        # Correlated systematic as nuisance
        if correlated_syst and (syst_scale > 0 or syst_tilt > 0):
            # Draw nuisance parameters (one per dataset)
            s0 = rng.normal(0, 1)  # Scale nuisance
            s1 = rng.normal(0, 1)  # Tilt nuisance

            # Compute systematic shift
            x_norm = (x - np.mean(x)) / (np.max(x) - np.min(x))  # Normalized x
            syst_shift = syst_scale * y_true * s0 + syst_tilt * y_true * x_norm * s1

            # Observed = true + syst + stat noise
            y_obs = rng.normal(y_true + syst_shift, stat_error)

            # Total uncertainty (for chi2 computation in fitter)
            # Store nuisance values for potential profiling
            syst_error = np.sqrt((syst_scale * y_true)**2 + (syst_tilt * y_true * np.abs(x_norm))**2)
            sigma = np.sqrt(stat_error**2 + syst_error**2)

            return {
                'y': y_obs,
                'y_true': y_true,
                'sigma': sigma,
                'stat_error': stat_error,
                'syst_error': syst_error,
                'type': data_type,
                'nuisance_s0': s0,
                'nuisance_s1': s1
            }
        else:
            y_obs = rng.normal(y_true, stat_error)
            sigma = stat_error

            return {
                'y': y_obs,
                'y_true': y_true,
                'sigma': sigma,
                'type': data_type
            }
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def generate_dataset_M0(test_config: Dict, scale_factor: float = 1.0,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Generate dataset under M0 (Rank-1 bottleneck).
    Both channels share the same R.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get true R
    R_cfg = test_config['R_true']
    R_true = complex_from_polar(R_cfg['r'], R_cfg['phi_deg'])

    # BW params
    bw_cfg = test_config.get('bw_params', {})
    bw_params = BWParams(**bw_cfg)

    # Channel A
    ch_a = test_config['channelA']
    nbins_a = ch_a.get('nbins', 40)
    x_range_a = ch_a.get('x_range', [-0.5, 1.5])
    x_a, edges_a = generate_x_bins(nbins_a, tuple(x_range_a))
    data_a = generate_channel_data(x_a, R_true, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_true

    # Channel B - SAME R under M0
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_range_b = ch_b.get('x_range', [-0.5, 1.5])
    x_b, edges_b = generate_x_bins(nbins_b, tuple(x_range_b))
    data_b = generate_channel_data(x_b, R_true, bw_params, ch_b, scale_factor, rng)
    data_b['x'] = x_b
    data_b['edges'] = edges_b
    data_b['R_true'] = R_true

    return {
        'mechanism': 'M0',
        'R_true_shared': R_true,
        'R_A': R_true,
        'R_B': R_true,
        'channelA': data_a,
        'channelB': data_b,
        'bw_params': bw_params,
        'test_name': test_config['name'],
        'scale_factor': scale_factor
    }


def generate_dataset_M1(test_config: Dict, scale_factor: float = 1.0,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Generate dataset under M1 (Unconstrained coherent).
    Channels have different R values.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get true R for channel A
    R_cfg = test_config['R_true']
    R_A = complex_from_polar(R_cfg['r'], R_cfg['phi_deg'])

    # Get delta for M1
    dR_cfg = test_config['deltaR_M1']
    r_B = R_cfg['r'] + dR_cfg['dr']
    phi_B = R_cfg['phi_deg'] + dR_cfg['dphi_deg']
    R_B = complex_from_polar(r_B, phi_B)

    # BW params
    bw_cfg = test_config.get('bw_params', {})
    bw_params = BWParams(**bw_cfg)

    # Channel A
    ch_a = test_config['channelA']
    nbins_a = ch_a.get('nbins', 40)
    x_range_a = ch_a.get('x_range', [-0.5, 1.5])
    x_a, edges_a = generate_x_bins(nbins_a, tuple(x_range_a))
    data_a = generate_channel_data(x_a, R_A, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_A

    # Channel B - DIFFERENT R under M1
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_range_b = ch_b.get('x_range', [-0.5, 1.5])
    x_b, edges_b = generate_x_bins(nbins_b, tuple(x_range_b))
    data_b = generate_channel_data(x_b, R_B, bw_params, ch_b, scale_factor, rng)
    data_b['x'] = x_b
    data_b['edges'] = edges_b
    data_b['R_true'] = R_B

    return {
        'mechanism': 'M1',
        'R_A': R_A,
        'R_B': R_B,
        'channelA': data_a,
        'channelB': data_b,
        'bw_params': bw_params,
        'test_name': test_config['name'],
        'scale_factor': scale_factor
    }


def generate_dataset_M4(test_config: Dict, scale_factor: float = 1.0,
                        dr: Optional[float] = None, dphi_deg: Optional[float] = None,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Generate dataset under M4 (Rank-2 / two-source).
    R_B = R_true + deltaR with configurable delta.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get true R for channel A
    R_cfg = test_config['R_true']
    R_A = complex_from_polar(R_cfg['r'], R_cfg['phi_deg'])

    # Get delta for M4 (use provided or default)
    if dr is None or dphi_deg is None:
        dR_cfg = test_config['deltaR_M4']
        dr = dR_cfg['dr']
        dphi_deg = dR_cfg['dphi_deg']

    r_B = R_cfg['r'] + dr
    phi_B = R_cfg['phi_deg'] + dphi_deg
    R_B = complex_from_polar(r_B, phi_B)

    # BW params
    bw_cfg = test_config.get('bw_params', {})
    bw_params = BWParams(**bw_cfg)

    # Channel A
    ch_a = test_config['channelA']
    nbins_a = ch_a.get('nbins', 40)
    x_range_a = ch_a.get('x_range', [-0.5, 1.5])
    x_a, edges_a = generate_x_bins(nbins_a, tuple(x_range_a))
    data_a = generate_channel_data(x_a, R_A, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_A

    # Channel B - PERTURBED R under M4
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_range_b = ch_b.get('x_range', [-0.5, 1.5])
    x_b, edges_b = generate_x_bins(nbins_b, tuple(x_range_b))
    data_b = generate_channel_data(x_b, R_B, bw_params, ch_b, scale_factor, rng)
    data_b['x'] = x_b
    data_b['edges'] = edges_b
    data_b['R_true'] = R_B

    return {
        'mechanism': 'M4',
        'R_A': R_A,
        'R_B': R_B,
        'deltaR': {'dr': dr, 'dphi_deg': dphi_deg},
        'channelA': data_a,
        'channelB': data_b,
        'bw_params': bw_params,
        'test_name': test_config['name'],
        'scale_factor': scale_factor
    }


def generate_dataset(test_config: Dict, mechanism: str,
                     scale_factor: float = 1.0,
                     seed: Optional[int] = None,
                     dr: Optional[float] = None,
                     dphi_deg: Optional[float] = None) -> Dict:
    """
    Generate a dataset for a given mechanism.

    Args:
        test_config: Test configuration from tests.json
        mechanism: One of 'M0', 'M1', 'M4'
        scale_factor: Multiplier for statistics level
        seed: Random seed
        dr, dphi_deg: Optional deltaR override for M4

    Returns:
        Dataset dict
    """
    rng = np.random.default_rng(seed)

    if mechanism == 'M0':
        return generate_dataset_M0(test_config, scale_factor, rng)
    elif mechanism == 'M1':
        return generate_dataset_M1(test_config, scale_factor, rng)
    elif mechanism == 'M4':
        return generate_dataset_M4(test_config, scale_factor, dr, dphi_deg, rng)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


if __name__ == "__main__":
    # Quick test
    import sys
    import os

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'tests_top3.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    for test in config['tests']:
        print(f"\nTesting data generation for: {test['name']}")

        for mech in ['M0', 'M1', 'M4']:
            data = generate_dataset(test, mech, scale_factor=1.0, seed=42)
            print(f"\n  {mech}:")
            print(f"    Channel A: {len(data['channelA']['y'])} bins, "
                  f"type={data['channelA']['type']}, "
                  f"mean count = {np.mean(data['channelA']['y']):.1f}")
            print(f"    Channel B: {len(data['channelB']['y'])} bins, "
                  f"type={data['channelB']['type']}, "
                  f"mean count = {np.mean(data['channelB']['y']):.1f}")
            if mech == 'M0':
                print(f"    R_shared = {data['R_true_shared']:.3f}")
            else:
                print(f"    R_A = {data['R_A']:.3f}, R_B = {data['R_B']:.3f}")
