#!/usr/bin/env python3
"""
sim_generate.py - Generate synthetic datasets for M0/M1/M4 mechanisms

Mechanisms:
- M0 (Rank-1): Same shared complex ratio R across channels
- M1 (Unconstrained coherent): Different ratios per channel
- M4 (Rank-2): Partial alignment, R_B = R_true + deltaR
"""

import numpy as np
from typing import Dict, Tuple, Optional
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
    background = b0 + b1 * x
    return signal + background


def generate_x_bins(nbins: int, x_range: Tuple[float, float] = (-0.5, 1.5)) -> Tuple[np.ndarray, np.ndarray]:
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
    b1 = bg_level * mean_scale * 0.1

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

    elif data_type == 'gaussian':
        # Gaussian with optional correlated systematics
        error_scale = channel_config.get('error_scale', 0.1)
        syst_scale = channel_config.get('syst_scale', 0.0)

        y_true = intensity
        stat_error = error_scale * np.sqrt(y_true)

        # Correlated systematic as nuisance
        if channel_config.get('correlated_syst', False) and syst_scale > 0:
            # Draw nuisance parameter
            s = rng.normal(0, 1)
            syst_shift = syst_scale * y_true * s
            y_obs = rng.normal(y_true + syst_shift, stat_error)
            # Total uncertainty includes systematic
            sigma = np.sqrt(stat_error**2 + (syst_scale * y_true)**2)
        else:
            y_obs = rng.normal(y_true, stat_error)
            sigma = stat_error
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return {
        'y': y_obs,
        'y_true': y_true,
        'sigma': sigma,
        'type': data_type
    }


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
    x_a, edges_a = generate_x_bins(nbins_a)
    data_a = generate_channel_data(x_a, R_true, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_true

    # Channel B - SAME R under M0
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_b, edges_b = generate_x_bins(nbins_b)
    data_b = generate_channel_data(x_b, R_true, bw_params, ch_b, scale_factor, rng)
    data_b['x'] = x_b
    data_b['edges'] = edges_b
    data_b['R_true'] = R_true

    return {
        'mechanism': 'M0',
        'R_true_shared': R_true,
        'channelA': data_a,
        'channelB': data_b,
        'bw_params': bw_params,
        'test_name': test_config['name']
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
    x_a, edges_a = generate_x_bins(nbins_a)
    data_a = generate_channel_data(x_a, R_A, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_A

    # Channel B - DIFFERENT R under M1
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_b, edges_b = generate_x_bins(nbins_b)
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
        'test_name': test_config['name']
    }


def generate_dataset_M4(test_config: Dict, scale_factor: float = 1.0,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """
    Generate dataset under M4 (Rank-2 / two-source).
    R_B = R_true + deltaR (smaller perturbation than M1).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get true R for channel A
    R_cfg = test_config['R_true']
    R_A = complex_from_polar(R_cfg['r'], R_cfg['phi_deg'])

    # Get delta for M4 (smaller than M1)
    dR_cfg = test_config['deltaR_M4']
    r_B = R_cfg['r'] + dR_cfg['dr']
    phi_B = R_cfg['phi_deg'] + dR_cfg['dphi_deg']
    R_B = complex_from_polar(r_B, phi_B)

    # BW params
    bw_cfg = test_config.get('bw_params', {})
    bw_params = BWParams(**bw_cfg)

    # Channel A
    ch_a = test_config['channelA']
    nbins_a = ch_a.get('nbins', 40)
    x_a, edges_a = generate_x_bins(nbins_a)
    data_a = generate_channel_data(x_a, R_A, bw_params, ch_a, scale_factor, rng)
    data_a['x'] = x_a
    data_a['edges'] = edges_a
    data_a['R_true'] = R_A

    # Channel B - SMALL perturbation under M4
    ch_b = test_config['channelB']
    nbins_b = ch_b.get('nbins', 40)
    x_b, edges_b = generate_x_bins(nbins_b)
    data_b = generate_channel_data(x_b, R_B, bw_params, ch_b, scale_factor, rng)
    data_b['x'] = x_b
    data_b['edges'] = edges_b
    data_b['R_true'] = R_B

    return {
        'mechanism': 'M4',
        'R_A': R_A,
        'R_B': R_B,
        'channelA': data_a,
        'channelB': data_b,
        'bw_params': bw_params,
        'test_name': test_config['name']
    }


def generate_dataset(test_config: Dict, mechanism: str,
                     scale_factor: float = 1.0,
                     seed: Optional[int] = None) -> Dict:
    """
    Generate a dataset for a given mechanism.

    Args:
        test_config: Test configuration from tests.json
        mechanism: One of 'M0', 'M1', 'M4'
        scale_factor: Multiplier for statistics level
        seed: Random seed

    Returns:
        Dataset dict
    """
    rng = np.random.default_rng(seed)

    if mechanism == 'M0':
        return generate_dataset_M0(test_config, scale_factor, rng)
    elif mechanism == 'M1':
        return generate_dataset_M1(test_config, scale_factor, rng)
    elif mechanism == 'M4':
        return generate_dataset_M4(test_config, scale_factor, rng)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


if __name__ == "__main__":
    # Quick test
    import sys

    with open('configs/tests.json', 'r') as f:
        config = json.load(f)

    test = config['tests'][0]
    print(f"Testing data generation for: {test['name']}")

    for mech in ['M0', 'M1', 'M4']:
        data = generate_dataset(test, mech, scale_factor=1.0, seed=42)
        print(f"\n{mech}:")
        print(f"  Channel A: {len(data['channelA']['y'])} bins, "
              f"mean count = {np.mean(data['channelA']['y']):.1f}")
        print(f"  Channel B: {len(data['channelB']['y'])} bins, "
              f"mean count = {np.mean(data['channelB']['y']):.1f}")
        if mech == 'M0':
            print(f"  R_shared = {data['R_true_shared']:.3f}")
        else:
            print(f"  R_A = {data['R_A']:.3f}, R_B = {data['R_B']:.3f}")
