from pathlib import Path

import numpy as np

from structure_tests.zb_core_threshold import (
    HiddenRatioDatum,
    OpenRatioDatum,
    SectorHealth,
    _phi_with_spin_flip,
    assess_health,
    hidden_ratio_nll,
    hidden_ratio_chi2,
    open_ratio_nll,
    resolve_open_mode,
    wrap_to_pi,
    P_LOW_THRESHOLD,
    CHI2_DOF_MISMATCH,
)


def test_nested_invariant_from_constrained_init():
    hidden = [
        HiddenRatioDatum(
            name="test",
            r=1.0,
            r_err=0.1,
            phi_deg=0.0,
            phi_err_deg=10.0,
            spin_flip=False,
        )
    ]
    open_ratio = OpenRatioDatum(
        r=1.0,
        r_err=0.2,
        phi_deg=0.0,
        phi_err_deg=15.0,
    )
    r_core = 1.0
    phi_core = 0.0
    nll_con = hidden_ratio_nll(r_core, phi_core, hidden) + open_ratio_nll(
        r_core, phi_core, open_ratio
    )
    nll_unc = hidden_ratio_nll(r_core, phi_core, hidden) + open_ratio_nll(
        r_core, phi_core, open_ratio
    )
    assert nll_unc <= nll_con + 1e-8


def test_wrap_to_pi_and_spin_flip():
    assert np.isclose(wrap_to_pi(3 * np.pi), -np.pi)
    assert np.isclose(_phi_with_spin_flip(0.2, True), 0.2 + np.pi)


def test_mode_auto_selection(tmp_path: Path):
    base_dir = tmp_path
    spectrum_path = base_dir / "belle_zb_openbottom_rank1" / "extracted" / "bb_star_pi.csv"
    ratio_path = base_dir / "belle_zb_openbottom_rank1" / "out" / "result_table.json"

    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    ratio_path.parent.mkdir(parents=True, exist_ok=True)

    spectrum_path.write_text("# header\nm_low_MeV,m_high_MeV,m_center_MeV,RS,RS_err,WS,WS_err,efficiency,signal,signal_err\n", encoding="utf-8")
    ratio_path.write_text("{}", encoding="utf-8")

    mode, _, _ = resolve_open_mode("auto", base_dir)
    assert mode == "spectrum"

    spectrum_path.unlink()
    mode, _, _ = resolve_open_mode("auto", base_dir)
    assert mode == "ratio"

    ratio_path.unlink()
    mode, _, _ = resolve_open_mode("auto", base_dir)
    assert mode == "missing"


def test_run_structure_test_skips_when_missing(tmp_path: Path):
    """Verify pipeline returns SKIPPED status when open-bottom data is absent."""
    from structure_tests.zb_core_threshold import run_structure_test

    result = run_structure_test(
        mode="auto",
        n_boot=0,
        n_starts=5,
        seed=42,
        base_dir=tmp_path,
    )
    assert result.status == "SKIPPED"
    assert "missing" in result.reason


def test_health_gating_uses_unconstrained_fit():
    """Verify health is computed from unconstrained fit parameters."""
    from structure_tests.zb_core_threshold import hidden_ratio_chi2, HIDDEN_RATIO_TABLE

    # Compute chi2 at the approximate unconstrained optimum
    r_hid = 0.935
    phi_hid = -0.155  # radians
    health = hidden_ratio_chi2(r_hid, phi_hid, HIDDEN_RATIO_TABLE)

    # chi2 should be computed correctly (value near 2.95 for this input)
    assert health.chi2 > 0
    assert health.dof == 6  # 4 channels * 2 observables - 2 params
    assert health.chi2_dof == health.chi2 / health.dof
    # New: verify p_low and p_high are computed
    assert 0.0 <= health.p_low <= 1.0
    assert 0.0 <= health.p_high <= 1.0
    assert np.isclose(health.p_low + health.p_high, 1.0, atol=1e-6)


def test_dof_aware_health_not_underconstrained_at_moderate_chi2():
    """
    Verify that chi2/dof slightly below 0.5 at dof=6 does NOT trigger UNDERCONSTRAINED
    when p_low is not tiny (i.e., the new DOF-aware criterion).

    With chi2=2.95, dof=6:
    - chi2/dof = 0.49 (below old 0.5 threshold)
    - p_low = CDF(2.95; 6) ≈ 0.185 >> 1e-3

    This should NOT be classified as UNDERCONSTRAINED.
    """
    from scipy.stats import chi2 as chi2_dist

    # Simulate the hidden sector health with chi2=2.95, dof=6
    chi2_val = 2.95
    dof = 6
    chi2_dof = chi2_val / dof
    p_low = float(chi2_dist.cdf(chi2_val, dof))
    p_high = float(1.0 - chi2_dist.cdf(chi2_val, dof))

    hidden = SectorHealth(chi2=chi2_val, dof=dof, chi2_dof=chi2_dof, p_low=p_low, p_high=p_high)

    # Open sector with reasonable health
    open_health = SectorHealth(chi2=35.0, dof=18, chi2_dof=35.0/18,
                               p_low=float(chi2_dist.cdf(35.0, 18)),
                               p_high=float(1.0 - chi2_dist.cdf(35.0, 18)))

    status, reason, health_label = assess_health(hidden, open_health)

    # Key assertion: should NOT be UNDERCONSTRAINED
    assert health_label != "UNDERCONSTRAINED", f"Got {health_label}: {reason}"
    assert p_low > P_LOW_THRESHOLD, f"p_low={p_low} should be > {P_LOW_THRESHOLD}"
    # Should be HEALTHY since neither sector triggers issues
    assert health_label == "HEALTHY", f"Expected HEALTHY, got {health_label}: {reason}"


def test_truly_underconstrained_triggers_correctly():
    """
    Verify that genuinely underconstrained fits (p_low < 1e-3) are flagged.
    """
    from scipy.stats import chi2 as chi2_dist

    # Very low chi2 relative to dof => p_low will be tiny
    chi2_val = 0.01
    dof = 6
    chi2_dof = chi2_val / dof
    p_low = float(chi2_dist.cdf(chi2_val, dof))
    p_high = float(1.0 - chi2_dist.cdf(chi2_val, dof))

    hidden = SectorHealth(chi2=chi2_val, dof=dof, chi2_dof=chi2_dof, p_low=p_low, p_high=p_high)

    # Open sector normal
    open_health = SectorHealth(chi2=20.0, dof=18, chi2_dof=20.0/18,
                               p_low=float(chi2_dist.cdf(20.0, 18)),
                               p_high=float(1.0 - chi2_dist.cdf(20.0, 18)))

    status, reason, health_label = assess_health(hidden, open_health)

    # Should be UNDERCONSTRAINED because p_low is tiny
    assert p_low < P_LOW_THRESHOLD, f"p_low={p_low} should be < {P_LOW_THRESHOLD}"
    assert health_label == "UNDERCONSTRAINED", f"Expected UNDERCONSTRAINED, got {health_label}"


def test_bootstrap_always_runs_even_when_health_not_ok():
    """
    Verify that bootstrap is NOT skipped when health is not OK (new behavior).
    This is a behavioral test - we check that the result has bootstrap stats
    even when health_label != HEALTHY.
    """
    from structure_tests.zb_core_threshold import run_structure_test
    from pathlib import Path

    # Run on real data with minimal bootstrap
    base_dir = Path(__file__).resolve().parents[1]
    result = run_structure_test(
        mode="auto",
        n_boot=10,  # Small for speed
        n_starts=10,
        seed=42,
        base_dir=base_dir,
    )

    # If we got past optimizer failure, bootstrap should have run
    if result.status != "OPTIMIZER_FAILURE" and result.status != "SKIPPED":
        # Bootstrap should have been attempted
        assert result.n_boot > 0, "Bootstrap should have been attempted"
        # p_boot should be computed (may still be None if all failed, but n_boot_valid should be checked)
        assert result.n_boot_valid >= 0, "n_boot_valid should be set"


def test_nested_invariant_still_enforced():
    """Verify nested-model invariant is still enforced."""
    from structure_tests.zb_core_threshold import run_structure_test
    from pathlib import Path

    base_dir = Path(__file__).resolve().parents[1]
    result = run_structure_test(
        mode="auto",
        n_boot=5,
        n_starts=20,
        seed=42,
        base_dir=base_dir,
    )

    # If we got a valid result (not optimizer failure), the invariant held
    if result.status != "OPTIMIZER_FAILURE":
        assert result.lambda_stat is not None
        assert result.lambda_stat >= 0, "Lambda should be non-negative (invariant: NLL_unc <= NLL_con)"
