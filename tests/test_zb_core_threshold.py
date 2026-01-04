from pathlib import Path

import numpy as np

from structure_tests.zb_core_threshold import (
    HiddenRatioDatum,
    OpenRatioDatum,
    _phi_with_spin_flip,
    hidden_ratio_nll,
    open_ratio_nll,
    resolve_open_mode,
    wrap_to_pi,
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
