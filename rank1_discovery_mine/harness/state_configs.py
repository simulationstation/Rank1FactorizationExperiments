"""
Exotic hadron state configurations for rank-1 tests.

Contains masses, widths, and fit windows for known exotic hadron doublets/families.
All masses and widths in MeV.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class StateConfig:
    """Configuration for a two-state rank-1 test."""
    state1_name: str
    state1_mass: float  # MeV
    state1_width: float  # MeV
    state2_name: str
    state2_mass: float  # MeV
    state2_width: float  # MeV
    fit_window: Tuple[float, float]  # (m_low, m_high) in MeV
    notes: str = ""


# Known exotic hadron configurations
STATE_CONFIGS: Dict[str, StateConfig] = {
    # LHCb Pentaquarks - Pc(4312) family
    "lhcb_pc_4312_extensions": StateConfig(
        state1_name="Pc(4440)",
        state1_mass=4440.3,
        state1_width=20.6,
        state2_name="Pc(4457)",
        state2_mass=4457.3,
        state2_width=6.4,
        fit_window=(4400, 4500),
        notes="LHCb PRL 122, 222001 (2019)"
    ),

    # LHCb Pcs (strange pentaquarks)
    "lhcb_pcs_jpsilambda_family": StateConfig(
        state1_name="Pcs(4338)",
        state1_mass=4338.2,
        state1_width=7.0,
        state2_name="Pcs(4459)",
        state2_mass=4459.0,
        state2_width=17.0,
        fit_window=(4300, 4500),
        notes="LHCb Sci. Bull. 66, 1278 (2021)"
    ),

    # LHCb X(4140) family in J/psi phi
    "lhcb_jpsiphi_x4140_family": StateConfig(
        state1_name="X(4140)",
        state1_mass=4146.5,
        state1_width=83.0,
        state2_name="X(4274)",
        state2_mass=4273.3,
        state2_width=56.2,
        fit_window=(4100, 4350),
        notes="LHCb PRD 95, 012002 (2017)"
    ),

    # CMS X(4140) family
    "cms_jpsiphi_x4140_family": StateConfig(
        state1_name="X(4140)",
        state1_mass=4146.5,
        state1_width=83.0,
        state2_name="X(4274)",
        state2_mass=4273.3,
        state2_width=56.2,
        fit_window=(4100, 4350),
        notes="CMS JHEP 04, 154 (2023)"
    ),

    # ATLAS X(4140) family
    "atlas_jpsiphi_x4140_family": StateConfig(
        state1_name="X(4140)",
        state1_mass=4146.5,
        state1_width=83.0,
        state2_name="X(4274)",
        state2_mass=4273.3,
        state2_width=56.2,
        fit_window=(4100, 4350),
        notes="ATLAS"
    ),

    # BESIII Zc(3900)/Zc(4020) family
    "besiii_zc3900_pijpsi_ddstar": StateConfig(
        state1_name="Zc(3900)",
        state1_mass=3887.2,
        state1_width=28.4,
        state2_name="Zc(4020)",
        state2_mass=4024.1,
        state2_width=13.0,
        fit_window=(3850, 4100),
        notes="BESIII PRL 110, 252001 (2013)"
    ),

    # BESIII Zc(4020)
    "besiii_zc4020_pihc_dstar_dstar": StateConfig(
        state1_name="Zc(3900)",
        state1_mass=3887.2,
        state1_width=28.4,
        state2_name="Zc(4020)",
        state2_mass=4024.1,
        state2_width=13.0,
        fit_window=(3850, 4100),
        notes="BESIII PRL 111, 242001 (2013)"
    ),

    # BESIII Y(4360)/Y(4660) family
    "besiii_y4360_y4660_pipipsi2s": StateConfig(
        state1_name="Y(4360)",
        state1_mass=4368.0,
        state1_width=96.0,
        state2_name="Y(4660)",
        state2_mass=4643.0,
        state2_width=72.0,
        fit_window=(4200, 4800),
        notes="BESIII PRL 118, 092001 (2017)"
    ),

    # BESIII Y(4230)/Y(4390)
    "besiii_y4230_4390_pipiomega": StateConfig(
        state1_name="Y(4230)",
        state1_mass=4222.0,
        state1_width=44.0,
        state2_name="Y(4390)",
        state2_mass=4391.0,
        state2_width=139.0,
        fit_window=(4150, 4500),
        notes="BESIII"
    ),

    # LHCb Zcs states
    "lhcb_zcs_4000_4220": StateConfig(
        state1_name="Zcs(4000)",
        state1_mass=4003.0,
        state1_width=131.0,
        state2_name="Zcs(4220)",
        state2_mass=4216.0,
        state2_width=233.0,
        fit_window=(3900, 4400),
        notes="LHCb PRL 127, 082001 (2021)"
    ),

    # BESIII Zcs(3985)
    "besiii_zcs_3985": StateConfig(
        state1_name="Zcs(3985)",
        state1_mass=3982.5,
        state1_width=12.8,
        state2_name="Zcs(4000)",
        state2_mass=4003.0,
        state2_width=131.0,
        fit_window=(3900, 4100),
        notes="BESIII PRL 126, 102001 (2021)"
    ),

    # LHCb X(2900) doublet
    "lhcb_x2900_doublet_dk": StateConfig(
        state1_name="X0(2900)",
        state1_mass=2866.0,
        state1_width=57.0,
        state2_name="X1(2900)",
        state2_mass=2904.0,
        state2_width=110.0,
        fit_window=(2750, 3000),
        notes="LHCb PRL 125, 242001 (2020)"
    ),

    # LHCb X(6900) family (fully-charm tetraquark)
    "lhcb_x6900_jpsijpsi": StateConfig(
        state1_name="X(6600)",
        state1_mass=6552.0,
        state1_width=124.0,
        state2_name="X(6900)",
        state2_mass=6886.0,
        state2_width=168.0,
        fit_window=(6400, 7100),
        notes="LHCb Sci. Bull. 65, 1983 (2020)"
    ),

    # CMS X(6900) family
    "cms_x6900_x7100_jpsijpsi": StateConfig(
        state1_name="X(6600)",
        state1_mass=6552.0,
        state1_width=124.0,
        state2_name="X(6900)",
        state2_mass=6886.0,
        state2_width=168.0,
        fit_window=(6400, 7100),
        notes="CMS"
    ),

    # ATLAS X(6900)
    "atlas_x6900_jpsijpsi": StateConfig(
        state1_name="X(6600)",
        state1_mass=6552.0,
        state1_width=124.0,
        state2_name="X(6900)",
        state2_mass=6886.0,
        state2_width=168.0,
        fit_window=(6400, 7100),
        notes="ATLAS"
    ),

    # Belle Zb(10610)/Zb(10650) family
    "belle_zb_hb_pipi": StateConfig(
        state1_name="Zb(10610)",
        state1_mass=10607.2,
        state1_width=18.4,
        state2_name="Zb(10650)",
        state2_mass=10652.2,
        state2_width=11.5,
        fit_window=(10550, 10700),
        notes="Belle PRL 108, 122001 (2012)"
    ),

    # Belle Y_b states
    "belle_yb_10750_10860": StateConfig(
        state1_name="Yb(10750)",
        state1_mass=10752.7,
        state1_width=35.5,
        state2_name="Y(10860)",
        state2_mass=10885.2,
        state2_width=37.4,
        fit_window=(10700, 10950),
        notes="Belle"
    ),

    # Belle ISR Y family
    "belle_isr_y_family": StateConfig(
        state1_name="Y(4260)",
        state1_mass=4230.0,
        state1_width=55.0,
        state2_name="Y(4360)",
        state2_mass=4368.0,
        state2_width=96.0,
        fit_window=(4150, 4500),
        notes="Belle"
    ),

    # LHCb Tcc+
    "lhcb_tcc_d0d0pi": StateConfig(
        state1_name="Tcc(3875)",
        state1_mass=3874.83,
        state1_width=0.41,
        state2_name="Tcc*(3876)",
        state2_mass=3876.0,
        state2_width=1.0,
        fit_window=(3870, 3880),
        notes="LHCb Nat. Phys. 18, 751 (2022)"
    ),

    # Control: X(3872) multi-channel
    "control_x3872_multichannnel": StateConfig(
        state1_name="X(3872)",
        state1_mass=3871.65,
        state1_width=1.19,
        state2_name="chi_c1(2P)",
        state2_mass=3871.7,
        state2_width=0.96,
        fit_window=(3865, 3880),
        notes="Control/calibration"
    ),

    # BESIII X(3872) lineshape
    "besiii_x3872_lineshape": StateConfig(
        state1_name="X(3872)",
        state1_mass=3871.65,
        state1_width=1.19,
        state2_name="chi_c1(2P)",
        state2_mass=3871.7,
        state2_width=0.96,
        fit_window=(3865, 3880),
        notes="BESIII"
    ),

    # LHCb X(3960) in Ds+Ds-
    "lhcb_x3960_dsd_s": StateConfig(
        state1_name="X(3960)",
        state1_mass=3960.0,
        state1_width=50.0,
        state2_name="chi_c0(2P)",
        state2_mass=3915.0,
        state2_width=20.0,
        fit_window=(3850, 4050),
        notes="LHCb PRD 106, 072002 (2022)"
    ),

    # LHCb X(4630) in J/psi phi
    "lhcb_x4630_jpsi_phi": StateConfig(
        state1_name="X(4500)",
        state1_mass=4506.0,
        state1_width=92.0,
        state2_name="X(4700)",
        state2_mass=4704.0,
        state2_width=120.0,
        fit_window=(4400, 4800),
        notes="LHCb"
    ),
}


def get_state_config(slug: str) -> Optional[StateConfig]:
    """Get state configuration by candidate slug."""
    return STATE_CONFIGS.get(slug)


def list_configured_states() -> List[str]:
    """List all configured state slugs."""
    return list(STATE_CONFIGS.keys())
