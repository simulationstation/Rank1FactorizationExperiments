# Relevant CMSSW Paths in Fork

## Generator Configurations

### FourMu Generators

| File | Status | Description |
|------|--------|-------------|
| `Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py` | FOUND | 4-muon gun (pT 1-200 GeV), uses Pythia8PtGun with AddAntiParticle |
| `Configuration/Generator/python/FourMuExtendedPt_1_200_pythia8_cfi.py` | FOUND | Extended version of 4-muon generator |

### EtaB Generators

| File | Status | Description |
|------|--------|-------------|
| `Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py` | FOUND | EtaB→J/ψJ/ψ at 14 TeV, with muon pT>1.8 GeV filter |
| `Configuration/Generator/python/EtaBToJpsiJpsi_forSTEAM_13TeV_cfi.py` | FOUND | Alternative 13 TeV version |
| `Configuration/Generator/python/EtaBToUpsilon1SUpsilon1S_forSTEAM_13TeV_cfi.py` | FOUND | EtaB→Υ(1S)Υ(1S) variant |

## NanoAOD/BPH Customizations

### Core BPH Integration

| File | Status | Description |
|------|--------|-------------|
| `PhysicsTools/NanoAOD/python/custom_bph_cff.py` | FOUND | Main BPH customization with functions: `nanoAOD_customizeBPH()`, `nanoAOD_customizeDiMuonBPH()`, etc. |

### BPHNano Module

| File | Status | Description |
|------|--------|-------------|
| `PhysicsTools/BPHNano/python/muons_cff.py` | FOUND | Muon selection and trigger matching |
| `PhysicsTools/BPHNano/python/MuMu_cff.py` | FOUND | DiMuon builder (J/ψ, ψ(2S)) |
| `PhysicsTools/BPHNano/python/tracks_cff.py` | FOUND | Track selection for B physics |
| `PhysicsTools/BPHNano/python/DiTrack_cff.py` | FOUND | Di-track builder |
| `PhysicsTools/BPHNano/python/BToKLL_cff.py` | FOUND | B→K μμ reconstruction |
| `PhysicsTools/BPHNano/python/BToTrkTrkLL_cff.py` | FOUND | B→hh μμ reconstruction |
| `PhysicsTools/BPHNano/python/V0_cff.py` | FOUND | V0 (Ks, Λ) reconstruction |
| `PhysicsTools/BPHNano/python/BToV0LL_cff.py` | FOUND | B→V0 μμ reconstruction |

### Production Configuration

| File | Status | Description |
|------|--------|-------------|
| `PhysicsTools/BPHNano/production/samples.yml` | FOUND | Dataset definitions for ParkingDoubleMuonLowMass (2022-2024) |
| `PhysicsTools/BPHNano/production/submit_on_crab.py` | FOUND | CRAB submission script |

## Heavy Flavor Analysis

| File | Status | Description |
|------|--------|-------------|
| `HeavyFlavorAnalysis/SpecificDecay/plugins/BPHHistoSpecificDecay.cc` | FOUND | Histogramming plugin for B decay mass spectra |

## BPHNano Plugins

| File | Description |
|------|-------------|
| `PhysicsTools/BPHNano/plugins/DiLeptonBuilder.cc` | Di-lepton candidate builder |
| `PhysicsTools/BPHNano/plugins/MuonTriggerSelector.cc` | Trigger-matched muon selector |
| `PhysicsTools/BPHNano/plugins/KinVtxFitter.cc` | Kinematic vertex fitter |
| `PhysicsTools/BPHNano/plugins/DiTrackBuilder.cc` | Di-track builder |
| `PhysicsTools/BPHNano/plugins/V0ReBuilder.cc` | V0 re-builder |
| `PhysicsTools/BPHNano/plugins/BToTrkLLBuilder.cc` | B→track μμ builder |
| `PhysicsTools/BPHNano/plugins/BToTrkTrkLLBuilder.cc` | B→hh μμ builder |
| `PhysicsTools/BPHNano/plugins/BToV0LLBuilder.cc` | B→V0 μμ builder |

## Key Configuration Details

### EtaB→J/ψJ/ψ Generator Features
```python
# Mass fixed to 9.4 GeV (eta_b mass)
'35:m0 = 9.4'
# Decay: eta_b → J/ψ J/ψ
'35:addChannel 1 1.00 100 443 443'
# J/ψ forced to μμ
'443:onIfMatch 13 -13'
# Muon filter: pT > 1.8 GeV, |η| < 2.5
```

### FourMu Gun Features
```python
# Two μ⁻ with AddAntiParticle = True → 4 muons total
ParticleID = [-13, -13]
AddAntiParticle = True
# pT range: 0.9 - 200 GeV
MinPt = 0.9
MaxPt = 200.0
# η range: |η| < 2.5
```

### Muon Selection (muons_cff.py)
```python
muonSelection = "pt > 3 && abs(eta) < 2.4"
HLTPaths = [
    "HLT_DoubleMu4_LowMass_Displaced",
    "HLT_DoubleMu4_3_LowMass",
    "HLT_Mu8", "HLT_Mu3_PFJet40",
    "HLT_Mu4_L1DoubleMu", "HLT_Mu0_L1DoubleMu"
]
```
