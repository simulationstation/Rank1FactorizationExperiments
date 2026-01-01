#!/bin/bash
# ============================================================
# CMS Rank-1 Pipeline - Complete LXPLUS Execution Script
# ============================================================
# Usage:
#   1. Copy this entire cms_rank1_pipeline folder to lxplus:
#      scp -r cms_rank1_pipeline username@lxplus.cern.ch:~/
#
#   2. SSH to lxplus:
#      ssh username@lxplus.cern.ch
#
#   3. Run this script:
#      cd ~/cms_rank1_pipeline && chmod +x run_on_lxplus.sh && ./run_on_lxplus.sh
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
LOGFILE="$HOME/cms_rank1_pipeline/logs/LXPLUS_RUNLOG.txt"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOGFILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGFILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGFILE"
}

# Initialize log
mkdir -p "$HOME/cms_rank1_pipeline/logs"
echo "# CMS Rank-1 Pipeline LXPLUS Run Log" > "$LOGFILE"
echo "# Started: $(date)" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"

# ============================================================
# 0) Preconditions Check
# ============================================================
log "=== PRECONDITIONS CHECK ==="

log "Hostname: $(hostname)"
echo "Hostname: $(hostname)" >> "$LOGFILE"

# Check CMSSW tools
if ! command -v scramv1 &> /dev/null; then
    # Try to source CMS environment
    if [ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]; then
        log "Sourcing CMS environment from CVMFS..."
        source /cvmfs/cms.cern.ch/cmsset_default.sh
    else
        error "CMSSW tools not found and CVMFS not available. Are you on lxplus?"
    fi
fi

# Verify tools
for tool in scramv1 cmsrel cmsDriver.py; do
    if command -v $tool &> /dev/null; then
        log "$tool: $(which $tool)"
    else
        error "$tool not found after sourcing CMS environment"
    fi
done

# Check ROOT
if command -v root &> /dev/null; then
    log "ROOT: $(which root)"
else
    warn "ROOT not in PATH (will be available after cmsenv)"
fi

# Check DAS client
if command -v dasgoclient &> /dev/null; then
    log "dasgoclient: $(which dasgoclient)"
    DAS_AVAILABLE=true
else
    warn "dasgoclient not found - real data workflow will be skipped"
    DAS_AVAILABLE=false
fi

# Check VOMS proxy
if command -v voms-proxy-info &> /dev/null; then
    PROXY_TIME=$(voms-proxy-info --timeleft 2>/dev/null || echo "0")
    if [ "$PROXY_TIME" -gt 3600 ]; then
        log "VOMS proxy valid for $PROXY_TIME seconds"
        PROXY_VALID=true
    else
        warn "VOMS proxy expired or not found"
        PROXY_VALID=false
    fi
else
    warn "voms-proxy-info not found"
    PROXY_VALID=false
fi

# ============================================================
# 1) Setup CMSSW Area
# ============================================================
log "=== SETTING UP CMSSW AREA ==="

WORKDIR="$HOME/cms_rank1_pipeline_run"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

log "Working directory: $WORKDIR"

# Create CMSSW release if not exists
if [ ! -d "CMSSW_14_0_13" ]; then
    log "Creating CMSSW_14_0_13..."
    cmsrel CMSSW_14_0_13
else
    log "CMSSW_14_0_13 already exists"
fi

cd CMSSW_14_0_13/src
eval `scramv1 runtime -sh`  # cmsenv equivalent

log "CMSSW environment set up: $CMSSW_BASE"

# Clone fork if not exists
if [ ! -d "fork_cmssw" ]; then
    log "Cloning CMSSW fork..."
    git clone --depth 1 https://github.com/simulationstation/cmssw fork_cmssw
else
    log "Fork already cloned"
fi

# Copy modules
log "Copying BPHNano modules..."
cp -r fork_cmssw/PhysicsTools/BPHNano PhysicsTools/ 2>/dev/null || mkdir -p PhysicsTools && cp -r fork_cmssw/PhysicsTools/BPHNano PhysicsTools/
cp fork_cmssw/PhysicsTools/NanoAOD/python/custom_bph_cff.py PhysicsTools/NanoAOD/python/ 2>/dev/null || true

# Copy generator configs
log "Copying generator configs..."
cp fork_cmssw/Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py Configuration/Generator/python/ 2>/dev/null || true
cp fork_cmssw/Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py Configuration/Generator/python/ 2>/dev/null || true

# Build
log "Building CMSSW (this may take a few minutes)..."
scram b -j8 2>&1 | tail -20

# Copy analysis configs
log "Copying analysis configs..."
mkdir -p "$WORKDIR/configs"
if [ -d "$SCRIPT_DIR/configs" ]; then
    cp -v "$SCRIPT_DIR/configs"/*.py "$WORKDIR/configs/" 2>/dev/null || true
fi

# Create output directories
mkdir -p "$WORKDIR/outputs/local_mc/fourmu"
mkdir -p "$WORKDIR/outputs/local_mc/dijpsi"
mkdir -p "$WORKDIR/outputs/local_mc/rank1_test_mc"
mkdir -p "$WORKDIR/outputs/real_data"

# ============================================================
# 2) MC Validation A: FourMu GEN-only
# ============================================================
log "=== MC VALIDATION A: FOURMU ==="

cd "$WORKDIR/CMSSW_14_0_13/src"

# Generate cmsDriver config
log "Generating FourMu cmsDriver config..."
cmsDriver.py Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py \
  --fileout file:fourmu_gen.root \
  --mc \
  --eventcontent GENRAW \
  --datatier GEN \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step GEN \
  --era Run3 \
  --python_filename fourmu_gen_cfg.py \
  --no_exec \
  -n 20000

# Run generation
log "Running FourMu generation (20000 events)..."
cmsRun fourmu_gen_cfg.py 2>&1 | tail -50

# Check output
if [ -f "fourmu_gen.root" ]; then
    FOURMU_SIZE=$(du -h fourmu_gen.root | cut -f1)
    log "FourMu GEN output: fourmu_gen.root ($FOURMU_SIZE)"
else
    error "FourMu generation failed - no output file"
fi

# Run analysis
log "Running FourMu histogram analysis..."
if [ -f "$WORKDIR/configs/fourmu_analysis_cfg.py" ]; then
    python3 "$WORKDIR/configs/fourmu_analysis_cfg.py" \
      --input fourmu_gen.root \
      --output fourmu_hist.root \
      --csv fourmu_hist.csv \
      --gen-only
else
    warn "fourmu_analysis_cfg.py not found, using inline analysis..."
    python3 << 'PYEOF'
import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

# Simple GEN analysis
f = ROOT.TFile.Open("fourmu_gen.root")
tree = f.Get("Events")

h_dimu = ROOT.TH1D("h_dimu", "Dimuon Mass;m(#mu#mu) [GeV];Events", 150, 0, 15)
h_4mu = ROOT.TH1D("h_4mu", "4-Muon Mass;m(4#mu) [GeV];Events", 100, 0, 50)

M_MU = 0.10566

for evt in tree:
    muons = []
    for i in range(evt.nGenPart):
        if abs(evt.GenPart_pdgId[i]) == 13 and evt.GenPart_status[i] == 1:
            pt = evt.GenPart_pt[i]
            eta = evt.GenPart_eta[i]
            phi = evt.GenPart_phi[i]
            charge = -1 if evt.GenPart_pdgId[i] > 0 else 1
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            E = np.sqrt(px**2 + py**2 + pz**2 + M_MU**2)
            muons.append((E, px, py, pz, charge))

    # Dimuon
    for i in range(len(muons)):
        for j in range(i+1, len(muons)):
            if muons[i][4] * muons[j][4] < 0:
                E = muons[i][0] + muons[j][0]
                px = muons[i][1] + muons[j][1]
                py = muons[i][2] + muons[j][2]
                pz = muons[i][3] + muons[j][3]
                m = np.sqrt(max(0, E**2 - px**2 - py**2 - pz**2))
                h_dimu.Fill(m)

    # 4-muon
    if len(muons) >= 4:
        E = sum(m[0] for m in muons[:4])
        px = sum(m[1] for m in muons[:4])
        py = sum(m[2] for m in muons[:4])
        pz = sum(m[3] for m in muons[:4])
        m = np.sqrt(max(0, E**2 - px**2 - py**2 - pz**2))
        h_4mu.Fill(m)

f.Close()

# Save
fout = ROOT.TFile("fourmu_hist.root", "RECREATE")
h_dimu.Write()
h_4mu.Write()
fout.Close()

# Write CSV
with open("fourmu_hist_dimu.csv", "w") as f:
    f.write("mass_GeV,counts,stat_err\n")
    for i in range(1, h_dimu.GetNbinsX()+1):
        f.write(f"{h_dimu.GetBinCenter(i):.4f},{h_dimu.GetBinContent(i):.0f},{h_dimu.GetBinError(i):.4f}\n")

with open("fourmu_hist_4mu.csv", "w") as f:
    f.write("mass_GeV,counts,stat_err\n")
    for i in range(1, h_4mu.GetNbinsX()+1):
        f.write(f"{h_4mu.GetBinCenter(i):.4f},{h_4mu.GetBinContent(i):.0f},{h_4mu.GetBinError(i):.4f}\n")

print("FourMu analysis complete")
PYEOF
fi

# Copy outputs
cp -v fourmu_gen.root fourmu_hist*.root fourmu_hist*.csv "$WORKDIR/outputs/local_mc/fourmu/" 2>/dev/null || true

log "FourMu outputs:"
ls -la "$WORKDIR/outputs/local_mc/fourmu/"

log "First 5 lines of FourMu dimuon CSV:"
head -5 "$WORKDIR/outputs/local_mc/fourmu/fourmu_hist_dimu.csv" 2>/dev/null || head -5 "$WORKDIR/outputs/local_mc/fourmu/fourmu_hist.csv" 2>/dev/null || echo "No CSV found"

# ============================================================
# 3) MC Validation B: EtaB→J/ψJ/ψ GEN-only
# ============================================================
log "=== MC VALIDATION B: DIJPSI ==="

cd "$WORKDIR/CMSSW_14_0_13/src"

# Generate cmsDriver config
log "Generating DiJpsi cmsDriver config..."
cmsDriver.py Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py \
  --fileout file:dijpsi_gen.root \
  --mc \
  --eventcontent GENRAW \
  --datatier GEN \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step GEN \
  --era Run3 \
  --python_filename dijpsi_gen_cfg.py \
  --no_exec \
  -n 50000

# Run generation
log "Running DiJpsi generation (50000 events)..."
cmsRun dijpsi_gen_cfg.py 2>&1 | tail -50

# Check output
if [ -f "dijpsi_gen.root" ]; then
    DIJPSI_SIZE=$(du -h dijpsi_gen.root | cut -f1)
    log "DiJpsi GEN output: dijpsi_gen.root ($DIJPSI_SIZE)"
else
    error "DiJpsi generation failed - no output file"
fi

# Run analysis
log "Running DiJpsi histogram analysis..."
if [ -f "$WORKDIR/configs/dijpsi_analysis_cfg.py" ]; then
    python3 "$WORKDIR/configs/dijpsi_analysis_cfg.py" \
      --input dijpsi_gen.root \
      --output dijpsi_hist.root \
      --csv dijpsi_hist.csv \
      --gen-only
else
    warn "dijpsi_analysis_cfg.py not found, using inline analysis..."
    python3 << 'PYEOF'
import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

f = ROOT.TFile.Open("dijpsi_gen.root")
tree = f.Get("Events")

h_jpsi = ROOT.TH1D("h_jpsi", "J/#psi Mass;m(#mu#mu) [GeV];Events", 100, 2.5, 3.5)
h_dijpsi = ROOT.TH1D("h_dijpsi", "Di-J/#psi Mass;m(J/#psi J/#psi) [GeV];Events", 180, 6, 15)
h_4mu = ROOT.TH1D("h_4mu", "4-Muon Mass;m(4#mu) [GeV];Events", 180, 6, 15)

M_MU = 0.10566
M_JPSI = 3.097

for evt in tree:
    muons = []
    jpsis = []

    for i in range(evt.nGenPart):
        pdg = evt.GenPart_pdgId[i]
        status = evt.GenPart_status[i]

        # J/psi
        if abs(pdg) == 443:
            pt = evt.GenPart_pt[i]
            eta = evt.GenPart_eta[i]
            phi = evt.GenPart_phi[i]
            mass = evt.GenPart_mass[i]
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
            jpsis.append((E, px, py, pz))
            h_jpsi.Fill(mass)

        # Final-state muons
        if abs(pdg) == 13 and status == 1:
            pt = evt.GenPart_pt[i]
            eta = evt.GenPart_eta[i]
            phi = evt.GenPart_phi[i]
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            E = np.sqrt(px**2 + py**2 + pz**2 + M_MU**2)
            muons.append((E, px, py, pz))

    # Di-J/psi
    if len(jpsis) >= 2:
        E = jpsis[0][0] + jpsis[1][0]
        px = jpsis[0][1] + jpsis[1][1]
        py = jpsis[0][2] + jpsis[1][2]
        pz = jpsis[0][3] + jpsis[1][3]
        m = np.sqrt(max(0, E**2 - px**2 - py**2 - pz**2))
        h_dijpsi.Fill(m)

    # 4-muon
    if len(muons) >= 4:
        E = sum(m[0] for m in muons[:4])
        px = sum(m[1] for m in muons[:4])
        py = sum(m[2] for m in muons[:4])
        pz = sum(m[3] for m in muons[:4])
        m = np.sqrt(max(0, E**2 - px**2 - py**2 - pz**2))
        h_4mu.Fill(m)

f.Close()

# Save
fout = ROOT.TFile("dijpsi_hist.root", "RECREATE")
h_jpsi.Write()
h_dijpsi.Write()
h_4mu.Write()
fout.Close()

# Write CSVs
with open("dijpsi_hist_jpsi.csv", "w") as f:
    f.write("mass_GeV,counts,stat_err\n")
    for i in range(1, h_jpsi.GetNbinsX()+1):
        f.write(f"{h_jpsi.GetBinCenter(i):.4f},{h_jpsi.GetBinContent(i):.0f},{h_jpsi.GetBinError(i):.4f}\n")

with open("dijpsi_hist_dijpsi.csv", "w") as f:
    f.write("mass_GeV,counts,stat_err\n")
    for i in range(1, h_dijpsi.GetNbinsX()+1):
        f.write(f"{h_dijpsi.GetBinCenter(i):.4f},{h_dijpsi.GetBinContent(i):.0f},{h_dijpsi.GetBinError(i):.4f}\n")

print("DiJpsi analysis complete")
PYEOF
fi

# Copy outputs
cp -v dijpsi_gen.root dijpsi_hist*.root dijpsi_hist*.csv "$WORKDIR/outputs/local_mc/dijpsi/" 2>/dev/null || true

log "DiJpsi outputs:"
ls -la "$WORKDIR/outputs/local_mc/dijpsi/"

log "First 5 lines of DiJpsi spectrum CSV:"
head -5 "$WORKDIR/outputs/local_mc/dijpsi/dijpsi_hist_dijpsi.csv" 2>/dev/null || head -5 "$WORKDIR/outputs/local_mc/dijpsi/dijpsi_hist.csv" 2>/dev/null || echo "No CSV found"

# ============================================================
# 4) Run Rank-1 Test on MC Outputs
# ============================================================
log "=== RANK-1 TEST ON MC OUTPUTS ==="

cd "$WORKDIR"

if [ -f "$WORKDIR/configs/cms_rank1_test.py" ]; then
    python3 "$WORKDIR/configs/cms_rank1_test.py" \
      --channel-a "$WORKDIR/outputs/local_mc/dijpsi/dijpsi_hist_dijpsi.csv" \
      --channel-b "$WORKDIR/outputs/local_mc/fourmu/fourmu_hist_dimu.csv" \
      --output "$WORKDIR/outputs/local_mc/rank1_test_mc/RESULT.md" \
      --bootstrap 100 2>&1 | tee "$WORKDIR/outputs/local_mc/rank1_test_mc/RESULT.txt"
else
    log "cms_rank1_test.py not found - skipping rank-1 test"
fi

# ============================================================
# 5) Real Data Attempt (if tools available)
# ============================================================
log "=== REAL DATA WORKFLOW ==="

REAL_DATA_SUCCESS=false

if [ "$DAS_AVAILABLE" = true ] && [ "$PROXY_VALID" = true ]; then
    log "DAS and VOMS proxy available - attempting real data workflow..."

    cd "$WORKDIR/CMSSW_14_0_13/src"

    # Query DAS
    log "Querying DAS for ParkingDoubleMuonLowMass files..."
    dasgoclient -query="file dataset=/ParkingDoubleMuonLowMass0/Run2022G-22Sep2023-v1/MINIAOD" --limit=20 > files_2022G.txt 2>&1

    if [ -s "files_2022G.txt" ]; then
        FIRST_FILE=$(head -1 files_2022G.txt)
        log "First file: $FIRST_FILE"

        # Test XRootD access
        log "Testing XRootD access..."
        if xrdfs root://cms-xrd-global.cern.ch/ stat "$FIRST_FILE" 2>/dev/null; then
            log "XRootD access successful"

            # Run NanoAOD+BPH production (limited events)
            if [ -f "$WORKDIR/configs/realdata_bphnano_cfg.py" ]; then
                log "Running NanoAOD+BPH production (10000 events)..."
                cmsRun "$WORKDIR/configs/realdata_bphnano_cfg.py" \
                  inputFiles="root://cms-xrd-global.cern.ch/$FIRST_FILE" \
                  maxEvents=10000 2>&1 | tail -50

                if [ -f "bphnano_output.root" ]; then
                    log "NanoAOD+BPH output created"

                    # Run histogram analysis
                    if [ -f "$WORKDIR/configs/realdata_hist_cfg.py" ]; then
                        python3 "$WORKDIR/configs/realdata_hist_cfg.py" \
                          --input bphnano_output.root \
                          --output realdata_hist.root \
                          --csv-prefix realdata

                        cp -v bphnano_output.root realdata_hist.root realdata*.csv "$WORKDIR/outputs/real_data/" 2>/dev/null
                        REAL_DATA_SUCCESS=true
                        log "Real data workflow completed!"
                    fi
                fi
            fi
        else
            warn "XRootD access failed - skipping real data"
        fi
    else
        warn "DAS query returned no files"
    fi
else
    warn "Skipping real data workflow (DAS: $DAS_AVAILABLE, Proxy: $PROXY_VALID)"
fi

# ============================================================
# 6) Final Report
# ============================================================
log "=== GENERATING FINAL REPORT ==="

cat > "$WORKDIR/reports/RUN_SUMMARY.md" << EOF
# CMS Rank-1 Pipeline Run Summary

## Run Information

- **Date**: $(date)
- **Hostname**: $(hostname)
- **CMSSW**: CMSSW_14_0_13
- **Working Directory**: $WORKDIR

## MC Workflows

### FourMu GEN-only
$(if [ -f "$WORKDIR/outputs/local_mc/fourmu/fourmu_gen.root" ]; then echo "- Status: SUCCESS"; else echo "- Status: FAILED"; fi)
$(ls -lh "$WORKDIR/outputs/local_mc/fourmu/" 2>/dev/null | grep -E "\.root|\.csv")

### EtaB→J/ψJ/ψ GEN-only
$(if [ -f "$WORKDIR/outputs/local_mc/dijpsi/dijpsi_gen.root" ]; then echo "- Status: SUCCESS"; else echo "- Status: FAILED"; fi)
$(ls -lh "$WORKDIR/outputs/local_mc/dijpsi/" 2>/dev/null | grep -E "\.root|\.csv")

### Rank-1 Test
$(if [ -f "$WORKDIR/outputs/local_mc/rank1_test_mc/RESULT.md" ]; then echo "- Status: COMPLETED"; cat "$WORKDIR/outputs/local_mc/rank1_test_mc/RESULT.md" | head -20; else echo "- Status: NOT RUN"; fi)

## Real Data Workflow

$(if [ "$REAL_DATA_SUCCESS" = true ]; then echo "- Status: SUCCESS"; ls -lh "$WORKDIR/outputs/real_data/"; else echo "- Status: SKIPPED (DAS: $DAS_AVAILABLE, Proxy: $PROXY_VALID)"; fi)

## Output Paths

- FourMu: \`$WORKDIR/outputs/local_mc/fourmu/\`
- DiJpsi: \`$WORKDIR/outputs/local_mc/dijpsi/\`
- Rank-1: \`$WORKDIR/outputs/local_mc/rank1_test_mc/\`
- Real Data: \`$WORKDIR/outputs/real_data/\`

## Next Steps

1. Increase event counts for better statistics
2. Run full GEN-SIM-RECO chain for realistic detector effects
3. Process more real data files via CRAB
4. Apply proper physics selections for exotic searches
EOF

log "Run summary written to: $WORKDIR/reports/RUN_SUMMARY.md"

# ============================================================
# Final Output
# ============================================================
log "============================================================"
log "PIPELINE COMPLETE"
log "============================================================"

log "Output paths:"
echo "  FourMu:   $WORKDIR/outputs/local_mc/fourmu/"
echo "  DiJpsi:   $WORKDIR/outputs/local_mc/dijpsi/"
echo "  Rank-1:   $WORKDIR/outputs/local_mc/rank1_test_mc/"
echo "  Real:     $WORKDIR/outputs/real_data/"

log "CSV files:"
find "$WORKDIR/outputs" -name "*.csv" -exec ls -lh {} \;

log "First 5 lines of each CSV:"
for csv in $(find "$WORKDIR/outputs" -name "*.csv"); do
    echo "=== $csv ==="
    head -5 "$csv"
    echo ""
done

log "Full log: $LOGFILE"
log "Done!"
