#!/bin/bash
set -e

# ============================================================
# CMS Rank-1 Pipeline - Docker Runner
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "CMS Rank-1 Pipeline - Docker Runner"
echo "============================================================"

# Check prerequisites
echo ""
echo "=== Checking prerequisites ==="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found"
    exit 1
fi
echo "Docker: $(docker --version)"

# Check CVMFS
if [ ! -d "/cvmfs/cms.cern.ch" ]; then
    echo "ERROR: CVMFS not mounted at /cvmfs/cms.cern.ch"
    exit 1
fi
echo "CVMFS: OK"

# Create outputs directory
mkdir -p outputs logs

# Build Docker image
echo ""
echo "=== Building Docker image ==="
docker build -t cmssw-rank1:genonly . || { echo "ERROR: Docker build failed"; exit 1; }

# Make entrypoint executable
chmod +x configs/entrypoint.sh configs/entrypoint_simple.sh 2>/dev/null || true

# Clear previous outputs
echo ""
echo "=== Clearing previous outputs ==="
rm -f outputs/*.root outputs/*.csv outputs/*.txt outputs/*.md 2>/dev/null || true

# Run container
echo ""
echo "=== Running container ==="
echo "Running GEN-ONLY CMSSW workflow (real cmsRun, not mock data)"
echo "This will take 10-30 minutes for event generation..."
echo ""

if ! docker run --rm \
    -v /cvmfs:/cvmfs:ro \
    -v "$SCRIPT_DIR/outputs:/outputs" \
    -v "$SCRIPT_DIR/configs:/configs:ro" \
    --user root \
    cmssw-rank1:genonly \
    bash -lc "/configs/entrypoint.sh"; then
    echo ""
    echo "ERROR: Container execution failed!"
    echo "Check logs in outputs/CONTAINER_RUNLOG.txt"
    exit 1
fi

# Verify outputs
echo ""
echo "============================================================"
echo "=== Verifying outputs ==="
echo "============================================================"

MISSING=0

for f in fourmu_gen.root dijpsi_gen.root fourmu_hist_4mu.csv dijpsi_hist_dijpsi.csv RANK1_RESULT.md CONTAINER_RUNLOG.txt; do
    if [ -f "outputs/$f" ]; then
        SIZE=$(du -h "outputs/$f" | cut -f1)
        echo "  OK: $f ($SIZE)"
    else
        echo "  MISSING: $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "WARNING: Some outputs are missing!"
else
    echo ""
    echo "All outputs present!"
fi

# Print CSV previews
echo ""
echo "=== CSV Previews ==="

for csv in outputs/*.csv; do
    if [ -f "$csv" ]; then
        echo ""
        echo "--- $(basename $csv) ---"
        head -10 "$csv"
    fi
done

# Print rank-1 result
echo ""
echo "=== Rank-1 Result ==="
cat outputs/RANK1_RESULT.md 2>/dev/null || echo "Not found"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Outputs in: $SCRIPT_DIR/outputs/"
echo "============================================================"
