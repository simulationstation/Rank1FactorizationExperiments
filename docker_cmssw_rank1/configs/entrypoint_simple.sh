#!/bin/bash
set -e

# ============================================================
# CMS Rank-1 Pipeline - Simplified Container Entrypoint
# Uses mock data generation (no CMSSW required)
# ============================================================

LOGFILE="/outputs/CONTAINER_RUNLOG.txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "CMS Rank-1 Pipeline - Container Execution (Mock Data Mode)"
echo "Started: $(date)"
echo "============================================================"

cd /work

# ============================================================
# Generate mock FourMu data
# ============================================================
echo ""
echo "============================================================"
echo "=== Generating Mock FourMu Data ==="
echo "============================================================"

python3 /configs/fourmu_analysis_cfg.py \
  --input dummy.root \
  --output /outputs/fourmu_hist.root \
  --csv /outputs/fourmu_hist.csv \
  --mock

# ============================================================
# Generate mock DiJpsi data
# ============================================================
echo ""
echo "============================================================"
echo "=== Generating Mock DiJpsi Data ==="
echo "============================================================"

python3 /configs/dijpsi_analysis_cfg.py \
  --input dummy.root \
  --output /outputs/dijpsi_hist.root \
  --csv /outputs/dijpsi_hist.csv \
  --mock

# ============================================================
# Run Rank-1 Test
# ============================================================
echo ""
echo "============================================================"
echo "=== Running Rank-1 Bottleneck Test ==="
echo "============================================================"

python3 /configs/cms_rank1_test.py \
  --channel-a /outputs/dijpsi_hist_dijpsi.csv \
  --channel-b /outputs/fourmu_hist_4mu.csv \
  --output /outputs/RANK1_RESULT.md \
  --bootstrap 100

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "============================================================"
echo "=== PIPELINE COMPLETE ==="
echo "============================================================"

echo ""
echo "=== Output files ==="
ls -lh /outputs/

echo ""
echo "=== First 10 lines of FourMu 4mu CSV ==="
head -10 /outputs/fourmu_hist_4mu.csv 2>/dev/null || echo "File not found"

echo ""
echo "=== First 10 lines of DiJpsi dijpsi CSV ==="
head -10 /outputs/dijpsi_hist_dijpsi.csv 2>/dev/null || echo "File not found"

echo ""
echo "=== Rank-1 Result ==="
cat /outputs/RANK1_RESULT.md 2>/dev/null || echo "File not found"

echo ""
echo "Completed: $(date)"
echo "============================================================"
