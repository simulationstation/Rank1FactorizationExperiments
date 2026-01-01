#!/bin/bash
# Status script for calibration runs

echo "=============================================="
echo "CALIBRATION STATUS"
echo "=============================================="
date
echo

# Check for running processes
echo "=== RUNNING PROCESSES ==="
ps aux | grep -E "run_medium|run_power|run_m4" | grep -v grep | head -5
echo

# Medium calibration
if [ -f out/medium_calibration.log ]; then
    echo "=== MEDIUM CALIBRATION ==="
    # Count trials
    dicharm_trials=$(grep -c "Trial.*Lambda" out/medium_calibration.log 2>/dev/null | head -1)
    echo "Di-charmonium trials: ~$dicharm_trials"

    # Check for completion
    if grep -q "MEDIUM CALIBRATION PASS" out/medium_calibration.log 2>/dev/null; then
        echo "Status: PASS"
    elif grep -q "FAIL" out/medium_calibration.log 2>/dev/null; then
        echo "Status: FAIL"
    else
        echo "Status: RUNNING"
    fi

    echo
    echo "Last 30 lines:"
    tail -30 out/medium_calibration.log
    echo
fi

# Power analysis
if [ -f out/power_analysis.log ]; then
    echo "=== POWER ANALYSIS ==="
    echo "Last 20 lines:"
    tail -20 out/power_analysis.log
    echo
fi

# M4 grid
if [ -f out/m4_grid.log ]; then
    echo "=== M4 GRID ==="
    echo "Last 20 lines:"
    tail -20 out/m4_grid.log
    echo
fi

# Check for output files
echo "=== OUTPUT FILES ==="
ls -la out/*.md out/*.csv 2>/dev/null | head -20

# Check for MEDIUM_PASS marker
if [ -f out/MEDIUM_PASS ]; then
    echo
    echo "*** MEDIUM CALIBRATION COMPLETE - READY FOR POWER/M4 ***"
fi
