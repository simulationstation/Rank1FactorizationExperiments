#!/bin/bash
# Memory watchdog for discovery factory
# Kills process if it exceeds memory threshold

TARGET_PID=$1
MAX_MEM_MB=${2:-6000}  # Default 6GB limit

if [ -z "$TARGET_PID" ]; then
    echo "Usage: $0 <PID> [MAX_MEM_MB]"
    exit 1
fi

echo "Monitoring PID $TARGET_PID with ${MAX_MEM_MB}MB limit"
echo "Log: discoveries/memory_watchdog.log"

while true; do
    if ! ps -p $TARGET_PID > /dev/null 2>&1; then
        echo "[$(date)] Process $TARGET_PID no longer exists. Exiting watchdog."
        exit 0
    fi

    # Get RSS in KB, convert to MB
    MEM_KB=$(ps -o rss= -p $TARGET_PID 2>/dev/null)
    if [ -n "$MEM_KB" ]; then
        MEM_MB=$((MEM_KB / 1024))
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] PID $TARGET_PID: ${MEM_MB}MB / ${MAX_MEM_MB}MB limit" >> discoveries/memory_watchdog.log

        if [ $MEM_MB -gt $MAX_MEM_MB ]; then
            echo "[$TIMESTAMP] ALERT: Memory ${MEM_MB}MB exceeds ${MAX_MEM_MB}MB limit!" | tee -a discoveries/memory_watchdog.log
            echo "[$TIMESTAMP] Killing process $TARGET_PID" | tee -a discoveries/memory_watchdog.log
            kill $TARGET_PID
            sleep 2
            if ps -p $TARGET_PID > /dev/null 2>&1; then
                echo "[$TIMESTAMP] Force killing $TARGET_PID" | tee -a discoveries/memory_watchdog.log
                kill -9 $TARGET_PID
            fi
            exit 1
        fi
    fi

    sleep 30
done
