#!/bin/sh
echo "[PHM] Starting docker container..."

python main.py

if [ -f /work/result.csv ]; then
    echo "[PHM] ✅ result.csv generated successfully"
else
    echo "[PHM] ❌ result.csv not found in /work/"
    exit 1
fi
