#!/usr/bin/env bash
# MedSeg-XAI: Clinical Dashboard Launcher (Linux / Mac / RunPod)
set -e

echo "====================================================================="
echo " Starting MedSeg-XAI Clinical AI & Saliency Audit Dashboard (React) "
echo "====================================================================="
echo ""
echo "Dashboard URL: http://localhost:8000"
echo ""

python3 api/dashboard_server.py
