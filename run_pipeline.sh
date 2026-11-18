#!/bin/zsh
# Train the four models one after another so they do not share the 16 GB GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/245hw3c/bin/python
mkdir -p results
$PY part1_eda.py
$PY part3_yolov8.py
$PY part2_faster_rcnn.py
$PY part4_deeplabv3.py
$PY part5_mask_rcnn.py
$PY part6_analysis.py
echo "pipeline finished"
