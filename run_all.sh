#!/bin/zsh
# Train every model, then build the table and figures.
# Output goes to the terminal and to logs/. A failed part does not stop the rest.
cd "$(dirname "$0")"
PY="${PYTHON:-/opt/anaconda3/envs/245hw3c/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
mkdir -p logs results checkpoints

for part in part3_yolov8 part2_faster_rcnn part4_deeplabv3 part5_mask_rcnn part6_analysis; do
  echo "=== START $part $(date) ==="
  "$PY" -u "$part.py" 2>&1 | tee "logs/$part.log"
  echo "=== END $part exit ${pipestatus[1]} $(date) ==="
done
echo "=== ALL DONE $(date) ==="
