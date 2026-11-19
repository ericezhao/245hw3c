# Urban Traffic Scene Understanding

Fine-tune and compare Faster R-CNN, YOLOv8s, DeepLabV3+, and Mask R-CNN on a 3,000 / 500 BDD100K mini split. The written comparison is `report.pdf`.

## Setup

Python 3.11 is required. This machine uses the conda env `245hw3c` (Apple M2 Pro, PyTorch MPS). A peer can recreate it with:

```bash
python3.11 -m venv dl_seg_env
source dl_seg_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon, the default PyTorch wheel already includes MPS. Check it:

```bash
python - << 'EOF'
import torch, torchvision, ultralytics, cv2
print("PyTorch", torch.__version__, "MPS", torch.backends.mps.is_available())
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("OpenCV", cv2.__version__)
EOF
```

### Windows + NVIDIA GPU

Install the CUDA build of PyTorch first, using the command for your CUDA version from https://pytorch.org/get-started/locally/, then the rest:

```powershell
py -3.11 -m venv dl_seg_env
dl_seg_env\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # swap in the line pytorch.org gives you
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and the card name. Then run `python prepare_bdd_mini.py` and `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`.

OpenMMLab (`mmcv`, `mmsegmentation`) does not train on MPS. `configs/deeplabv3plus_bdd.py` is the CUDA config from the handout. Part 4 trains the same DeepLabV3+ / ResNet-50 design with `segmentation-models-pytorch`.

## Data

The handout URL `https://dl.cv.ethz.ch/bdd100k/data/bdd100k_mini_coco.zip` does not resolve. We build two mini splits:

- **Detection / instance (Parts 2, 3, 5):** `prepare_bdd_det_mini.py` — official BDD box2d labels + 100k images → `data/bdd100k_det_mini`
- **Semantic (Part 4):** `prepare_bdd_mini.py` — 10k frames + trainId masks → `data/bdd100k_mini` and `seg_masks/`

```bash
python prepare_bdd_det_mini.py   # official boxes for detectors
python prepare_bdd_mini.py       # semantic masks for DeepLab
```

Image files, masks, checkpoints, and YOLO runs are gitignored.

## Run

From the repo root, with the env activated:

```bash
python part1_eda.py
python part3_yolov8.py        # 50 epochs, batch 8, early stop patience 10
python part2_faster_rcnn.py   # 10 epochs, batch 2
python part4_deeplabv3.py     # 12 epochs, batch 2 (batch 1 breaks ASPP BatchNorm)
python part5_mask_rcnn.py     # 10 epochs, batch 1
python part6_analysis.py
```

Each training script resumes from `checkpoints/` if a checkpoint is already there. `--max-steps N` runs a short smoke test and skips evaluation. `--eval-only` reloads the checkpoint and only evaluates.

## Outputs

| Path | What it is |
| --- | --- |
| `results/eda_samples.png` | Boxes and masks on six training images |
| `results/class_distribution.png` | Instance counts |
| `results/metrics_table.csv` | mAP, mIoU, FPS, parameters |
| `results/fps_vs_map.png` | Speed-accuracy scatter |
| `results/*_predictions.json` | Validation detections used for failures |
| `report.pdf` | The five-section writeup |

GPU FPS in the table is measured on whichever GPU trained the models; each `results/*_metrics.json` records it under `gpu`. YOLOv8's CPU number is the ONNX Runtime benchmark from Part 3. The other CPU numbers are PyTorch.
