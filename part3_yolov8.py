"""Part 3: convert BDD mini to YOLO labels, train YOLOv8s, export ONNX."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from bdd_common import CLASS_NAMES, DATA_ROOT, RESULTS, ROOT, device_label, evaluate_coco, get_device, write_metrics

YOLO_ROOT = ROOT / "yolo_bdd"
YAML_PATH = ROOT / "configs" / "bdd100k_mini.yaml"


def coco_to_yolo(split: str) -> None:
    ann_path = DATA_ROOT / "annotations" / f"instances_{split}.json"
    with ann_path.open() as handle:
        coco = json.load(handle)
    by_image = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    image_dir = YOLO_ROOT / split / "images"
    label_dir = YOLO_ROOT / split / "labels"
    # Drop leftovers from older splits so Ultralytics does not train/eval extras.
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    src_dir = DATA_ROOT / "images" / split

    for info in coco["images"]:
        src = src_dir / info["file_name"]
        dst = image_dir / info["file_name"]
        if not dst.exists():
            # Windows refuses symlinks without Developer Mode; hard links work on NTFS.
            try:
                os.symlink(src, dst)
            except OSError:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        width, height = info["width"], info["height"]
        label_path = label_dir / f"{Path(info['file_name']).stem}.txt"
        lines = []
        for ann in by_image[info["id"]]:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            cx = min(max((x + w / 2) / width, 0.0), 1.0)
            cy = min(max((y + h / 2) / height, 0.0), 1.0)
            nw = min(max(w / width, 0.0), 1.0)
            nh = min(max(h / height, 0.0), 1.0)
            cls = int(ann["category_id"]) - 1
            if nw <= 0 or nh <= 0 or not (0 <= cls < len(CLASS_NAMES)):
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"{split}: {len(coco['images'])} images -> {image_dir}")


def write_yaml() -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    YAML_PATH.write_text(
        "\n".join(
            [
                f"path: {YOLO_ROOT}",
                "train: train/images",
                "val: val/images",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                names,
                "",
            ]
        )
    )
    print("wrote", YAML_PATH)


def latest_map50(run_dir: Path) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0.0
    import pandas as pd

    frame = pd.read_csv(csv_path)
    frame.columns = [col.strip() for col in frame.columns]
    column = "metrics/mAP50(B)"
    if column not in frame.columns:
        matches = [col for col in frame.columns if "mAP50" in col]
        column = matches[0]
    return float(frame[column].max())


def benchmark_onnx(onnx_path: Path, imgsz: int) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy})
    start = time.perf_counter()
    steps = 30
    for _ in range(steps):
        session.run(None, {input_name: dummy})
    return steps / (time.perf_counter() - start)


def benchmark_mps(weights: Path, device, device_arg, imgsz: int) -> float:
    model = YOLO(str(weights))
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, imgsz=imgsz, device=device_arg, verbose=False)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 20
    for _ in range(steps):
        model.predict(dummy, imgsz=imgsz, device=device_arg, verbose=False)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    model_tag = Path(args.model).stem  # yolov8s
    run_dir = ROOT / "yolo_runs" / f"bdd_{model_tag}"

    RESULTS.mkdir(parents=True, exist_ok=True)
    coco_to_yolo("train")
    coco_to_yolo("val")
    write_yaml()

    device = get_device()
    device_arg = "mps" if device.type == "mps" else (0 if device.type == "cuda" else "cpu")
    weights = run_dir / "weights" / "best.pt"

    last_pt = run_dir / "weights" / "last.pt"
    done_epochs = 0
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        import pandas as pd

        logged = pd.read_csv(results_csv)
        done_epochs = len(logged)
    # Ultralytics sets epoch to -1 once a run ends, including early stopping.
    finished = last_pt.exists() and torch.load(last_pt, map_location="cpu", weights_only=False).get("epoch", -1) == -1

    if not args.eval_only and done_epochs < args.epochs and not finished:
        if last_pt.exists() and done_epochs > 0:
            print(f"resuming YOLOv8 from epoch {done_epochs}")
            model = YOLO(str(last_pt))
            model.train(resume=True)
        else:
            model = YOLO(args.model)
            model.train(
                data=str(YAML_PATH),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device_arg,
                project=str(ROOT / "yolo_runs"),
                name=f"bdd_{model_tag}",
                patience=args.patience,
                cos_lr=True,
                close_mosaic=10,
                copy_paste=0.1,
                amp=device.type == "cuda",
                workers=8 if device.type == "cuda" else 2,
                seed=245,
                exist_ok=True,
                plots=True,
            )

    if not weights.exists():
        raise FileNotFoundError(weights)

    trained = YOLO(str(weights))
    onnx_path = Path(trained.export(format="onnx", imgsz=args.imgsz))
    ultra_map50 = latest_map50(run_dir)
    coco_metrics = dump_coco_predictions(trained, device_arg, args.imgsz)
    map50 = coco_metrics["map50"] or ultra_map50
    n_params = sum(p.numel() for p in trained.model.parameters()) / 1e6
    gpu_fps = benchmark_mps(weights, device, device_arg, args.imgsz)
    cpu_fps = benchmark_onnx(onnx_path, args.imgsz)
    copy_plots(run_dir)
    write_metrics(
        "yolov8",
        {
            "model": model_tag.upper().replace("YOLOV", "YOLOv"),
            "mAP@0.5": map50,
            "mAP": coco_metrics["map"],
            "mAP_small": coco_metrics["map_small"],
            "mAP_medium": coco_metrics["map_medium"],
            "mAP_large": coco_metrics["map_large"],
            "per_class_ap50": coco_metrics["per_class_ap50"],
            "ultralytics_map50": ultra_map50,
            "mIoU": None,
            "FPS (GPU)": gpu_fps,
            "FPS (CPU)": cpu_fps,
            "onnx_fps": cpu_fps,
            "Params (M)": round(n_params, 2),
            "device": str(device),
            "gpu": device_label(device),
            "onnx": str(onnx_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
        },
    )
    print(f"{model_tag} mAP@0.5 {map50:.4f} | GPU {gpu_fps:.1f} FPS | ONNX CPU {cpu_fps:.1f} FPS")


def copy_plots(run_dir: Path) -> None:
    import shutil

    if not run_dir.exists():
        return
    for path in run_dir.glob("*.png"):
        target = RESULTS / f"yolo_{path.name}"
        shutil.copy2(path, target)


def dump_coco_predictions(model: YOLO, device_arg, imgsz: int) -> dict:
    """Re-score the val split with pycocotools so small-object AP matches Part 2."""
    ann_path = DATA_ROOT / "annotations" / "instances_val.json"
    with ann_path.open() as handle:
        coco = json.load(handle)
    stem_to_id = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}
    paths = [str(DATA_ROOT / "images" / "val" / img["file_name"]) for img in coco["images"]]
    results = []
    for pred in model.predict(
        paths, stream=True, conf=0.001, iou=0.7, imgsz=imgsz,
        device=device_arg, verbose=False, max_det=300,
    ):
        image_id = stem_to_id[Path(pred.path).stem]
        if pred.boxes is None or len(pred.boxes) == 0:
            continue
        boxes = pred.boxes.xyxy.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()
        classes = pred.boxes.cls.cpu().numpy()
        order = np.argsort(-scores)[:100]
        for index in order:
            x1, y1, x2, y2 = [float(v) for v in boxes[index]]
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(classes[index]) + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(scores[index]),
                }
            )
    (RESULTS / "yolov8_predictions.json").write_text(json.dumps(results))
    return evaluate_coco(results, "bbox")


if __name__ == "__main__":
    main()
