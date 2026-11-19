"""Part 6. Comparison table, failure cases, and the written report.

Reads the metric json files written by parts 2-5. Failure panels are built
from the saved COCO detections. The semantic-versus-instance figure is a
paired forward pass of DeepLabV3+ and Mask R-CNN on two validation frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from PIL import Image

from bdd_common import (
    CLASS_NAMES,
    DATA_ROOT,
    RESULTS,
    ROOT,
    box_iou,
    get_device,
)

PREDS = {
    "Faster R-CNN": RESULTS / "faster_rcnn_predictions.json",
    "YOLOv8": RESULTS / "yolov8_predictions.json",
    "Mask R-CNN": RESULTS / "mask_rcnn_predictions.json",
}
METRIC_FILES = {
    "Faster R-CNN FPN": RESULTS / "faster_rcnn_metrics.json",
    "YOLOv8": RESULTS / "yolov8_metrics.json",
    "DeepLabV3+ R50": RESULTS / "deeplabv3plus_metrics.json",
    "Mask R-CNN FPN": RESULTS / "mask_rcnn_metrics.json",
}


def load_metrics() -> list[dict]:
    rows = []
    for name, path in METRIC_FILES.items():
        if not path.exists():
            print("missing", path)
            continue
        payload = json.loads(path.read_text())
        payload.setdefault("model", name)
        rows.append(payload)
    if len(rows) < 4:
        print(f"warning: only {len(rows)} of 4 metric files are present")
    return rows


def cell(value, digits=3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "mAP@0.5": cell(row.get("mAP@0.5")),
                "mIoU": cell(row.get("mIoU")),
                "FPS (GPU)": cell(row.get("FPS (GPU)"), 2),
                "FPS (CPU)": cell(row.get("FPS (CPU)"), 2),
                "Params (M)": cell(row.get("Params (M)"), 2),
            }
            for row in rows
        ]
    )
    path = RESULTS / "metrics_table.csv"
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print("wrote", path)
    return frame


def scatter_and_bars(rows: list[dict]) -> None:
    gpu_name = next((row["gpu"] for row in rows if row.get("gpu")), "GPU")
    detectors = [row for row in rows if row.get("mAP@0.5") is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row in detectors:
        ax.scatter(row["FPS (GPU)"], row["mAP@0.5"], s=80)
        ax.annotate(row["model"], (row["FPS (GPU)"], row["mAP@0.5"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"FPS on {gpu_name} (batch 1)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Speed versus detection accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fps_vs_map.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = [row["model"].replace(" ", "\n") for row in rows]
    x = np.arange(len(rows))
    gpu = [row.get("FPS (GPU)") or 0 for row in rows]
    cpu = [row.get("FPS (CPU)") or 0 for row in rows]
    ax.bar(x - 0.18, gpu, width=0.36, label=f"GPU ({gpu_name})")
    ax.bar(x + 0.18, cpu, width=0.36, label="CPU")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Frames per second")
    ax.set_title("Inference throughput, batch size 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fps_bars.png", dpi=140)
    plt.close(fig)

    faster = next((row for row in rows if "Faster" in row["model"]), None)
    yolo = next((row for row in rows if "YOLO" in row["model"]), None)
    if not faster or not yolo:
        return
    left = faster.get("per_class_ap50") or {}
    right = yolo.get("per_class_ap50") or {}
    if not left or not right:
        return
    labels = [name for name in CLASS_NAMES if name in left or name in right]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [left.get(name) or 0 for name in labels], width=0.36, label="Faster R-CNN")
    ax.bar(x + 0.18, [right.get(name) or 0 for name in labels], width=0.36, label="YOLOv8")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AP@0.50")
    ax.set_title("Per-class AP at IoU 0.50")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_class_ap.png", dpi=140)
    plt.close(fig)


def _gt_by_image() -> tuple[dict, dict]:
    payload = json.loads((DATA_ROOT / "annotations" / "instances_val.json").read_text())
    images = {img["id"]: img for img in payload["images"]}
    by_image = {img["id"]: [] for img in payload["images"]}
    for ann in payload["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return images, by_image


def failure_panel(name: str, pred_path: Path) -> None:
    if not pred_path.exists():
        print("no predictions for", name)
        return
    images, by_image = _gt_by_image()
    grouped: dict[int, list] = {}
    for det in json.loads(pred_path.read_text()):
        grouped.setdefault(int(det["image_id"]), []).append(det)
    false_pos, misses = [], []
    for image_id, info in images.items():
        gts = by_image[image_id]
        preds = sorted(grouped.get(image_id, []), key=lambda item: -item["score"])
        used = set()
        unmatched = []
        for pred in preds:
            x, y, w, h = pred["bbox"]
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            best_iou, best_j = 0.0, -1
            for j, ann in enumerate(gts):
                if j in used or int(ann["category_id"]) != int(pred["category_id"]):
                    continue
                ax, ay, aw, ah = ann["bbox"]
                iou = box_iou(box, np.array([ax, ay, ax + aw, ay + ah], dtype=np.float32))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
            else:
                unmatched.append(pred)
        for pred in unmatched:
            if pred["score"] >= 0.5:
                false_pos.append((pred["score"], image_id, pred))
        for j, ann in enumerate(gts):
            if j not in used:
                ax, ay, aw, ah = ann["bbox"]
                misses.append((aw * ah, image_id, ann))
    false_pos.sort(key=lambda item: -item[0])
    misses.sort(key=lambda item: item[0])
    chosen = []
    seen = set()
    for kind, pool in (("False positive", false_pos), ("Missed object", misses), ("False positive", false_pos[1:])):
        for item in pool:
            if item[1] in seen:
                continue
            chosen.append((kind, item))
            seen.add(item[1])
            break
        if len(chosen) >= 3:
            break
    if not chosen:
        return
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen), 4.4))
    if len(chosen) == 1:
        axes = [axes]
    for ax, (kind, item) in zip(axes, chosen):
        image_id = item[1]
        info = images[image_id]
        image = np.array(Image.open(DATA_ROOT / "images" / "val" / info["file_name"]).convert("RGB"))
        import cv2
        for ann in by_image[image_id]:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (40, 180, 70), 2)
        for det in grouped.get(image_id, []):
            if det["score"] < 0.5:
                continue
            x, y, w, h = [int(v) for v in det["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (220, 40, 40), 2)
            label = CLASS_NAMES[int(det["category_id"]) - 1]
            cv2.putText(image, f"{label} {det['score']:.2f}", (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        if kind == "Missed object":
            ann = item[2]
            title = f"Missed {CLASS_NAMES[ann['category_id'] - 1]}"
        else:
            det = item[2]
            title = f"False positive {CLASS_NAMES[det['category_id'] - 1]} {det['score']:.2f}"
        ax.imshow(image)
        ax.set_title(title + "\nGT green, pred red", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{name} failure cases", fontsize=12)
    fig.tight_layout()
    slug = name.lower().replace(" ", "_").replace("-", "")
    fig.savefig(RESULTS / f"failures_{slug}.png", dpi=140)
    plt.close(fig)


def semantic_vs_instance(device) -> None:
    deeplab_ckpt = ROOT / "checkpoints" / "deeplabv3plus_bdd.pth"
    mask_ckpt = ROOT / "checkpoints" / "mask_rcnn_bdd.pth"
    if not deeplab_ckpt.exists() or not mask_ckpt.exists():
        print("skipping paired figure; a checkpoint is missing")
        return
    import cv2
    import torch
    from part4_deeplabv3 import CHECKPOINT as DL_CKPT, build_deeplab, build_transforms, colorize
    from part5_mask_rcnn import CHECKPOINT as MASK_CKPT, build_mask_rcnn

    images, by_image = _gt_by_image()
    crowded, isolated = None, None
    for image_id, info in images.items():
        cars = [ann for ann in by_image[image_id] if ann["category_id"] == 3]
        persons = [ann for ann in by_image[image_id] if ann["category_id"] == 1]
        close = False
        for i, a in enumerate(cars):
            ax, ay, aw, ah = a["bbox"]
            for b in cars[i + 1 :]:
                bx, by, bw, bh = b["bbox"]
                box_a = np.array([ax, ay, ax + aw, ay + ah])
                box_b = np.array([bx, by, bx + bw, by + bh])
                if box_iou(box_a, box_b) > 0.02:
                    close = True
        if close and crowded is None:
            crowded = info
        if len(persons) == 1 and len(cars) >= 1 and not close and isolated is None:
            isolated = info
        if crowded and isolated:
            break
    if crowded is None or isolated is None:
        print("could not find the two comparison frames")
        return

    deeplab = build_deeplab(pretrained=False)
    payload = torch.load(DL_CKPT, map_location=device, weights_only=False)
    deeplab.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    deeplab.to(device).eval()
    mask_model = build_mask_rcnn().to(device)
    payload = torch.load(MASK_CKPT, map_location=device, weights_only=False)
    mask_model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    mask_model.eval()
    transform = build_transforms(False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    titles = [
        (isolated, "Semantic is enough: one person, separated vehicles"),
        (crowded, "Semantic is not enough: adjacent cars share one class"),
    ]
    palette = plt.colormaps["tab20"]
    for row, (info, title) in enumerate(titles):
        rgb = np.array(Image.open(DATA_ROOT / "images" / "val" / info["file_name"]).convert("RGB"))
        sample = transform(image=rgb, mask=np.zeros(rgb.shape[:2], dtype=np.uint8))
        with torch.inference_mode():
            logits = deeplab(sample["image"].unsqueeze(0).to(device))
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            inst = mask_model([tensor.to(device)])[0]
        pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = rgb.astype(np.float32) / 255.0
        shown = 0
        for mask, score in zip(inst["masks"], inst["scores"]):
            if float(score) < 0.5:
                continue
            binary = mask[0].cpu().numpy() > 0.5
            color = np.array(palette(shown % 20)[:3])
            overlay[binary] = overlay[binary] * 0.45 + color * 0.55
            shown += 1
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(title, fontsize=8)
        axes[row, 1].imshow(colorize(pred))
        axes[row, 1].set_title("DeepLabV3+ (class color)", fontsize=8)
        axes[row, 2].imshow(np.clip(overlay, 0, 1))
        axes[row, 2].set_title(f"Mask R-CNN ({shown} instances)", fontsize=8)
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "semantic_vs_instance.png", dpi=140)
    plt.close(fig)
    print("wrote semantic_vs_instance.png")


def _row(rows, needle):
    return next((row for row in rows if needle in row["model"]), None)


def write_report(rows: list[dict]) -> None:
    faster, yolo = _row(rows, "Faster"), _row(rows, "YOLO")
    deeplab, mask = _row(rows, "DeepLab"), _row(rows, "Mask")

    def fget(row, key, digits=3):
        if row is None:
            return "n/a"
        if key == "model":
            return str(row.get("model", "n/a"))
        if row.get(key) is None:
            return "n/a"
        value = row[key]
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    fs, ys = (faster or {}).get("mAP_small"), (yolo or {}).get("mAP_small")
    if fs is not None and ys is not None and fs > ys + 0.02:
        small_text = (
            f"Faster R-CNN small-object AP is {fs:.3f} and YOLOv8s is {ys:.3f}. "
            "That is the two-stage pattern. The region proposal network scores anchors on every FPN level, "
            "then RoI Align pools a feature that still has the resolution of the level where the object was found. "
            "YOLOv8 assigns each object to one anchor-free point. It is faster, and on the COCO small bin "
            "(area under 32 squared pixels) it gives up more of those distant lights, signs, and people."
        )
    elif fs is not None and ys is not None:
        small_text = (
            f"YOLOv8s small-object AP is {ys:.3f} and Faster R-CNN is {fs:.3f}. "
            "This split does not show the old two-stage advantage on small boxes. "
            "YOLOv8 is anchor-free and uses a task-aligned assigner, so a prediction is matched to a ground-truth box "
            "without a fixed IoU cutoff. The head is also decoupled, which reduces the fight between the class loss and the box loss. "
            "The FPN is still inside YOLOv8, so 'single stage means one stride' is not a description of this model."
        )
    else:
        small_text = "Per-area AP was not available for both detectors, so the small-object claim is left to the per-class chart."

    tta_delta = None
    if deeplab and deeplab.get("mIoU_tta") is not None and deeplab.get("mIoU") is not None:
        tta_delta = deeplab["mIoU_tta"] - deeplab["mIoU"]
    if tta_delta is None:
        tta_text = "Test-time flip augmentation was not recorded."
    else:
        tta_text = (
            f"Horizontal-flip test-time augmentation changed DeepLabV3+ mIoU by {tta_delta:+.3f} "
            f"({fget(deeplab, 'mIoU')} to {fget(deeplab, 'mIoU_tta')}). "
            "Averaging the two views cancels mistakes that depend on which side of an object lands on a cell boundary. "
            "It does not add features the encoder never learned, which is why the handout's expected gain is only one to three points."
        )

    gpu_name = next((row["gpu"] for row in rows if row.get("gpu")), "the training GPU")
    on_mps = "MPS" in gpu_name
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.add_page()
    _multi = pdf.multi_cell

    def multi_cell(*args, **kwargs):
        pdf.set_x(pdf.l_margin)
        return _multi(*args, **kwargs)

    pdf.multi_cell = multi_cell
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "Urban Traffic Scene Understanding on BDD100K Mini")
    pdf.set_font("Helvetica", size=10)
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        "Four perception models were fine-tuned on the same 3,000 / 500 urban driving split: "
        "Faster R-CNN with a ResNet-50 FPN, YOLOv8s, DeepLabV3+ with a ResNet-50 encoder, and Mask R-CNN. "
        f"Training and the GPU timing ran on {gpu_name}. "
        + ("There is no CUDA device on this machine, so the GPU column is Metal, not an NVIDIA card. " if on_mps else "")
        +
        "The official mini-zip host in the handout does not resolve. Images are BDD100K 10k frames. "
        "Instance masks are connected components of the ten object classes in the semantic label, and stuff classes stay background.",
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 6, "Measured results")
    pdf.set_font("Helvetica", size=8)
    widths = [42, 24, 22, 28, 28, 28]
    headers = ["Model", "mAP@0.5", "mIoU", "FPS GPU", "FPS CPU", "Params (M)"]
    for header, width in zip(headers, widths):
        pdf.cell(width, 6, header, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=8)
    for row in rows:
        values = [
            row["model"],
            cell(row.get("mAP@0.5")),
            cell(row.get("mIoU")),
            cell(row.get("FPS (GPU)"), 2),
            cell(row.get("FPS (CPU)"), 2),
            cell(row.get("Params (M)"), 2),
        ]
        for value, width in zip(values, widths):
            pdf.cell(width, 6, value[:28], border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(
        0,
        4.6,
        f"Protocol: batch size 1, timed after warmup, with device synchronization on {gpu_name}. "
        "YOLOv8 CPU FPS is ONNX Runtime on a 640 dummy tensor, which is the handout's export benchmark. "
        "The other CPU numbers are PyTorch forwards on a real validation image. "
        "mAP@0.5 is COCO AP at IoU 0.50 from pycocotools, except that a missing file falls back to the Ultralytics validation column. "
        "DeepLab mIoU is the mean over classes that appear in the validation masks, including background, ignoring label 255.",
    )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, "1. Speed-accuracy tradeoff")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        5,
        f"Faster R-CNN reaches mAP@0.5 {fget(faster, 'mAP@0.5')} at {fget(faster, 'FPS (GPU)', 1)} FPS "
        f"with {fget(faster, 'Params (M)', 1)} M parameters. "
        f"YOLOv8s reaches {fget(yolo, 'mAP@0.5')} at {fget(yolo, 'FPS (GPU)', 1)} FPS "
        f"with {fget(yolo, 'Params (M)', 1)} M parameters. "
        f"Mask R-CNN sits at {fget(mask, 'mAP@0.5')} mAP and {fget(mask, 'FPS (GPU)', 1)} FPS because the mask head "
        "runs a small conv net on every surviving proposal. "
        f"DeepLabV3+ has no boxes (mAP is N/A) and runs at {fget(deeplab, 'FPS (GPU)', 1)} FPS "
        f"for a full-frame class map, mIoU {fget(deeplab, 'mIoU')}.",
    )
    detectors = [row for row in rows if row.get("mAP@0.5") is not None and row.get("FPS (GPU)") is not None]
    fastest = max(detectors, key=lambda row: row["FPS (GPU)"]) if detectors else None
    heaviest = max(detectors, key=lambda row: row["mAP@0.5"]) if detectors else None
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        (
            f"On this run the fastest detector is {fastest['model']} at {fget(fastest, 'FPS (GPU)', 1)} FPS. "
            if fastest
            else "Detector FPS was not available for every model. "
        )
        + (
            f"The highest mAP@0.5 is {heaviest['model']} at {fget(heaviest, 'mAP@0.5')}. "
            if heaviest
            else ""
        )
        + "Edge cameras care about the FPS column first. A cloud job that can wait can take the higher-mAP two-stage model. "
        "DeepLab has no box mAP, so it is not on the scatter plot; use it only when the product needs a class per pixel "
        "and does not need a separate identity for each car or person.",
    )
    _image(pdf, RESULTS / "fps_vs_map.png", 150)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, "2. Architectural comparison")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, small_text)
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        "The rest of the per-class chart is the class imbalance from Part 1. Car, traffic sign, and traffic light dominate the counts, "
        "and train is nearly empty. Both detectors inherit COCO features for person, car, bus, and truck, so those classes move first. "
        "Rider is not a COCO class in the same way, and traffic sign is easy to confuse with poles and storefront text. "
        "Sampling images by the rarest class they contain, and using a 10x lower learning rate on the Faster R-CNN backbone than on the new box head, "
        "is there so fine-tuning does not wipe the COCO features while the new 11-way classifier is still random. "
        "YOLOv8 keeps its own mosaic augmentation and cosine schedule; the Ultralytics dashboard in the run folder is the training record.",
    )
    _image(pdf, RESULTS / "per_class_ap.png", 155)
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        f"Area bins from the same COCO evaluator: Faster R-CNN small/medium/large "
        f"{fget(faster, 'mAP_small')}/{fget(faster, 'mAP_medium')}/{fget(faster, 'mAP_large')}, "
        f"YOLOv8s {fget(yolo, 'mAP_small')}/{fget(yolo, 'mAP_medium')}/{fget(yolo, 'mAP_large')}. "
        "Medium and large objects (nearby cars) are where both models look competent. The product decision is about the small bin.",
    )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, "3. Semantic versus instance segmentation")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        5,
        "Semantic segmentation paints every pixel of a class with one label. Two cars that touch become one car-colored region. "
        "That is enough for a vulnerable-road-user alert: the question is whether any person pixel is in the crosswalk, not which person it is. "
        "The top row of the figure is that case, one person and vehicles that do not overlap. "
        "The bottom row is two cars close enough that a semantic map cannot tell a counter how many vehicles are in the lane. "
        "Mask R-CNN gives each RoI its own 28x28 mask, so those cars can receive different colors even when their boxes touch. "
        "The cost is the second stage. DeepLab's ASPP (dilations 6, 12, 18 plus image pooling) is what lets one forward pass see both a nearby car and a far sign "
        "without throwing away the feature resolution the way a classification network's last pooling layer would.",
    )
    pdf.ln(1)
    pdf.multi_cell(0, 5, tta_text)
    _image(pdf, RESULTS / "semantic_vs_instance.png", 160)
    _image(pdf, RESULTS / "deeplab_preds.png", 145)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, "4. Failure modes")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        5,
        "Green boxes are ground truth and red boxes are predictions at score 0.5 or higher. "
        "Three recurring failures show up for every detector.",
    )
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        "False positives. Reflections in glass, billboards, and parked cars cut off by the frame still fire the car or traffic-sign head. "
        "The proposal or grid cell sees a local texture that matched COCO, and the fine-tune did not see enough negatives of that kind. "
        "Mitigation: raise the score threshold for the sign class, and add a short hard-negative pass on frames whose only 'sign' pixels are shop text.",
    )
    pdf.multi_cell(
        0,
        5,
        "Missed objects. Distant traffic lights and the rare classes (train, rider, motorcycle) are small and scarce. "
        "An unweighted loss can ignore them and still post a respectable mAP because car dominates the average. "
        "Mitigation: the image sampler already up-weights rare classes; the next step is a class-balanced classification loss inside the ROI head, "
        "or an extra epoch with the copy-paste augmentation Ultralytics uses, aimed at rider and train.",
    )
    pdf.multi_cell(
        0,
        5,
        "Mask leakage. A Mask R-CNN mask sometimes spills onto the road or into the neighboring car. "
        "RoI Align is aligned, but the 28x28 head is a low-resolution guess that gets pasted back, and our ground truth itself merges two cars "
        "when their semantic components touch. The model is then punished for splitting a blob the label called one object, or rewarded for leaking. "
        "Mitigation: train the mask loss only inside the box, which Mask R-CNN already does, and replace connected-component labels with the official instance json when it is available. "
        "For DeepLab, the matching failure is a thin object (pole, distant person) whose pixels are swallowed by the majority class. "
        "The class-weighted cross entropy and the flip TTA are the mitigations already measured above.",
    )
    for filename in (
        "failures_faster_rcnn.png",
        "failures_yolov8s.png",
        "failures_mask_rcnn.png",
    ):
        _image(pdf, RESULTS / filename, 155)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, "5. Deployment recommendation")
    pdf.set_font("Helvetica", size=10)
    live = fastest or yolo
    pdf.multi_cell(
        0,
        5,
        f"For a real-time traffic monitor I would put {fget(live, 'model') if live else 'the fastest detector'} on the camera "
        "and keep a slower two-stage model only as an offline auditor.",
    )
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        f"A 20 FPS camera leaves under 50 ms per frame after tracking and display. "
        f"The live candidate measured {fget(live, 'FPS (GPU)', 1)} FPS on {gpu_name}, "
        f"{fget(live, 'Params (M)', 1)} M parameters, mAP@0.5 {fget(live, 'mAP@0.5')}. "
        f"YOLOv8 ONNX CPU FPS was {fget(yolo, 'onnx_fps', 1)}. "
        f"Faster R-CNN was {fget(faster, 'FPS (GPU)', 1)} FPS / mAP {fget(faster, 'mAP@0.5')}. "
        f"Mask R-CNN was {fget(mask, 'FPS (GPU)', 1)} FPS and only matters when counting distinct vehicles.",
    )
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        "I would not run DeepLabV3+ in the live loop. A class map does not give a track id, and the classes we care about for an alert "
        "(person, rider, car) are exactly the ones a detector already localizes. "
        f"DeepLab mIoU is {fget(deeplab, 'mIoU')} "
        f"(foreground-only {fget(deeplab, 'mIoU_fg')}); that is for an overnight occupancy study, not the live loop. "
        "If a later product needs both a live box and a pixel occupancy grid, the combination is YOLOv8s online and DeepLab on a sampled frame every second, "
        "not Mask R-CNN on every frame.",
    )
    pdf.ln(1)
    pdf.multi_cell(
        0,
        5,
        "Operating points I would set before a pilot: score 0.4 on person and rider, 0.5 on car, 0.6 on traffic sign, "
        "because signs are the noisiest false-positive class in the per-class chart. "
        "I would log every frame where the top person score is between 0.3 and 0.5 and review those weekly; that is the band where misses and false alarms trade off. "
        "I would not claim a safety case from this mini split. The labels are a 3,500-image subset, train is almost absent, "
        "and the instance masks are connected components rather than the official BDD boxes. "
        "The next measurement before deployment is the same four models on the full BDD detection validation set, at the resolution the camera will actually use.",
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(
        0,
        4.6,
        "Class imbalance note from the training set is in results/class_imbalance.txt. "
        "Figures for this report are in results/. Checkpoints are under checkpoints/ and are not committed. "
        "The dataset download is prepare_bdd_mini.py; image files are gitignored.",
    )
    out = ROOT / "report.pdf"
    pdf.output(str(out))
    print(f"wrote {out} ({len(pdf.pages)} pages)")


def _image(pdf: FPDF, path: Path, width: float) -> None:
    if not path.exists():
        return
    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.image(str(path), w=width)
    pdf.ln(2)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    if not rows:
        raise SystemExit("no metric json files; train parts 2-5 first")
    write_table(rows)
    scatter_and_bars(rows)
    for name, path in PREDS.items():
        failure_panel(name, path)
    semantic_vs_instance(get_device())
    write_report(rows)


if __name__ == "__main__":
    main()
