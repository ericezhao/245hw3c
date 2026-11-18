"""Shared dataset, device, and metric helpers for the BDD100K mini homework."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "bdd100k_mini"
RESULTS = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
NUM_CLASSES = 1 + len(CLASS_NAMES)  # background + 10 BDD classes


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_split(split: str) -> tuple[list[dict], dict[int, list[dict]]]:
    ann_path = DATA_ROOT / "annotations" / f"instances_{split}.json"
    with ann_path.open() as handle:
        coco = json.load(handle)
    by_image = {image["id"]: [] for image in coco["images"]}
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return coco["images"], by_image


def decode_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        if isinstance(rles, list):
            rle = mask_utils.merge(rles)
        else:
            rle = rles
    else:
        rle = dict(segmentation)
        counts = rle["counts"]
        if isinstance(counts, str):
            rle["counts"] = counts.encode("ascii")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def collate(batch):
    return tuple(zip(*batch))


def _boxes_and_labels(anns, width, height, with_masks=False):
    boxes, labels, masks = [], [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            continue
        x1 = float(np.clip(x, 0, width - 1))
        y1 = float(np.clip(y, 0, height - 1))
        x2 = float(np.clip(x + w, 0, width))
        y2 = float(np.clip(y + h, 0, height))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        label = int(ann["category_id"])
        if label < 1 or label >= NUM_CLASSES:
            continue
        if with_masks:
            mask = decode_mask(ann["segmentation"], height, width)
            if int(mask.sum()) < 1:
                continue
            masks.append(mask)
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
    return boxes, labels, masks


class BDDDetectionDataset(Dataset):
    def __init__(self, split: str, require_objects: bool = False, augment: bool = False):
        self.root = DATA_ROOT / "images" / split
        self.augment = augment
        images, by_image = load_split(split)
        if require_objects:
            images = [img for img in images if by_image[img["id"]]]
        self.images = images
        self.by_image = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        tensor = TF.to_tensor(image)
        width, height = info["width"], info["height"]
        boxes, labels, _ = _boxes_and_labels(
            self.by_image[info["id"]], width, height, with_masks=False
        )
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
        if self.augment and box_tensor.numel() and torch.rand(1).item() < 0.5:
            tensor = tensor.flip(-1)
            width = tensor.shape[-1]
            flipped = box_tensor.clone()
            flipped[:, 0] = width - box_tensor[:, 2]
            flipped[:, 2] = width - box_tensor[:, 0]
            box_tensor = flipped
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([info["id"]]),
        }
        return tensor, target


class BDDInstanceDataset(Dataset):
    def __init__(self, split: str, require_objects: bool = False, augment: bool = False):
        self.root = DATA_ROOT / "images" / split
        self.augment = augment
        images, by_image = load_split(split)
        if require_objects:
            images = [img for img in images if by_image[img["id"]]]
        self.images = images
        self.by_image = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        tensor = TF.to_tensor(image)
        width, height = info["width"], info["height"]
        boxes, labels, masks = _boxes_and_labels(
            self.by_image[info["id"]], width, height, with_masks=True
        )
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
            mask_tensor = torch.tensor(np.stack(masks), dtype=torch.uint8)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        if self.augment and box_tensor.numel() and torch.rand(1).item() < 0.5:
            tensor = tensor.flip(-1)
            width = tensor.shape[-1]
            flipped = box_tensor.clone()
            flipped[:, 0] = width - box_tensor[:, 2]
            flipped[:, 2] = width - box_tensor[:, 0]
            box_tensor = flipped
            mask_tensor = mask_tensor.flip(-1)
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor([info["id"]]),
        }
        return tensor, target


def class_weights_for_sampler(dataset: BDDDetectionDataset) -> list[float]:
    """Up-weight images that contain rare classes. Weight is max 1/sqrt(freq)."""
    freq = {i: 0 for i in range(1, NUM_CLASSES)}
    for info in dataset.images:
        for ann in dataset.by_image[info["id"]]:
            cat = int(ann["category_id"])
            if cat in freq:
                freq[cat] += 1
    weights = []
    for info in dataset.images:
        anns = dataset.by_image[info["id"]]
        if not anns:
            weights.append(0.0)
            continue
        weights.append(
            max((1.0 / max(freq.get(int(a["category_id"]), 1), 1)) ** 0.5 for a in anns)
        )
    return weights


def write_metrics(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}_metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return path


def count_parameters(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_fps(step, device: torch.device, warmup: int = 5, iters: int = 20) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            step()
        sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            step()
        sync(device)
        elapsed = time.perf_counter() - start
    return iters / elapsed if elapsed > 0 else 0.0


def evaluate_coco(results: list[dict], iou_type: str = "bbox") -> dict:
    """Score detections with pycocotools. ``map50`` is AP at IoU 0.50."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(DATA_ROOT / "annotations" / "instances_val.json"))
    empty = {
        "map": 0.0,
        "map50": 0.0,
        "map75": 0.0,
        "map_small": None,
        "map_medium": None,
        "map_large": None,
        "per_class_ap50": {},
    }
    if not results:
        print(f"no {iou_type} predictions")
        return empty
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    def stat(index: int):
        value = float(evaluator.stats[index])
        return None if value < 0 else value

    per_class = {}
    precision = evaluator.eval["precision"]
    for index, cat_id in enumerate(evaluator.params.catIds):
        column = precision[0, :, index, 0, -1]
        valid = column[column > -1]
        name = CLASS_NAMES[cat_id - 1] if 1 <= int(cat_id) <= len(CLASS_NAMES) else str(cat_id)
        per_class[name] = float(valid.mean()) if valid.size else None
    return {
        "map": stat(0),
        "map50": stat(1),
        "map75": stat(2),
        "map_small": stat(3),
        "map_medium": stat(4),
        "map_large": stat(5),
        "per_class_ap50": per_class,
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
