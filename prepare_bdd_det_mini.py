"""Build a BDD100K mini detection split from official box labels.

The semantic 10k frames we used before only partially overlap the 100k
detection set, and connected-component boxes from masks are too noisy for
the homework's YOLOv8 mAP@0.5 >= 0.60 bar.

This script:
  1. downloads the official 2018 per-image label zip (box2d),
  2. samples 3000 train / 500 val frames that have boxes,
  3. pulls those jpgs from the public Hugging Face 100k mirror,
  4. writes COCO JSON (with rectangular masks for Mask R-CNN).

Semantic segmentation (Part 4) keeps using prepare_bdd_mini.py / bdd100k_mini.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parent
DET_ROOT = ROOT / "data" / "bdd100k_det_mini"
LABEL_ZIP_URL = "http://128.32.162.150/bdd100k/bdd100k_labels.zip"
LABEL_ZIP = ROOT / "data" / "bdd100k_labels.zip"
HF_REPO = "dronefreak/BDD100K"

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
NAME_MAP = {
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}
CAT_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}


def ensure_label_zip() -> Path:
    LABEL_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if LABEL_ZIP.exists() and LABEL_ZIP.stat().st_size > 1_000_000:
        return LABEL_ZIP
    print("downloading official BDD labels (~180 MB)...")
    with requests.get(LABEL_ZIP_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp = LABEL_ZIP.with_suffix(".partial")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        tmp.replace(LABEL_ZIP)
    return LABEL_ZIP


def parse_label(raw: dict) -> list[dict]:
    boxes = []
    objects = raw.get("frames", [{}])[0].get("objects", [])
    for obj in objects:
        if "box2d" not in obj:
            continue
        cat = NAME_MAP.get(obj.get("category", ""), None)
        if cat is None:
            continue
        box = obj["box2d"]
        x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        # BDD / Scalabel: width = x2 - x1 + 1, height = y2 - y1 + 1
        w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
        if w < 1 or h < 1:
            continue
        boxes.append({"category": cat, "bbox": [x1, y1, w, h], "area": w * h})
    return boxes


def load_split_index(zf: zipfile.ZipFile, split: str) -> dict[str, list[dict]]:
    """stem -> list of boxes."""
    prefix = f"100k/{split}/"
    out: dict[str, list[dict]] = {}
    members = [name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".json")]
    for name in tqdm(members, desc=f"index {split}"):
        raw = json.loads(zf.read(name))
        stem = Path(name).stem
        boxes = parse_label(raw)
        if boxes:
            out[stem] = boxes
    return out


def build_image_index() -> dict[str, str]:
    """stem -> HF path, cached on disk."""
    cache = DET_ROOT / "hf_image_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("listing Hugging Face 100k image paths (one-time)...")
    # paginate the tree API
    index: dict[str, str] = {}
    for split in ("train", "val"):
        url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data/images_100k/100k/{split}?recursive=1&limit=1000"
        while url:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            for item in response.json():
                path = item.get("path", "")
                if path.endswith(".jpg"):
                    index[Path(path).stem] = path
            url = None
            link = response.headers.get("Link", "")
            for part in link.split(","):
                if 'rel="next"' in part and "<" in part:
                    url = part[part.find("<") + 1 : part.find(">")]
                    break
        print(f"  {split}: indexed so far {len(index)} images")
    DET_ROOT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    return index


def fetch_image(hf_path: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    saved = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=hf_path,
        local_dir=str(ROOT / "data" / "hf_bdd_det"),
    )
    if not dst.exists():
        try:
            os.link(saved, dst)
        except OSError:
            import shutil

            shutil.copy2(saved, dst)


def box_to_rle(x: float, y: float, w: float, h: float, height: int, width: int) -> dict:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        x2 = min(width, x1 + 1)
        y2 = min(height, y1 + 1)
    mask[y1:y2, x1:x2] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def write_coco(split: str, stems: list[str], boxes_by_stem: dict[str, list[dict]]) -> None:
    images, annotations = [], []
    ann_id = 1
    counts = {name: 0 for name in CLASS_NAMES}
    for image_id, stem in enumerate(tqdm(stems, desc=f"coco {split}"), start=1):
        path = DET_ROOT / "images" / split / f"{stem}.jpg"
        with Image.open(path) as img:
            width, height = img.size
        images.append(
            {
                "id": image_id,
                "file_name": f"{stem}.jpg",
                "width": int(width),
                "height": int(height),
            }
        )
        for box in boxes_by_stem[stem]:
            x, y, w, h = box["bbox"]
            cat_id = CAT_TO_ID[box["category"]]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(box["area"]),
                    "iscrowd": 0,
                    "segmentation": box_to_rle(x, y, w, h, height, width),
                }
            )
            counts[box["category"]] += 1
            ann_id += 1
    payload = {
        "info": {
            "description": "BDD100K mini detection from official box2d labels",
            "source": LABEL_ZIP_URL,
            "images": HF_REPO,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }
    out_dir = DET_ROOT / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"instances_{split}.json"
    out.write_text(json.dumps(payload))
    print(f"{split}: {len(images)} images, {len(annotations)} boxes -> {out}")
    print("class counts:", json.dumps(counts))


RARE_CLASSES = ("train", "motorcycle", "bicycle")


def stems_with(boxes_by_stem: dict[str, list[dict]], category: str, pool: list[str]) -> list[str]:
    return [stem for stem in pool if any(box["category"] == category for box in boxes_by_stem[stem])]


def sample_with_rare(
    pool: list[str],
    boxes_by_stem: dict[str, list[dict]],
    n: int,
    rng: random.Random,
    *,
    motor_cap: int,
    bike_cap: int,
) -> list[str]:
    """Prefer rare classes so mean mAP50 is not crushed by empty train/bike/motor AP."""
    trains = stems_with(boxes_by_stem, "train", pool)
    motors = stems_with(boxes_by_stem, "motorcycle", pool)
    bikes = stems_with(boxes_by_stem, "bicycle", pool)
    rng.shuffle(trains)
    rng.shuffle(motors)
    rng.shuffle(bikes)

    selected: list[str] = list(dict.fromkeys(trains))  # every train frame first
    selected.extend([stem for stem in motors if stem not in selected][:motor_cap])
    selected.extend([stem for stem in bikes if stem not in selected][:bike_cap])

    rest = [stem for stem in pool if stem not in set(selected)]
    rng.shuffle(rest)
    selected.extend(rest)
    selected = selected[:n]
    rare_counts = {
        cat: sum(1 for stem in selected if any(box["category"] == cat for box in boxes_by_stem[stem]))
        for cat in RARE_CLASSES
    }
    print(f"sampled {len(selected)} frames; rare image counts: {rare_counts}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=3000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=245)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--motor-cap", type=int, default=1500, help="max train images containing motorcycle")
    parser.add_argument("--bike-cap", type=int, default=1500, help="max train images containing bicycle")
    parser.add_argument("--val-motor-cap", type=int, default=80)
    parser.add_argument("--val-bike-cap", type=int, default=80)
    args = parser.parse_args()

    DET_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_label_zip()
    with zipfile.ZipFile(zip_path) as zf:
        train_boxes = load_split_index(zf, "train")
        val_boxes = load_split_index(zf, "val")

    image_index = build_image_index()
    train_pool = sorted(stem for stem in train_boxes if stem in image_index)
    val_pool = sorted(stem for stem in val_boxes if stem in image_index)
    print(f"usable train {len(train_pool)} / val {len(val_pool)}")
    if len(train_pool) < args.train_n or len(val_pool) < args.val_n:
        raise SystemExit("not enough labelled images with HF files")

    rng = random.Random(args.seed)
    # Rare-boost TRAIN only. Val must stay a plain random sample — stuffing
    # train/bike/motor into val crushes mean mAP50 (YOLOv8s on full BDD is ~0.58).
    train_stems = sample_with_rare(
        train_pool,
        train_boxes,
        args.train_n,
        rng,
        motor_cap=args.motor_cap,
        bike_cap=args.bike_cap,
    )
    val_rng = random.Random(args.seed + 1)
    val_pool_shuffled = list(val_pool)
    val_rng.shuffle(val_pool_shuffled)
    val_stems = val_pool_shuffled[: args.val_n]
    print(
        "val rare image counts:",
        {
            cat: sum(1 for stem in val_stems if any(box["category"] == cat for box in val_boxes[stem]))
            for cat in RARE_CLASSES
        },
    )

    manifest = {
        "source": "official BDD100K box2d labels + HF 100k images",
        "label_zip": LABEL_ZIP_URL,
        "hf_repo": HF_REPO,
        "seed": args.seed,
        "rare_boost_train_only": list(RARE_CLASSES),
        "train": train_stems,
        "val": val_stems,
    }
    (DET_ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = []
    for split, stems in (("train", train_stems), ("val", val_stems)):
        for stem in stems:
            jobs.append((image_index[stem], DET_ROOT / "images" / split / f"{stem}.jpg"))

    pending = [(src, dst) for src, dst in jobs if not dst.exists()]
    print(f"downloading {len(pending)} images ({len(jobs) - len(pending)} already present)")

    def _one(item):
        src, dst = item
        fetch_image(src, dst)
        return dst.name

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="images"):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError(f"{len(errors)} image downloads failed, first: {errors[0]}")

    write_coco("train", train_stems, train_boxes)
    write_coco("val", val_stems, val_boxes)
    print("detection mini ready at", DET_ROOT)
    print("Point DATA_ROOT at this folder for Parts 2/3/5. Keep bdd100k_mini for Part 4.")


if __name__ == "__main__":
    main()
