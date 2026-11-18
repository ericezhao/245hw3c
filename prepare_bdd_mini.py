"""Build the homework's BDD100K mini split.

URLs (https://dl.cv.ethz.ch/bdd100k/data/bdd100k_mini_coco.zip, https://bdd-data.berkeley.edu/)
do not work. This script downloads real BDD100K 10k images and
Cityscapes-style trainId masks from the public Hugging Face copy, keeps
3,000 train and 500 val images, and writes COCO instance JSON.

Instance masks are connected components of the 10 object classes, after a
5x5 closing; pieces under 200 px or 6 px per side are dropped. Stuff
classes (road, sky, building, ...) stay background. Pixel value 255 is
the official ignore label and is not turned into an object.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

REPO = "liqijia/BDD100K-Semantic-Segmentation"
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "bdd100k_mini"
HF_DIR = ROOT / "data" / "hf_bdd_sem"

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
# Official BDD / Cityscapes trainId -> our category id (background is 0).
TRAINID_TO_CAT = {
    11: 1,  # person
    12: 2,  # rider
    13: 3,  # car
    14: 4,  # truck
    15: 5,  # bus
    16: 6,  # train
    17: 7,  # motorcycle
    18: 8,  # bicycle
    6: 9,   # traffic light
    7: 10,  # traffic sign
}
# Components under ~14x14 px are mostly fragments of occluded objects and
# make the box labels noisy. A small closing joins pieces split by poles.
MIN_AREA = 200
MIN_SIDE = 6
CLOSE_KERNEL = np.ones((5, 5), np.uint8)


def list_repo_files(repo_path: str) -> list[str]:
    url = (
        "https://huggingface.co/api/datasets/"
        f"{REPO}/tree/main/{repo_path}?limit=1000"
    )
    names: list[str] = []
    while url:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        payload = response.json()
        for item in payload:
            if item.get("type") == "file":
                names.append(item["path"])
        url = None
        link = response.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part and ">" in part:
                url = part[part.find("<") + 1 : part.find(">")]
                break
    return names


def choose_stems(paths: list[str], n: int, seed: int) -> list[str]:
    stems = sorted(
        Path(path).stem for path in paths if path.endswith(".jpg")
    )
    rng = random.Random(seed)
    rng.shuffle(stems)
    if len(stems) < n:
        raise RuntimeError(f"only found {len(stems)} images, needed {n}")
    return stems[:n]


def fetch_file(filename: str) -> Path:
    last_error: Exception | None = None
    for _ in range(4):
        try:
            saved = hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(HF_DIR),
            )
            return Path(saved)
        except Exception as exc:  # rate limits and dropped connections
            last_error = exc
    raise RuntimeError(f"failed to download {filename}: {last_error}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def download_split(split: str, stems: list[str], workers: int) -> None:
    image_dir = DATA_ROOT / "images" / split
    mask_dir = DATA_ROOT / "raw_train_id" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for stem in stems:
        jobs.append(
            (
                f"data/images_10k/10k/{split}/shard_000/{stem}.jpg",
                image_dir / f"{stem}.jpg",
            )
        )
        jobs.append(
            (
                f"data/seg_maps/labels/{split}/shard_000/{stem}_train_id.png",
                mask_dir / f"{stem}.png",
            )
        )

    pending = [(src, dst) for src, dst in jobs if not dst.exists()]
    if not pending:
        print(f"{split}: all {len(stems)} pairs already on disk")
        return

    print(f"{split}: downloading {len(pending)} files ({len(stems)} images)")

    def _one(item: tuple[str, Path]) -> str:
        filename, dst = item
        saved = fetch_file(filename)
        link_or_copy(saved, dst)
        return filename

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc=split):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        print(f"{split}: {len(errors)} failures, first: {errors[0]}")
        raise RuntimeError(f"{split} download incomplete ({len(errors)} files)")


def mask_to_annotations(mask: np.ndarray, image_id: int, ann_id: int) -> tuple[list[dict], int]:
    annotations = []
    for train_id, cat_id in TRAINID_TO_CAT.items():
        binary = (mask == train_id).astype(np.uint8)
        if int(binary.sum()) < MIN_AREA:
            continue
        grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, CLOSE_KERNEL)
        count, labels = cv2.connectedComponents(grouped, connectivity=8)
        for lab in range(1, count):
            # keep only the real class pixels, not the ones closing filled in
            component = (labels == lab) & (binary > 0)
            area = int(component.sum())
            if area < MIN_AREA:
                continue
            ys, xs = np.nonzero(component)
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
            if w < MIN_SIDE or h < MIN_SIDE:
                continue
            component = np.asfortranarray(component, dtype=np.uint8)
            encoded = mask_utils.encode(component)
            encoded["counts"] = encoded["counts"].decode("ascii")
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(area),
                    "iscrowd": 0,
                    "segmentation": encoded,
                }
            )
            ann_id += 1
    return annotations, ann_id


def mapped_semantic(mask: np.ndarray) -> np.ndarray:
    """0 background, 1-10 object classes, 255 ignore."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    ignore = mask == 255
    for train_id, cat_id in TRAINID_TO_CAT.items():
        out[mask == train_id] = cat_id
    out[ignore] = 255
    return out


def write_coco(split: str, stems: list[str]) -> None:
    images = []
    annotations = []
    ann_id = 1
    seg_dir = ROOT / "seg_masks" / split
    seg_dir.mkdir(parents=True, exist_ok=True)
    class_counts = {name: 0 for name in CLASS_NAMES}

    for image_id, stem in enumerate(tqdm(stems, desc=f"coco {split}"), start=1):
        mask_path = DATA_ROOT / "raw_train_id" / split / f"{stem}.png"
        mask = np.array(Image.open(mask_path))
        height, width = mask.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": f"{stem}.jpg",
                "width": int(width),
                "height": int(height),
            }
        )
        anns, ann_id = mask_to_annotations(mask, image_id, ann_id)
        annotations.extend(anns)
        for ann in anns:
            class_counts[CLASS_NAMES[ann["category_id"] - 1]] += 1
        Image.fromarray(mapped_semantic(mask)).save(seg_dir / f"{stem}.png")

    payload = {
        "info": {
            "description": "BDD100K mini reconstructed for HW3c",
            "source": REPO,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(CLASS_NAMES)
        ],
    }
    ann_dir = DATA_ROOT / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out = ann_dir / f"instances_{split}.json"
    with out.open("w") as handle:
        json.dump(payload, handle)
    print(
        f"{split}: {len(images)} images, {len(annotations)} instances -> {out}"
    )
    print("class counts:", json.dumps(class_counts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=3000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=245)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--coco-only", action="store_true",
                        help="rebuild annotations from the masks already on disk")
    args = parser.parse_args()

    manifest_path = DATA_ROOT / "split_manifest.json"
    if args.coco_only:
        manifest = json.loads(manifest_path.read_text())
        write_coco("train", manifest["train"])
        write_coco("val", manifest["val"])
        return

    print("listing train images...")
    train_files = list_repo_files("data/images_10k/10k/train/shard_000")
    print("listing val images...")
    val_files = list_repo_files("data/images_10k/10k/val/shard_000")
    train_stems = choose_stems(train_files, args.train_n, args.seed)
    val_stems = choose_stems(val_files, args.val_n, args.seed + 1)
    print(f"selected {len(train_stems)} train and {len(val_stems)} val")
    print("train sample:", ", ".join(train_stems[:3]))

    manifest = {
        "repo": REPO,
        "seed": args.seed,
        "train": train_stems,
        "val": val_stems,
        "note": (
            "Official bdd100k_mini_coco.zip host does not resolve. "
            "Images are BDD100K 10k frames. Instances are connected "
            "components of the 10 object classes in the semantic mask "
            "(5x5 closing, min 200 px area, min 6 px side)."
        ),
    }
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    download_split("train", train_stems, args.workers)
    download_split("val", val_stems, args.workers)
    if not args.skip_coco:
        write_coco("train", train_stems)
        write_coco("val", val_stems)
    print("dataset ready at", DATA_ROOT)


if __name__ == "__main__":
    main()
