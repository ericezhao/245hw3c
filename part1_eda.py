"""Part 1: visualize BDD100K mini boxes/masks and the class imbalance."""

from __future__ import annotations

import json
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bdd_common import CLASS_NAMES, DATA_ROOT, RESULTS, decode_mask

RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ann_path = DATA_ROOT / "annotations" / "instances_train.json"
    with ann_path.open() as handle:
        coco = json.load(handle)
    images = {img["id"]: img for img in coco["images"]}
    by_image: dict[int, list] = {img_id: [] for img_id in images}
    counts = Counter()
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
        counts[CLASS_NAMES[ann["category_id"] - 1]] += 1

    # Prefer images that actually contain objects so the figure is informative.
    candidates = [img_id for img_id, anns in by_image.items() if anns]
    rng = np.random.default_rng(245)
    chosen = rng.choice(candidates, size=6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    img_dir = DATA_ROOT / "images" / "train"
    for ax, img_id in zip(axes.flatten(), chosen):
        info = images[int(img_id)]
        bgr = cv2.imread(str(img_dir / info["file_name"]))
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for ann in by_image[int(img_id)]:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mask = decode_mask(ann["segmentation"], info["height"], info["width"])
            colored = np.zeros_like(image)
            colored[:, :, 1] = mask * 180
            image = cv2.addWeighted(image, 1.0, colored, 0.4, 0)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Image ID {int(img_id)}", fontsize=10)
    fig.tight_layout()
    eda_path = RESULTS / "eda_samples.png"
    fig.savefig(eda_path, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(counts.most_common(), columns=["Class", "Count"])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Class", y="Count", hue="Class", palette="Blues_d", legend=False, ax=ax)
    ax.set_title("Annotation Count per Class (BDD100K Mini Train)")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    dist_path = RESULTS / "class_distribution.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)

    total = sum(counts.values())
    lines = [
        "BDD100K mini train instance counts",
        f"images: {len(images)}",
        f"instances: {total}",
        "",
    ]
    for name, count in counts.most_common():
        lines.append(f"{name:16s} {count:6d}  {100.0 * count / total:5.2f}%")
    lines += [
        "",
        "Class imbalance: car, traffic sign, and traffic light dominate. "
        "train has only a few dozen instances, so an unweighted loss will "
        "under-fit that class and the other rare road users (rider, motorcycle, bicycle). "
        "Detection training uses inverse-sqrt image sampling so rare classes "
        "appear more often. Semantic training uses inverse-sqrt pixel-frequency class weights.",
    ]
    note_path = RESULTS / "class_imbalance.txt"
    note_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {eda_path}")
    print(f"saved {dist_path}")


if __name__ == "__main__":
    main()
