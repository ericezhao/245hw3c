"""Part 5: fine-tune Mask R-CNN and draw a separate color for each instance."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from bdd_common import (
    CHECKPOINT_DIR,
    NUM_CLASSES,
    RESULTS,
    BDDInstanceDataset,
    benchmark_fps,
    collate,
    count_parameters,
    evaluate_coco,
    device_label,
    get_device,
    write_metrics,
)

CHECKPOINT = CHECKPOINT_DIR / "mask_rcnn_bdd.pth"


def build_mask_rcnn(num_classes: int = NUM_CLASSES):
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, num_classes)
    model.transform.min_size = (512,)
    model.transform.max_size = 896
    return model


def move_targets(targets, device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("device", device)
    model = build_mask_rcnn().to(device)
    print(f"parameters: {count_parameters(model):.2f} M")

    start_epoch = 0
    if CHECKPOINT.exists():
        payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model" in payload:
            model.load_state_dict(payload["model"])
            start_epoch = int(payload.get("epoch", 0))
        else:
            model.load_state_dict(payload)
        print(f"loaded {CHECKPOINT} at epoch {start_epoch}")
    elif args.eval_only:
        raise SystemExit(f"missing {CHECKPOINT}")

    if not args.eval_only and start_epoch < args.epochs:
        dataset = BDDInstanceDataset("train", require_objects=True, augment=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
        )
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        for _ in range(start_epoch):
            scheduler.step()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total = 0.0
            steps = 0
            for images, targets in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
                images = [image.to(device) for image in images]
                targets = move_targets(targets, device)
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
            scheduler.step()
            print(f"Epoch {epoch + 1:02d} | Loss: {total / max(steps, 1):.4f}")
            if device.type == "mps":
                torch.mps.empty_cache()
            if args.max_steps == 0:
                torch.save({"model": model.state_dict(), "epoch": epoch + 1}, CHECKPOINT)
                print("saved", CHECKPOINT, "epoch", epoch + 1)

    if args.max_steps:
        print("smoke test finished; skipping visualization")
        return

    visualize(model, device)
    val_ds = BDDInstanceDataset("val")
    model.eval()
    image, _ = val_ds[0]

    def step():
        model([image.to(device)])

    gpu_fps = benchmark_fps(step, device, warmup=2, iters=8)
    cpu = torch.device("cpu")
    cpu_model = build_mask_rcnn()
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to(cpu).eval()

    def cpu_step():
        cpu_model([image])

    cpu_fps = benchmark_fps(cpu_step, cpu, warmup=1, iters=3)
    box_metrics = evaluate_boxes(model, val_ds, device)
    write_metrics(
        "mask_rcnn",
        {
            "model": "Mask R-CNN FPN",
            "mAP@0.5": box_metrics["map50"],
            "mAP": box_metrics["map"],
            "mAP_small": box_metrics["map_small"],
            "mAP_medium": box_metrics["map_medium"],
            "mAP_large": box_metrics["map_large"],
            "per_class_ap50": box_metrics["per_class_ap50"],
            "mIoU": None,
            "FPS (GPU)": gpu_fps,
            "FPS (CPU)": cpu_fps,
            "Params (M)": round(count_parameters(model), 2),
            "device": str(device),
            "gpu": device_label(device),
            "epochs": args.epochs,
        },
    )


@torch.no_grad()
def evaluate_boxes(model, dataset, device) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    results = []
    for images, targets in tqdm(loader, desc="mask-rcnn eval", leave=False):
        pred = model([images[0].to(device)])[0]
        image_id = int(targets[0]["image_id"].item())
        scores = pred["scores"].detach().cpu()
        boxes = pred["boxes"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        if len(scores) > 100:
            keep = torch.topk(scores, 100).indices
            scores, boxes, labels = scores[keep], boxes[keep], labels[keep]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    (RESULTS / "mask_rcnn_predictions.json").write_text(json.dumps(results))
    return evaluate_coco(results, "bbox")


@torch.no_grad()
def visualize(model, device, n: int = 4) -> None:
    model.eval()
    dataset = BDDInstanceDataset("val")
    palette = plt.colormaps["tab20"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, index in zip(axes, range(n)):
        image, _ = dataset[index]
        pred = model([image.to(device)])[0]
        canvas = image.permute(1, 2, 0).cpu().numpy().copy()
        shown = 0
        for mask, score in zip(pred["masks"], pred["scores"]):
            if float(score) < 0.5:
                continue
            binary = mask[0].cpu().numpy() > 0.5
            color = np.array(palette(shown % 20)[:3])
            canvas[binary] = canvas[binary] * 0.5 + color * 0.5
            shown += 1
        ax.imshow(np.clip(canvas, 0, 1))
        ax.set_title(f"{shown} instances")
        ax.axis("off")
    fig.tight_layout()
    path = RESULTS / "instance_masks.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
