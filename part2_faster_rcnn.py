"""Part 2: fine-tune Faster R-CNN ResNet-50 FPN on BDD100K mini."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from bdd_common import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    RESULTS,
    BDDDetectionDataset,
    benchmark_fps,
    class_weights_for_sampler,
    collate,
    count_parameters,
    evaluate_coco,
    device_label,
    get_device,
    write_metrics,
)

CHECKPOINT = CHECKPOINT_DIR / "faster_rcnn_bdd.pth"


def build_faster_rcnn(num_classes: int = NUM_CLASSES):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # 1280x720 frames are downscaled so a 16 GB machine can hold batch 2.
    model.transform.min_size = (640,)
    model.transform.max_size = 1024
    return model


def move_targets(targets, device):
    moved = []
    for target in targets:
        item = {key: value.to(device) for key, value in target.items()}
        moved.append(item)
    return moved


def train(model, loader, optimizer, device, max_steps: int) -> float:
    model.train()
    total = 0.0
    steps = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [image.to(device) for image in images]
        targets = move_targets(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            print("non-finite loss, skipping batch", {k: float(v) for k, v in loss_dict.items()})
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)


@torch.no_grad()
def predict(model, dataset, device, max_images: int = 0):
    model.eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    results = []
    seen = 0
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = [image.to(device) for image in images]
        preds = model(images)
        for pred, target in zip(preds, targets):
            image_id = int(target["image_id"].item())
            for box, score, label in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
            seen += 1
            if max_images and seen >= max_images:
                return results
    return results


def load_checkpoint(model, device) -> int:
    if not CHECKPOINT.exists():
        return 0
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"])
        return int(payload.get("epoch", 0))
    model.load_state_dict(payload)
    return 0


def save_checkpoint(model, epoch: int) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch}, CHECKPOINT)
    print("saved", CHECKPOINT, "epoch", epoch)


def save_prediction_figure(model, dataset, device) -> None:
    model.eval()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    with torch.inference_mode():
        for ax, index in zip(axes, range(4)):
            image, _ = dataset[index]
            pred = model([image.to(device)])[0]
            canvas = image.permute(1, 2, 0).cpu().numpy().copy()
            ax.imshow(canvas)
            for box, score, label in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
                if float(score) < 0.5:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                name = CLASS_NAMES[int(label) - 1] if 1 <= int(label) <= len(CLASS_NAMES) else str(int(label))
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 4, 0), f"{name} {float(score):.2f}", color="yellow", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    path = RESULTS / "faster_rcnn_preds.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


def measure_fps(model, dataset, device) -> dict:
    model.eval()
    image, _ = dataset[0]
    cpu_model = None

    def gpu_step():
        model([image.to(device)])

    gpu_fps = benchmark_fps(gpu_step, device, warmup=3, iters=10)

    cpu = torch.device("cpu")
    cpu_model = build_faster_rcnn()
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to(cpu).eval()

    def cpu_step():
        cpu_model([image])

    cpu_fps = benchmark_fps(cpu_step, cpu, warmup=1, iters=5)
    del cpu_model
    return {"fps_gpu": gpu_fps, "fps_cpu": cpu_fps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max-steps", type=int, default=0, help="stop each epoch early; 0 means full data")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    model = build_faster_rcnn().to(device)
    print(f"parameters: {count_parameters(model):.2f} M")
    train_ds = BDDDetectionDataset("train", require_objects=True, augment=True)
    val_ds = BDDDetectionDataset("val")
    if args.eval_only and not CHECKPOINT.exists():
        raise SystemExit(f"missing {CHECKPOINT}")
    start_epoch = load_checkpoint(model, device)

    if not args.eval_only and start_epoch < args.epochs:
        weights = class_weights_for_sampler(train_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate,
        )
        backbone, head = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone if name.startswith("backbone") else head).append(param)
        optimizer = SGD(
            [
                {"params": backbone, "lr": args.lr * 0.1},
                {"params": head, "lr": args.lr},
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        for _ in range(start_epoch):
            scheduler.step()
        history = []
        for epoch in range(start_epoch, args.epochs):
            loss = train(model, loader, optimizer, device, args.max_steps)
            scheduler.step()
            history.append(loss)
            print(f"Epoch {epoch + 1:02d} | Loss: {loss:.4f}")
            if args.max_steps == 0:
                save_checkpoint(model, epoch + 1)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        if history and args.max_steps == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(start_epoch + 1, start_epoch + len(history) + 1), history, marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Faster R-CNN training loss")
            fig.tight_layout()
            fig.savefig(RESULTS / "faster_rcnn_loss.png", dpi=140)
            plt.close(fig)

    if args.max_steps:
        print("smoke test finished; skipping eval")
        return

    results = predict(model, val_ds, device)
    pred_path = RESULTS / "faster_rcnn_predictions.json"
    pred_path.write_text(json.dumps(results))
    metrics = evaluate_coco(results, "bbox")
    save_prediction_figure(model, val_ds, device)
    speed = measure_fps(model, val_ds, device)
    map50 = metrics["map50"]
    write_metrics(
        "faster_rcnn",
        {
            "model": "Faster R-CNN FPN",
            "mAP@0.5": map50,
            "mAP": metrics["map"],
            "mAP_small": metrics["map_small"],
            "mAP_medium": metrics["map_medium"],
            "mAP_large": metrics["map_large"],
            "per_class_ap50": metrics["per_class_ap50"],
            "mIoU": None,
            "FPS (GPU)": speed["fps_gpu"],
            "FPS (CPU)": speed["fps_cpu"],
            "Params (M)": round(count_parameters(model), 2),
            "device": str(device),
            "gpu": device_label(device),
            "epochs": args.epochs,
        },
    )
    (RESULTS / "faster_rcnn_map.txt").write_text(f"mAP@0.5 {map50:.4f}\n")


if __name__ == "__main__":
    main()
