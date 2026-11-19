"""Part 4: DeepLabV3+ semantic segmentation on the 10 BDD object classes.

MMSegmentation is not installable on this Apple Silicon machine, so the
same architecture (ResNet-50 encoder, ASPP, decoder) is trained with
segmentation-models-pytorch. Pixel labels are the official BDD trainIds
mapped to background + the 10 homework classes. 255 is the ignore label.
"""

from __future__ import annotations

import argparse

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bdd_common import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    SEG_DATA_ROOT,
    NUM_CLASSES,
    RESULTS,
    ROOT,
    benchmark_fps,
    count_parameters,
    device_label,
    get_device,
    write_metrics,
)

CHECKPOINT = CHECKPOINT_DIR / "deeplabv3plus_bdd.pth"
HEIGHT, WIDTH = 384, 640
IGNORE = 255


def build_deeplab(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    # Output stride 16 with ASPP rates 6/12/18, matching DeepLabV3+.
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        encoder_output_stride=16,
        decoder_atrous_rates=(6, 12, 18),
        in_channels=3,
        classes=NUM_CLASSES,
    )


def build_transforms(train: bool):
    ops = []
    if train:
        ops.append(A.HorizontalFlip(p=0.5))
        ops.append(A.RandomBrightnessContrast(p=0.3))
    ops.extend(
        [
            A.Resize(HEIGHT, WIDTH, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(ops)


class SegDataset(Dataset):
    def __init__(self, split: str, train: bool):
        self.image_dir = SEG_DATA_ROOT / "images" / split
        self.mask_dir = ROOT / "seg_masks" / split
        self.paths = sorted(self.image_dir.glob("*.jpg"))
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = np.array(Image.open(path).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{path.stem}.png"))
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


def class_weights(split: str = "train") -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for path in (ROOT / "seg_masks" / split).glob("*.png"):
        mask = np.array(Image.open(path))
        valid = mask != IGNORE
        counts += np.bincount(mask[valid].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    print("seg class pixels", {("background" if i == 0 else CLASS_NAMES[i - 1]): int(c) for i, c in enumerate(counts)})
    return torch.tensor(weights, dtype=torch.float32)


def confusion(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE
    pred = pred[valid].ravel()
    target = target[valid].ravel()
    keep = (target >= 0) & (target < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
    hist = np.bincount(
        NUM_CLASSES * target[keep].astype(np.int64) + pred[keep].astype(np.int64),
        minlength=NUM_CLASSES ** 2,
    )
    return hist.reshape(NUM_CLASSES, NUM_CLASSES)


def miou_from_conf(matrix: np.ndarray, skip_background: bool = False) -> float:
    intersection = np.diag(matrix)
    union = matrix.sum(1) + matrix.sum(0) - intersection
    present = union > 0
    if skip_background and present.size:
        present[0] = False
    iou = np.divide(intersection, np.maximum(union, 1))
    return float(iou[present].mean()) if present.any() else 0.0


def colorize(mask: np.ndarray) -> np.ndarray:
    cmap = plt.colormaps["tab20"](np.linspace(0, 1, NUM_CLASSES))[:, :3]
    safe = np.clip(mask, 0, NUM_CLASSES - 1)
    color = (cmap[safe] * 255).astype(np.uint8)
    color[mask == IGNORE] = 0
    return color


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        pred = logits.argmax(1).cpu().numpy()
        matrix += confusion(pred, masks.numpy())
    score = miou_from_conf(matrix)
    fg = miou_from_conf(matrix, skip_background=True)
    print(f"mIoU {score:.4f} (foreground-only {fg:.4f})")
    return score, fg


@torch.no_grad()
def evaluate_tta(model, loader, device) -> float:
    """Average the prediction with its horizontal flip. No extra training."""
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        flipped = model(torch.flip(images, dims=[-1]))
        logits = 0.5 * (logits + torch.flip(flipped, dims=[-1]))
        pred = logits.argmax(1).cpu().numpy()
        matrix += confusion(pred, masks.numpy())
    return miou_from_conf(matrix)


def save_overlays(model, device) -> None:
    model.eval()
    dataset = SegDataset("val", train=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for column, index in enumerate((0, 1, 2)):
        image, mask = dataset[index]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(1)[0].cpu().numpy()
        raw = image.permute(1, 2, 0).numpy()
        raw = raw * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        raw = np.clip(raw, 0, 1)
        gt = mask.numpy()
        axes[0, column].imshow(raw)
        axes[0, column].set_title("image")
        axes[1, column].imshow(colorize(gt))
        axes[1, column].set_title("ground truth")
        axes[2, column].imshow(colorize(pred))
        axes[2, column].set_title("DeepLabV3+")
        for row in range(3):
            axes[row, column].axis("off")
    fig.tight_layout()
    path = RESULTS / "deeplab_preds.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2, help="must be >= 2; ASPP BatchNorm fails on batch 1")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 2:
        raise SystemExit("DeepLabV3+ ASPP BatchNorm needs batch size >= 2")
    device = get_device()
    print("device", device)
    model = build_deeplab(pretrained=True).to(device)
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
        weights = class_weights().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)
        encoder, head = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (encoder if name.startswith("encoder") else head).append(param)
        optimizer = AdamW(
            [{"params": encoder, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        for _ in range(start_epoch):
            scheduler.step()
        loader = DataLoader(
            SegDataset("train", train=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total = 0.0
            steps = 0
            for images, masks in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                loss = criterion(model(images), masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
            scheduler.step()
            print(f"Epoch {epoch + 1:02d} | Loss: {total / max(steps, 1):.4f}")
            if args.max_steps == 0:
                torch.save({"model": model.state_dict(), "epoch": epoch + 1}, CHECKPOINT)

    if args.max_steps:
        print("smoke test finished; skipping eval")
        return

    val_loader = DataLoader(SegDataset("val", train=False), batch_size=2, shuffle=False, num_workers=0)
    score, fg_score = evaluate(model, val_loader, device)
    tta_score = evaluate_tta(model, val_loader, device)
    print(f"flip TTA mIoU {tta_score:.4f} (delta {tta_score - score:+.4f})")
    save_overlays(model, device)
    sample, _ = SegDataset("val", train=False)[0]

    def step():
        model(sample.unsqueeze(0).to(device))

    gpu_fps = benchmark_fps(step, device, warmup=3, iters=10)
    cpu = torch.device("cpu")
    cpu_model = build_deeplab(pretrained=False)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.to(cpu).eval()

    def cpu_step():
        cpu_model(sample.unsqueeze(0))

    cpu_fps = benchmark_fps(cpu_step, cpu, warmup=1, iters=5)
    write_metrics(
        "deeplabv3plus",
        {
            "model": "DeepLabV3+ R50",
            "mAP@0.5": None,
            "mIoU": score,
            "mIoU_fg": fg_score,
            "mIoU_tta": tta_score,
            "FPS (GPU)": gpu_fps,
            "FPS (CPU)": cpu_fps,
            "Params (M)": round(count_parameters(model), 2),
            "device": str(device),
            "gpu": device_label(device),
            "epochs": args.epochs,
        },
    )


if __name__ == "__main__":
    main()
