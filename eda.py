from collections import Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from data.loader import load_images_path, create_dataloaders

def plot_class_distribution(split_name, pixel_counts, total_pixels, out_dir="documentation"):
    classes = sorted(pixel_counts.keys())
    percents = [(pixel_counts[c] / total_pixels) * 100 for c in classes]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(classes, percents)
    plt.title(f"Pixel Class Distribution - {split_name}")
    plt.xlabel("Class ID")
    plt.ylabel("Percentage (%)")
    plt.xticks(classes)

    for bar, pct in zip(bars, percents):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()

    out_path = Path(out_dir) / f"class_distribution_{split_name.lower()}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def analyze_loader_splits():
    config = Config()

    images_path, labels_path = load_images_path(max_per_class=config.MAX_IMAGES_PER_CLASS)
    train_loader, val_loader, test_loader, _ = create_dataloaders(images_path, labels_path, config)

    for name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        img_sizes = Counter()
        mask_sizes = Counter()
        pixel_counts = Counter()

        for imgs, masks in loader:
            b, c, h, w = imgs.shape
            img_sizes[(c, h, w)] += b

            for m in masks:
                mask_sizes[m.shape] += 1

            vals, counts = torch.unique(masks, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                pixel_counts[int(v)] += int(c)

        total_pixels = sum(pixel_counts.values())

        print(f"\n=== {name} ===")
        print("Batches:", len(loader))
        print("Images:", len(loader.dataset))
        print("Image sizes (C,H,W):", dict(img_sizes))
        print("Mask sizes (H,W):", dict(mask_sizes))

        print("Pixel distribution by class id:")
        for k in sorted(pixel_counts.keys()):
            pct = (pixel_counts[k] / total_pixels) * 100 if total_pixels else 0
            print(f"  {k}: {pixel_counts[k]} pixels ({pct:.2f}%)")

        # generar gráfica
        plot_class_distribution(name, pixel_counts, total_pixels)

if __name__ == "__main__":
    analyze_loader_splits()
