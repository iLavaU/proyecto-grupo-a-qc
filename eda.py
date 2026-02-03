# eda_with_loader_splits.py
from collections import Counter
import numpy as np
import torch

from config import Config
from data.loader import load_images_path, create_dataloaders

def analyze_loader_splits():
    config = Config()

    # misma carga que en main.py
    images_path, labels_path = load_images_path(max_per_class=config.MAX_IMAGES_PER_CLASS)
    train_loader, val_loader, test_loader, _ = create_dataloaders(images_path, labels_path, config)

    for name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        img_sizes = Counter()
        mask_sizes = Counter()
        pixel_counts = Counter()

        for imgs, masks in loader:
            # imgs: [B,3,H,W], masks: [B,H,W] (long)
            b, c, h, w = imgs.shape
            img_sizes[(c, h, w)] += b

            # mask sizes
            for m in masks:
                mask_sizes[m.shape] += 1

            # distribución de clases por píxel
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

if __name__ == "__main__":
    analyze_loader_splits()
