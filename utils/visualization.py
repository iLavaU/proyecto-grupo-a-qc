"""
Visualization utilities for analysis and presentation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from config import settings


def plot_sample_images(images, labels, class_names, n_samples=10, figsize=(15, 6)):
    """
    Plot a sample of images from the dataset.

    Args:
        images: Array of images
        labels: Array of labels (numeric)
        class_names: List of class names
        n_samples: Number of samples to plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, n_samples // 2, figsize=figsize)
    axes = axes.flatten()

    # Randomly select samples
    indices = np.random.choice(len(images), n_samples, replace=False)

    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        axes[i].set_title(f"{class_names[labels[idx]]}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Sample Images from EuroSAT Dataset", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def un_normalize_image(img_tensor, mean, std):
    """
    Un-normalize a tensor image.

    Args:
        img_tensor: Normalized image tensor (C, H, W) or numpy array
        mean: Mean used for normalization
        std: Std used for normalization
    Returns:
        numpy.ndarray: Un-normalized image (H, W, C)
    """
    if hasattr(img_tensor, "detach"):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.asarray(img_tensor)

    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)  # Convert to HWC

    img = (img * std) + mean  # Un-normalize
    img = np.clip(img, 0, 1)  # Clip to [0, 1]
    return img


def plot_sample_image_from_dataloader(dataloader, figsize=(15, 6), same_plot=True):
    """
    Plot a batch of images and masks from a DataLoader.

    Args:
        dataloader: PyTorch DataLoader yielding (images, masks)
        batch_size: Number of samples to plot from a single batch
        figsize: Figure size
        same_plot: If True, show image+mask side-by-side in the same row.
                   If False, show each image and mask in separate figures.
    """
    # Take a single batch
    batch_images, batch_labels = next(iter(dataloader))

    # Move to CPU and convert to numpy if needed
    if hasattr(batch_images, "detach"):
        batch_images = batch_images.detach().cpu().numpy()
    if hasattr(batch_labels, "detach"):
        batch_labels = batch_labels.detach().cpu().numpy()

    img, mask = batch_images[0], batch_labels[0]

    img = un_normalize_image(img, mean=settings.MEAN, std=settings.STD)

    # Prepare mask for display (grayscale)
    if mask.ndim == 3 and mask.shape[0] in (1, 3):
        mask = mask.transpose(1, 2, 0)
    mask = np.squeeze(mask)

    print("\nDisplaying a sample image and its corresponding mask from the DataLoader...")
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    if same_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Rezise images if too large
        if img.shape[0] >= 300 or img.shape[1] >= 300:
            pass  # Add resizing logic if needed

        axes[0].imshow(img)
        axes[0].set_title("Sample Image", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Corresponding Mask", fontsize=12)
        axes[1].axis("off")

        plt.suptitle("Sample Image and Mask from DataLoader", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title("Sample Image", fontsize=12)
        plt.axis("off")
        plt.show()

        plt.figure(figsize=figsize)
        plt.imshow(mask, cmap="gray")
        plt.title("Corresponding Mask", fontsize=12)
        plt.axis("off")
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(12, 10)):
    """
    Plot confusion matrix with labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to percentages
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_xlabel("Predicted Label", fontsize=12)

    # Plot percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={"label": "Percentage (%)"},
    )
    ax2.set_title("Confusion Matrix (Percentages)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=12)
    ax2.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.show()

    return cm


def plot_classification_report(y_true, y_pred, class_names, figsize=(10, 8)):
    """
    Display classification report as a heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Extract metrics for each class
    metrics = ["precision", "recall", "f1-score"]
    data = []

    for class_name in class_names:
        if class_name in report:
            data.append([report[class_name]["precision"], report[class_name]["recall"], report[class_name]["f1-score"]])

    data = np.array(data)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data.T,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=class_names,
        yticklabels=metrics,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Score"},
    )

    ax.set_title("Classification Report by Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Also print text report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 60 + "\n")


def plot_class_distribution(labels, class_names, title="Class Distribution"):
    """
    Plot the distribution of classes in the dataset.

    Args:
        labels: Array of numeric labels
        class_names: List of class names
        title: Plot title
    """
    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(unique)), counts, color="steelblue", alpha=0.7)

    # Customize
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names, figsize=(12, 6)):
    """
    Plot accuracy for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    # Calculate per-class accuracy
    accuracies = []

    for i, class_name in enumerate(class_names):
        # Get indices for this class
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies.append(class_acc * 100)
        else:
            accuracies.append(0)

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(class_names)), accuracies, color="steelblue", alpha=0.7)

    # Color code by performance
    for i, bar in enumerate(bars):
        if accuracies[i] >= 80:
            bar.set_color("green")
        elif accuracies[i] >= 60:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    # Customize
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim([0, 105])

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{accuracies[i]:.1f}%", ha="center", va="bottom", fontsize=9
        )

    # Add horizontal line at overall accuracy
    overall_acc = (y_pred == y_true).mean() * 100
    ax.axhline(y=overall_acc, color="black", linestyle="--", alpha=0.5, label=f"Overall Accuracy: {overall_acc:.1f}%")
    ax.legend()

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_results(test_results):
    """
    Create a comprehensive visualization of test results.

    Args:
        test_results: Dictionary containing test results from test_model()
    """
    y_true = test_results["targets"]
    y_pred = test_results["predictions"]
    class_names = test_results["class_names"]
    accuracy = test_results["accuracy"]

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 1. Confusion Matrix
    print("\n1. Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names)

    # 2. Classification Report
    print("\n2. Classification Report...")
    plot_classification_report(y_true, y_pred, class_names)

    # 3. Per-Class Accuracy
    print("\n3. Per-Class Accuracy...")
    plot_per_class_accuracy(y_true, y_pred, class_names)

    print("\n" + "=" * 60)
    print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60 + "\n")


def plot_training_history(trainer, figsize=(12, 5)):
    """
    Plot training history from PyTorch Lightning trainer.

    Args:
        trainer: PyTorch Lightning Trainer instance
        figsize: Figure size
    """
    # This would require storing metrics during training
    # For now, we'll leave this as a placeholder
    print("Training history plotting not yet implemented.")
    print("Use TensorBoard for detailed training visualization:")
    print("  tensorboard --logdir lightning_logs/")


def save_results_summary(test_results, filename="results_summary.txt"):
    """
    Save a text summary of results.

    Args:
        test_results: Dictionary containing test results
        filename: Output filename
    """
    y_true = test_results["targets"]
    y_pred = test_results["predictions"]
    class_names = test_results["class_names"]
    accuracy = test_results["accuracy"]

    with open(filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HYBRID QUANTUM CLASSIFIER - TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 60 + "\n")
        cm = confusion_matrix(y_true, y_pred)

        # Write header
        f.write("True\\Pred  ")
        for name in class_names:
            f.write(f"{name[:8]:>8s} ")
        f.write("\n")

        # Write rows
        for i, name in enumerate(class_names):
            f.write(f"{name[:10]:10s} ")
            for j in range(len(class_names)):
                f.write(f"{cm[i,j]:8d} ")
            f.write("\n")

    print(f"\nResults saved to: {filename}")
