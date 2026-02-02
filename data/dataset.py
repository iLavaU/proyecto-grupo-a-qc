"""
EuroSAT Dataset class for satellite image classification
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FloodNetDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT satellite imagery.

    This dataset loads satellite images directly without dimensionality reduction,
    allowing the CNN to extract features from the raw images.

    Args:
        images (numpy.ndarray): Array of images with shape (N, H, W, C)
        labels (numpy.ndarray): Array of numeric labels with shape (N,)
        transform (callable, optional): Transform to apply to images
    """

    def __init__(self, images_paths, labels_paths, transform=None):
        """
        Initialize the dataset.

        Args:
            images_paths: List or array of image file paths
            labels_paths: List or array of corresponding label file paths
            transform: Optional torchvision transforms
        """
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.transform = transform

        # Default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transform()

    def __len__(self):
        """Return the total number of samples"""
        return len(self.images_paths)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, label) where image_tensor is a torch.Tensor
        """
        # Load image and label
        img_path = self.images_paths[idx]
        label_path = self.labels_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")  # Assuming label is a grayscale mask

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, mask

    @staticmethod
    def get_default_transform(image_size=(64, 64)):
        """
        Get default image transformations.

        This includes:
        1. Converting numpy array to PIL Image
        2. Resizing to target size
        3. Converting to tensor
        4. Normalizing with ImageNet statistics

        Args:
            image_size: Target image size (height, width)

        Returns:
            transforms.Compose: Composition of transforms
        """
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # ImageNet normalization (standard for pre-trained models)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_class_distribution(self):
        """
        Calculate the distribution of classes in the dataset.

        Returns:
            dict: Mapping from label to count
        """
        unique, counts = torch.unique(torch.tensor(self.labels_paths), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __repr__(self):
        """String representation of the dataset"""
        return (
            f"EuroSATDataset(samples={len(self)}, "
            f"image_shape={self.images_paths[0].shape}, "
            f"num_classes={len(set(self.labels_paths))})"
        )
