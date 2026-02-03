"""
Test suite for DataLoader and Dataset functionality
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import FloodNetDataset, MaskToLong
from data.loader import create_dataloaders, load_images_path
from config import settings


class TestFloodNetDataset:
    """Test FloodNetDataset class"""
    
    @pytest.fixture
    def sample_paths(self):
        """Get sample image and label paths from actual dataset"""
        try:
            image_files, label_files = load_images_path(
                data_dir=Path("FloodNet"),
                max_per_class=10  # Only load a few samples for testing
            )
            return image_files[:5], label_files[:5]  # Use only 5 samples
        except FileNotFoundError:
            pytest.skip("FloodNet dataset not found. Please download the dataset first.")
    
    def test_dataset_initialization(self, sample_paths):
        """Test that dataset can be initialized"""
        image_paths, label_paths = sample_paths
        dataset = FloodNetDataset(image_paths, label_paths)
        
        assert len(dataset) == len(image_paths)
        assert dataset.images_paths == image_paths
        assert dataset.labels_paths == label_paths
    
    def test_dataset_getitem(self, sample_paths):
        """Test that __getitem__ returns correct tensor shapes and types"""
        image_paths, label_paths = sample_paths
        dataset = FloodNetDataset(
            image_paths, 
            label_paths,
            image_transforms=FloodNetDataset.get_image_transform(image_size=(64, 64)),
            label_transforms=FloodNetDataset.get_label_transform(image_size=(64, 64))
        )
        
        img, mask = dataset[0]
        
        # Check image tensor properties
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert img.shape == (3, 64, 64), f"Image shape should be (3, 64, 64), got {img.shape}"
        assert img.dtype == torch.float32, f"Image should be float32, got {img.dtype}"
        
        # Check mask tensor properties
        assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
        assert mask.shape == (64, 64), f"Mask shape should be (64, 64), got {mask.shape}"
        assert mask.dtype == torch.int64, f"Mask should be int64 (long), got {mask.dtype}"
        
        # Check value ranges
        assert mask.min() >= 0, "Mask values should be >= 0"
        assert mask.max() < len(settings.CLASS_MAPPING), f"Mask max value should be < {len(settings.CLASS_MAPPING)}"
    
    def test_image_transform(self):
        """Test that image transform produces correct output"""
        transform = FloodNetDataset.get_image_transform(image_size=(64, 64))
        
        # Create dummy PIL image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        
        transformed = transform(dummy_img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 64, 64)
        assert transformed.dtype == torch.float32
    
    def test_label_transform(self):
        """Test that label transform produces integer tensor"""
        transform = FloodNetDataset.get_label_transform(image_size=(64, 64))
        
        # Create dummy PIL mask
        from PIL import Image
        import numpy as np
        dummy_mask = Image.fromarray(np.random.randint(0, 10, (128, 128), dtype=np.uint8))
        
        transformed = transform(dummy_mask)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (64, 64), f"Expected (64, 64), got {transformed.shape}"
        assert transformed.dtype == torch.int64, f"Expected int64, got {transformed.dtype}"
        assert len(transformed.shape) == 2, "Mask should be 2D (no channel dimension)"
    
    def test_mask_to_long_transform(self):
        """Test MaskToLong custom transform"""
        transform = MaskToLong()
        
        # Create a dummy tensor with channel dimension
        dummy_tensor = torch.randint(0, 10, (1, 64, 64), dtype=torch.uint8)
        
        transformed = transform(dummy_tensor)
        
        assert transformed.shape == (64, 64), "Channel dimension should be removed"
        assert transformed.dtype == torch.int64, "Should be converted to long"


class TestDataLoaders:
    """Test DataLoader creation and functionality"""
    
    @pytest.fixture
    def sample_paths(self):
        """Get sample image and label paths from actual dataset"""
        try:
            image_files, label_files = load_images_path(
                data_dir=Path("FloodNet"),
                max_per_class=50  # Load more samples for dataloader tests
            )
            return image_files, label_files
        except FileNotFoundError:
            pytest.skip("FloodNet dataset not found. Please download the dataset first.")
    
    def test_create_dataloaders(self, sample_paths):
        """Test that dataloaders are created successfully"""
        image_paths, label_paths = sample_paths
        
        train_loader, val_loader, test_loader, label_map = create_dataloaders(
            image_paths, 
            label_paths, 
            settings
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(label_map, dict)
    
    def test_dataloader_batch_iteration(self, sample_paths):
        """Test that we can iterate through dataloader batches"""
        image_paths, label_paths = sample_paths
        
        # Create a simple dataset for testing
        dataset = FloodNetDataset(
            image_paths[:20],  # Use only 20 samples
            label_paths[:20],
            image_transforms=FloodNetDataset.get_image_transform(image_size=(64, 64)),
            label_transforms=FloodNetDataset.get_label_transform(image_size=(64, 64))
        )
        
        # Create dataloader with num_workers=0 to avoid pickling issues in tests
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        # Get first batch
        images, masks = next(iter(dataloader))
        
        # Check batch shapes
        assert images.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {images.shape}"
        assert masks.shape == (4, 64, 64), f"Expected (4, 64, 64), got {masks.shape}"
        
        # Check batch types
        assert images.dtype == torch.float32
        assert masks.dtype == torch.int64
        
        # Check value ranges
        assert masks.min() >= 0
        assert masks.max() < len(settings.CLASS_MAPPING)
    
    def test_dataloader_split_sizes(self, sample_paths):
        """Test that dataset splits have correct sizes"""
        image_paths, label_paths = sample_paths
        
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            image_paths, 
            label_paths, 
            settings
        )
        
        total_batches = len(train_loader) + len(val_loader) + len(test_loader)
        assert total_batches > 0, "Should have at least one batch across all loaders"
        
        # Verify that train set is larger than val and test
        assert len(train_loader) >= len(val_loader), "Train set should be larger than val set"
        assert len(train_loader) >= len(test_loader), "Train set should be larger than test set"
    
    def test_dataloader_no_data_leakage(self, sample_paths):
        """Test that train/val/test splits don't overlap (basic check)"""
        image_paths, label_paths = sample_paths
        
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            image_paths, 
            label_paths, 
            settings
        )
        
        # Calculate total samples
        total_train = len(train_loader.dataset)
        total_val = len(val_loader.dataset)
        total_test = len(test_loader.dataset)
        total = total_train + total_val + total_test
        
        # Should equal original dataset size
        assert total == len(image_paths), "Split sizes should sum to original dataset size"


class TestTransformPipeline:
    """Test the complete transform pipeline"""
    
    def test_image_normalization(self):
        """Test that images are properly normalized"""
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        
        transform = FloodNetDataset.get_image_transform(image_size=(64, 64))
        transformed = transform(dummy_img)
        
        # After normalization with ImageNet stats, values should be roughly in [-3, 3] range
        assert transformed.min() >= -5, "Normalized values too low"
        assert transformed.max() <= 5, "Normalized values too high"
    
    def test_label_no_normalization(self):
        """Test that labels are NOT normalized (should remain as class indices)"""
        from PIL import Image
        import numpy as np
        
        # Create a dummy label mask with specific class values
        class_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        dummy_mask = Image.fromarray(
            np.random.choice(class_values, (128, 128)).astype(np.uint8)
        )
        
        transform = FloodNetDataset.get_label_transform(image_size=(64, 64))
        transformed = transform(dummy_mask)
        
        # Values should still be in the original class range
        unique_values = torch.unique(transformed)
        assert all(v in class_values for v in unique_values.tolist()), \
            "Label values should remain as original class indices"
    
    def test_resize_consistency(self):
        """Test that resize produces consistent output sizes"""
        from PIL import Image
        import numpy as np
        
        sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        for size in sizes:
            dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
            dummy_mask = Image.fromarray(np.random.randint(0, 10, (300, 400), dtype=np.uint8))
            
            img_transform = FloodNetDataset.get_image_transform(image_size=size)
            mask_transform = FloodNetDataset.get_label_transform(image_size=size)
            
            img_transformed = img_transform(dummy_img)
            mask_transformed = mask_transform(dummy_mask)
            
            assert img_transformed.shape == (3, size[0], size[1]), \
                f"Image size mismatch for {size}"
            assert mask_transformed.shape == (size[0], size[1]), \
                f"Mask size mismatch for {size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
