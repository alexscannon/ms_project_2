"""
Shared pytest fixtures for embedding validation tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Dict, Tuple, Any


class MockDINOModel(nn.Module):
    """
    Lightweight mock of DINO model for testing.

    Mimics the DINO model interface with:
    - embed_dim attribute
    - forward() that returns (batch_size, embed_dim) embeddings
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple linear layer for deterministic output
        self.linear = nn.Linear(3 * 224 * 224, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning embeddings.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Embeddings of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]
        # Flatten and take first embed_dim*input_features for linear layer
        x_flat = x.view(batch_size, -1)
        # Pad or truncate to expected input size
        expected_features = 3 * 224 * 224
        if x_flat.shape[1] < expected_features:
            padding = torch.zeros(batch_size, expected_features - x_flat.shape[1], device=x.device)
            x_flat = torch.cat([x_flat, padding], dim=1)
        else:
            x_flat = x_flat[:, :expected_features]
        return self.linear(x_flat)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing embedding extraction.

    Generates deterministic random images with known metadata structure.
    """

    def __init__(self, n_samples: int = 50, n_classes: int = 5, image_size: int = 224):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.image_size = image_size
        self.samples = []

        for i in range(n_samples):
            class_id = i % n_classes
            superclass_id = class_id // 2
            self.samples.append({
                'subclass_id': class_id,
                'subclass_name': f'class_{class_id}',
                'superclass_id': superclass_id,
                'superclass_name': f'super_{superclass_id}',
                'source': 'cifar100' if i < n_samples // 2 else 'genai_novel_subclass',
                'split': 'train' if i < n_samples * 0.8 else 'test',
                'image_path': f'image_{i}.png'
            })

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get item with deterministic random image based on index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, metadata_dict)
        """
        # Generate deterministic random image based on index
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)
        metadata = self.samples[idx].copy()
        return image, metadata


@pytest.fixture
def mock_dino_model():
    """Create a mock DINO model with embed_dim=768."""
    model = MockDINOModel(embed_dim=768)
    model.eval()
    return model


@pytest.fixture
def mock_dino_model_small():
    """Create a smaller mock DINO model (embed_dim=384) for testing."""
    model = MockDINOModel(embed_dim=384)
    model.eval()
    return model


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset with 50 samples."""
    return SyntheticDataset(n_samples=50, n_classes=5)


@pytest.fixture
def synthetic_dataset_small():
    """Create a very small synthetic dataset for quick tests."""
    return SyntheticDataset(n_samples=10, n_classes=2)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing save functionality."""
    np.random.seed(42)
    return np.random.randn(100, 768).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Generate sample metadata list for testing save functionality."""
    return [
        {
            'subclass_id': i % 10,
            'subclass_name': f'class_{i % 10}',
            'superclass_id': i % 5,
            'superclass_name': f'super_{i % 5}',
            'source': 'cifar100' if i < 50 else 'genai_novel_subclass',
            'split': 'train',
            'image_path': f'img_{i}.png'
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_metadata_with_tensors():
    """
    Generate sample metadata with torch.Tensor values.

    Used to test sanitization of tensor values to Python scalars.
    """
    return [
        {
            'subclass_id': torch.tensor(i % 10),
            'subclass_name': f'class_{i % 10}',
            'superclass_id': torch.tensor(i % 5),
            'superclass_name': f'super_{i % 5}',
            'source': 'cifar100' if i < 50 else 'genai_novel_subclass',
            'split': 'train',
            'image_path': f'img_{i}.png'
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_label_mappings():
    """Generate sample label mappings for testing."""
    return {
        'subclass_to_id': {f'class_{i}': i for i in range(10)},
        'id_to_subclass': {i: f'class_{i}' for i in range(10)},
        'superclass_to_id': {f'super_{i}': i for i in range(5)},
        'id_to_superclass': {i: f'super_{i}' for i in range(5)}
    }


@pytest.fixture
def dino_transform():
    """Standard DINO preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(32),  # CIFAR size
        transforms.Resize(224),  # DINO input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory for save tests."""
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def device():
    """Get appropriate device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
