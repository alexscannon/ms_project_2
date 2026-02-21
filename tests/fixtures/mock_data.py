"""
Mock data generators for testing without real DINO model or CIFAR-100 data.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def create_mock_genai_structure(base_path: Path, n_images_per_class: int = 3) -> Path:
    """
    Create mock GenAI directory structure with tiny images.

    Structure:
    base_path/
        novel_subclasses/
            aquatic_mammals/
                new_aquatic_class/
                    img_0.png, img_1.png, ...
        novel_superclasses/
            mythical_creatures/
                dragon/
                    img_0.png, ...

    Args:
        base_path: Root directory to create structure in
        n_images_per_class: Number of images to create per class

    Returns:
        The base_path for chaining
    """
    # Novel subclasses (existing CIFAR superclasses)
    novel_sub = base_path / "novel_subclasses"
    superclasses = ["aquatic_mammals", "fish", "flowers"]

    for sc in superclasses:
        subclass_dir = novel_sub / sc / f"new_{sc}_class"
        subclass_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_images_per_class):
            # Create small colored images
            img = Image.new('RGB', (64, 64), color=(i * 50 % 256, i * 30 % 256, i * 20 % 256))
            img.save(subclass_dir / f"img_{i}.png")

    # Novel superclasses (entirely new)
    novel_super = base_path / "novel_superclasses"
    new_superclasses = ["mythical_creatures"]

    for sc in new_superclasses:
        subclass_dir = novel_super / sc / "dragon"
        subclass_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_images_per_class):
            img = Image.new('RGB', (64, 64), color=(i * 40 % 256, i * 60 % 256, i * 80 % 256))
            img.save(subclass_dir / f"img_{i}.png")

    return base_path


def generate_semantic_embeddings(
    n_samples: int,
    n_classes: int,
    embed_dim: int = 768,
    intra_class_std: float = 0.1,
    inter_class_dist: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic embeddings with class structure.

    Creates embeddings where samples from the same class cluster together.
    Useful for testing semantic validity without real DINO model.

    Args:
        n_samples: Total number of samples to generate
        n_classes: Number of distinct classes
        embed_dim: Embedding dimensionality
        intra_class_std: Standard deviation within classes (smaller = tighter clusters)
        inter_class_dist: Distance scale between class centroids
        seed: Random seed for reproducibility

    Returns:
        embeddings: (n_samples, embed_dim) array of L2-normalized embeddings
        labels: (n_samples,) array of class indices
    """
    np.random.seed(seed)

    # Generate class centroids
    centroids = np.random.randn(n_classes, embed_dim) * inter_class_dist

    # Normalize centroids (cosine similarity friendly)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Generate samples around centroids
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    embeddings = []
    labels = []

    for class_idx in range(n_classes):
        # Add extra sample to first classes if there's remainder
        n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)

        class_samples = centroids[class_idx] + np.random.randn(
            n_class_samples, embed_dim
        ) * intra_class_std
        embeddings.append(class_samples)
        labels.extend([class_idx] * n_class_samples)

    embeddings = np.vstack(embeddings).astype(np.float32)
    labels = np.array(labels)

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings, labels


def generate_degenerate_embeddings(n_samples: int, embed_dim: int = 768) -> np.ndarray:
    """
    Generate degenerate embeddings for testing failure detection.

    Creates embeddings that are all zeros (collapsed representation).

    Args:
        n_samples: Number of samples
        embed_dim: Embedding dimensionality

    Returns:
        Zero array of shape (n_samples, embed_dim)
    """
    return np.zeros((n_samples, embed_dim), dtype=np.float32)


def generate_embeddings_with_nans(
    n_samples: int,
    embed_dim: int = 768,
    nan_fraction: float = 0.01,
    seed: int = 42
) -> np.ndarray:
    """
    Generate embeddings with some NaN values for testing validation.

    Args:
        n_samples: Number of samples
        embed_dim: Embedding dimensionality
        nan_fraction: Fraction of values to set as NaN
        seed: Random seed

    Returns:
        Embeddings with some NaN values
    """
    np.random.seed(seed)
    embeddings = np.random.randn(n_samples, embed_dim).astype(np.float32)

    # Randomly set some values to NaN
    n_nans = int(n_samples * embed_dim * nan_fraction)
    nan_indices = np.random.choice(n_samples * embed_dim, n_nans, replace=False)
    flat = embeddings.flatten()
    flat[nan_indices] = np.nan
    embeddings = flat.reshape(n_samples, embed_dim)

    return embeddings
