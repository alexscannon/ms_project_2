import os
import json
from pathlib import Path

from typing import List, Dict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging

from src.utils import create_next_experiment_dir


logger = logging.getLogger("msproject")


class DINOEmbeddingExtractor:
    """
    Extract embeddings using DINOv3 model.
    """
    def __init__(self, model: torch.nn.Module, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.embedding_dim = self.model.embed_dim

        logger.info(f"Embedding Extractor successfully loaded on {self.device}...")
        logger.info(f"Embedding dimension: {self.embedding_dim}")


    @torch.no_grad()
    def extract_embeddings(self, dataset, batch_size=32, num_workers=4):
        """
        Extract embeddings for entire dataset.

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size for inference
            num_workers: DataLoader workers

        Returns:
            embeddings: np.array of shape (N, embedding_dim)
            metadata_list: List of metadata dicts
        """
        print(f"\n {'=' * 60} \n Extracting embeddings...\n {'=' * 60}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        all_embeddings = []
        all_metadata = []

        logger.info(f"Extracting embeddings for {len(dataset)} images...")
        for images, metadata_batch in tqdm(dataloader, desc="Processing batches"):
            images = images.to(self.device) # move images to same device as encoder

            embeddings = self.model(images) # Forward pass - get CLS token
            all_embeddings.append(embeddings.cpu().numpy()) # Move to CPU and store

            if isinstance(metadata_batch, dict):
                keys = list(metadata_batch.keys())
                # zip(*values) iterates through the batch dimension of all values simultaneously
                batch_list = [
                    dict(zip(keys, values))
                    for values in zip(*metadata_batch.values())
                ]
                all_metadata.extend(batch_list)
            else:
                # Fallback if metadata is just a list (e.g. custom collate)
                all_metadata.extend(metadata_batch)

        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)

        return embeddings, all_metadata

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict],
        label_mappings: Dict,
        output_dir: Path = Path('/home/alex/data/embeddings'),
    ):
        """
        Save embeddings and metadata.

        Args:
            embeddings: (N, D) array
            metadata_list: List of metadata dicts
            label_mappings: Dict with label mappings
            output_dir: Output directory
        """
        print(f"\n {'=' * 60} \n Saving results...\n {'=' * 60}")

        output_dir = create_next_experiment_dir(base_dir=output_dir, prefix="set_")

        # Save embeddings
        embeddings_filename = f"embeddings.npy"
        embeddings_path = output_dir / embeddings_filename
        np.save(embeddings_path, embeddings)

        print(f"Saved embeddings to {embeddings_path}")
        print(f"  Shape: {embeddings.shape}")

        # 2. Save Metadata (.csv)
        # Helper to convert Tensors to Python scalars
        def sanitize(val):
            if hasattr(val, 'item'):
                return val.item()
            return val

        # Create DataFrame robustly
        processed_metadata = []
        for i, meta in enumerate(metadata_list):
            processed_metadata.append({
                'index': i,
                'subclass_id': sanitize(meta.get('subclass_id')),
                'subclass_name': meta.get('subclass_name'),
                'superclass_id': sanitize(meta.get('superclass_id')),
                'superclass_name': meta.get('superclass_name'),
                'source': meta.get('source'),
                'split': meta.get('split'),
                'image_path': str(meta.get('image_path', ''))
            })

        metadata_df = pd.DataFrame(processed_metadata)

        metadata_filename = f"metadata.csv"
        metadata_path = output_dir / metadata_filename
        metadata_df.to_csv(metadata_path, index=False)

        logger.info(f"Saved metadata to {metadata_path}...")

        # Save label mappings as JSON
        mappings_path = output_dir / f"label_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(label_mappings, f, indent=2)
        logger.info(f"Saved label mappings to {mappings_path}")

        print(f"\n {'=' * 60} \n Successfully extracted embeddings...\n {'=' * 60}")

        return output_dir