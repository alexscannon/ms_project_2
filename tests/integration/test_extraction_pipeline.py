"""
Integration tests for the full embedding extraction pipeline.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.embedding_extractor import DINOEmbeddingExtractor


class TestExtractionPipeline:
    """Integration tests for the full extraction pipeline."""

    def test_extract_save_load_roundtrip(self, mock_dino_model, synthetic_dataset, tmp_path):
        """Test: extract -> save -> load produces consistent results."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        # Extract
        embeddings, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset,
            batch_size=8,
            num_workers=0
        )

        # Save
        label_mappings = {
            'subclass_to_id': {f'class_{i}': i for i in range(5)},
            'id_to_subclass': {i: f'class_{i}' for i in range(5)},
            'superclass_to_id': {f'super_{i}': i for i in range(3)},
            'id_to_superclass': {i: f'super_{i}' for i in range(3)}
        }

        output_dir = extractor.save_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            label_mappings=label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        # Load
        loaded_embeddings = np.load(output_dir / "embeddings.npy")
        loaded_metadata = pd.read_csv(output_dir / "metadata.csv")

        # Verify consistency
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        assert len(loaded_metadata) == len(metadata)

    def test_embedding_metadata_index_alignment(self, mock_dino_model, synthetic_dataset, tmp_path):
        """Verify saved metadata 'index' column matches embedding array indices."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset,
            batch_size=8,
            num_workers=0
        )

        label_mappings = {
            'subclass_to_id': {},
            'id_to_subclass': {},
            'superclass_to_id': {},
            'id_to_superclass': {}
        }

        output_dir = extractor.save_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            label_mappings=label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        loaded_metadata = pd.read_csv(output_dir / "metadata.csv")

        # Verify index column is sequential starting from 0
        expected_indices = list(range(len(embeddings)))
        actual_indices = loaded_metadata['index'].tolist()
        assert expected_indices == actual_indices

    def test_batch_size_does_not_affect_results(self, mock_dino_model, synthetic_dataset_small):
        """Verify different batch sizes produce identical embeddings."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        # Extract with different batch sizes
        embeddings_bs1, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=1,
            num_workers=0
        )

        embeddings_bs4, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        embeddings_bs8, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=8,
            num_workers=0
        )

        # All should be identical
        np.testing.assert_allclose(embeddings_bs1, embeddings_bs4, rtol=1e-5)
        np.testing.assert_allclose(embeddings_bs1, embeddings_bs8, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    def test_gpu_cpu_embedding_consistency(self, synthetic_dataset_small):
        """Verify embeddings are consistent between GPU and CPU extraction."""
        from tests.conftest import MockDINOModel

        # CPU extraction
        model_cpu = MockDINOModel(embed_dim=768)
        model_cpu.eval()
        extractor_cpu = DINOEmbeddingExtractor(model=model_cpu, device='cpu')

        embeddings_cpu, _ = extractor_cpu.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # GPU extraction
        model_gpu = MockDINOModel(embed_dim=768)
        model_gpu = model_gpu.cuda()
        model_gpu.eval()
        extractor_gpu = DINOEmbeddingExtractor(model=model_gpu, device='cuda')

        embeddings_gpu, _ = extractor_gpu.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # Should be close (may have minor floating point differences)
        np.testing.assert_allclose(embeddings_cpu, embeddings_gpu, rtol=1e-4, atol=1e-6)


class TestDataLoaderIntegration:
    """Tests for DataLoader integration with dataset."""

    def test_dataloader_no_shuffle_preserves_order(self, synthetic_dataset):
        """Verify shuffle=False maintains consistent ordering."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            synthetic_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )

        # Collect metadata in order
        collected_indices = []
        for _, metadata_batch in dataloader:
            for path in metadata_batch['image_path']:
                # Extract index from image_path like "image_5.png"
                idx = int(path.split('_')[1].split('.')[0])
                collected_indices.append(idx)

        # Should be sequential
        expected = list(range(len(synthetic_dataset)))
        assert collected_indices == expected

    def test_dataloader_metadata_batch_structure(self, synthetic_dataset_small):
        """Verify batched metadata has dict-of-lists structure."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            synthetic_dataset_small,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        for images, metadata_batch in dataloader:
            # metadata_batch should be a dict where each value is a list/tensor
            assert isinstance(metadata_batch, dict)

            # Check expected keys
            expected_keys = {
                'subclass_id', 'subclass_name', 'superclass_id',
                'superclass_name', 'source', 'split', 'image_path'
            }
            assert expected_keys.issubset(set(metadata_batch.keys()))

            # Each value should have batch_size elements
            batch_size = images.shape[0]
            for key, value in metadata_batch.items():
                if isinstance(value, torch.Tensor):
                    assert len(value) == batch_size
                elif isinstance(value, (list, tuple)):
                    assert len(value) == batch_size

            break  # Only check first batch


class TestEmbeddingQuality:
    """Tests for embedding quality and validity."""

    def test_embeddings_are_not_all_identical(self, mock_dino_model, synthetic_dataset_small):
        """Verify different inputs produce different embeddings."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # Check that not all embeddings are identical
        unique_embeddings = np.unique(embeddings, axis=0)
        assert len(unique_embeddings) > 1, "All embeddings are identical"

    def test_embedding_norms_are_reasonable(self, mock_dino_model, synthetic_dataset_small):
        """Verify embedding norms are within reasonable range."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        norms = np.linalg.norm(embeddings, axis=1)

        # Norms should be finite and positive
        assert np.all(np.isfinite(norms))
        assert np.all(norms > 0)

        # Norms shouldn't be extremely large or small (sanity check)
        assert norms.mean() < 1000, "Embedding norms are unusually large"

    def test_embeddings_have_dimension_variance(self, mock_dino_model, synthetic_dataset_small):
        """Verify embedding dimensions have variance (not collapsed)."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # Check variance per dimension
        dim_variances = embeddings.var(axis=0)

        # Most dimensions should have some variance
        zero_variance_dims = np.sum(dim_variances == 0)
        total_dims = embeddings.shape[1]

        # Allow up to 10% zero-variance dimensions
        assert zero_variance_dims / total_dims < 0.1, \
            f"{zero_variance_dims}/{total_dims} dimensions have zero variance"


class TestMultipleExtractions:
    """Tests for consistency across multiple extractions."""

    def test_repeated_extraction_is_deterministic(self, mock_dino_model, synthetic_dataset_small):
        """Verify repeated extractions produce identical results."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        # Multiple extractions
        results = []
        for _ in range(3):
            embeddings, _ = extractor.extract_embeddings(
                dataset=synthetic_dataset_small,
                batch_size=4,
                num_workers=0
            )
            results.append(embeddings)

        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
