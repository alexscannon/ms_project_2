"""
Unit tests for DINOEmbeddingExtractor class.
"""

import json
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.embedding_extractor import DINOEmbeddingExtractor


class TestDINOEmbeddingExtractorInit:
    """Tests for DINOEmbeddingExtractor initialization."""

    def test_init_with_mock_model_cpu(self, mock_dino_model):
        """Verify extractor initializes correctly with mock model on CPU."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        assert extractor.device == torch.device('cpu')
        assert extractor.embedding_dim == 768
        assert extractor.model is mock_dino_model

    def test_init_with_small_model(self, mock_dino_model_small):
        """Verify extractor correctly detects embedding dimension."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model_small, device='cpu')

        assert extractor.embedding_dim == 384

    def test_init_device_auto_detection(self, mock_dino_model):
        """Verify device auto-detection when device=None."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device=None)

        # Should default to CUDA if available, else CPU
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert extractor.device == expected_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_with_cuda_device(self, mock_dino_model):
        """Verify extractor initializes correctly with CUDA device."""
        mock_dino_model = mock_dino_model.cuda()
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cuda')

        assert extractor.device == torch.device('cuda')


class TestExtractEmbeddings:
    """Tests for the extract_embeddings method."""

    def test_output_shape_matches_dataset_size(self, mock_dino_model, synthetic_dataset):
        """Verify embeddings shape is (N, embedding_dim) where N = dataset size."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset,
            batch_size=8,
            num_workers=0
        )

        assert embeddings.shape == (len(synthetic_dataset), 768)

    def test_output_shape_with_different_batch_sizes(self, mock_dino_model, synthetic_dataset_small):
        """Verify output shape is consistent regardless of batch_size."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        for batch_size in [1, 2, 4, 8]:
            embeddings, _ = extractor.extract_embeddings(
                dataset=synthetic_dataset_small,
                batch_size=batch_size,
                num_workers=0
            )
            assert embeddings.shape == (len(synthetic_dataset_small), 768)

    def test_no_nan_values_in_embeddings(self, mock_dino_model, synthetic_dataset_small):
        """Verify no NaN values in extracted embeddings."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"

    def test_no_inf_values_in_embeddings(self, mock_dino_model, synthetic_dataset_small):
        """Verify no Inf values in extracted embeddings."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        assert not np.isinf(embeddings).any(), "Embeddings contain Inf values"

    def test_embeddings_have_nonzero_variance(self, mock_dino_model, synthetic_dataset_small):
        """Verify embeddings are not all zeros or constant (not collapsed)."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        assert embeddings.std() > 0, "Embeddings have zero variance (collapsed)"

    def test_metadata_length_matches_embeddings(self, mock_dino_model, synthetic_dataset):
        """Verify metadata list has same length as embeddings."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset,
            batch_size=8,
            num_workers=0
        )

        assert len(metadata) == embeddings.shape[0]

    def test_metadata_contains_required_keys(self, mock_dino_model, synthetic_dataset_small):
        """Verify each metadata dict contains required keys."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        _, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        required_keys = {
            'subclass_id', 'subclass_name', 'superclass_id',
            'superclass_name', 'source', 'split', 'image_path'
        }

        for i, meta in enumerate(metadata):
            assert required_keys.issubset(meta.keys()), \
                f"Metadata at index {i} missing keys: {required_keys - set(meta.keys())}"

    def test_embedding_metadata_alignment(self, mock_dino_model, synthetic_dataset_small):
        """Verify embeddings and metadata are correctly aligned (same order)."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        _, metadata = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # Verify metadata order matches dataset order
        for i in range(len(synthetic_dataset_small)):
            expected_meta = synthetic_dataset_small.samples[i]
            actual_meta = metadata[i]

            assert actual_meta['subclass_id'] == expected_meta['subclass_id'], \
                f"Misaligned at index {i}: expected {expected_meta['subclass_id']}, got {actual_meta['subclass_id']}"
            assert actual_meta['image_path'] == expected_meta['image_path'], \
                f"Misaligned at index {i}: image_path mismatch"

    def test_determinism_same_seed(self, mock_dino_model, synthetic_dataset_small):
        """Verify same embeddings with same random seed."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        # First extraction
        torch.manual_seed(42)
        embeddings1, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        # Second extraction
        torch.manual_seed(42)
        embeddings2, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)

    def test_embeddings_dtype_is_float(self, mock_dino_model, synthetic_dataset_small):
        """Verify embeddings are returned as float32."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        embeddings, _ = extractor.extract_embeddings(
            dataset=synthetic_dataset_small,
            batch_size=4,
            num_workers=0
        )

        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64


class TestSaveEmbeddings:
    """Tests for the save_embeddings method."""

    def test_creates_output_directory(self, mock_dino_model, sample_embeddings,
                                      sample_metadata, sample_label_mappings, tmp_path):
        """Verify output directory is created."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        assert output_dir.exists()

    def test_saves_embeddings_npy_file(self, mock_dino_model, sample_embeddings,
                                        sample_metadata, sample_label_mappings, tmp_path):
        """Verify embeddings.npy is created with correct content."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        embeddings_path = output_dir / "embeddings.npy"
        assert embeddings_path.exists()

        loaded_embeddings = np.load(embeddings_path)
        np.testing.assert_array_equal(loaded_embeddings, sample_embeddings)

    def test_saved_embeddings_shape_preserved(self, mock_dino_model, sample_embeddings,
                                               sample_metadata, sample_label_mappings, tmp_path):
        """Verify saved embeddings maintain original shape."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        loaded_embeddings = np.load(output_dir / "embeddings.npy")
        assert loaded_embeddings.shape == sample_embeddings.shape

    def test_saves_metadata_csv_file(self, mock_dino_model, sample_embeddings,
                                      sample_metadata, sample_label_mappings, tmp_path):
        """Verify metadata.csv is created."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        metadata_path = output_dir / "metadata.csv"
        assert metadata_path.exists()

    def test_metadata_csv_has_correct_columns(self, mock_dino_model, sample_embeddings,
                                               sample_metadata, sample_label_mappings, tmp_path):
        """Verify CSV has expected columns."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        df = pd.read_csv(output_dir / "metadata.csv")
        expected_columns = {
            'index', 'subclass_id', 'subclass_name', 'superclass_id',
            'superclass_name', 'source', 'split', 'image_path'
        }
        assert expected_columns.issubset(set(df.columns))

    def test_metadata_csv_row_count(self, mock_dino_model, sample_embeddings,
                                     sample_metadata, sample_label_mappings, tmp_path):
        """Verify CSV has correct number of rows."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        df = pd.read_csv(output_dir / "metadata.csv")
        assert len(df) == len(sample_metadata)

    def test_saves_label_mappings_json(self, mock_dino_model, sample_embeddings,
                                        sample_metadata, sample_label_mappings, tmp_path):
        """Verify label_mappings.json is created."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        mappings_path = output_dir / "label_mappings.json"
        assert mappings_path.exists()

        with open(mappings_path) as f:
            loaded_mappings = json.load(f)

        # Check structure is preserved (keys converted to strings in JSON)
        assert 'subclass_to_id' in loaded_mappings
        assert 'superclass_to_id' in loaded_mappings

    def test_tensor_values_sanitized_in_metadata(self, mock_dino_model, sample_embeddings,
                                                   sample_metadata_with_tensors,
                                                   sample_label_mappings, tmp_path):
        """Verify torch.Tensor values are converted to Python scalars."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata_with_tensors,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        df = pd.read_csv(output_dir / "metadata.csv")

        # Check that values are proper integers, not tensor representations
        assert df['subclass_id'].dtype in [np.int64, np.int32, int]
        assert df['superclass_id'].dtype in [np.int64, np.int32, int]

    def test_returns_output_directory_path(self, mock_dino_model, sample_embeddings,
                                            sample_metadata, sample_label_mappings, tmp_path):
        """Verify method returns the created output directory."""
        extractor = DINOEmbeddingExtractor(model=mock_dino_model, device='cpu')

        output_dir = extractor.save_embeddings(
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            label_mappings=sample_label_mappings,
            output_dir=tmp_path / "embeddings"
        )

        assert isinstance(output_dir, Path)
        assert output_dir.exists()
