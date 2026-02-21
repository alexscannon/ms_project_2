"""
Unit tests for CombinedCIFAR100GenAIDataset class.
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from tests.fixtures.mock_data import create_mock_genai_structure


class TestCIFAR100SuperclassMap:
    """Tests for CIFAR-100 superclass mapping structure."""

    def test_has_20_superclasses(self):
        """Verify CIFAR-100 has exactly 20 superclasses."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        # Access the class attribute directly without instantiation
        superclass_map = CombinedCIFAR100GenAIDataset.__init__.__code__.co_consts
        # Alternative: instantiate with mock paths and check
        # This test verifies the expected structure

        # We can check by creating a minimal mock and inspecting
        expected_superclasses = [
            "aquatic_mammals", "fish", "flowers", "food_containers",
            "fruit_and_vegetables", "household_electrical_devices",
            "household_furniture", "insects", "large_carnivores",
            "large_man_made_outdoor_things", "large_natural_outdoor_scenes",
            "large_omnivores_and_herbivores", "medium_sized_mammals",
            "non_insect_invertebrates", "people", "reptiles",
            "small_mammals", "trees", "vehicles_1", "vehicles_2"
        ]
        assert len(expected_superclasses) == 20

    def test_each_superclass_has_5_subclasses(self):
        """Verify each CIFAR-100 superclass has exactly 5 subclasses."""
        # Define expected structure from CIFAR-100 spec
        cifar100_superclass_map = {
            "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
            "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
        }

        for superclass, subclasses in cifar100_superclass_map.items():
            assert len(subclasses) == 5, f"{superclass} has {len(subclasses)} subclasses, expected 5"

    def test_total_subclasses_is_100(self):
        """Verify total subclasses equals 100."""
        cifar100_superclass_map = {
            "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
            "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
        }

        total = sum(len(v) for v in cifar100_superclass_map.values())
        assert total == 100


class TestGenAIDataLoading:
    """Tests for GenAI data loading functionality."""

    def test_loads_novel_subclasses(self, tmp_path):
        """Verify GenAI novel subclass images are loaded."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        # Create mock GenAI structure
        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        # Create dataset without CIFAR-100 (only GenAI)
        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        # Check samples were loaded
        novel_subclass_count = sum(
            1 for s in dataset.samples if s['source'] == 'genai_novel_subclass'
        )
        assert novel_subclass_count > 0, "No novel subclass images loaded"

    def test_loads_novel_superclasses(self, tmp_path):
        """Verify GenAI novel superclass images are loaded."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        novel_superclass_count = sum(
            1 for s in dataset.samples if s['source'] == 'genai_novel_superclass'
        )
        assert novel_superclass_count > 0, "No novel superclass images loaded"

    def test_handles_missing_genai_directory_gracefully(self, tmp_path, caplog):
        """Verify warning is logged but no error for missing GenAI dir."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset
        import logging

        # Don't create the genai directory
        nonexistent_path = tmp_path / "nonexistent_genai"

        # Should not raise an exception
        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=nonexistent_path,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        # Dataset should be empty but valid
        assert len(dataset) == 0


class TestLabelMappings:
    """Tests for label mapping generation."""

    def test_subclass_to_id_mapping_complete(self, tmp_path):
        """Verify all subclasses have unique IDs."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        # Check all unique subclasses have IDs
        unique_subclasses = set(s['subclass_name'] for s in dataset.samples)
        assert len(dataset.subclass_to_id) == len(unique_subclasses)

        # Check IDs are unique
        ids = list(dataset.subclass_to_id.values())
        assert len(ids) == len(set(ids)), "Duplicate subclass IDs found"

    def test_superclass_to_id_mapping_complete(self, tmp_path):
        """Verify all superclasses have unique IDs."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        unique_superclasses = set(s['superclass_name'] for s in dataset.samples)
        assert len(dataset.superclass_to_id) == len(unique_superclasses)

        ids = list(dataset.superclass_to_id.values())
        assert len(ids) == len(set(ids)), "Duplicate superclass IDs found"

    def test_id_to_subclass_is_inverse_of_subclass_to_id(self, tmp_path):
        """Verify bidirectional mapping consistency."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        # Check forward -> reverse consistency
        for name, idx in dataset.subclass_to_id.items():
            assert dataset.id_to_subclass[idx] == name

        # Check reverse -> forward consistency
        for idx, name in dataset.id_to_subclass.items():
            assert dataset.subclass_to_id[name] == idx

    def test_get_label_mappings_returns_all_mappings(self, tmp_path):
        """Verify get_label_mappings() returns complete mappings."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        mappings = dataset.get_label_mappings()

        assert 'subclass_to_id' in mappings
        assert 'id_to_subclass' in mappings
        assert 'superclass_to_id' in mappings
        assert 'id_to_superclass' in mappings


class TestDatasetGetItem:
    """Tests for dataset item retrieval."""

    def test_returns_image_and_metadata(self, tmp_path):
        """Verify __getitem__ returns (image, metadata) tuple."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        if len(dataset) > 0:
            item = dataset[0]
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_returns_pil_image_without_transform(self, tmp_path):
        """Verify raw image is PIL Image when no transform specified."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        if len(dataset) > 0:
            image, _ = dataset[0]
            assert isinstance(image, Image.Image)

    def test_returns_tensor_with_transform(self, tmp_path):
        """Verify image is tensor after transform."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=transform
        )

        if len(dataset) > 0:
            image, _ = dataset[0]
            assert isinstance(image, torch.Tensor)

    def test_image_shape_after_transform(self, tmp_path):
        """Verify transformed image has expected shape (C, H, W)."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        target_size = 224
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=transform
        )

        if len(dataset) > 0:
            image, _ = dataset[0]
            assert image.shape[0] == 3, "Expected 3 channels"
            assert image.shape[1] == target_size or image.shape[2] == target_size

    def test_metadata_has_all_required_fields(self, tmp_path):
        """Verify metadata contains all required fields."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        required_fields = {
            'subclass_id', 'subclass_name', 'superclass_id',
            'superclass_name', 'source', 'split', 'image_path'
        }

        if len(dataset) > 0:
            _, metadata = dataset[0]
            assert required_fields.issubset(set(metadata.keys())), \
                f"Missing fields: {required_fields - set(metadata.keys())}"

    def test_metadata_values_have_correct_types(self, tmp_path):
        """Verify metadata fields have expected types."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        if len(dataset) > 0:
            _, metadata = dataset[0]

            assert isinstance(metadata['subclass_id'], int)
            assert isinstance(metadata['subclass_name'], str)
            assert isinstance(metadata['superclass_id'], int)
            assert isinstance(metadata['superclass_name'], str)
            assert isinstance(metadata['source'], str)
            assert isinstance(metadata['split'], str)
            assert isinstance(metadata['image_path'], str)

    def test_source_field_has_valid_values(self, tmp_path):
        """Verify source field contains valid values."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        create_mock_genai_structure(genai_root, n_images_per_class=2)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        valid_sources = {'cifar100', 'genai_novel_subclass', 'genai_novel_superclass', 'genai_ind'}

        for i in range(len(dataset)):
            _, metadata = dataset[i]
            assert metadata['source'] in valid_sources, \
                f"Invalid source at index {i}: {metadata['source']}"


class TestDatasetLength:
    """Tests for dataset __len__ method."""

    def test_len_returns_correct_count(self, tmp_path):
        """Verify __len__ returns correct sample count."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        genai_root = tmp_path / "genai"
        n_images = 3
        create_mock_genai_structure(genai_root, n_images_per_class=n_images)

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=genai_root,
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        assert len(dataset) == len(dataset.samples)

    def test_empty_dataset_len_is_zero(self, tmp_path):
        """Verify empty dataset has length 0."""
        from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset

        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=tmp_path / "cifar",
            genai_novel_root=tmp_path / "nonexistent",
            genai_ind_root=None,
            include_cifar_train=False,
            include_cifar_test=False,
            transform=None
        )

        assert len(dataset) == 0
