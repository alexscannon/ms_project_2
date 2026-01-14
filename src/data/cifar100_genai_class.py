import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms, datasets
import logging


logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger("msproject")


class CombinedCIFAR100GenAIDataset(Dataset):
    """
    Unified dataset combining CIFAR-100 and GenAI images.

    Applies identical preprocessing to both sources.
    Returns images with rich metadata for later analysis.

    Attributes:
        cifar100_superclass_map: Mapping of CIFAR-100 superclasses to their subclasses.
        subclass_to_superclass: Reverse mapping from subclass to superclass.
        samples: List of all loaded samples with metadata.
        subclass_to_id: Mapping from subclass name to numeric ID.
        superclass_to_id: Mapping from superclass name to numeric ID.
    """

    def __init__(
        self,
        cifar100_root: Path,
        genai_novel_root: Path,
        genai_ind_root: Optional[Path] = None,
        include_cifar_train: bool = True,
        include_cifar_test: bool = True,
        transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize the combined CIFAR-100 and GenAI dataset.

        Args:
            cifar100_root: Root directory containing CIFAR-100 data.
            genai_novel_root: Root directory for novel subclasses/superclasses GenAI data.
                Expected structure: {novel_subclasses,novel_superclasses}/{superclass}/{subclass}/*.png
            genai_ind_root: Root directory for in-distribution GenAI CIFAR-100 data.
                Expected structure: {superclass}/{subclass}/*.png. If None, skips loading.
            include_cifar_train: Whether to include CIFAR-100 training split.
            include_cifar_test: Whether to include CIFAR-100 test split.
            transform: Optional transform to apply to images.

        Returns:
            None
        """
        print(f"\n {'=' * 60} \n Loading datasets...\n {'=' * 60}")
        self.transform = transform

        # CIFAR100 superclass mapping (cleaned)
        self.cifar100_superclass_map = {
            "aquatic_mammals":                ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish":                           ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers":                        ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food_containers":                ["bottle", "bowl", "can", "cup", "plate"],
            "fruit_and_vegetables":           ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household_electrical_devices":   ["clock", "keyboard", "lamp", "telephone", "television"],
            "household_furniture":            ["bed", "chair", "couch", "table", "wardrobe"],
            "insects":                        ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large_carnivores":               ["bear", "leopard", "lion", "tiger", "wolf"],
            "large_man_made_outdoor_things":  ["bridge", "castle", "house", "road", "skyscraper"],
            "large_natural_outdoor_scenes":   ["cloud", "forest", "mountain", "plain", "sea"],
            "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium_sized_mammals":           ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non_insect_invertebrates":       ["crab", "lobster", "snail", "spider", "worm"],
            "people":                         ["baby", "boy", "girl", "man", "woman"],
            "reptiles":                       ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small_mammals":                  ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees":                          ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles_1":                     ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles_2":                     ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
        }

        # Build reverse mapping: subclass -> superclass
        self.subclass_to_superclass = {}
        for superclass, subclasses in self.cifar100_superclass_map.items():
            for subclass in subclasses:
                self.subclass_to_superclass[subclass] = superclass

        # ==================== Load data ==================== #
        self.samples = []

        # Load CIFAR100
        if include_cifar_train:
            self._load_cifar100(root=cifar100_root, train=True)
        if include_cifar_test:
            self._load_cifar100(root=cifar100_root, train=False)

        # Load GenAI Image Dataset
        self._load_genai(genai_novel_root=genai_novel_root, genai_ind_root=genai_ind_root)

        # Build label mappings
        self._build_label_mappings()

        print(f"Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  CIFAR100: {sum(1 for s in self.samples if s['source'] == 'cifar100')}")
        print(f"  GenAI novel subclasses: {sum(1 for s in self.samples if s['source'] == 'genai_novel_subclass')}")
        print(f"  GenAI novel superclasses: {sum(1 for s in self.samples if s['source'] == 'genai_novel_superclass')}")
        print(f"  GenAI in-distribution: {sum(1 for s in self.samples if s['source'] == 'genai_ind')}")
        print(f"  Unique subclasses: {len(self.subclass_to_id)}")
        print(f"  Unique superclasses: {len(self.superclass_to_id)}")

        logger.info(f"Superclass list: {str(self.superclass_to_id)}")



    def _load_cifar100(self, root: Path, train: bool) -> None:
        """
        Load CIFAR-100 dataset with superclass labels.

        Args:
            root: Root directory containing CIFAR-100 data.
            train: If True, loads training split; otherwise loads test split.

        Returns:
            None
        """
        split_name = "train" if train else "test"

        cifar = datasets.CIFAR100(
            root=root,
            train=train,
            download=True
        )

        for idx in range(len(cifar)):
            img, label = cifar[idx]
            subclass_name = cifar.classes[label]
            superclass_name = self.subclass_to_superclass.get(subclass_name, "unknown")

            self.samples.append({
                'image': img,
                'subclass_name': subclass_name,
                'superclass_name': superclass_name,
                'source': 'cifar100',
                'split': split_name,
                'image_path': f'cifar100_{split_name}_{idx}.png'
            })


    def _load_genai(
        self, genai_novel_root: Path, genai_ind_root: Optional[Path]
    ) -> None:
        """
        Load GenAI images from hierarchical directory structures.

        Args:
            genai_novel_root: Root directory for novel subclasses and superclasses.
                Expected structure: {novel_subclasses,novel_superclasses}/{superclass}/{subclass}/*.png
            genai_ind_root: Root directory for in-distribution GenAI CIFAR-100 data.
                Expected structure: {superclass}/{subclass}/*.png. If None, skips loading.

        Returns:
            None
        """
        novel_superclass_names = []

        # Load novel subclasses (under existing CIFAR-100 superclasses)
        novel_sub_path = genai_novel_root / 'novel_subclasses'
        if novel_sub_path.exists():
            for superclass_dir in novel_sub_path.iterdir():
                if not superclass_dir.is_dir():
                    logger.warning(f"{superclass_dir} is not a directory ...")
                    continue
                superclass_name = superclass_dir.name

                for subclass_dir in superclass_dir.iterdir():
                    if not subclass_dir.is_dir():
                        logger.warning(f"{subclass_dir} is not a directory ...")
                        continue
                    subclass_name = subclass_dir.name

                    for img_file in subclass_dir.glob('*.png'):
                        self.samples.append({
                            'image': Image.open(img_file).convert('RGB'),
                            'subclass_name': subclass_name,
                            'superclass_name': superclass_name,
                            'source': 'genai_novel_subclass',
                            'split': 'genai',
                            'image_path': str(img_file.relative_to(genai_novel_root))
                        })
        else:
            logger.warning(
                f"GenAI novel SUBCLASS images directory does not exist at {novel_sub_path}"
            )

        # Load novel superclasses (entirely new superclasses not in CIFAR-100)
        novel_super_path = genai_novel_root / 'novel_superclasses'
        if novel_super_path.exists():
            for superclass_dir in novel_super_path.iterdir():
                if not superclass_dir.is_dir():
                    logger.warning(f"{superclass_dir} is not a directory ...")
                    continue
                superclass_name = superclass_dir.name
                novel_superclass_names.append(superclass_name)

                for subclass_dir in superclass_dir.iterdir():
                    if not subclass_dir.is_dir():
                        logger.warning(f"{subclass_dir} is not a directory ...")
                        continue
                    subclass_name = subclass_dir.name

                    for img_file in subclass_dir.glob('*.png'):
                        self.samples.append({
                            'image': Image.open(img_file).convert('RGB'),
                            'subclass_name': subclass_name,
                            'superclass_name': superclass_name,
                            'source': 'genai_novel_superclass',
                            'split': 'genai',
                            'image_path': str(img_file.relative_to(genai_novel_root))
                        })
        else:
            logger.warning(
                f"GenAI novel SUPERCLASS images directory does not exist at {novel_super_path}"
            )

        # Load in-distribution GenAI CIFAR-100 data (synthetic recreations)
        if genai_ind_root is not None:
            if genai_ind_root.exists():
                for superclass_dir in genai_ind_root.iterdir():
                    if not superclass_dir.is_dir():
                        logger.warning(f"{superclass_dir} is not a directory ...")
                        continue
                    superclass_name = superclass_dir.name

                    for subclass_dir in superclass_dir.iterdir():
                        if not subclass_dir.is_dir():
                            logger.warning(f"{subclass_dir} is not a directory ...")
                            continue
                        subclass_name = subclass_dir.name

                        for img_file in subclass_dir.glob('*.png'):
                            self.samples.append({
                                'image': Image.open(img_file).convert('RGB'),
                                'subclass_name': subclass_name,
                                'superclass_name': superclass_name,
                                'source': 'genai_ind',
                                'split': 'genai',
                                'image_path': str(img_file.relative_to(genai_ind_root))
                            })
            else:
                logger.warning(
                    f"GenAI in-distribution images directory does not exist at {genai_ind_root}"
                )

        logger.info(f"Novel Superclass names: {str(novel_superclass_names)}")

    def _build_label_mappings(self) -> None:
        """
        Create numeric ID mappings for subclasses and superclasses.

        Populates self.subclass_to_id, self.id_to_subclass, self.superclass_to_id,
        and self.id_to_superclass dictionaries. Also adds numeric IDs to each sample.

        Returns:
            None
        """
        # Collect all unique subclasses and superclasses
        all_subclasses = sorted(set(s['subclass_name'] for s in self.samples))
        all_superclasses = sorted(set(s['superclass_name'] for s in self.samples))

        # Create mappings
        self.subclass_to_id = {name: idx for idx, name in enumerate(all_subclasses)}
        self.id_to_subclass = {idx: name for name, idx in self.subclass_to_id.items()}

        self.superclass_to_id = {name: idx for idx, name in enumerate(all_superclasses)}
        self.id_to_superclass = {idx: name for name, idx in self.superclass_to_id.items()}

        # Add numeric IDs to samples
        for sample in self.samples:
            sample['subclass_id'] = self.subclass_to_id[sample['subclass_name']]
            sample['superclass_id'] = self.superclass_to_id[sample['superclass_name']]

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - image: Transformed image tensor.
                - metadata: Dictionary with subclass_id, subclass_name, superclass_id,
                  superclass_name, source, split, and image_path.
        """
        sample = self.samples[idx]
        image = sample['image']

        if self.transform:
            image = self.transform(image)

        metadata = {
            'subclass_id': sample['subclass_id'],
            'subclass_name': sample['subclass_name'],
            'superclass_id': sample['superclass_id'],
            'superclass_name': sample['superclass_name'],
            'source': sample['source'],
            'split': sample['split'],
            'image_path': sample['image_path']
        }

        return image, metadata

    def get_label_mappings(self) -> Dict[str, Dict]:
        """
        Return label mappings for saving with embeddings.

        Returns:
            Dict containing subclass_to_id, id_to_subclass, superclass_to_id,
            and id_to_superclass mappings.
        """
        return {
            'subclass_to_id': self.subclass_to_id,
            'id_to_subclass': self.id_to_subclass,
            'superclass_to_id': self.superclass_to_id,
            'id_to_superclass': self.id_to_superclass
        }