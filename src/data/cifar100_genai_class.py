import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
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
    Unified dataset combining CIFAR100 and GenAI images.

    Applies identical preprocessing to both sources.
    Returns images with rich metadata for later analysis.
    """

    def __init__(
        self,
        cifar100_root: Path = Path('/home/alex/data'),
        genai_root: Path = Path('/home/alex/data/ms_cifar100_ai_data_cleaned'),
        include_cifar_train: bool = True,
        include_cifar_test: bool = True,
        transform = None
    ):
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
        self._load_genai(genai_root=genai_root)

        # Build label mappings
        self._build_label_mappings()

        print(f"Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  CIFAR100: {sum(1 for s in self.samples if s['source'] == 'cifar100')}")
        print(f"  GenAI novel subclasses: {sum(1 for s in self.samples if s['source'] == 'genai_novel_subclass')}")
        print(f"  GenAI novel superclasses: {sum(1 for s in self.samples if s['source'] == 'genai_novel_superclass')}")
        print(f"  Unique subclasses: {len(self.subclass_to_id)}")
        print(f"  Unique superclasses: {len(self.superclass_to_id)}")

        logger.info(f"Superclass list: {str(self.superclass_to_id)}")



    def _load_cifar100(self, root: Path, train: bool):
        """
        Load CIFAR100 dataset with superclass labels.
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


    def _load_genai(self, genai_root: Path):
        """
        Load GenAI images from hierarchical directory structure.
        """

        # Load novel subclasses (under existing superclasses)
        novel_sub_path = genai_root / 'novel_subclasses'
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
                            'image_path': str(img_file.relative_to(genai_root))
                        })
        else:
            logger.warning(f"GenAI novel SUBCLASS images directory does not exist at {novel_sub_path}")

        # Load novel superclasses
        novel_super_path = genai_root / 'novel_superclasses'
        superclass_names = []
        if novel_super_path.exists():
            for superclass_dir in novel_super_path.iterdir():
                if not superclass_dir.is_dir():
                    logger.warning(f"{superclass_dir} is not a directory ...")
                    continue
                superclass_name = superclass_dir.name
                superclass_names.append(superclass_name)
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
                            'image_path': str(img_file.relative_to(genai_root))
                        })
        else:
            logger.warning(f"GenAI novel SUPERCLASS images directory does not exist at {novel_super_path}")

        logger.info(f"Novel Superclass names: {str(superclass_names)}")

    def _build_label_mappings(self):
        """
        Create numeric ID mappings for subclasses and superclasses.
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


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
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


    def get_label_mappings(self):
        """Return label mappings for saving."""
        return {
            'subclass_to_id': self.subclass_to_id,
            'id_to_subclass': self.id_to_subclass,
            'superclass_to_id': self.superclass_to_id,
            'id_to_superclass': self.id_to_superclass
        }