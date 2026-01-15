import time
import os
import random
import logging
import yaml
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import version
from torch import device as TorchDevice

from src.loggers.wandb_logger import WandBLogger


logger = logging.getLogger("msproject")
logging.getLogger("PIL").setLevel(logging.WARNING)


def setup_experiment(config: DictConfig):
    load_dotenv()
    setup_logging(config)
    set_seed(config.seed)

    logger.info(f"Successfully loaded project configuration...")
    # logger.info(f"Loaded Configuration: {OmegaConf.to_yaml(config)}")

    device = get_device(config)
    # wand_logger = WandBLogger(config)

    # return device, wand_logger
    return device

def setup_logging(global_config: DictConfig) -> None:
    """

    """
    # Load logging configuration
    logging_config_location = "/home/alex/repos/ms_project/configs/logging/logger.yaml"
    with open(logging_config_location) as f_in:
        logging_config = yaml.safe_load(f_in)

    # Create logs directory if it doesn't exist
    os.makedirs(global_config.logs_location, exist_ok=True)

    # Set log filename
    log_filename = os.path.join(global_config.logs_location, f"msproject_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging_config['handlers']['file']['filename'] = os.path.join(global_config.logs_location, log_filename)

    logging.config.dictConfig(logging_config)

    for lib in ["numba", "umap", "pynndescent", "matplotlib", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seeds for reproducibility.
    Args:
        seed (int): The seed value to set.
    Returns:
        None
    """

    random.seed(seed) # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed
    os.environ["PYTHONHASHSEED"] = str(seed) # Set a fixed value for the hash seed, seeds for data loading operations

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # set torch (GPU) seed
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True # set cudnn to deterministic mode
        torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

    # Document the environment for future reproducibility
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {version.cuda if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"Random Seed: {seed}")


def get_device(config: DictConfig) -> torch.device:
    """
    Get the device to use for the experiment.
    Args:
        config (DictConfig): Configuration object
    Returns:
        device (torch.device): Device to use for the experiment
    """
    if torch.cuda.is_available() and hasattr(config, 'device') and config.device == "gpu":
            device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.warning("No GPU detected. Reverting to CPU...")
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


def create_next_experiment_dir(base_dir="DINOv2", prefix="set_"):
    """
    Scans base_dir for folders starting with prefix, finds the highest number,
    and creates the next incremented directory.
    """
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)  # Create DINOv2 if it doesn't exist

    max_id = 0

    # Iterate over existing directories to find the highest number
    for item in path.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                # Extract the number from the folder name (e.g., "set_5" -> 5)
                num = int(item.name.replace(prefix, ""))
                max_id = max(max_id, num)
            except ValueError:
                continue  # Skip folders that don't end in a number

    # Define the new directory name
    new_experiment_dir = path / f"{prefix}{max_id + 1}"

    # Create the directory
    new_experiment_dir.mkdir()

    return new_experiment_dir