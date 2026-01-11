import logging
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ============================================
# Basic DINOvX Setup and Feature Extraction
# ============================================

logger = logging.getLogger("msproject")
logging.getLogger("PIL").setLevel(logging.WARNING)

def load_dino_model(model_size: str, device: torch.device, dino_version: int) -> nn.Module:
    """
    Load a DINOvX model from torch hub.
        v2: (https://ai.meta.com/dinov2/
        v3: https://ai.meta.com/dinov3/

    DINOv2 Available sizes (https://github.com/facebookresearch/dinov2):
        - 'small': ViT-S/14 (21M parameters)
        - 'base': ViT-B/14 (86M parameters)
        - 'large': ViT-L/14 (300M parameters)
        - 'giant': ViT-g/14 (1.1B parameters)

    DINOv3 Available sizes (https://github.com/facebookresearch/dinov3):
        - 'small': ViT-S/16 (21M parameters)
        - 'base': ViT-B/16 (86M parameters)
        - 'large': ViT-L/16 (300M parameters)
        - 'large-plus': ViT-H+/16 (300M parameters)
        - 'giant': ViT-g/16 (6.7B parameters)
    """

    try:
        repo_name, model_map = "", {}
        if dino_version == 2:
            repo_name = "facebookresearch/dinov2"
            model_map = {
                'small': 'dinov2_vits14',
                'base': 'dinov2_vitb14',
                'large': 'dinov2_vitl14',
                'giant': 'dinov2_vitg14'
            }

        elif dino_version == 3:
            repo_name = "facebookresearch/dinov3"
            model_map = {
                'small': 'dinov3_vits16',
                'base': 'dinov3_vitb16',
                'large': 'dinov3_vitl16',
                'large-plus': 'dinov3_vith16plus',
                'giant': 'dinov3_vit7b16'
            }

        model_name = model_map.get(model_size, 'dinov2_vitb14')

        # Load the model from Facebook Research's repository
        logger.info(f"Loading dino model {model_name} from repo {repo_name}...")
        model = torch.hub.load(
            repo_or_dir=repo_name,
            model=model_name
        )
        logger.info("Successfully loaded DINO model...")

        model.to(device) # put model on device
        model.eval() # set model to evaluation mode

        return model

    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
