import json
import logging
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
from torchvision.transforms import v2 as transforms

from src.utils import setup_experiment
from src.models.backbone.DINO import load_dino_model
from src.analysis.visualization import run_visualization_suite, load_analysis_data
from src.analysis.group_metrics import calculate_group_metrics
from src.analysis.artifact_analysis import run_artifact_analysis
from src.analysis.source_leakage_analysis import run_source_leakage_analysis
from src.data.cifar100_genai_class import CombinedCIFAR100GenAIDataset
from src.data.embedding_extractor import DINOEmbeddingExtractor

logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger("msproject")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    print(f"\n {'=' * 60} \n Beginning Experiment...\n {'=' * 60}")
    # ============================ EXPERIMENTAL SET UP ============================ #
    # Load .env variables + set up logger + set random seed + set device + initialize logging
    device = setup_experiment(config)


    # ============================ LOAD ENCODER ============================ #
    # Load pre-trained (not fine-tuned) DINO vit
    encoder = load_dino_model(
        model_size=config.model.backbone.model_size,
        device=config.device,
        dino_version=config.model.dino_version
    )

    DATA_DIR = Path(config.data_dir)
    EMBEDDINGS_DIR = DATA_DIR / "embeddings" / f"DINOv{config.model.dino_version}"


    # ============================ CREATE EMBEDDINGS ============================ #
    if config.gen_embeddings:
        # --------------------- LOAD DATASETS --------------------- #
        # DINOv2/v3 use patch size 14, so input must be divisible by 14 (e.g., 224, 518)
        DINO_INPUT_SIZE = config.model.backbone.expected_input_size
        CIFAR100_IMG_SIZE = config.data.image_size

        transform = transforms.Compose([
            # Step 1: Downsample all images to CIFAR100's native 32×32 resolution to equalize quality
            transforms.Resize(size=CIFAR100_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            # Step 2: Upsample to DINOv3 expected input size
            transforms.Resize(size=DINO_INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            # Convert to tensor [C, H, W] in range [0, 1]
            transforms.ToTensor(),
            # Step 3: Normalize with ImageNet stats (DINOv2 training stats)
            transforms.Normalize(
                mean=config.model.backbone.mean,
                std=config.model.backbone.std
            )
        ])

        genai_novel_dir = DATA_DIR / config.genai_novel_dir
        genai_ind_dir = DATA_DIR / config.genai_ind_dir
        dataset = CombinedCIFAR100GenAIDataset(
            cifar100_root=DATA_DIR,
            genai_novel_root=genai_novel_dir,
            genai_ind_root=genai_ind_dir,
            include_cifar_train=True,
            include_cifar_test=True,
            transform=transform
        )

        # --------------------- EXTRACT EMBEDDINGS --------------------- #
        # Initialize extractor
        extractor = DINOEmbeddingExtractor(model=encoder, device=device)

        # Extract embeddings
        embeddings, metadata = extractor.extract_embeddings(
            dataset=dataset,
            batch_size=32,  # Adjust based on your GPU memory
            num_workers=4
        )

        # Save embeddings, embedding metadata, label_mappings
        EMBEDDINGS_DIR = extractor.save_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            label_mappings=dataset.get_label_mappings(),
            output_dir=EMBEDDINGS_DIR,
        )
        logger.info(f"EMBEDDINGS_DIR: {EMBEDDINGS_DIR}")

    else:
        EMBEDDINGS_DIR = EMBEDDINGS_DIR / f"set_{config.set_num}"
        logger.info(f"Skipping embedding extractions. Extracted embeddings already exist at location {EMBEDDINGS_DIR}")


    # =================================== Analysis =================================== #
    print(f"\n {'=' * 60} \n Analyzing embeddings...\n {'=' * 60}")

    # 1. Load Embeddings and embeddings' metadata
    emb_path = EMBEDDINGS_DIR / "embeddings.npy"
    meta_path = EMBEDDINGS_DIR / "metadata.csv"

    if not emb_path.exists() or not meta_path.exists():
        logger.error(f"Could not find embedding data in {EMBEDDINGS_DIR}")
        return

    embeddings, metadata = load_analysis_data(emb_path, meta_path)

    # 1.) Run visualization report
    run_visualization_suite(
        embeddings_dir=EMBEDDINGS_DIR,
        embeddings=embeddings,
        metadata=metadata
    )

    # 2.) Compute quantitative clustering/grouping metrics + RQ2 novel class analysis
    df_results, rq2_results = calculate_group_metrics(embeddings=embeddings, metadata=metadata)
    df_results.to_csv(EMBEDDINGS_DIR / "quantitative_results.csv", index=False)

    # Save RQ2 results
    def make_serializable(obj):
        """Recursively convert numpy types to Python native types for JSON."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        return obj

    rq2_path = EMBEDDINGS_DIR / "rq2_novel_class_results.json"
    with open(rq2_path, 'w') as f:
        serializable_rq2 = make_serializable(rq2_results)
        json.dump(serializable_rq2, f, indent=2)
    logger.info(f"RQ2 results saved to {rq2_path}")

    # 3.) Artifact Check Analysis (includes validity tests)
    run_artifact_analysis(EMBEDDINGS_DIR)

    # 4.) Source Leakage Analysis
    run_source_leakage_analysis(EMBEDDINGS_DIR)

    from src.analysis.balanced_analysis import run_balanced_analysis
    run_balanced_analysis(EMBEDDINGS_DIR)

    print(f"\n {'*' * 10}  Experiment Complete  {'*' * 10}\n")

if __name__ == "__main__":
    main()