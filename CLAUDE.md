# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating whether state-of-the-art foundation models (DINOv2/v3) can generalize to out-of-distribution data without fine-tuning. The experiment extracts embeddings from CIFAR-100 and synthetic (GenAI) images, then analyzes their geometric relationships to assess semantic coherence.

## Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate project_env
cp .env_template .env  # Then fill in WANDB_API_KEY and GH_TOKEN
```

### Running the Experiment
```bash
python3 main.py
```

Configuration is managed via Hydra. Override settings with:
```bash
python3 main.py device=cpu model.dino_version=3 data=tiny_imagenet
python3 main.py set_num=2 generate_embeddings=false  # Use cached embeddings
```

### Running Tests
```bash
pytest tests/                    # All tests
pytest tests/unit/ -v            # Unit tests only
pytest tests/integration/ -v     # Integration tests only

# Manual validation on existing embeddings
python tests/validation/validate_embeddings.py data/embeddings/DINOv2/set_5
```

## Architecture

### Configuration (Hydra)
- `configs/config.yaml` - Main config with defaults for model, data, and logging
- `configs/data/` - Data configs (cifar100.yaml, tiny_imagenet.yaml)
- `configs/model/backbone/` - Model configs (dinov2.yaml, dinov3.yaml)
- Key settings: `data_dir`, `output_dir`, `set_num`, `generate_embeddings`, `batch_size`

### Pipeline Flow (main.py)
1. **Setup** (`src/utils.py:setup_experiment`) - Loads .env, configures logging, sets seed, detects device
2. **Load Encoder** (`src/models/backbone/DINO.py:load_dino_model`) - Loads DINOv2 or v3 from torch hub
3. **Extract Embeddings** (if `generate_embeddings=true`):
   - `src/data/cifar100_genai_class.py:CombinedCIFAR100GenAIDataset` - Unifies CIFAR-100 + GenAI images with superclass/subclass metadata
   - `src/data/embedding_extractor.py:DINOEmbeddingExtractor` - Batch extraction, saves to `embeddings.npy` + `metadata.csv`
4. **Analysis** (5 sequential modules):
   - `src/analysis/visualization.py` - UMAP projections, per-superclass plots, source separation visualization
   - `src/analysis/group_metrics.py` - Silhouette scores, intra/inter-class distances, k-NN retrieval accuracy, RQ2 novel class metrics
   - `src/analysis/artifact_analysis.py` - Validity tests including source classification, centroid alignment, cross-source retrieval
   - `src/analysis/source_leakage_analysis.py` - Retrieval source bias and superclass confusion analysis
   - `src/analysis/balanced_analysis.py` - Balanced dataset tests (downsamples CIFAR to match GenAI for fair comparison)

### Data Structure
GenAI images expected at `{data_dir}/genai_cleaned/` with structure:
```
novel_subclasses/{superclass}/{subclass}/*.png
novel_superclasses/{superclass}/{subclass}/*.png
```

Optional in-distribution GenAI data at `{data_dir}/synthetic_cifar100_research/cifar100_512x512_master/`:
```
{superclass}/{subclass}/*.png
```

### Data Source Identifiers
The `source` column in metadata identifies data origin:
- `cifar100` - Real CIFAR-100 images (train + test splits)
- `genai_novel_subclass` - GenAI images of new subclasses under existing CIFAR-100 superclasses
- `genai_novel_superclass` - GenAI images of entirely new superclasses (e.g., biopunk_weaponry, fungi, geological_structures)
- `genai_ind` - In-distribution GenAI CIFAR-100 synthetic recreations (optional)

### Key Data Groups
- **Baseline (CIFAR100)**: Real images from CIFAR-100 train/test
- **Group A (Novel Subclass)**: GenAI images of new subclasses under existing superclasses
- **Group B (Novel Superclass)**: GenAI images of entirely new superclasses

### Outputs
Embeddings and results saved to `{data_dir}/embeddings/DINOv{version}/set_{N}/`:
- `embeddings.npy` - (N, embedding_dim) array
- `metadata.csv` - Per-sample labels: index, subclass_id, subclass_name, superclass_id, superclass_name, source, split, image_path
- `label_mappings.json` - Subclass/superclass ID mappings
- `plots/` - UMAP visualizations including source separation check
- `quantitative_results.csv` - Metrics by data group
- `rq2_novel_class_results.json` - Research Question 2 novel class metrics
- `validity_analysis_results.json` - Artifact detection test results
- `source_leakage_results.json` - Source bias and confusion analysis results
- `balanced_analysis_results.json` - Balanced dataset test results

### Validity Tests
The analysis pipeline includes tests to detect potential confounds:
- **Source Classification**: Can a classifier predict real vs synthetic from embeddings? (>70% accuracy = warning)
- **Within-Superclass Separation**: Do CIFAR and GenAI centroids of same superclass diverge? (>0.3 cosine distance = warning)
- **Retrieval Source Bias**: Do GenAI images disproportionately retrieve other GenAI images?
- **Embedding Dimension Analysis**: Are source differences concentrated in specific dimensions? (top 10 dims >50% = warning)
- **Balanced Analysis**: Re-runs validity tests on downsampled balanced dataset to control for class imbalance

### Data Preprocessing Scripts
Located in `data/genai_scripts/`:
- `clean_genai_imgs.py` - Strips metadata, applies center crop, resizes to 1024x1024
- `sync_genai_imgs_to_cifar100.py` - Resizes images to 32x32 for CIFAR-100 style matching

Helper notebooks in `scripts/`:
- `data_cleaning.ipynb` - Interactive data cleaning
- `generate_synthetic_cifar100_imgs.ipynb` - Generate synthetic in-distribution CIFAR-100 data

## Testing

### Test Structure
```
tests/
├── conftest.py                      # Shared fixtures (MockDINOModel, SyntheticDataset)
├── fixtures/mock_data.py            # Synthetic data generators
├── unit/
│   ├── test_embedding_extractor.py  # DINOEmbeddingExtractor tests
│   └── test_dataset.py              # CombinedCIFAR100GenAIDataset tests
├── integration/
│   └── test_extraction_pipeline.py  # End-to-end pipeline tests
└── validation/
    └── validate_embeddings.py       # Manual validation CLI script
```

### Key Test Areas
- **Numerical validity**: NaN/Inf detection, variance checks, duplicate detection
- **Metadata alignment**: Embedding-metadata correspondence, required fields
- **Determinism**: Same input produces same output
- **Semantic structure**: Within-class > between-class similarity

## Context Management

**Token Budget Awareness**
- Check context periodically with `/context` during long tasks
- Target staying under 70% to preserve reasoning quality
- At 80%: wrap up current subtask and checkpoint
- Never let auto-compact trigger

**Checkpoint Protocol**
When approaching 70% context usage:
1. Complete current atomic task
2. Create checkpoint: "Document current state to `checkpoint.md`"
3. Run `/clear`
4. Reload: "Read `checkpoint.md` and continue"

**File Reading Strategy**
Don't ask me to read entire large files. Instead:
- "Read the authentication logic in auth.py" (specific function/section)
- "Grep for uses of the User class" (targeted search)
- "Show me lines 100-200 of config.py" (specific range)

**Tool Output Management**
For commands that produce large output:
- Pipe through head: `pytest | head -50` (first 50 lines)
- Filter to relevant parts: `docker logs app | grep ERROR`
- Save to file and summarize: `command > output.txt`, then "summarize `output.txt`"

## General Coding Best Practices
- Python functions should always have
  - Type hints (e.g., my_name: str)
  - Doc strings that include:
    1. Description of the function
    2. Arguments and a description of the arguments
    3. What the function returns and the description of the returned variable
    4. (Optional) What exception the function raises
- Remove unused imports from files
