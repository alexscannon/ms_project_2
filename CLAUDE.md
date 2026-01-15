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
```

## Architecture

### Configuration (Hydra)
- `configs/config.yaml` - Main config with defaults for model, data, and logging
- Model config selects backbone (dinov2/dinov3) and version
- All machine-specific paths (data_dir, output_dir, logs_location) must be updated in configs

### Pipeline Flow (main.py)
1. **Setup** (`src/utils.py:setup_experiment`) - Loads .env, configures logging, sets seed, detects device
2. **Load Encoder** (`src/models/backbone/DINO.py:load_dino_model`) - Loads DINOv2 or v3 from torch hub
3. **Extract Embeddings** (if not cached):
   - `src/data/cifar100_genai_class.py:CombinedCIFAR100GenAIDataset` - Unifies CIFAR-100 + GenAI images with superclass/subclass metadata
   - `src/data/embedding_extractor.py:DINOEmbeddingExtractor` - Batch extraction, saves to `embeddings.npy` + `metadata.csv`
4. **Analysis**:
   - `src/analysis/visualization.py` - UMAP projections, per-superclass plots, source separation visualization
   - `src/analysis/group_metrics.py` - Silhouette scores, intra/inter-class distances, k-NN retrieval accuracy
   - `src/analysis/artifact_analysis.py` - Validity tests including source classification, within-superclass separation, cross-source retrieval
   - `src/analysis/source_leakage_analysis.py` - Retrieval source bias and superclass confusion analysis
5. **Helper Scripts**:
   - `data_cleaning.ipynb` – python notebook for performing simple cleaning of image data
   - `generate_synthetic_cifar100_imgs.ipynb` – python notebook for generating synthetic in-distribution CIFAR100 data to test GenAI artifacts

### Data Structure
GenAI images expected at `{data_dir}/ms_cifar100_ai_data_cleaned/` with structure:
```
novel_subclasses/{superclass}/{subclass}/*.png
novel_superclasses/{superclass}/{subclass}/*.png
```

### Key Data Groups
- **Baseline (CIFAR100)**: Real images from CIFAR-100 train/test
- **Group A (Novel Subclass)**: GenAI images of new subclasses under existing superclasses
- **Group B (Novel Superclass)**: GenAI images of entirely new superclasses

### Outputs
Embeddings and results saved to `{data_dir}/embeddings/DINOv{version}/`:
- `embeddings.npy` - (N, embedding_dim) array
- `metadata.csv` - Per-sample labels and source info
- `plots/` - UMAP visualizations including source separation check
- `quantitative_results.csv` - Metrics by data group
- `validity_analysis_results.json` - Artifact detection test results
- `source_leakage_results.json` - Source bias and confusion analysis results

### Validity Tests
The analysis pipeline includes tests to detect potential confounds:
- **Source Classification**: Can a classifier predict real vs synthetic from embeddings? (>70% accuracy = warning)
- **Within-Superclass Separation**: Do CIFAR and GenAI centroids of same superclass diverge?
- **Retrieval Source Bias**: Do GenAI images disproportionately retrieve other GenAI images?
- **Embedding Dimension Analysis**: Are source differences concentrated in specific dimensions?

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