# Masters Project: SOTA Encoder Generalization

## Introduction
**Project Central Question**

Can extremely well-trained foundation models (e.g., Meta's DINOv2) be used as the encoder in continual learning scenarios without requiring some degree of fine-tuning on Out-of-Distribution data ("OOD"). Put differently, can state-of-the-art ("SOTA") foundation models remove the necessity for the many continual learning strategies which prevent catastrophic forgetting?

**Testing Central Question**

Determining the answer to the project's central question will be done by observing the geometric structure of computed embeddings for both synthetic data which the foundation model has never been exposed to and data which the encoder was trained on. Specific questions to answer:
1. Are synthetic embeddings of existing sub-classes geometrically close to real, pre-trained and existing sub-class embeddings?
2. Are synthetically generated images computed embeddings, which belong to existing super-classes, geometrically close to existing embeddings of real super-classes?
3. If we generate synthetic images of a completely novel sub-class, which belong to an existing super-class, will the synthetic examples be clustered together and meaningfully separate from other existing sub-class clusters of real existing super-classes and closer to other existing sub-class clusters of the same super-class than to other sub-classes of other super-classes?

## How to Run Experiment
1. Clone repository
```bash
git clone git@github.com:alexscannon/ms_project_2.git
```
2. Update all the machine specific path attributes in the `~/ms_project_2/configs/` directory
3. Navigate to the project's root directory
```bash
cd ms_project_2
```
4. Fill in `.env` variables with personal information
5. Create a conda virtual environment with required packages
```bash
conda env create -f environment.yml
```
6. Create a `.env` file in project's root directory
```bash
cp .env_template .env
```
7. Activate conda virtual environment
```bash
conda activate project_env
```
8.  Run project (from project root)
```bash
python3 main.py
```

## Misc. Notes
- This project (specifically the embedding generation) requires the use a GPU. This project was ran on a NVIDIA RTX 3090 (VRAM – 24GB).