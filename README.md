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

