# Validity Analysis Explanation

This document explains the validity tests designed to detect potential confounds in the DINOv2/v3 embedding analysis experiment.

## Background: Why These Tests Matter

The experiment's central question is: **Can foundation models (DINOv2/v3) generalize to out-of-distribution (OOD) synthetic images without fine-tuning?**

To answer this, we extract embeddings from both real images (CIFAR-100) and synthetic images (GenAI), then analyze whether they cluster by semantic content (e.g., "fish", "flowers") rather than by source (real vs. synthetic).

However, two potential confounds could invalidate the results:

1. **Training Data Overlap**: Both DINOv2 and the GenAI models may have been trained on similar web-scale data, meaning the synthetic images aren't truly "out-of-distribution."

2. **Generation Artifacts**: Diffusion models produce images with characteristic visual signatures (texture patterns, color distributions, edge characteristics) that DINOv2 might detect and encode, causing clustering by source rather than semantics.

The validity tests below are designed to detect whether these confounds are present in the embeddings.

---

## Test Descriptions

### Test 1: Cross-Domain Nearest Neighbor Retrieval

**Question**: When a GenAI image searches for its nearest neighbor in CIFAR-100, does it find an image of the same superclass?

**Method**:
- Train a k-NN model on CIFAR-100 embeddings only
- Query with GenAI images
- Measure: Does the retrieved CIFAR image share the same superclass?

**Interpretation**:
- Random chance baseline: ~5% (1/20 superclasses)
- High accuracy (>50%): Good semantic alignment across sources
- Low accuracy (<20%): Sources may not share semantic structure

**Your Result**: 51.25% - Suggests moderate semantic alignment.

---

### Test 2: Centroid Alignment

**Question**: For superclasses that exist in both CIFAR and GenAI, do the class centroids point in similar directions?

**Method**:
- For each shared superclass, compute the mean embedding (centroid) for CIFAR and GenAI separately
- Build a similarity matrix comparing all CIFAR centroids to all GenAI centroids
- Compare diagonal (same superclass, e.g., CIFAR-fish vs GenAI-fish) to off-diagonal (different superclass, e.g., CIFAR-fish vs GenAI-flowers)

**Interpretation**:
- **Same-class similarity**: How similar are embeddings of the same semantic category across sources?
- **Different-class similarity**: Baseline similarity between unrelated categories
- **Delta** (same - different): The "semantic strength" signal
  - Delta > 0.15: Strong semantic alignment
  - Delta < 0.15: Weak alignment, source artifacts may dominate

**Your Result**: Delta = 0.264 - Indicates semantic structure is preserved.

---

### Test 3: Source Classification (Artifact Detection)

**Question**: Can a simple classifier predict whether an embedding came from CIFAR or GenAI?

**Method**:
- Train a logistic regression classifier: embedding → source label
- Use 5-fold cross-validation to measure accuracy
- Identify which embedding dimensions are most discriminative

**Interpretation**:
- Baseline (random): 50%
- Accuracy 50-55%: Embeddings are source-agnostic (good)
- Accuracy 55-70%: Moderate source signal (caution)
- Accuracy >70%: Strong source signal (warning - artifacts detected)

**Why This Matters**: If a linear classifier can easily distinguish sources, the embeddings contain information about "how the image was made" rather than just "what the image contains." This confounds semantic analysis.

**Your Result**: 98.98% accuracy - **Critical warning**. The embeddings almost perfectly encode source identity.

---

### Test 4: Within-Superclass Source Separation

**Question**: For images of the same semantic category, how far apart are the CIFAR and GenAI clusters?

**Method**:
- For each superclass present in both sources (e.g., "fish"):
  - Compute the CIFAR centroid
  - Compute the GenAI centroid
  - Measure cosine distance between them

**Interpretation**:
- Low distance (<0.15): Same-class images cluster together regardless of source (good)
- Moderate distance (0.15-0.30): Some source separation (caution)
- High distance (>0.30): Source dominates over semantics (warning)

**Why This Matters**: If "CIFAR fish" and "GenAI fish" are far apart in embedding space, the model treats them as different categories despite having the same semantic label.

**Your Result**: Mean separation = 0.579 - **Critical warning**. Same-superclass images are very far apart based on source.

---

### Test 5: Source-Agnostic Cross-Source Retrieval

**Question**: When forced to retrieve across sources, do images find semantically similar matches?

**Method**:
- For each GenAI image, find its nearest CIFAR neighbor (forced cross-source)
- For each CIFAR image, find its nearest GenAI neighbor (forced cross-source)
- Measure: Does the retrieved image share the same superclass?
- Check for asymmetry between the two directions

**Interpretation**:
- High accuracy in both directions: Strong semantic alignment
- Asymmetry >10%: One source may have artifacts biasing retrieval
- Large distance values: Sources are far apart in embedding space

**Your Result**:
- GenAI → CIFAR: 24.2%
- CIFAR → GenAI: 18.7%
- Mean distance GenAI→CIFAR: 0.49, CIFAR→GenAI: 0.75

The asymmetric distances suggest CIFAR images are "further" from GenAI than vice versa.

---

### Test 6: Embedding Dimension Analysis

**Question**: Are source differences concentrated in specific embedding dimensions, or spread across all dimensions?

**Method**:
- Compute class-balanced mean embedding for CIFAR and GenAI
- Calculate per-dimension differences
- Measure what fraction of total difference is explained by the top 10 dimensions

**Interpretation**:
- Concentrated (>50% in top 10): Specific features encode source - could potentially be removed
- Spread (<50% in top 10): Source signal is distributed throughout the embedding

**Your Result**: Top 10 dimensions account for only 4.4% of difference - the source signal is spread throughout, making it difficult to remove.

---

### Test 7: Retrieval Source Bias (Source Leakage)

**Question**: When images retrieve their nearest neighbors from the combined dataset, do they disproportionately retrieve same-source images?

**Method**:
- For each image, find k=5 nearest neighbors in the combined CIFAR+GenAI dataset
- Measure: What fraction of neighbors are from the same source?
- Compare to expected fraction based on dataset proportions

**Interpretation**:
- Expected: Proportional to dataset size (e.g., if 99% CIFAR, expect ~99% CIFAR neighbors)
- Bias = Observed - Expected
- High bias (>15%): Images cluster by source, not just semantics

**Your Result**:
- GenAI images retrieve GenAI neighbors 92.3% of the time
- Expected rate: 1.25%
- **Bias: +91%** - GenAI images almost exclusively find other GenAI images

---

### Test 8: Superclass Confusion Analysis

**Question**: Do CIFAR and GenAI images make the same classification errors?

**Method**:
- Use k-NN to predict superclass for each image
- Build confusion matrices for CIFAR and GenAI separately
- Compare error patterns between sources

**Interpretation**:
- Similar confusion patterns: Model treats both sources consistently
- Different confusion patterns: Sources have different semantic structures or artifacts

**Your Result**: Mean confusion difference = 0.0026 - Very similar error patterns, suggesting semantic structure is consistent within each source.

---

## Summary of Findings

| Test | Metric | Result | Interpretation |
|------|--------|--------|----------------|
| Cross-Domain Retrieval | Accuracy | 51.25% | Moderate semantic alignment |
| Centroid Alignment | Delta | 0.264 | Strong semantic structure |
| Source Classification | Accuracy | 98.98% | **CRITICAL: Source artifacts detected** |
| Within-Superclass Separation | Distance | 0.579 | **CRITICAL: Source dominates semantics** |
| Cross-Source Retrieval | Accuracy | 18-24% | Weak cross-source matching |
| Dimension Analysis | Concentration | 4.4% | Source signal is distributed |
| Retrieval Source Bias | Bias | +91% | **CRITICAL: Extreme source clustering** |
| Confusion Analysis | Difference | 0.003 | Consistent error patterns |

---

## Conclusion

The validity tests reveal a fundamental problem: **DINOv2 embeddings encode source identity (real vs. synthetic) so strongly that it dominates over semantic content.**

Key evidence:
1. A linear classifier achieves 99% accuracy distinguishing sources
2. Same-superclass images from different sources are very far apart (0.58 cosine distance)
3. GenAI images almost exclusively retrieve other GenAI images (91% bias)

This means the original research question cannot be cleanly answered with this data. The apparent semantic clustering within GenAI images may be real, but the lack of cross-source integration suggests DINOv2 treats synthetic images as a fundamentally different domain.

### Possible Explanations

1. **Resolution/preprocessing artifacts**: CIFAR-100 images are 32x32 upsampled, while GenAI images are high-res downsampled then upsampled. This creates different interpolation artifacts.

2. **Diffusion model signatures**: GenAI images may share subtle patterns from the generation process that DINOv2 detects.

3. **Distribution shift**: The visual statistics of GenAI images (color, texture, composition) may differ systematically from natural photographs.

### Recommendations

1. Investigate the top discriminative dimensions (858, 566, 590, etc.) to understand what features encode source identity
2. Test with GenAI images generated at native 32x32 resolution to eliminate preprocessing confounds
3. Apply domain adaptation techniques to reduce source signal before semantic analysis
4. Consider using a different encoder that may be less sensitive to synthetic artifacts
