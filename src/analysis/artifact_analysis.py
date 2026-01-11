import json
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("artifact_analysis")

def load_data(embeddings_dir):
    """Load embeddings and metadata."""
    emb_path = embeddings_dir / "embeddings.npy"
    meta_path = embeddings_dir / "metadata.csv"

    if not emb_path.exists():
        logger.error(f"Embeddings not found at {emb_path}")
        return None, None

    logger.info(f"Loading data from {embeddings_dir}...")
    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    return embeddings, metadata


def source_classification_test(embeddings_norm, metadata, mask_cifar, mask_genai):
    """
    Test 3: Source Classification Test (Artifact Detection)

    Train a linear classifier to predict source (CIFAR vs GenAI) from embeddings.
    If accuracy >> 50%, embeddings contain detectable source-specific artifacts.
    """
    print(f"\n[Test 3] Source Classification (Can we predict Real vs Fake from embedding?)")

    # Prepare data: combine CIFAR and GenAI
    combined_mask = mask_cifar | mask_genai
    X = embeddings_norm[combined_mask]
    y = metadata.loc[combined_mask, 'source'].apply(
        lambda s: 0 if s == 'cifar100' else 1
    ).values

    # Train logistic regression with cross-validation
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"*  5-Fold CV Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
    print(f"*  Baseline (random): 50.00%")

    # Interpretation
    if mean_acc > 0.70:
        print("*  WARNING: Strong source signal detected in embeddings!")
        print("*  Interpretation: Embeddings may cluster by generation artifacts.")
    elif mean_acc > 0.55:
        print("*  CAUTION: Moderate source signal detected.")
        print("*  Interpretation: Some artifact influence, but semantic signal may dominate.")
    else:
        print("*  GOOD: Weak/no source signal detected.")
        print("*  Interpretation: Embeddings appear source-agnostic.")

    # Find most discriminative dimensions
    clf.fit(X, y)
    coef = clf.coef_[0]
    top_dims = np.argsort(np.abs(coef))[-10:][::-1]

    print(f"*  Top 10 discriminative dimensions: {top_dims.tolist()}")
    print(f"*  Their coefficients: {coef[top_dims].round(3).tolist()}")

    return {
        'source_classification_accuracy': mean_acc,
        'source_classification_std': std_acc,
        'top_discriminative_dims': top_dims.tolist()
    }


def within_superclass_source_separation(embeddings_norm, metadata):
    """
    Test 4: Within-Superclass Source Separation

    For superclasses present in both CIFAR and GenAI, measure how far apart
    the CIFAR and GenAI centroids are. High separation indicates artifacts
    are pushing same-superclass samples apart based on source.
    """
    print(f"\n[Test 4] Within-Superclass Source Separation")

    # Get all GenAI sources (both novel subclass and novel superclass)
    mask_cifar = metadata['source'] == 'cifar100'
    mask_genai = metadata['source'].str.contains('genai')

    all_superclasses = metadata['superclass_name'].unique()

    separations = []
    results_by_superclass = {}

    for sc in all_superclasses:
        c_mask = mask_cifar & (metadata['superclass_name'] == sc)
        g_mask = mask_genai & (metadata['superclass_name'] == sc)

        if not c_mask.any() or not g_mask.any():
            continue

        # Compute centroids
        c_cent = embeddings_norm[c_mask].mean(axis=0)
        c_cent = c_cent / np.linalg.norm(c_cent)  # Re-normalize

        g_cent = embeddings_norm[g_mask].mean(axis=0)
        g_cent = g_cent / np.linalg.norm(g_cent)  # Re-normalize

        # Cosine distance between centroids
        cosine_dist = 1 - np.dot(c_cent, g_cent)
        separations.append(cosine_dist)
        results_by_superclass[sc] = cosine_dist

    if not separations:
        print("*  ERROR: No overlapping superclasses found.")
        return {}

    mean_sep = np.mean(separations)
    std_sep = np.std(separations)
    max_sep = np.max(separations)
    max_sep_class = list(results_by_superclass.keys())[np.argmax(separations)]

    print(f"*  Analyzed {len(separations)} shared superclasses")
    print(f"*  Mean cosine distance (CIFAR centroid <-> GenAI centroid): {mean_sep:.4f}")
    print(f"*  Std deviation: {std_sep:.4f}")
    print(f"*  Max separation: {max_sep:.4f} (superclass: {max_sep_class})")

    # Interpretation (cosine distance ranges from 0 to 2, typically 0-1 for similar items)
    if mean_sep > 0.3:
        print("*  WARNING: High within-superclass separation detected!")
        print("*  Interpretation: Source artifacts may be dominating semantic clustering.")
    elif mean_sep > 0.15:
        print("*  CAUTION: Moderate within-superclass separation.")
    else:
        print("*  GOOD: Low within-superclass separation.")
        print("*  Interpretation: CIFAR and GenAI samples of same superclass are close.")

    return {
        'mean_within_superclass_separation': mean_sep,
        'std_within_superclass_separation': std_sep,
        'separations_by_superclass': results_by_superclass
    }


def source_agnostic_retrieval_test(embeddings_norm, metadata, mask_cifar, mask_genai):
    """
    Test 5: Source-Agnostic Retrieval Test

    Force cross-source retrieval: For each GenAI image, find nearest CIFAR neighbor.
    For each CIFAR image, find nearest GenAI neighbor.
    Measure: Does the retrieved cross-source neighbor share the same superclass?
    """
    print(f"\n[Test 5] Source-Agnostic Cross-Source Retrieval")

    cifar_emb = embeddings_norm[mask_cifar]
    cifar_super = metadata.loc[mask_cifar, 'superclass_name'].values

    genai_emb = embeddings_norm[mask_genai]
    genai_super = metadata.loc[mask_genai, 'superclass_name'].values

    # GenAI -> CIFAR retrieval
    nn_cifar = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_cifar.fit(cifar_emb)
    distances_g2c, indices_g2c = nn_cifar.kneighbors(genai_emb)
    retrieved_super_g2c = cifar_super[indices_g2c.flatten()]
    acc_g2c = (retrieved_super_g2c == genai_super).mean()

    # CIFAR -> GenAI retrieval
    nn_genai = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_genai.fit(genai_emb)
    distances_c2g, indices_c2g = nn_genai.kneighbors(cifar_emb)
    retrieved_super_c2g = genai_super[indices_c2g.flatten()]
    acc_c2g = (retrieved_super_c2g == cifar_super).mean()

    n_superclasses = metadata['superclass_name'].nunique()
    random_baseline = 1.0 / n_superclasses

    print(f"*  GenAI -> CIFAR retrieval accuracy: {acc_g2c:.2%}")
    print(f"*  CIFAR -> GenAI retrieval accuracy: {acc_c2g:.2%}")
    print(f"*  Random baseline: {random_baseline:.2%} ({n_superclasses} superclasses)")

    # Asymmetry check
    asymmetry = abs(acc_g2c - acc_c2g)
    print(f"*  Retrieval asymmetry: {asymmetry:.2%}")

    if asymmetry > 0.10:
        print("*  CAUTION: Significant asymmetry in cross-source retrieval.")
        print("*  Interpretation: One direction may be biased by artifacts or distribution shift.")

    return {
        'genai_to_cifar_accuracy': acc_g2c,
        'cifar_to_genai_accuracy': acc_c2g,
        'retrieval_asymmetry': asymmetry,
        'mean_distance_g2c': distances_g2c.mean(),
        'mean_distance_c2g': distances_c2g.mean()
    }


def embedding_dimension_analysis(embeddings_norm, metadata, mask_cifar, mask_genai):
    """
    Test 6: Embedding Dimension Analysis

    Identify which embedding dimensions differ most between CIFAR and GenAI.
    Large per-dimension differences suggest source-specific encoding.
    """
    print(f"\n[Test 6] Embedding Dimension Analysis")

    # Compute class-balanced means to avoid class imbalance confounds
    # For each source, compute per-superclass means, then average

    def get_balanced_mean(mask):
        superclasses = metadata.loc[mask, 'superclass_name'].unique()
        superclass_means = []
        for sc in superclasses:
            sc_mask = mask & (metadata['superclass_name'] == sc)
            if sc_mask.any():
                superclass_means.append(embeddings_norm[sc_mask].mean(axis=0))
        if superclass_means:
            return np.mean(superclass_means, axis=0)
        return None

    cifar_mean = get_balanced_mean(mask_cifar)
    genai_mean = get_balanced_mean(mask_genai)

    if cifar_mean is None or genai_mean is None:
        print("*  ERROR: Could not compute balanced means.")
        return {}

    # Per-dimension difference
    dim_diff = genai_mean - cifar_mean
    abs_diff = np.abs(dim_diff)

    # Statistics
    mean_abs_diff = abs_diff.mean()
    max_abs_diff = abs_diff.max()
    max_diff_dim = np.argmax(abs_diff)

    # Top 10 most different dimensions
    top_diff_dims = np.argsort(abs_diff)[-10:][::-1]

    print(f"*  Mean absolute per-dimension difference: {mean_abs_diff:.4f}")
    print(f"*  Max absolute difference: {max_abs_diff:.4f} (dimension {max_diff_dim})")
    print(f"*  Top 10 differing dimensions: {top_diff_dims.tolist()}")
    print(f"*  Their differences: {dim_diff[top_diff_dims].round(4).tolist()}")

    # Check if differences are concentrated or spread
    top10_contribution = abs_diff[top_diff_dims].sum() / abs_diff.sum()
    print(f"*  Top 10 dims account for {top10_contribution:.1%} of total difference")

    if top10_contribution > 0.5:
        print("*  WARNING: Difference concentrated in few dimensions.")
        print("*  Interpretation: Specific features may encode source information.")
    else:
        print("*  Differences are spread across many dimensions.")

    return {
        'mean_abs_dimension_diff': mean_abs_diff,
        'max_abs_dimension_diff': max_abs_diff,
        'top_differing_dimensions': top_diff_dims.tolist(),
        'top10_concentration': top10_contribution
    }


def run_artifact_analysis(embeddings_dir_str):
    """
    Run comprehensive artifact and validity analysis.

    Tests included:
    - Test 1: Cross-domain nearest neighbor retrieval (GenAI -> CIFAR)
    - Test 2: Centroid alignment between sources
    - Test 3: Source classification (can we predict real vs fake?)
    - Test 4: Within-superclass source separation
    - Test 5: Source-agnostic cross-source retrieval
    - Test 6: Embedding dimension analysis
    """
    embeddings_dir = Path(embeddings_dir_str)
    embeddings, metadata = load_data(embeddings_dir)
    if embeddings is None:
        return None

    # Normalize embeddings (Crucial for Cosine Distance)
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')

    # Identify Masks - include ALL GenAI sources
    mask_cifar = metadata['source'] == 'cifar100'
    mask_genai_novel_sub = metadata['source'] == 'genai_novel_subclass'
    mask_genai_novel_super = metadata['source'] == 'genai_novel_superclass'
    mask_genai_all = mask_genai_novel_sub | mask_genai_novel_super

    if not mask_genai_all.any():
        print("Error: No GenAI data found.")
        return None

    print(f"\n{'='*60}\n VALIDITY ANALYSIS: ARTIFACT & SOURCE DETECTION\n{'='*60}")
    print(f"CIFAR-100 Samples: {mask_cifar.sum()}")
    print(f"GenAI Novel Subclass Samples: {mask_genai_novel_sub.sum()}")
    print(f"GenAI Novel Superclass Samples: {mask_genai_novel_super.sum()}")
    print(f"Total GenAI Samples: {mask_genai_all.sum()}")

    # Collect all results
    all_results = {}

    # =========================================================
    # TEST 1: Nearest Neighbor (GenAI -> CIFAR)
    # =========================================================
    print(f"\n[Test 1] Nearest Neighbor Retrieval (GenAI Query -> Real Database)")

    # 1. Train Nearest Neighbor on CIFAR ONLY
    cifar_emb = embeddings_norm[mask_cifar]
    cifar_super = metadata.loc[mask_cifar, 'superclass_name'].values

    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(cifar_emb, cifar_super)

    # 2. Query using GenAI images (novel subclass only for this test)
    if mask_genai_novel_sub.any():
        genai_emb = embeddings_norm[mask_genai_novel_sub]
        genai_super = metadata.loc[mask_genai_novel_sub, 'superclass_name'].values

        # 3. Find closest REAL image for every FAKE image
        distances, indices = knn.kneighbors(genai_emb)
        retrieved_superclass = cifar_super[indices.flatten()]

        # 4. Calculate Accuracy (Does the retrieved Real image share the Superclass?)
        matches = (retrieved_superclass == genai_super)
        accuracy = matches.mean()

        print(f"*  Cross-Domain Accuracy: {accuracy:.2%} (Baseline Random Chance: ~5.00%)")
        all_results['test1_cross_domain_accuracy'] = accuracy
    else:
        print("*  Skipped: No novel subclass data available")

    # =========================================================
    # TEST 2: Centroid Alignment (Robust Version)
    # =========================================================
    print(f"\n[Test 2] Centroid Alignment (Vector Angle between Real & Fake)")

    # 1. Get the UNION of all superclasses present in the metadata
    all_superclasses = np.unique(metadata['superclass_name'])

    cifar_centroids = []
    genai_centroids = []
    used_superclasses = []

    print(f"*  Scanning {len(all_superclasses)} potential superclasses...")

    for sc in all_superclasses:
        # --- CIFAR Check ---
        c_mask = (metadata['source'] == 'cifar100') & (metadata['superclass_name'] == sc)
        if not c_mask.any():
            continue

        # --- GenAI Check (include both novel subclass and novel superclass) ---
        g_mask = mask_genai_all & (metadata['superclass_name'] == sc)
        if not g_mask.any():
            continue

        # --- Safe Calculation ---
        c_cent = embeddings_norm[c_mask].mean(axis=0)
        g_cent = embeddings_norm[g_mask].mean(axis=0)

        cifar_centroids.append(c_cent)
        genai_centroids.append(g_cent)
        used_superclasses.append(sc)

    if len(used_superclasses) > 0:
        # Convert to numpy for matrix operations
        cifar_centroids = np.array(cifar_centroids)
        genai_centroids = np.array(genai_centroids)

        print(f"*  Successfully aligned {len(used_superclasses)} shared superclasses.")

        # Compute Sim Matrix (Rows=Real, Cols=Fake)
        sim_matrix = cosine_similarity(cifar_centroids, genai_centroids)

        # Diagonal = Same Superclass (Fish vs Fish)
        same_class_sim = np.diag(sim_matrix).mean()

        # Off-Diagonal = Different Superclass (Fish vs Flowers)
        mask_off = ~np.eye(sim_matrix.shape[0], dtype=bool)
        diff_class_sim = sim_matrix[mask_off].mean()

        print(f"*  Avg Similarity (Same Superclass):      {same_class_sim:.4f}")
        print(f"*  Avg Similarity (Different Superclass): {diff_class_sim:.4f}")

        delta = same_class_sim - diff_class_sim
        print(f"*  Delta (Semantic Strength):             {delta:.4f}")

        if delta > 0.15:
            print("*  RESULT: STRONG Semantic Alignment.")
        else:
            print("*  RESULT: WEAK Alignment.")

        all_results['test2_same_class_similarity'] = same_class_sim
        all_results['test2_diff_class_similarity'] = diff_class_sim
        all_results['test2_semantic_delta'] = delta
    else:
        print("*  Skipped: No overlapping superclasses found between CIFAR and GenAI.")

    # =========================================================
    # TEST 3: Source Classification (NEW)
    # =========================================================
    test3_results = source_classification_test(
        embeddings_norm, metadata, mask_cifar, mask_genai_all
    )
    all_results.update(test3_results)

    # =========================================================
    # TEST 4: Within-Superclass Source Separation (NEW)
    # =========================================================
    test4_results = within_superclass_source_separation(embeddings_norm, metadata)
    all_results.update(test4_results)

    # =========================================================
    # TEST 5: Source-Agnostic Retrieval (NEW)
    # =========================================================
    test5_results = source_agnostic_retrieval_test(
        embeddings_norm, metadata, mask_cifar, mask_genai_all
    )
    all_results.update(test5_results)

    # =========================================================
    # TEST 6: Embedding Dimension Analysis (NEW)
    # =========================================================
    test6_results = embedding_dimension_analysis(
        embeddings_norm, metadata, mask_cifar, mask_genai_all
    )
    all_results.update(test6_results)

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*60}\n VALIDITY ANALYSIS SUMMARY\n{'='*60}")

    # Compute overall validity score
    warnings = 0
    if all_results.get('source_classification_accuracy', 0) > 0.70:
        warnings += 1
        print("WARNING: High source classification accuracy detected")
    if all_results.get('mean_within_superclass_separation', 0) > 0.3:
        warnings += 1
        print("WARNING: High within-superclass source separation detected")
    if all_results.get('retrieval_asymmetry', 0) > 0.10:
        warnings += 1
        print("WARNING: Asymmetric cross-source retrieval detected")
    if all_results.get('top10_concentration', 0) > 0.5:
        warnings += 1
        print("WARNING: Source differences concentrated in few dimensions")

    if warnings == 0:
        print("\nOVERALL: No major validity concerns detected.")
    elif warnings <= 2:
        print(f"\nOVERALL: {warnings} potential validity concern(s) - review recommended.")
    else:
        print(f"\nOVERALL: {warnings} validity warnings - results may be confounded by artifacts.")

    # Save results to JSON
    results_path = embeddings_dir / "validity_analysis_results.json"
    # Convert numpy types for JSON serialization
    serializable_results = {}
    for k, v in all_results.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable_results[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, dict):
            serializable_results[k] = {
                str(dk): float(dv) if isinstance(dv, (np.floating, np.integer)) else dv
                for dk, dv in v.items()
            }
        else:
            serializable_results[k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    # Update path to your embeddings folder
    run_artifact_analysis("/home/alex/data/embeddings/DINOv2")