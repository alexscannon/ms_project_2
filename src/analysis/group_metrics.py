import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# Import your existing loader
from .visualization import load_analysis_data

# Configure Logger
logger = logging.getLogger("msproject")
logging.basicConfig(level=logging.INFO)

def calculate_group_metrics(embeddings, metadata):
    """
    Calculates semantic coherence metrics broken down by data source.
    """

    # 1. Define Groups based on 'source' column
    # Adjust these string matches if your exact source strings differ in the CSV
    groups = {
        'Baseline (CIFAR100)': metadata['source'] == 'cifar100',
        'Group A (Novel Subclass)': metadata['source'] == 'genai_novel_subclass',
        'Group B (Novel Superclass)': metadata['source'] == 'genai_novel_superclass',
        'Group B (GenAI CIFAR100)': metadata['source'] == 'genai_ind'
    }

    results = []

    # Pre-compute normalized embeddings for Cosine Distance calculations
    # Cosine Distance = 1 - Cosine Similarity
    # If vectors are normalized, Cosine Sim = dot product.
    # Dist = 1 - (u . v)
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')

    logger.info(f"--------- COMPUTING METRICS --------")

    # ====================================================
    # Metric 1: Global Retrieval Accuracy (k-NN)
    # ====================================================
    # Hypothesis: Does a GenAI image retrieve a neighbor of the SAME class?

    # We fit the Nearest Neighbor on ALL data (Global Space)
    # This tests if the model confuses a GenAI concept with a CIFAR concept.
    knn = KNeighborsClassifier(n_neighbors=2, metric='cosine') # k=2 because nearest is itself
    knn.fit(embeddings_norm, metadata['subclass_id'])

    # Find neighbors for all points
    distances, indices = knn.kneighbors(embeddings_norm)

    # The nearest neighbor (index 0) is the point itself. We want index 1.
    neighbor_indices = indices[:, 1]
    neighbor_classes = metadata['subclass_id'].iloc[neighbor_indices].values
    true_classes = metadata['subclass_id'].values

    # Boolean array: Did it find the correct class?
    correct_retrieval = (neighbor_classes == true_classes)

    # ====================================================
    # Metric 2 & 3: Coherence (Intra/Inter) & Silhouette
    # ====================================================

    for group_name, mask in groups.items():
        if not mask.any():
            logger.warning(f"Skipping {group_name}: No samples found.")
            continue

        logger.info(f"Processing {group_name} ({mask.sum()} samples)...")

        # Subset data
        group_emb = embeddings_norm[mask]
        group_labels = metadata.loc[mask, 'subclass_id'].values

        # --- A. Silhouette Score ---
        # A value near +1 indicates samples are far away from the neighboring clusters.
        # 0 indicates that the sample is on or very close to the decision boundary.
        # (Sample size limited to 10k for speed, if dataset is huge)
        sample_size = min(len(group_emb), 10000)
        sil_score = silhouette_score(
            group_emb,
            group_labels,
            metric='cosine',
            sample_size=sample_size
        )

        # --- B. Intra-Class Distance (Compactness) ---
        # Mean distance between points and their class centroid
        unique_classes = np.unique(group_labels)
        intra_dists = []
        class_centroids = []

        for cls in unique_classes:
            cls_mask = (group_labels == cls)
            cls_emb = group_emb[cls_mask]

            # Centroid
            centroid = cls_emb.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid) # Re-normalize
            class_centroids.append(centroid)

            # Distances from points to centroid
            # cosine dist = 1 - dot_product (since normalized)
            dists = 1 - np.dot(cls_emb, centroid)
            intra_dists.extend(dists)

        mean_intra_dist = np.mean(intra_dists)

        # --- C. Inter-Class Distance (Separation) ---
        # Mean distance between class centroids
        class_centroids = np.array(class_centroids)
        if len(class_centroids) > 1:
            # Pairwise distance between centroids
            centroid_dists = pairwise_distances(class_centroids, metric='cosine')
            # Get upper triangle only (exclude diagonal 0s and duplicates)
            triu_indices = np.triu_indices(len(class_centroids), k=1)
            mean_inter_dist = np.mean(centroid_dists[triu_indices])
        else:
            mean_inter_dist = 0.0

        # --- D. Retrieval Accuracy for this Group ---
        group_acc = correct_retrieval[mask].mean()

        # Store Result
        results.append({
            'Group': group_name,
            'Silhouette Score': sil_score,
            'Intra-Class Dist (Lower is Better)': mean_intra_dist,
            'Inter-Class Dist (Higher is Better)': mean_inter_dist,
            'Ratio (Inter/Intra)': mean_inter_dist / mean_intra_dist if mean_intra_dist > 0 else 0,
            'Retrieval Acc (k=1)': group_acc
        })

    df_results = pd.DataFrame(results)

    print(f"\n {'=' * 60} \n FINAL RESULTS TABLE...\n {'=' * 60}")
    print(df_results.round(4).to_string(index=False))

    # Run RQ2 Novel Class Analysis
    rq2_results = calculate_novel_class_metrics(embeddings_norm, metadata)

    return df_results, rq2_results


def calculate_novel_class_metrics(
    embeddings_norm: np.ndarray,
    metadata: pd.DataFrame
) -> dict:
    """
    Calculate metrics specifically for RQ2: Novel Class Semantic Coherence.

    Tests whether DINOv2 produces semantically coherent embeddings for classes
    it has never seen (novel subclasses and superclasses).

    Args:
        embeddings_norm: L2-normalized embeddings.
        metadata: DataFrame with source, superclass_name, subclass_name columns.

    Returns:
        dict: Results for RQ2 analysis.
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"\n{'='*60}")
    print(" RQ2 ANALYSIS: NOVEL CLASS SEMANTIC COHERENCE")
    print(f"{'='*60}")

    results = {}

    # Define masks
    mask_cifar = metadata['source'] == 'cifar100'
    mask_novel_sub = metadata['source'] == 'genai_novel_subclass'
    mask_novel_super = metadata['source'] == 'genai_novel_superclass'

    # =========================================================
    # TEST 1: Novel Subclass → Parent Superclass Retrieval (RQ2)
    # =========================================================
    print(f"\n[RQ2 Test 1] Novel Subclass → Parent Superclass Retrieval")
    print("*  Question: Do novel subclasses retrieve CIFAR samples from their parent superclass?")

    if mask_novel_sub.any() and mask_cifar.any():
        # Get CIFAR embeddings and superclass labels
        cifar_emb = embeddings_norm[mask_cifar]
        cifar_superclass = metadata.loc[mask_cifar, 'superclass_name'].values

        # Get novel subclass embeddings and their expected parent superclass
        novel_sub_emb = embeddings_norm[mask_novel_sub]
        novel_sub_superclass = metadata.loc[mask_novel_sub, 'superclass_name'].values

        # Find nearest CIFAR neighbor for each novel subclass image
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(cifar_emb)
        distances, indices = nn.kneighbors(novel_sub_emb)

        # Check if retrieved CIFAR sample has same superclass
        retrieved_superclass = cifar_superclass[indices.flatten()]
        correct_superclass = (retrieved_superclass == novel_sub_superclass)
        superclass_retrieval_acc = correct_superclass.mean()

        # Per-superclass breakdown
        unique_superclasses = np.unique(novel_sub_superclass)
        per_superclass_acc = {}
        for sc in unique_superclasses:
            sc_mask = novel_sub_superclass == sc
            if sc_mask.sum() > 0:
                per_superclass_acc[sc] = correct_superclass[sc_mask].mean()

        print(f"*  Overall superclass retrieval accuracy: {superclass_retrieval_acc:.2%}")
        print(f"*  Random baseline: {1/len(unique_superclasses):.2%} ({len(unique_superclasses)} superclasses)")
        print(f"*  Per-superclass breakdown:")
        for sc, acc in sorted(per_superclass_acc.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {sc}: {acc:.2%}")

        if superclass_retrieval_acc > 0.5:
            print("*  RESULT: Novel subclasses align well with parent superclass semantics.")
        else:
            print("*  RESULT: Weak alignment - novel subclasses may not fit expected superclass.")

        results['novel_subclass_superclass_retrieval_acc'] = superclass_retrieval_acc
        results['novel_subclass_per_superclass_acc'] = per_superclass_acc
    else:
        print("*  SKIPPED: No novel subclass or CIFAR data available.")

    # =========================================================
    # TEST 2: Within-Class Similarity Comparison (RQ2)
    # =========================================================
    print(f"\n[RQ2 Test 2] Within-Class Similarity: Novel vs CIFAR Baseline")
    print("*  Question: Do novel classes have similar within-class coherence as CIFAR classes?")

    def compute_intra_class_distances(emb, labels):
        """Compute mean intra-class distance for each class."""
        unique_labels = np.unique(labels)
        class_intra_dists = {}

        for lbl in unique_labels:
            cls_mask = labels == lbl
            cls_emb = emb[cls_mask]

            if len(cls_emb) < 2:
                continue

            # Compute centroid
            centroid = cls_emb.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Mean distance to centroid
            dists = 1 - np.dot(cls_emb, centroid)
            class_intra_dists[lbl] = dists.mean()

        return class_intra_dists

    # CIFAR baseline
    cifar_labels = metadata.loc[mask_cifar, 'subclass_name'].values
    cifar_intra = compute_intra_class_distances(embeddings_norm[mask_cifar], cifar_labels)
    cifar_mean_intra = np.mean(list(cifar_intra.values())) if cifar_intra else 0

    # Novel subclasses
    if mask_novel_sub.any():
        novel_sub_labels = metadata.loc[mask_novel_sub, 'subclass_name'].values
        novel_sub_intra = compute_intra_class_distances(
            embeddings_norm[mask_novel_sub], novel_sub_labels
        )
        novel_sub_mean_intra = np.mean(list(novel_sub_intra.values())) if novel_sub_intra else 0
    else:
        novel_sub_mean_intra = 0
        novel_sub_intra = {}

    # Novel superclasses (use subclass-level for fair comparison)
    if mask_novel_super.any():
        novel_super_labels = metadata.loc[mask_novel_super, 'subclass_name'].values
        novel_super_intra = compute_intra_class_distances(
            embeddings_norm[mask_novel_super], novel_super_labels
        )
        novel_super_mean_intra = np.mean(list(novel_super_intra.values())) if novel_super_intra else 0
    else:
        novel_super_mean_intra = 0
        novel_super_intra = {}

    print(f"*  CIFAR-100 mean intra-class distance: {cifar_mean_intra:.4f} (baseline)")
    print(f"*  Novel Subclass mean intra-class distance: {novel_sub_mean_intra:.4f}")
    print(f"*  Novel Superclass mean intra-class distance: {novel_super_mean_intra:.4f}")

    # Express as percentage of baseline
    if cifar_mean_intra > 0:
        novel_sub_ratio = novel_sub_mean_intra / cifar_mean_intra
        novel_super_ratio = novel_super_mean_intra / cifar_mean_intra
        print(f"*  Novel Subclass coherence: {novel_sub_ratio:.1%} of CIFAR baseline")
        print(f"*  Novel Superclass coherence: {novel_super_ratio:.1%} of CIFAR baseline")

        if novel_sub_ratio < 1.2 and novel_super_ratio < 1.2:
            print("*  RESULT: Novel classes have comparable within-class coherence to CIFAR.")
        elif novel_sub_ratio < 1.5 and novel_super_ratio < 1.5:
            print("*  RESULT: Novel classes have slightly lower coherence than CIFAR.")
        else:
            print("*  RESULT: Novel classes are less coherent than CIFAR baseline.")

        results['cifar_mean_intra_class_dist'] = cifar_mean_intra
        results['novel_subclass_mean_intra_class_dist'] = novel_sub_mean_intra
        results['novel_superclass_mean_intra_class_dist'] = novel_super_mean_intra
        results['novel_subclass_coherence_ratio'] = novel_sub_ratio
        results['novel_superclass_coherence_ratio'] = novel_super_ratio

    # =========================================================
    # TEST 3: Per-Novel-Superclass Coherence Breakdown (RQ2)
    # =========================================================
    print(f"\n[RQ2 Test 3] Per-Novel-Superclass Coherence Breakdown")
    print("*  Question: Which novel superclasses are most/least coherent?")

    if mask_novel_super.any():
        novel_super_data = metadata[mask_novel_super]
        unique_novel_superclasses = novel_super_data['superclass_name'].unique()

        superclass_coherence = {}

        for sc in unique_novel_superclasses:
            sc_mask = mask_novel_super & (metadata['superclass_name'] == sc)
            sc_emb = embeddings_norm[sc_mask]
            sc_subclass_labels = metadata.loc[sc_mask, 'subclass_name'].values

            if len(sc_emb) < 2:
                continue

            # Compute silhouette score for this superclass
            unique_subclasses = np.unique(sc_subclass_labels)
            if len(unique_subclasses) > 1:
                # Use subclass labels for silhouette
                from sklearn.metrics import silhouette_score
                try:
                    sil = silhouette_score(sc_emb, sc_subclass_labels, metric='cosine')
                except ValueError:
                    sil = 0.0
            else:
                sil = 0.0

            # Compute overall intra-superclass distance
            centroid = sc_emb.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            intra_dist = (1 - np.dot(sc_emb, centroid)).mean()

            superclass_coherence[sc] = {
                'silhouette': sil,
                'intra_dist': intra_dist,
                'n_samples': len(sc_emb),
                'n_subclasses': len(unique_subclasses)
            }

        print(f"*  Analyzed {len(superclass_coherence)} novel superclasses:")
        for sc, metrics in sorted(superclass_coherence.items(), key=lambda x: x[1]['silhouette'], reverse=True):
            print(f"     - {sc}: silhouette={metrics['silhouette']:.3f}, "
                  f"intra_dist={metrics['intra_dist']:.4f}, "
                  f"samples={metrics['n_samples']}, subclasses={metrics['n_subclasses']}")

        results['novel_superclass_coherence'] = superclass_coherence
    else:
        print("*  SKIPPED: No novel superclass data available.")

    print(f"\n{'='*60}")
    print(" RQ2 ANALYSIS COMPLETE")
    print(f"{'='*60}")

    return results

