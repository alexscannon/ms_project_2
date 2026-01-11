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
        'Group B (Novel Superclass)': metadata['source'] == 'genai_novel_superclass'
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

    return df_results

