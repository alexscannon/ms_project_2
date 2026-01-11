import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import logging

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

def run_artifact_analysis(embeddings_dir_str):
    embeddings_dir = Path(embeddings_dir_str)
    embeddings, metadata = load_data(embeddings_dir)
    if embeddings is None: return

    # Normalize embeddings (Crucial for Cosine Distance)
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')

    # Identify Masks
    mask_cifar = metadata['source'] == 'cifar100'
    mask_genai = metadata['source'] == 'genai_novel_subclass' # Group A

    if not mask_genai.any():
        print("Error: No Group A (Novel Subclass) data found.")
        return

    print(f"\n{'='*60}\n ANALYSIS: CROSS-DOMAIN RETRIEVAL (Artifact Check)\n{'='*60}")
    print(f"CIFAR-100 Samples: {mask_cifar.sum()}")
    print(f"GenAI Group A Samples: {mask_genai.sum()}")

    # =========================================================
    # TEST 1: Nearest Neighbor (GenAI -> CIFAR)
    # =========================================================
    print(f"\n[Test 1] Nearest Neighbor Retrieval (GenAI Query -> Real Database)")

    # 1. Train Nearest Neighbor on CIFAR ONLY
    cifar_emb = embeddings_norm[mask_cifar]
    cifar_super = metadata.loc[mask_cifar, 'superclass_name'].values

    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(cifar_emb, cifar_super)

    # 2. Query using GenAI images
    genai_emb = embeddings_norm[mask_genai]
    genai_super = metadata.loc[mask_genai, 'superclass_name'].values

    # 3. Find closest REAL image for every FAKE image
    distances, indices = knn.kneighbors(genai_emb)
    retrieved_superclass = cifar_super[indices.flatten()]

    # 4. Calculate Accuracy (Does the retrieved Real image share the Superclass?)
    matches = (retrieved_superclass == genai_super)
    accuracy = matches.mean()

    print(f"*  Cross-Domain Accuracy: {accuracy:.2%} (Baseline Random Chance: ~5.00%)")

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
            # This superclass is not in CIFAR (e.g. it's a novel superclass from Group B)
            continue

        # --- GenAI Check ---
        g_mask = (metadata['source'] == 'genai_novel_subclass') & (metadata['superclass_name'] == sc)
        if not g_mask.any():
            # This superclass exists in CIFAR but is missing from your GenAI set
            continue

        # --- Safe Calculation ---
        # We now know both masks have at least 1 sample.
        c_cent = embeddings_norm[c_mask].mean(axis=0)
        g_cent = embeddings_norm[g_mask].mean(axis=0)

        cifar_centroids.append(c_cent)
        genai_centroids.append(g_cent)
        used_superclasses.append(sc)

    # Convert to numpy for matrix operations
    cifar_centroids = np.array(cifar_centroids)
    genai_centroids = np.array(genai_centroids)

    if len(used_superclasses) == 0:
        print("ERROR: No overlapping superclasses found between CIFAR and GenAI Group A.")
        return

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
        print("\n✅ RESULT: STRONG Semantic Alignment.\n")
    else:
        print("\n⚠️ RESULT: WEAK Alignment.\n")


if __name__ == "__main__":
    # Update path to your embeddings folder
    run_artifact_analysis("/home/alex/data/embeddings/DINOv2")