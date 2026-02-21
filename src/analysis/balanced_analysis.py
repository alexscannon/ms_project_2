"""
Balanced Dataset Analysis

Runs validity tests on a balanced subset of CIFAR and GenAI data
to check if class imbalance affects the results.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("msproject")


def create_balanced_subset(embeddings, metadata, random_state=42):
    """
    Create a balanced subset by downsampling CIFAR to match GenAI count.
    """
    np.random.seed(random_state)

    mask_cifar = metadata['source'] == 'cifar100'
    mask_genai = ~mask_cifar

    n_cifar = mask_cifar.sum()
    n_genai = mask_genai.sum()

    print(f"Original dataset: CIFAR={n_cifar}, GenAI={n_genai}")

    # Get indices
    cifar_indices = np.where(mask_cifar)[0]
    genai_indices = np.where(mask_genai)[0]

    # Downsample CIFAR to match GenAI
    sampled_cifar_indices = np.random.choice(cifar_indices, size=n_genai, replace=False)

    # Combine indices
    balanced_indices = np.concatenate([sampled_cifar_indices, genai_indices])
    np.random.shuffle(balanced_indices)

    # Create balanced dataset
    balanced_embeddings = embeddings[balanced_indices]
    balanced_metadata = metadata.iloc[balanced_indices].reset_index(drop=True)

    n_cifar_balanced = (balanced_metadata['source'] == 'cifar100').sum()
    n_genai_balanced = (balanced_metadata['source'] != 'cifar100').sum()
    print(f"Balanced dataset: CIFAR={n_cifar_balanced}, GenAI={n_genai_balanced}")

    return balanced_embeddings, balanced_metadata


def run_balanced_source_classification(embeddings_norm, metadata):
    """
    Test: Source Classification on balanced data
    """
    print(f"\n[Balanced Test 1] Source Classification")

    is_cifar = (metadata['source'] == 'cifar100').values
    y = (~is_cifar).astype(int)  # 0 = CIFAR, 1 = GenAI

    print(f"*  Class distribution: CIFAR={is_cifar.sum()}, GenAI={(~is_cifar).sum()}")

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings_norm, y, cv=cv, scoring='accuracy')

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"*  5-Fold CV Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
    print(f"*  Baseline (random): 50.00%")

    if mean_acc > 0.70:
        print("*  WARNING: Strong source signal persists even with balanced data")
    elif mean_acc > 0.55:
        print("*  CAUTION: Moderate source signal detected")
    else:
        print("*  GOOD: Weak/no source signal")

    return {
        'balanced_source_classification_accuracy': mean_acc,
        'balanced_source_classification_std': std_acc
    }


def run_balanced_retrieval_bias(embeddings_norm, metadata, k=5):
    """
    Test: Retrieval Source Bias on balanced data
    """
    print(f"\n[Balanced Test 2] Retrieval Source Bias (k={k})")

    is_cifar = (metadata['source'] == 'cifar100').values

    n_cifar = is_cifar.sum()
    n_genai = (~is_cifar).sum()
    n_total = len(metadata)

    expected_same_source = 0.5  # With balanced data, expected is 50%

    print(f"*  Dataset: CIFAR={n_cifar} (50%), GenAI={n_genai} (50%)")

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings_norm)
    distances, indices = nn.kneighbors(embeddings_norm)

    neighbor_indices = indices[:, 1:]  # Exclude self

    # Calculate same-source retrieval rate
    same_source_counts = []
    for i in range(len(metadata)):
        neighbors = neighbor_indices[i]
        if is_cifar[i]:
            same_source = is_cifar[neighbors].sum()
        else:
            same_source = (~is_cifar[neighbors]).sum()
        same_source_counts.append(same_source / k)

    same_source_counts = np.array(same_source_counts)

    cifar_same_rate = same_source_counts[is_cifar].mean()
    genai_same_rate = same_source_counts[~is_cifar].mean()

    cifar_bias = cifar_same_rate - expected_same_source
    genai_bias = genai_same_rate - expected_same_source
    avg_bias = (abs(cifar_bias) + abs(genai_bias)) / 2

    print(f"*  CIFAR same-source rate: {cifar_same_rate:.1%} (expected: 50%)")
    print(f"*  GenAI same-source rate: {genai_same_rate:.1%} (expected: 50%)")
    print(f"*  CIFAR bias: {cifar_bias:+.1%}")
    print(f"*  GenAI bias: {genai_bias:+.1%}")
    print(f"*  Average bias: {avg_bias:.1%}")

    if avg_bias > 0.15:
        print("*  WARNING: Strong source clustering persists")
    elif avg_bias > 0.05:
        print("*  CAUTION: Moderate source clustering")
    else:
        print("*  GOOD: Sources are well-mixed")

    return {
        'balanced_cifar_same_source_rate': cifar_same_rate,
        'balanced_genai_same_source_rate': genai_same_rate,
        'balanced_cifar_bias': cifar_bias,
        'balanced_genai_bias': genai_bias,
        'balanced_average_bias': avg_bias
    }


def run_balanced_within_superclass_separation(embeddings_norm, metadata):
    """
    Test: Within-Superclass Source Separation on balanced data
    """
    print(f"\n[Balanced Test 3] Within-Superclass Source Separation")

    mask_cifar = metadata['source'] == 'cifar100'
    mask_genai = ~mask_cifar

    all_superclasses = metadata['superclass_name'].unique()

    separations = []
    results_by_superclass = {}

    for sc in all_superclasses:
        c_mask = mask_cifar & (metadata['superclass_name'] == sc)
        g_mask = mask_genai & (metadata['superclass_name'] == sc)

        if not c_mask.any() or not g_mask.any():
            continue

        c_cent = embeddings_norm[c_mask].mean(axis=0)
        c_cent = c_cent / np.linalg.norm(c_cent)

        g_cent = embeddings_norm[g_mask].mean(axis=0)
        g_cent = g_cent / np.linalg.norm(g_cent)

        cosine_dist = 1 - np.dot(c_cent, g_cent)
        separations.append(cosine_dist)
        results_by_superclass[sc] = cosine_dist

    if not separations:
        print("*  ERROR: No overlapping superclasses found")
        return {}

    mean_sep = np.mean(separations)
    std_sep = np.std(separations)

    print(f"*  Analyzed {len(separations)} shared superclasses")
    print(f"*  Mean separation: {mean_sep:.4f}")
    print(f"*  Std deviation: {std_sep:.4f}")

    if mean_sep > 0.3:
        print("*  WARNING: High within-superclass separation persists")
    elif mean_sep > 0.15:
        print("*  CAUTION: Moderate separation")
    else:
        print("*  GOOD: Low separation")

    return {
        'balanced_mean_within_superclass_separation': mean_sep,
        'balanced_std_within_superclass_separation': std_sep,
        'balanced_separations_by_superclass': results_by_superclass
    }


def run_balanced_analysis(embeddings_dir_str):
    """
    Run validity analysis on a balanced subset of the data.
    """
    embeddings_dir = Path(embeddings_dir_str)

    # Load data
    emb_path = embeddings_dir / "embeddings.npy"
    meta_path = embeddings_dir / "metadata.csv"

    if not emb_path.exists():
        print(f"Error: Embeddings not found at {emb_path}")
        return None

    print(f"\n{'='*60}")
    print(" BALANCED DATASET VALIDITY ANALYSIS")
    print(f"{'='*60}")

    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)

    # Create balanced subset
    balanced_emb, balanced_meta = create_balanced_subset(embeddings, metadata)

    # Normalize
    balanced_emb_norm = normalize(balanced_emb, axis=1, norm='l2')

    all_results = {}

    # Run tests
    test1 = run_balanced_source_classification(balanced_emb_norm, balanced_meta)
    all_results.update(test1)

    test2 = run_balanced_retrieval_bias(balanced_emb_norm, balanced_meta)
    all_results.update(test2)

    test3 = run_balanced_within_superclass_separation(balanced_emb_norm, balanced_meta)
    all_results.update(test3)

    # Summary
    print(f"\n{'='*60}")
    print(" BALANCED ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print(f"\nComparison (Imbalanced -> Balanced):")
    print(f"*  Source Classification: 98.98% -> {all_results.get('balanced_source_classification_accuracy', 0):.2%}")
    print(f"*  Within-Superclass Separation: 0.579 -> {all_results.get('balanced_mean_within_superclass_separation', 0):.4f}")
    print(f"*  Retrieval Bias: 91% -> {all_results.get('balanced_average_bias', 0):.1%}")

    # Save results
    results_path = embeddings_dir / "balanced_analysis_results.json"

    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, dict):
            serializable[k] = {str(dk): float(dv) for dk, dv in v.items()}
        else:
            serializable[k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_balanced_analysis(sys.argv[1])
    else:
        print("Usage: python balanced_analysis.py <embeddings_dir>")
