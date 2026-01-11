"""
Source Leakage Analysis Module

This module provides tests to detect potential training data overlap effects
between the foundation model (DINOv2/v3) and GenAI-generated images.

Tests included:
1. Retrieval Source Bias Test - Do GenAI images disproportionately retrieve other GenAI images?
2. Superclass Confusion Matrix by Source - Do CIFAR and GenAI have different error patterns?
3. Cross-Model Consistency Test - [FUTURE] Requires tracking which GenAI model produced each image
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("msproject")


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


def retrieval_source_bias_test(embeddings_norm, metadata, k=5):
    """
    Test 1: Retrieval Source Bias Test

    For each image, find its k nearest neighbors in the combined dataset.
    Measure what fraction of neighbors come from the same source vs different source.

    If GenAI images disproportionately retrieve other GenAI images (beyond what's
    expected from dataset proportions), this indicates source-based clustering
    that could confound semantic analysis.

    Args:
        embeddings_norm: L2-normalized embeddings
        metadata: DataFrame with 'source' column
        k: Number of neighbors to consider

    Returns:
        dict with test results
    """
    print(f"\n[Source Leakage Test 1] Retrieval Source Bias (k={k})")

    # Create binary source labels
    is_cifar = (metadata['source'] == 'cifar100').values
    is_genai = ~is_cifar

    n_cifar = is_cifar.sum()
    n_genai = is_genai.sum()
    n_total = len(metadata)

    # Expected proportions if no bias
    expected_cifar_prop = n_cifar / n_total
    expected_genai_prop = n_genai / n_total

    print(f"*  Dataset composition: CIFAR={n_cifar} ({expected_cifar_prop:.1%}), GenAI={n_genai} ({expected_genai_prop:.1%})")

    # Fit nearest neighbors on all data
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')  # k+1 because first neighbor is self
    nn.fit(embeddings_norm)
    distances, indices = nn.kneighbors(embeddings_norm)

    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]  # Shape: (n_samples, k)

    # For each sample, count how many neighbors are from same source
    same_source_counts = []
    for i in range(len(metadata)):
        neighbors = neighbor_indices[i]
        if is_cifar[i]:
            same_source = is_cifar[neighbors].sum()
        else:
            same_source = is_genai[neighbors].sum()
        same_source_counts.append(same_source / k)

    same_source_counts = np.array(same_source_counts)

    # Compute statistics by source
    cifar_same_source_rate = same_source_counts[is_cifar].mean()
    genai_same_source_rate = same_source_counts[is_genai].mean()

    # Expected same-source rates if no bias
    expected_cifar_same = expected_cifar_prop
    expected_genai_same = expected_genai_prop

    # Bias = observed - expected
    cifar_bias = cifar_same_source_rate - expected_cifar_same
    genai_bias = genai_same_source_rate - expected_genai_same

    print(f"*  CIFAR images: {cifar_same_source_rate:.1%} of neighbors are CIFAR (expected: {expected_cifar_same:.1%})")
    print(f"*  GenAI images: {genai_same_source_rate:.1%} of neighbors are GenAI (expected: {expected_genai_same:.1%})")
    print(f"*  CIFAR retrieval bias: {cifar_bias:+.1%}")
    print(f"*  GenAI retrieval bias: {genai_bias:+.1%}")

    # Interpretation
    avg_bias = (abs(cifar_bias) + abs(genai_bias)) / 2
    if avg_bias > 0.15:
        print("*  WARNING: Strong source retrieval bias detected!")
        print("*  Interpretation: Images cluster by source, not just by semantic content.")
    elif avg_bias > 0.05:
        print("*  CAUTION: Moderate source retrieval bias detected.")
    else:
        print("*  GOOD: Low source retrieval bias.")
        print("*  Interpretation: Retrieval appears source-agnostic.")

    return {
        'cifar_same_source_rate': cifar_same_source_rate,
        'genai_same_source_rate': genai_same_source_rate,
        'cifar_retrieval_bias': cifar_bias,
        'genai_retrieval_bias': genai_bias,
        'average_retrieval_bias': avg_bias,
        'expected_cifar_proportion': expected_cifar_prop,
        'expected_genai_proportion': expected_genai_prop
    }


def superclass_confusion_analysis(embeddings_norm, metadata):
    """
    Test 2: Superclass Confusion Matrix by Source

    For each image, predict its superclass using k-NN.
    Compare the confusion patterns between CIFAR and GenAI.

    If error patterns differ systematically, it suggests the model
    treats the two sources differently (potential distribution shift).

    Args:
        embeddings_norm: L2-normalized embeddings
        metadata: DataFrame with 'source' and 'superclass_name' columns

    Returns:
        dict with confusion analysis results
    """
    print(f"\n[Source Leakage Test 2] Superclass Confusion Analysis")

    is_cifar = (metadata['source'] == 'cifar100').values
    is_genai = ~is_cifar

    superclass_labels = metadata['superclass_name'].values
    unique_superclasses = np.unique(superclass_labels)
    n_classes = len(unique_superclasses)

    # Create label encoding
    label_to_idx = {label: idx for idx, label in enumerate(unique_superclasses)}
    y_true = np.array([label_to_idx[label] for label in superclass_labels])

    # Predict superclass using k-NN (k=5, majority vote)
    nn = NearestNeighbors(n_neighbors=6, metric='cosine')  # 6 = 5 neighbors + self
    nn.fit(embeddings_norm)
    distances, indices = nn.kneighbors(embeddings_norm)

    # Exclude self
    neighbor_indices = indices[:, 1:]

    # Predict by majority vote
    y_pred = []
    for i in range(len(metadata)):
        neighbor_labels = y_true[neighbor_indices[i]]
        # Majority vote
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        y_pred.append(unique[np.argmax(counts)])
    y_pred = np.array(y_pred)

    # Compute accuracy by source
    cifar_acc = (y_pred[is_cifar] == y_true[is_cifar]).mean()
    genai_acc = (y_pred[is_genai] == y_true[is_genai]).mean()

    print(f"*  CIFAR k-NN accuracy: {cifar_acc:.2%}")
    print(f"*  GenAI k-NN accuracy: {genai_acc:.2%}")
    print(f"*  Accuracy gap: {abs(cifar_acc - genai_acc):.2%}")

    # Compute confusion matrices
    cm_cifar = confusion_matrix(y_true[is_cifar], y_pred[is_cifar], labels=range(n_classes))
    cm_genai = confusion_matrix(y_true[is_genai], y_pred[is_genai], labels=range(n_classes))

    # Normalize confusion matrices
    cm_cifar_norm = cm_cifar.astype(float) / (cm_cifar.sum(axis=1, keepdims=True) + 1e-10)
    cm_genai_norm = cm_genai.astype(float) / (cm_genai.sum(axis=1, keepdims=True) + 1e-10)

    # Compute difference in error patterns
    # Only consider off-diagonal elements (errors)
    mask = ~np.eye(n_classes, dtype=bool)
    error_diff = np.abs(cm_cifar_norm - cm_genai_norm)[mask]
    mean_error_diff = error_diff.mean()
    max_error_diff = error_diff.max()

    print(f"*  Mean confusion pattern difference: {mean_error_diff:.4f}")
    print(f"*  Max confusion pattern difference: {max_error_diff:.4f}")

    # Find most different class pairs
    diff_matrix = np.abs(cm_cifar_norm - cm_genai_norm)
    np.fill_diagonal(diff_matrix, 0)  # Ignore diagonal
    max_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    max_pair = (unique_superclasses[max_idx[0]], unique_superclasses[max_idx[1]])

    print(f"*  Largest confusion difference: {max_pair[0]} -> {max_pair[1]}")

    # Interpretation
    if mean_error_diff > 0.1:
        print("*  WARNING: Significant difference in confusion patterns!")
        print("*  Interpretation: CIFAR and GenAI may have different semantic structures.")
    elif mean_error_diff > 0.05:
        print("*  CAUTION: Moderate difference in confusion patterns.")
    else:
        print("*  GOOD: Similar confusion patterns across sources.")
        print("*  Interpretation: Model treats both sources similarly.")

    return {
        'cifar_knn_accuracy': cifar_acc,
        'genai_knn_accuracy': genai_acc,
        'accuracy_gap': abs(cifar_acc - genai_acc),
        'mean_confusion_difference': mean_error_diff,
        'max_confusion_difference': max_error_diff,
        'max_diff_class_pair': max_pair
    }


def run_source_leakage_analysis(embeddings_dir_str):
    """
    Run comprehensive source leakage analysis.

    Tests included:
    - Test 1: Retrieval Source Bias
    - Test 2: Superclass Confusion Analysis
    - Test 3: Cross-Model Consistency (FUTURE - requires model tracking)

    Args:
        embeddings_dir_str: Path to embeddings directory

    Returns:
        dict with all test results
    """
    embeddings_dir = Path(embeddings_dir_str)
    embeddings, metadata = load_data(embeddings_dir)
    if embeddings is None:
        return None

    # Normalize embeddings
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')

    print(f"\n{'='*60}\n SOURCE LEAKAGE ANALYSIS\n{'='*60}")
    print(f"Total samples: {len(metadata)}")
    print(f"CIFAR samples: {(metadata['source'] == 'cifar100').sum()}")
    print(f"GenAI samples: {(metadata['source'] != 'cifar100').sum()}")

    all_results = {}

    # Test 1: Retrieval Source Bias
    test1_results = retrieval_source_bias_test(embeddings_norm, metadata, k=5)
    all_results.update({f'test1_{k}': v for k, v in test1_results.items()})

    # Test 2: Superclass Confusion Analysis
    test2_results = superclass_confusion_analysis(embeddings_norm, metadata)
    all_results.update({f'test2_{k}': v for k, v in test2_results.items()})

    # Test 3: Cross-Model Consistency (FUTURE)
    print(f"\n[Source Leakage Test 3] Cross-Model Consistency")
    print("*  SKIPPED: GenAI model source not tracked in current data.")
    print("*  To enable: Reorganize data to track which model (ChatGPT/Gemini) produced each image.")

    # Summary
    print(f"\n{'='*60}\n SOURCE LEAKAGE ANALYSIS SUMMARY\n{'='*60}")

    warnings = 0
    if all_results.get('test1_average_retrieval_bias', 0) > 0.15:
        warnings += 1
        print("WARNING: High source retrieval bias detected")
    if all_results.get('test2_accuracy_gap', 0) > 0.10:
        warnings += 1
        print("WARNING: Large accuracy gap between sources")
    if all_results.get('test2_mean_confusion_difference', 0) > 0.1:
        warnings += 1
        print("WARNING: Different confusion patterns across sources")

    if warnings == 0:
        print("\nOVERALL: No major source leakage concerns detected.")
    else:
        print(f"\nOVERALL: {warnings} potential source leakage concern(s) detected.")

    # Save results
    results_path = embeddings_dir / "source_leakage_results.json"

    # Convert numpy types for JSON serialization
    serializable_results = {}
    for k, v in all_results.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable_results[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, tuple):
            serializable_results[k] = list(v)
        else:
            serializable_results[k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    # Update path to your embeddings folder
    run_source_leakage_analysis("/home/alex/data/embeddings/DINOv2")
