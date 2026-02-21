#!/usr/bin/env python3
"""
Manual validation script for DINO embeddings.

This standalone script performs sanity checks on extracted embeddings that
complement the automated tests and artifact_analysis.py. It focuses on:
1. Numerical validity (NaN, Inf, variance)
2. Embedding-metadata alignment
3. Basic semantic sanity (not duplicating artifact_analysis.py)

Usage:
    python tests/validation/validate_embeddings.py /path/to/embeddings/set_X
    python tests/validation/validate_embeddings.py /path/to/embeddings/set_X --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def check_numerical_validity(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Check for NaN, Inf, and zero variance.

    Args:
        embeddings: (N, D) array of embeddings

    Returns:
        Dict with validation results
    """
    results = {
        'n_samples': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'has_nan': bool(np.isnan(embeddings).any()),
        'n_nan_values': int(np.isnan(embeddings).sum()),
        'has_inf': bool(np.isinf(embeddings).any()),
        'n_inf_values': int(np.isinf(embeddings).sum()),
        'global_mean': float(embeddings.mean()),
        'global_std': float(embeddings.std()),
        'per_dim_std_min': float(embeddings.std(axis=0).min()),
        'per_dim_std_max': float(embeddings.std(axis=0).max()),
        'zero_variance_dims': int((embeddings.std(axis=0) == 0).sum()),
    }

    # Check for duplicate embeddings
    unique_embeddings = np.unique(embeddings, axis=0)
    results['n_duplicate_rows'] = embeddings.shape[0] - len(unique_embeddings)

    return results


def check_metadata_alignment(embeddings: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify embeddings and metadata are properly aligned.

    Args:
        embeddings: (N, D) array of embeddings
        metadata: DataFrame with sample metadata

    Returns:
        Dict with alignment check results
    """
    results = {
        'n_embeddings': embeddings.shape[0],
        'n_metadata_rows': len(metadata),
        'aligned': embeddings.shape[0] == len(metadata),
    }

    # Check for missing values in critical columns
    required_cols = ['subclass_id', 'subclass_name', 'superclass_id',
                     'superclass_name', 'source', 'split']

    for col in required_cols:
        if col in metadata.columns:
            results[f'{col}_missing'] = int(metadata[col].isna().sum())
        else:
            results[f'{col}_missing'] = 'COLUMN_NOT_FOUND'

    # Check index column
    if 'index' in metadata.columns:
        expected_indices = list(range(len(metadata)))
        actual_indices = metadata['index'].tolist()
        results['index_sequential'] = expected_indices == actual_indices

    return results


def check_embedding_norms(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Check L2 norms of embeddings.

    Args:
        embeddings: (N, D) array of embeddings

    Returns:
        Dict with norm statistics
    """
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'norm_min': float(norms.min()),
        'norm_max': float(norms.max()),
        'near_unit_norm': bool(np.allclose(norms, 1.0, atol=0.1)),
    }


def check_class_separation(embeddings: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick sanity check on class separation (complementary to artifact_analysis).

    Computes within-class vs between-class cosine similarities to verify
    semantic structure is present in embeddings.

    Args:
        embeddings: (N, D) array of embeddings
        metadata: DataFrame with sample metadata

    Returns:
        Dict with separation metrics
    """
    embeddings_norm = normalize(embeddings, axis=1, norm='l2')

    results = {}

    # Compute mean cosine similarity within vs between superclasses
    superclasses = metadata['superclass_name'].unique()

    within_sims = []
    between_sims = []

    np.random.seed(42)  # For reproducibility

    # Sample for efficiency (only check first 5 superclasses)
    for sc in superclasses[:5]:
        mask = metadata['superclass_name'] == sc
        sc_emb = embeddings_norm[mask]

        if len(sc_emb) < 2:
            continue

        # Within-class similarity (sample pairs)
        n_pairs = min(100, len(sc_emb))
        for _ in range(n_pairs):
            i, j = np.random.choice(len(sc_emb), 2, replace=False)
            within_sims.append(np.dot(sc_emb[i], sc_emb[j]))

    # Between-class similarity (random pairs from different classes)
    for _ in range(200):
        sc1, sc2 = np.random.choice(superclasses, 2, replace=False)
        emb1 = embeddings_norm[metadata['superclass_name'] == sc1]
        emb2 = embeddings_norm[metadata['superclass_name'] == sc2]

        if len(emb1) > 0 and len(emb2) > 0:
            i = np.random.randint(len(emb1))
            j = np.random.randint(len(emb2))
            between_sims.append(np.dot(emb1[i], emb2[j]))

    if within_sims and between_sims:
        results['mean_within_class_sim'] = float(np.mean(within_sims))
        results['mean_between_class_sim'] = float(np.mean(between_sims))
        results['separation_delta'] = float(np.mean(within_sims) - np.mean(between_sims))
        results['semantic_structure_detected'] = results['separation_delta'] > 0.05

    return results


def check_source_distribution(metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Check distribution of samples across sources.

    Args:
        metadata: DataFrame with sample metadata

    Returns:
        Dict with source distribution info
    """
    if 'source' not in metadata.columns:
        return {'error': 'source column not found'}

    source_counts = metadata['source'].value_counts().to_dict()
    return {
        'source_counts': source_counts,
        'n_sources': len(source_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate DINO embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/validation/validate_embeddings.py data/embeddings/DINOv2/set_5
    python tests/validation/validate_embeddings.py data/embeddings/DINOv2/set_5 --verbose
        """
    )
    parser.add_argument('embeddings_dir', type=Path,
                        help='Directory containing embeddings.npy and metadata.csv')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed output')
    args = parser.parse_args()

    embeddings_path = args.embeddings_dir / 'embeddings.npy'
    metadata_path = args.embeddings_dir / 'metadata.csv'

    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found")
        sys.exit(1)

    print(f"Validating embeddings in: {args.embeddings_dir}")
    print("=" * 60)

    # Load data
    print("Loading data...")
    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")

    # Run checks
    print("\nRunning validation checks...")
    numerical_results = check_numerical_validity(embeddings)
    alignment_results = check_metadata_alignment(embeddings, metadata)
    norm_results = check_embedding_norms(embeddings)
    separation_results = check_class_separation(embeddings, metadata)
    source_results = check_source_distribution(metadata)

    # Report
    all_passed = True

    print("\n" + "=" * 60)
    print("[1] NUMERICAL VALIDITY")
    print("=" * 60)
    print(f"  Shape: ({numerical_results['n_samples']}, {numerical_results['embedding_dim']})")

    if numerical_results['has_nan']:
        print(f"  FAIL: Found {numerical_results['n_nan_values']} NaN values")
        all_passed = False
    else:
        print("  PASS: No NaN values")

    if numerical_results['has_inf']:
        print(f"  FAIL: Found {numerical_results['n_inf_values']} Inf values")
        all_passed = False
    else:
        print("  PASS: No Inf values")

    if numerical_results['zero_variance_dims'] > 0:
        pct = 100 * numerical_results['zero_variance_dims'] / numerical_results['embedding_dim']
        print(f"  WARN: {numerical_results['zero_variance_dims']} dimensions have zero variance ({pct:.1f}%)")
    else:
        print("  PASS: All dimensions have non-zero variance")

    if numerical_results['n_duplicate_rows'] > 0:
        print(f"  WARN: {numerical_results['n_duplicate_rows']} duplicate embedding rows")
    else:
        print("  PASS: No duplicate embeddings")

    if args.verbose:
        print(f"  Global mean: {numerical_results['global_mean']:.6f}")
        print(f"  Global std: {numerical_results['global_std']:.6f}")
        print(f"  Per-dim std range: [{numerical_results['per_dim_std_min']:.6f}, {numerical_results['per_dim_std_max']:.6f}]")

    print("\n" + "=" * 60)
    print("[2] METADATA ALIGNMENT")
    print("=" * 60)

    if alignment_results['aligned']:
        print(f"  PASS: {alignment_results['n_embeddings']} embeddings = {alignment_results['n_metadata_rows']} metadata rows")
    else:
        print(f"  FAIL: Mismatch - {alignment_results['n_embeddings']} embeddings vs {alignment_results['n_metadata_rows']} metadata rows")
        all_passed = False

    for col in ['subclass_id', 'subclass_name', 'superclass_id', 'superclass_name', 'source', 'split']:
        key = f'{col}_missing'
        if key in alignment_results:
            if alignment_results[key] == 0:
                if args.verbose:
                    print(f"  PASS: {col} has no missing values")
            elif alignment_results[key] == 'COLUMN_NOT_FOUND':
                print(f"  FAIL: {col} column not found")
                all_passed = False
            else:
                print(f"  WARN: {col} has {alignment_results[key]} missing values")

    if 'index_sequential' in alignment_results:
        if alignment_results['index_sequential']:
            print("  PASS: Index column is sequential (0 to N-1)")
        else:
            print("  WARN: Index column is not sequential")

    print("\n" + "=" * 60)
    print("[3] EMBEDDING NORMS")
    print("=" * 60)
    print(f"  Mean norm: {norm_results['norm_mean']:.4f}")
    print(f"  Std norm: {norm_results['norm_std']:.4f}")
    print(f"  Range: [{norm_results['norm_min']:.4f}, {norm_results['norm_max']:.4f}]")

    if norm_results['near_unit_norm']:
        print("  INFO: Embeddings are approximately unit normalized")

    print("\n" + "=" * 60)
    print("[4] SOURCE DISTRIBUTION")
    print("=" * 60)
    if 'source_counts' in source_results:
        for source, count in source_results['source_counts'].items():
            pct = 100 * count / numerical_results['n_samples']
            print(f"  {source}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("[5] SEMANTIC STRUCTURE (Quick Check)")
    print("=" * 60)
    if 'semantic_structure_detected' in separation_results:
        print(f"  Within-class similarity: {separation_results['mean_within_class_sim']:.4f}")
        print(f"  Between-class similarity: {separation_results['mean_between_class_sim']:.4f}")
        print(f"  Separation delta: {separation_results['separation_delta']:.4f}")
        if separation_results['semantic_structure_detected']:
            print("  PASS: Semantic structure detected (within > between)")
        else:
            print("  WARN: Weak semantic structure (delta < 0.05)")
    else:
        print("  SKIP: Not enough data for semantic analysis")

    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED: All critical checks passed")
        print("=" * 60)
    else:
        print("VALIDATION FAILED: Some checks failed (see above)")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
