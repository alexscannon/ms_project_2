import logging
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
from sklearn.decomposition import PCA

# Configure logger
logger = logging.getLogger("msproject")

# Silence Numba and UMAP internal loggers
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("umap").setLevel(logging.WARNING)
logging.getLogger("pynndescent").setLevel(logging.WARNING)


# 1. Define a custom distinct palette (20 colors for CIFAR-100 Superclasses)
# These hex codes are chosen to be maximally distinct.
DISTINCT_COLORS_20 = [
    "#e6194b", # Red
    "#3cb44b", # Green
    "#ffe119", # Yellow
    "#4363d8", # Blue
    "#f58231", # Orange
    "#911eb4", # Purple
    "#46f0f0", # Cyan
    "#f032e6", # Magenta
    "#bcf60c", # Lime
    "#fabebe", # Pink
    "#008080", # Teal
    "#e6beff", # Lavender
    "#9a6324", # Brown
    "#fffac8", # Beige
    "#800000", # Maroon
    "#aaffc3", # Mint
    "#808000", # Olive
    "#ffd8b1", # Apricot
    "#000075", # Navy
    "#808080"  # Grey
]


def load_analysis_data(embeddings_path, metadata_path):
    """
    Load the saved .npy and .csv files efficiently.
    """
    logger.info(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    logger.info(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)

    return embeddings, metadata

def compute_projections(embeddings, method='umap', random_state=42):
    """
    Reduce dimensionality to 2D for visualization.
    """
    logger.info(f"Computing {method.upper()} projection for {len(embeddings)} samples...")

    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        # UMAP is generally better for visualizing clusters
        # metric='cosine' is often better for deep learning embeddings
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            random_state=None,
            n_jobs=-1 # Use all CPU cores
        )

    projections = reducer.fit_transform(embeddings)
    return projections

def plot_projections(
    df,
    x_col,
    y_col,
    hue_col,
    style_col=None,
    title="",
    save_path=None,
    palette=None,
    alpha=0.6,
    s=10
):
    """
    Two-Layer Plotting: Faint Real background, Bold GenAI foreground.
    """
    plt.figure(figsize=(14, 10))

    # Check if we are doing the "Real vs GenAI" comparison
    is_genai_split = False
    if style_col == 'source':
        is_genai_split = True
        # Identify GenAI data
        # Adjust logic if your GenAI source names differ
        is_genai_mask = ~df['source'].str.contains('cifar100', case=False)

        df_real = df[~is_genai_mask].copy()
        df_genai = df[is_genai_mask].copy()

    # -------------------------------------------------------
    # Logic A: Standard Plot (No GenAI splitting)
    # -------------------------------------------------------
    if not is_genai_split:
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col, style=style_col,
            palette=palette, alpha=alpha, s=s, linewidth=0
        )

    # -------------------------------------------------------
    # Logic B: Enhanced Two-Layer Plot (Real vs GenAI)
    # -------------------------------------------------------
    else:
        # 1. Determine Color Palette based on ALL data (so keys match)
        #    This ensures "Shark" is Blue in both layers
        if palette is None:
            n_colors = df[hue_col].nunique()
            unique_cats = sorted(df[hue_col].unique())
            # Use custom distinct colors if many categories, else tab10
            color_list = DISTINCT_COLORS_20 if n_colors > 10 else sns.color_palette("tab10")
            palette = dict(zip(unique_cats, color_list[:n_colors]))

        # LAYER 1: Real Data (Background)
        # - High transparency (alpha=0.3)
        # - Normal size (s=s)
        # - No edges
        sns.scatterplot(
            data=df_real,
            x=x_col, y=y_col,
            hue=hue_col,
            marker='o', # Circle
            palette=palette,
            alpha=0.2,      # <--- Very Faint
            s=s * 1.5,      # Slightly larger base to fill gaps
            linewidth=0,
            legend=True     # Keep legend for colors
        )

        # LAYER 2: GenAI Data (Foreground)
        # - Fully Opaque (alpha=1.0)
        # - Huge Size (s * 4)
        # - Distinct Marker ('X')
        # - Thick Black Edge
        sns.scatterplot(
            data=df_genai,
            x=x_col, y=y_col,
            hue=hue_col,
            marker='X',     # Big X
            palette=palette,
            alpha=1.0,      # <--- Fully Solid
            s=s * 8,        # <--- Massive
            edgecolor='black', # <--- distinct border
            linewidth=1.5,
            legend=False    # Don't duplicate color legend
        )

        # Add a custom legend entry for the "GenAI" marker style
        # (Since we turned off the GenAI plot legend to avoid duplicate colors)
        from matplotlib.lines import Line2D
        custom_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=8, label='Real (CIFAR-100)', alpha=0.5),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=12, label='GenAI (Synthesized)')
        ]

        # Get existing legend (colors)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # Combine: Color Legend + Shape Legend
        ax.legend(handles=handles + custom_handles, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def run_visualization_suite(embeddings_dir: Path, embeddings, metadata):
    """
    Main function to run the full visualization report.
    """

    # 1. Compute 2D Projection (UMAP)
    # Using random_state=42 for reproducibility (will warn about n_jobs=1, which is fine)
    projections = compute_projections(embeddings, method='umap')

    metadata['x'] = projections[:, 0] # type: ignore
    metadata['y'] = projections[:, 1] # type: ignore

    logger.info("Generating plots...")

    plots_dir = embeddings_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- PLOT 1: All Superclasses (The "Big Picture") ---
    plot_projections(
        metadata, 'x', 'y',
        hue_col='superclass_name',
        title="Global UMAP: CIFAR-100 Superclasses",
        save_path=plots_dir / "global_superclasses.png",
        # Use our custom 20-color palette automatically
        palette=None
    )

    # --- PLOT 2: Real vs GenAI (Global Distribution) ---
    # [UPDATED] Added style_col='source' to trigger the foreground/background split
    plot_projections(
        metadata, 'x', 'y',
        hue_col='source',       # Color by source (Blue vs Orange)
        style_col='source',     # <--- KEY FIX: Triggers Two-Layer Plotting
        title="Global UMAP: Real vs GenAI Distribution",
        save_path=plots_dir / "global_source_distribution.png",
        palette="tab10"
    )

    # --- PLOT 3: Drill-Down by Superclass (20 plots) ---
    superclass_dir = plots_dir / "superclasses"
    superclass_dir.mkdir(exist_ok=True)

    unique_supers = metadata['superclass_name'].unique()

    for super_name in unique_supers:
        subset = metadata[metadata['superclass_name'] == super_name]

        plot_projections(
            subset, 'x', 'y',
            hue_col='subclass_name',
            style_col='source', # Triggers GenAI highlighting
            title=f"Superclass: {super_name}",
            save_path=superclass_dir / f"{super_name}_cluster.png",
            palette="tab10"
        )

    # --- PLOT 4: GenAI Only ---
    # Good to check internal structure of just the generated data
    is_genai = ~metadata['source'].str.contains('cifar100', case=False)
    genai_subset = metadata[is_genai]

    if not genai_subset.empty:
        plot_projections(
            genai_subset, 'x', 'y',
            hue_col='superclass_name',
            title="GenAI Only Clusters",
            save_path=plots_dir / "genai_only_clusters.png",
            palette=None # Uses custom 20-color palette
        )

    logger.info(f"✅ Visualization suite complete! Plots saved to {plots_dir}")