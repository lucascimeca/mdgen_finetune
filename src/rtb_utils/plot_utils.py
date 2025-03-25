import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb

from scipy.stats import wasserstein_distance
from scipy.stats import entropy


# For Jensen-Shannon Divergence
def js_divergence(samples_p, samples_q, bins=100, epsilon=1e-10):
    """Compute JS divergence between two sample sets."""
    # Combine samples to determine shared bin edges
    all_samples = np.concatenate([samples_p, samples_q])
    bin_edges = np.linspace(np.min(all_samples), np.max(all_samples), bins)

    # Compute histograms
    hist_p, _ = np.histogram(samples_p, bins=bin_edges, density=True)
    hist_q, _ = np.histogram(samples_q, bins=bin_edges, density=True)

    # Normalize and add epsilon to avoid zeros
    hist_p = hist_p / hist_p.sum() + epsilon
    hist_q = hist_q / hist_q.sum() + epsilon

    # Compute mixture and JS divergence
    m = 0.5 * (hist_p + hist_q)
    return 0.5 * (entropy(hist_p, m) + entropy(hist_q, m))


def compare_distributions(dist1, dist2):
    # Convert tensors to numpy arrays
    target_dist_np = dist1.numpy().flatten()
    logr_np = dist2.numpy().flatten()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create KDE plots with fill
    sns.kdeplot(target_dist_np, color='blue', label='Target Distribution', fill=True, alpha=0.3)
    sns.kdeplot(logr_np, color='orange', label='Logr Distribution', fill=True, alpha=0.3)

    # Add rug plots for raw values
    sns.rugplot(target_dist_np, color='blue', height=0.05, alpha=0.5)
    sns.rugplot(logr_np, color='orange', height=0.05, alpha=0.5)

    # Calculate EMD score
    emd_score = wasserstein_distance(target_dist_np, logr_np)

    # Style plot
    plt.title(f'Distribution Comparison (EMD Score: {emd_score:.3f})')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()

    jsd = js_divergence(target_dist_np, logr_np)

    return {
        "dist_comparison": wandb.Image(plt),
        "distribution_score/emd": emd_score,
        "distribution_score/jsd": jsd
    }


def plot_relative_distance_distributions(xyz, n_plots, target_dist, sample_size=100):
    """
    Args:
        xyz: Tensor/array of shape (..., 3)
        n_plots: Number of comparison columns in grid
        target_dist: Reference distribution with same shape as xyz
        sample_size: Points to sample for distance calculations (kept low for efficiency)
    """

    # Convert to numpy (if needed) and extract coordinates
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.cpu().numpy()
    if not isinstance(target_dist, np.ndarray):
        target_dist = target_dist.cpu().numpy()

    xyz_np = xyz.reshape(-1, 3)
    target_np = target_dist.reshape(-1, 3)

    sample_size = min(len(xyz), sample_size)

    fig, axes = plt.subplots(3, n_plots, figsize=(4 * n_plots, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # Cache for storing metrics
    all_emd = {0: [], 1: [], 2: []}
    all_jsd = {0: [], 1: [], 2: []}

    for coord_idx in range(3):  # X/Y/Z rows
        for plot_idx in range(n_plots):  # Columns
            ax = axes[coord_idx, plot_idx]

            # Randomly select points from both distributions
            xyz_sample = xyz_np[np.random.choice(len(xyz_np), sample_size, replace=False)]
            target_sample = target_np[np.random.choice(len(target_np), sample_size, replace=False)]

            # Calculate coordinate-specific distances
            xyz_dists = np.abs(xyz_sample[:, coord_idx, None] - xyz_sample[:, coord_idx])
            target_dists = np.abs(target_sample[:, coord_idx, None] - target_sample[:, coord_idx])

            # Flatten upper triangle (excluding diagonal)
            triu = np.triu_indices_from(xyz_dists, k=1)
            xyz_flat = xyz_dists[triu].flatten()
            target_flat = target_dists[triu].flatten()

            # Plot distributions
            sns.kdeplot(target_flat, ax=ax, color='blue', label='Target', fill=True, alpha=0.3)
            sns.kdeplot(xyz_flat, ax=ax, color='orange', label='Samples', fill=True, alpha=0.3)

            # Calculate metrics
            emd = wasserstein_distance(target_flat, xyz_flat)
            jsd = js_divergence(target_flat, xyz_flat)

            ax.set_title(f"{['X', 'Y', 'Z'][coord_idx]} | EMD: {emd:.2f}\nJSD: {jsd:.2f}")
            ax.set_xlabel('Coordinate Distance')
            ax.legend()

            # Store metrics
            all_emd[coord_idx].append(emd)
            all_jsd[coord_idx].append(jsd)

    # Calculate median scores
    median_metrics = {
        'median_emd_x': np.median(all_emd[0]),
        'median_emd_y': np.median(all_emd[1]),
        'median_emd_z': np.median(all_emd[2]),
        'median_jsd_x': np.median(all_jsd[0]),
        'median_jsd_y': np.median(all_jsd[1]),
        'median_jsd_z': np.median(all_jsd[2]),
    }

    return {
        "relative_distance_distributions": wandb.Image(fig),
        **median_metrics
    }



