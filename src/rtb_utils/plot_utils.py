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


def plot_xyz_distributions(xyz, n_plots, target_dist):
    """
    Args:
        xyz: Tensor (or array) of shape (..., 3)
        n_plots: Number of columns to show in the grid
        target_dist: Reference distribution to compare against
    """
    # Convert to numpy (if needed) and extract coordinates
    xyz_np = xyz  # If xyz is already NumPy, this is a no-op
    x_data = xyz_np[..., 0].flatten()
    y_data = xyz_np[..., 1].flatten()
    z_data = xyz_np[..., 2].flatten()
    target_np = target_dist.flatten()

    # Create figure grid
    fig, axes = plt.subplots(3, n_plots, figsize=(4 * n_plots, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # a little more vertical spacing

    # Plot each coordinate row
    for coord_idx, (coord_data, coord_name) in enumerate(zip(
            [x_data, y_data, z_data], ['X', 'Y', 'Z']
    )):
        for plot_idx in range(n_plots):
            ax = axes[coord_idx, plot_idx]

            # Subsample data for multiple plots
            subset = np.random.choice(
                coord_data,
                size=min(len(coord_data), 2000),  # Limit points for visibility
                replace=False
            )

            # Plot distributions
            sns.kdeplot(target_np, ax=ax, color='blue', label='Target', fill=True, alpha=0.3)
            sns.kdeplot(subset, ax=ax, color='orange', label='Samples', fill=True, alpha=0.3)

            # Add metrics
            emd = wasserstein_distance(target_np, subset)
            jsd = js_divergence(target_np, subset)
            ax.set_title(f"{coord_name} | EMD: {emd:.3f}\nJSD: {jsd:.3f}")
            ax.set_xlabel('Coordinate Value')

            # Add legend so user can see which distribution is which
            ax.legend()

    return {
        "xyz_distributions": wandb.Image(fig),
        "xyz_median_emd": np.median([wasserstein_distance(target_np, d)
                                     for d in [x_data, y_data, z_data]]),
        "xyz_median_jsd": np.median([js_divergence(target_np, d)
                                     for d in [x_data, y_data, z_data]])
    }



