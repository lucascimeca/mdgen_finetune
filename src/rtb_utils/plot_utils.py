import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb

from scipy.stats import wasserstein_distance
from scipy.stats import entropy


# For Jensen-Shannon Divergence
def js_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


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
