import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb

from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


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
    # median_metrics = {
    #     'median_emd_x': np.median(all_emd[0]),
    #     'median_emd_y': np.median(all_emd[1]),
    #     'median_emd_z': np.median(all_emd[2]),
    #     'median_jsd_x': np.median(all_jsd[0]),
    #     'median_jsd_y': np.median(all_jsd[1]),
    #     'median_jsd_z': np.median(all_jsd[2]),
    # }

    return {
        "relative_distance_distributions": wandb.Image(fig),
        # **median_metrics
    }


def compute_tica(X, lag=1):
    # Center the data.
    X_centered = X - np.mean(X, axis=0)
    T = X_centered.shape[0] - lag
    # Build time-lagged pairs:
    X0 = X_centered[:-lag]
    Xlag = X_centered[lag:]

    # Estimate the instantaneous (zero-lag) covariance matrix.
    C0 = np.dot(X0.T, X0) / T
    # Estimate the time-lagged covariance matrix.
    Clag = np.dot(X0.T, Xlag) / T

    # Use pseudoinverse for C0 to avoid singularity issues.
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(C0) @ Clag)

    # Sort eigenvectors (and eigenvalues) by eigenvalue magnitude (largest first).
    idx = np.argsort(-np.abs(eigvals))
    return eigvecs[:, idx], eigvals[idx]


def plot_TICA(samples_torsions, target_torsion, lag_time=1, point_size=15, alpha=0.5):
    """
    Create a 2D TICA plot that overlaps the distributions of the
    target torsions and sampled torsions.

    Both torsions are given as numpy arrays of shape (B, 4, 7, 2) where:
        B   - batch dimension (e.g. frames in a trajectory)
        4   - could be different subunits or portions
        7   - number of torsion angles per unit
        2   - the (sin, cos) representation of each angle

    The function flattens the last three dimensions (4*7*2 = 56 features per frame)
    then computes a TICA projection (using the target torsions as the reference time series)
    and projects both the target and sampled torsions onto the first two TICA components.

    Parameters:
        samples_torsions (np.ndarray): Sampled torsions, shape (B,4,7,2).
        target_torsion   (np.ndarray): Target torsions, shape (B,4,7,2).
        lag_time (int): Lag time (in time-steps) for TICA computation. Defaults to 1.
        point_size (int or float): Marker size for scatter plot.
        alpha (float): Transparency of the scatter plot markers.

    Returns:
        dict: A dictionary with a key "TICA" containing the plot as a wandb.Image.
    """
    # Flatten each sample from (4,7,2) to a vector of 56 features.
    B = target_torsion.shape[0]
    X_target = target_torsion.reshape(B, -1)  # shape: (B, 56)
    X_samples = samples_torsions.reshape(B, -1)  # shape: (B, 56)

    # Compute TICA components using the target torsions as the reference.
    # (Assumes the time ordering of the B samples is meaningful.)
    v, eigvals = compute_tica(X_target, lag=lag_time)

    # Choose the first two tICs to project onto.
    # Note: v is (D, D) so we take the first two columns and retain only the real parts.
    V2 = v[:, :2].real  # shape: (56, 2)

    # Project the data onto these two tICs.
    X_target_tica = np.dot(X_target, V2)  # shape: (B, 2)
    X_samples_tica = np.dot(X_samples, V2)  # shape: (B, 2)

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the target distribution.
    ax.scatter(X_target_tica[:, 0], X_target_tica[:, 1],
               s=point_size, c='blue', alpha=alpha, label='Target')
    # Plot the sampled distribution.
    ax.scatter(X_samples_tica[:, 0], X_samples_tica[:, 1],
               s=point_size, c='red', alpha=alpha, label='Sampled')

    ax.set_xlabel('tIC1')
    ax.set_ylabel('tIC2')
    ax.set_title('TICA Projection of Torsions')
    ax.legend()
    plt.tight_layout()

    # Return the figure wrapped as a wandb.Image.
    return {
        "TICA": wandb.Image(fig),
    }


def plot_TICA_PCA(samples_torsions, target_torsion, point_size=15, alpha=0.5, scale=False):
    """
    Create a 2D projection using PCA which, with lag=1, approximates the TICA projection.

    Both torsions are given as numpy arrays of shape (B, 4, 7, 2), where:
      - B: Batch dimension (e.g., frames)
      - 4: Possibly different subunits or segments
      - 7: Number of torsion angles per subunit
      - 2: (sin, cos) representation of each angle

    The function flattens each sample from (4,7,2) to a vector of 56 features,
    robustly scales the data, and then applies PCA to compute a 2D projection.
    It then overlaps the target distribution (blue) with the sampled distribution (red)
    in a scatter plot.

    Parameters:
      samples_torsions (np.ndarray): Sampled torsions, shape (B,4,7,2).
      target_torsion   (np.ndarray): Target torsions, shape (B,4,7,2).
      point_size (int or float): Marker size for the scatter plot.
      alpha (float): Transparency of the markers.

    Returns:
      dict: A dictionary with key "TICA" containing the plot as a wandb.Image.
    """

    # Flatten the torsion data so that each sample becomes a 56-dimensional vector.
    B = target_torsion.shape[0]
    X_target = target_torsion.reshape(B, -1)  # shape: (B, 56)
    X_samples = samples_torsions.reshape(B, -1)  # shape: (B, 56)

    if scale:
        # Robust Scaling to make the PCA more robust to outliers.
        scaler = RobustScaler()
        X_target = scaler.fit_transform(X_target)
        X_samples = scaler.transform(X_samples)

    # Apply PCA directly to compute the principal components.
    # With lag=1 TICA, much of the time-structure is not emphasized,
    # so PCA often gives a similar projection.
    pca = PCA(n_components=2)
    X_target_pca = pca.fit_transform(X_target)
    X_samples_pca = pca.transform(X_samples)

    # Create the scatter plot to compare the target and sampled projections.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_target_pca[:, 0], X_target_pca[:, 1],
               s=point_size, c='blue', alpha=alpha, label='Target')
    ax.scatter(X_samples_pca[:, 0], X_samples_pca[:, 1],
               s=point_size, c='red', alpha=alpha, label='Sampled')

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('PCA Projection (Approximates TICA with lag=1)')
    ax.legend()
    plt.tight_layout()

    label = "TICA" if not scale else "TICA_scaled"
    return {label: wandb.Image(fig)}