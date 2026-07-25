"""
Population-weighted K-means clustering for primary demand points.

The script uses population as sample weights and longitude/latitude as the two clustering features. 
"""

from __future__ import annotations

import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Kmeans_output"
ELBOW_PLOT_PATH = OUTPUT_DIR / "population_weighted_kmeans_elbow.png"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Location of representative demand points.
POPULATION = {
    1: 69332.0,
    2: 69165.0,
    3: 188500.0,
    4: 127475.0,
    5: 45797.0,
    6: 75444.0,
    7: 96310.0,
    8: 117049.0,
    9: 31733.0,
    10: 80988.0,
    11: 78607.0,
    12: 69218.0,
}

DEMAND_LOCATION = {
    1: [121.493, 31.376],
    2: [121.479, 31.348],
    3: [121.393, 31.308],
    4: [121.378, 31.355],
    5: [121.424, 31.331],
    6: [121.473, 31.325],
    7: [121.483, 31.403],
    8: [121.446, 31.338],
    9: [121.343, 31.476],
    10: [121.353, 31.407],
    11: [121.438, 31.383],
    12: [121.414, 31.435],
}

DEMAND_IDS = sorted(DEMAND_LOCATION)
DEMAND_POINTS = np.array([DEMAND_LOCATION[j] for j in DEMAND_IDS], dtype=float)
POPULATION_WEIGHTS = np.array([POPULATION[j] for j in DEMAND_IDS], dtype=float)


def weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run weighted K-means and return labels, centers, and weighted inertia."""
    if n_clusters < 1 or n_clusters > len(points):
        raise ValueError("n_clusters must be between 1 and the number of demand points")

    max_iter = 300
    tol = 1e-12
    n_init = 100
    rng = np.random.default_rng(42)
    best_labels = None
    best_centers = None
    best_inertia = math.inf
    probabilities = weights / weights.sum()

    for _ in range(n_init):
        center_indices = rng.choice(
            len(points),
            size=n_clusters,
            replace=False,
            p=probabilities,
        )
        centers = points[center_indices].copy()

        for _ in range(max_iter):
            squared_distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = squared_distances.argmin(axis=1)
            new_centers = centers.copy()

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if not np.any(mask):
                    replacement = rng.choice(len(points), p=probabilities)
                    new_centers[cluster_id] = points[replacement]
                    continue
                cluster_weights = weights[mask]
                new_centers[cluster_id] = np.average(
                    points[mask],
                    axis=0,
                    weights=cluster_weights,
                )

            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if center_shift <= tol:
                break

        squared_distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(np.sum(weights * squared_distances.min(axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("Weighted K-means failed to produce a solution")

    ordered_old_labels = sorted(
        range(len(best_centers)),
        key=lambda label: (best_centers[label, 0], -best_centers[label, 1]),
    )
    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(ordered_old_labels)
    }
    best_labels = np.array(
        [label_map[int(label)] for label in best_labels],
        dtype=int,
    )
    best_centers = best_centers[ordered_old_labels]
    return best_labels, best_centers, best_inertia


def plot_elbow(
    cluster_counts: list[int],
    inertias: list[float],
    out_path: Path,
) -> None:
    """Plot the elbow curve for weighted K-means."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=600)
    ax.plot(
        cluster_counts,
        inertias,
        marker="o",
        markersize=5,
        linewidth=1.6,
        color="#4F81BD",
    )
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Weighted within-cluster sum of squares")
    ax.set_xticks(cluster_counts)
    ax.grid(True, linestyle=(0, (4, 4)), linewidth=0.8, color="#d0d0d0")
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(
    demand_ids: list[int],
    points: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
) -> None:
    """Plot weighted K-means clustering result in longitude-latitude space."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 15,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=600)
    ax.set_axisbelow(True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    weight_span = weights.max() - weights.min()
    if weight_span > 0:
        normalized_weights = (weights - weights.min()) / weight_span
    else:
        normalized_weights = np.zeros_like(weights)
    marker_sizes = 45 + 300 * np.sqrt(normalized_weights)

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=marker_sizes[mask],
            color=colors[cluster_id % len(colors)],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=f"Cluster {cluster_id + 1}",
            zorder=3,
        )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=110,
        color="white",
        edgecolors="black",
        linewidths=1.2,
        marker="X",
        label="Cluster center",
        zorder=4,
    )

    for idx, demand_id in enumerate(demand_ids):
        ax.annotate(
            chr(64 + demand_id),
            (points[idx, 0], points[idx, 1]),
            textcoords="offset points",
            xytext=(9.5, 8.5),
            fontsize=11,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=5,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    all_x = np.concatenate([points[:, 0], centers[:, 0]])
    all_y = np.concatenate([points[:, 1], centers[:, 1]])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_padding = 0.12 * (x_max - x_min) if x_max > x_min else 0.01
    y_padding = 0.12 * (y_max - y_min) if y_max > y_min else 0.01
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.grid(True, linestyle=(0, (4, 4)), linewidth=0.8, color="#d0d0d0")
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.1)
    legend = ax.legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1,
        markerscale=0.75,
        fontsize=15,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([70])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    max_clusters = len(DEMAND_IDS)
    cluster_counts = list(range(1, max_clusters + 1))
    inertias = []

    for n_clusters in cluster_counts:
        _, _, inertia = weighted_kmeans(
            DEMAND_POINTS,
            POPULATION_WEIGHTS,
            n_clusters=n_clusters,
        )
        inertias.append(inertia)
        print(f"k={n_clusters:2d}, weighted inertia={inertia:.8f}")

    plot_elbow(cluster_counts, inertias, ELBOW_PLOT_PATH)
    print(f"Elbow plot saved to: {ELBOW_PLOT_PATH}")


if __name__ == "__main__":
    main()