import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# ****************************plotting functions*******************************

def plot_2d(points, title="2D Plot", labels=None, save_path=None,
            point_size=2, show_legend=True):
    """
    Plot point cloud in 2D using x and y axes.
    If labels are given, clusters are shown with different colors.
    """
    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(points[:, 0], points[:, 1], s=point_size)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_points = points[labels == label]

            if label == -1:
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=point_size,
                    c="black",
                    label="Noise"
                )
            else:
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=point_size,
                    color=cmap(i),
                    label=f"Cluster {label}"
                )

        # Show legend only when it is useful
        if show_legend and len(unique_labels) <= 15:
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
                markerscale=3,
                frameon=True
            )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_3d(points, title="3D Plot", labels=None, save_path=None,
            point_size=1, show_legend=True):
    """
    Plot point cloud in 3D using x, y, z axes.
    If labels are given, clusters are shown with different colors.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_points = points[labels == label]

            if label == -1:
                ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    cluster_points[:, 2],
                    s=point_size,
                    c="black",
                    label="Noise"
                )
            else:
                ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    cluster_points[:, 2],
                    s=point_size,
                    color=cmap(i),
                    label=f"Cluster {label}"
                )

        # Show legend only when it is useful
        if show_legend and len(unique_labels) <= 12:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
                markerscale=4,
                frameon=True
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_selected_cluster_check(all_points, selected_points, title="Selected Cluster Check", save_path=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(all_points[:, 0], all_points[:, 1], s=1, c="lightgray", label="All above-ground points")
    plt.scatter(selected_points[:, 0], selected_points[:, 1], s=4, c="maroon", label="Selected cluster")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
   

    # ****************************Task 1: Ground Level Estimation ***********************

def get_ground_level(pcd, bins=150, plot=False, save_path=None, dataset_name="dataset"):
    """
    Estimate ground level from z-values using histogram.
    Ground is assumed to be the densest horizontal layer in z.

    1.	Extract z-coordinates from all points
    2.	Create histogram with specified number of bins (default: 150)
    3.	Identify bin with maximum frequency (densest z-level)
    4.	Return midpoint of that bin as ground level

    """
    z_values = pcd[:, 2]

    counts, bin_edges = np.histogram(z_values, bins=bins)
    max_bin_index = np.argmax(counts)

    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(z_values, bins=bins, edgecolor="black")
        plt.axvline(
            ground_level,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Ground level = {ground_level:.3f}"
        )
        plt.title(f"Histogram of z-values - {dataset_name}")
        plt.xlabel("z")
        plt.ylabel("Frequency")
        plt.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    return ground_level

# ****************************Task 2: Find Optimal Eps Using Knee Plot ***********************

def compute_k_distance(points, min_samples=5):
    """
    Compute sorted k-nearest neighbor distances for knee plot.

    1.	Fit NearestNeighbors with k = min_samples (5)
    2.	Query distances to k nearest neighbors for each point
    3.	Extract the maximum distance (farthest neighbor) in each point's k-neighborhood
    4.	Sort distances in ascending order
    5.	Return sorted array

    """
    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(points)
    distances, _ = nbrs.kneighbors(points)

    k_distances = np.sort(distances[:, -1])
    return k_distances


def find_knee_index(k_distances):
    """
    Find knee point index in k-distance plot using KneeLocator.

    1.	Use KneeLocator on k-distance curve
    2.	Search for "knee"—the point of maximum curvature
    3.	Parameters:
        a)	curve="convex": Expect upward-curving (convex) k-distance plot
        b)	direction="increasing": Distance increases with sorted point index
        c)	S=1: Smoothing factor (higher = smoother curve, may miss subtle knee)
        d)  interp_method="interp1d": Linear interpolation between discrete k-distance values.
    4.	Return index where knee is detected

    """
    # KneeLocator method
    i = np.arange(len(k_distances))
    knee = KneeLocator(
        i,
        k_distances,
        S=1,
        curve="convex",
        direction="increasing",
        interp_method="interp1d"
    )

    return int(knee.knee)


def get_optimal_eps(points, min_samples=5, plot=False, save_path=None, dataset_name="dataset"):
    """
    Compute optimal DBSCAN eps from knee plot.
    """
    k_distances = compute_k_distance(points, min_samples=min_samples)
    knee_index = find_knee_index(k_distances)
    optimal_eps = k_distances[knee_index]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.axvline(knee_index, linestyle="--", label=f"Knee index = {knee_index}")
        plt.axhline(optimal_eps, linestyle="--", label=f"Optimal eps = {optimal_eps:.3f}")
        plt.scatter(knee_index, optimal_eps)  
        plt.title(f"DBSCAN Knee Plot - {dataset_name}")
        plt.xlabel("Sorted points")
        plt.ylabel(f"{min_samples}-NN distance")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    return optimal_eps 


def run_dbscan(points, eps, min_samples=5):
    """
    Run DBSCAN clustering.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points)
    return labels

# ****************************Task 3: Find Catenary Cluster ***********************

def find_catenary_cluster(points, labels):
    """
    Identify catenary cluster.
    Catenary cluster is expected to be the largest cluster in x-y span and above ground level.
    """
    valid_labels = [label for label in np.unique(labels) if label != -1]

    if len(valid_labels) == 0:
        raise ValueError("No valid clusters found. Only noise is present.")

    best_label = None
    best_score = -1
    best_points = None
    best_bounds = None

    for label in valid_labels:
        cluster_points = points[labels == label]

        min_x = np.min(cluster_points[:, 0])
        max_x = np.max(cluster_points[:, 0])
        min_y = np.min(cluster_points[:, 1])
        max_y = np.max(cluster_points[:, 1])
        min_z = np.min(cluster_points[:, 2])
        max_z = np.max(cluster_points[:, 2])
        mean_z = np.mean(cluster_points[:, 2])

        x_span = max_x - min_x
        y_span = max_y - min_y
        z_span = max_z - min_z
        n_points = len(cluster_points)

        score = x_span * y_span

        if score > best_score:
            best_score = score
            best_label = label
            best_points = cluster_points
            best_bounds = {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "x_span": x_span,
                "y_span": y_span,
                "z_span": z_span,
                "mean_z": mean_z,
                "n_points": n_points
            }

    if best_label is None:
        raise ValueError("No suitable catenary cluster found.")

    return best_label, best_points, best_bounds


# ****************************Complete Processing Pipeline*************************
def process_dataset(dataset_path, output_dir="", min_samples=5,
                    show_2d=True, show_3d=True):
    """
    Complete processing pipeline for one dataset.
    """
    base_path = r"/Users/senbagaabiramikumar/Library/CloudStorage/OneDrive-LuleåUniversityofTechnology/IAI & eMaint"
    output_dir = os.path.join(base_path, "output_point_clouds_results_3")
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print("=" * 70)
    print(f"Processing: {dataset_name}")

    # Load data
    pcd = np.load(dataset_path)
    print(f"Point cloud shape: {pcd.shape}")

    # visualize original point cloud before any processing
    if show_2d:
        plot_2d(
            pcd,
            title=f"Original Point Cloud 2D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_original_2d.png"),
            point_size=1,
            show_legend=False
        )

    if show_3d:
        plot_3d(
            pcd,
            title=f"Original Point Cloud 3D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_original_3d.png"),
            point_size=1,
            show_legend=False
        )

    # task 1: estimate ground level and filter points above ground
    ground_level = get_ground_level(
        pcd,
        bins=150,
        plot=True,
        save_path=os.path.join(output_dir, f"{dataset_name}_histogram.png"),
        dataset_name=dataset_name
    )
    print(f"Estimated ground level: {ground_level:.3f}")

    if dataset_name == "dataset2":
        pcd_above_ground = pcd[pcd[:, 2] > ground_level + 0.5]  # add larger margin for dataset2 to ensure we are above ground
    else:
        pcd_above_ground = pcd[pcd[:, 2] > ground_level + 0.3]  # add small margin to ensure we are above ground (0.1 / 0.15 / 0.3)

    print(f"Points above ground: {pcd_above_ground.shape[0]}")

    if show_2d:
        plot_2d(
            pcd_above_ground,
            title=f"Above Ground 2D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_above_ground_2d.png"),
            point_size=1,
            show_legend=False
        )

    if show_3d:
        plot_3d(
            pcd_above_ground,
            title=f"Above Ground 3D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_above_ground_3d.png"),
            point_size=1,
            show_legend=False
        )

    # task 2: find optimal eps using knee plot and run DBSCAN
    optimal_eps = get_optimal_eps(
        pcd_above_ground,
        min_samples=min_samples,
        plot=True,
        save_path=os.path.join(output_dir, f"{dataset_name}_knee_plot.png"),
        dataset_name=dataset_name
    )
    # dataset-specific eps handling
    if dataset_name == "dataset2":
        final_eps = optimal_eps * 0.7  # use smaller eps for dataset2 to avoid merging clusters (0.75)
    else:
        final_eps = optimal_eps

    print(f"Optimal eps from knee: {optimal_eps:.3f}")
    print(f"Final eps used for DBSCAN: {final_eps:.3f}")

    labels = run_dbscan(pcd_above_ground, eps=final_eps, min_samples=min_samples)

    if show_2d:
        plot_2d(
            pcd_above_ground,
            title=f"DBSCAN Clusters 2D - {dataset_name}",
            labels=labels,
            save_path=os.path.join(output_dir, f"{dataset_name}_clusters_2d.png"),
            point_size=2,
            show_legend=True
        )

    if show_3d:
        plot_3d(
            pcd_above_ground,
            title=f"DBSCAN Clusters 3D - {dataset_name}",
            labels=labels,
            save_path=os.path.join(output_dir, f"{dataset_name}_clusters_3d.png"),
            point_size=1,
            show_legend=True
        )  

    # task 3: find catenary cluster and compute bounds
    catenary_label, catenary_points, bounds = find_catenary_cluster(pcd_above_ground, labels)

    plot_selected_cluster_check(pcd_above_ground, catenary_points, title=f"Selected Catenary Cluster Check - {dataset_name}")

    print(f"Catenary cluster label: {catenary_label}")
    print(f"min(x): {bounds['min_x']:.3f}")
    print(f"min(y): {bounds['min_y']:.3f}")
    print(f"max(x): {bounds['max_x']:.3f}")
    print(f"max(y): {bounds['max_y']:.3f}")

    if show_2d:
        plot_2d(
            catenary_points,
            title=f"Catenary Cluster 2D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_catenary_2d.png"),
            point_size=2,
            show_legend=False
        )

    if show_3d:
        plot_3d(
            catenary_points,
            title=f"Catenary Cluster 3D - {dataset_name}",
            save_path=os.path.join(output_dir, f"{dataset_name}_catenary_3d.png"),
            point_size=2,
            show_legend=False
        )
     

    results = {
            "dataset_name": dataset_name,
            "ground_level": ground_level,
            "optimal_eps": optimal_eps,
            "final_eps": final_eps,
            "catenary_label": catenary_label,
            "min_x": bounds["min_x"],
            "min_y": bounds["min_y"],
            "max_x": bounds["max_x"],
            "max_y": bounds["max_y"]
         }

    return results


# ****************************Main Execution*************************
if __name__ == "__main__":
    dataset_paths = [
        "/Users/senbagaabiramikumar/Library/CloudStorage/OneDrive-LuleåUniversityofTechnology/IAI & eMaint/Data/Lidar_assignment-1/dataset1.npy",
        "/Users/senbagaabiramikumar/Library/CloudStorage/OneDrive-LuleåUniversityofTechnology/IAI & eMaint/Data/Lidar_assignment-1/dataset2.npy"
    ]

    all_results = []

    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            result = process_dataset(
                dataset_path,
                output_dir="",
                min_samples=5,
                show_2d=False,   # set False if you do not want 2D plots
                show_3d=True    # set False if you do not want 3D plots
            )
            all_results.append(result)
        else:
            print(f"File not found: {dataset_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for result in all_results:
        print(f"\nDataset: {result['dataset_name']}")
        print(f"Ground level: {result['ground_level']:.3f}")
        print(f"Optimal eps from knee: {result['optimal_eps']:.3f}")
        print(f"Final eps used: {result['final_eps']:.3f}")
        print(f"min(x): {result['min_x']:.3f}")
        print(f"min(y): {result['min_y']:.3f}")
        print(f"max(x): {result['max_x']:.3f}")
        print(f"max(y): {result['max_y']:.3f}")
        