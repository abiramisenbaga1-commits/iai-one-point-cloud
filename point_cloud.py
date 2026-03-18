import os
import numpy as np
import matplotlib.pyplot as plt

# ****************************plotting functions****************************

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

    results = {
            "dataset_name": dataset_name,
            "ground_level": ground_level,
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
        