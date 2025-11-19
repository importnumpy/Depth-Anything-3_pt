"""
Utility Functions for Point Cloud Scanning
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_logging(output_dir: str) -> None:
    """Setup logging to both console and file"""
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = os.path.join(output_dir, "processing.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized: {log_file}")


def merge_point_clouds(
    point_clouds: List[o3d.geometry.PointCloud],
    voxel_size: float = 0.005,
) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds"""
    merged = o3d.geometry.PointCloud()

    for pcd in point_clouds:
        merged += pcd

    # Downsample
    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)

    return merged


def save_loop_closures_json(loop_closures, output_path: str) -> None:
    """Save loop closures to JSON file"""
    data = []
    for lc in loop_closures:
        data.append({
            "frame_i": int(lc.frame_i),
            "frame_j": int(lc.frame_j),
            "transformation": lc.transformation.tolist(),
            "score": float(lc.score),
            "visual_score": float(lc.visual_score),
            "geometric_score": float(lc.geometric_score),
            "inlier_rmse": float(lc.inlier_rmse),
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_trajectory_visualization(
    poses_before: List[np.ndarray],
    poses_after: List[np.ndarray],
    loop_closures: List,
    output_dir: str,
) -> None:
    """Save trajectory visualization"""
    if not HAS_MATPLOTLIB:
        logging.warning("Matplotlib not available, skipping trajectory visualization")
        return

    # Extract positions
    pos_before = np.array([p[:3, 3] for p in poses_before])
    pos_after = np.array([p[:3, 3] for p in poses_after])

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    # Before optimization (2D)
    ax1 = fig.add_subplot(131)
    ax1.plot(pos_before[:, 0], pos_before[:, 1], 'b-', alpha=0.5, label='Before')
    ax1.scatter(pos_before[0, 0], pos_before[0, 1], c='g', s=100, marker='o', label='Start')
    ax1.scatter(pos_before[-1, 0], pos_before[-1, 1], c='r', s=100, marker='x', label='End')

    # Draw loop closures
    for lc in loop_closures:
        i, j = lc.frame_i, lc.frame_j
        ax1.plot([pos_before[i, 0], pos_before[j, 0]],
                [pos_before[i, 1], pos_before[j, 1]],
                'r--', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Before Optimization')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # After optimization (2D)
    ax2 = fig.add_subplot(132)
    ax2.plot(pos_after[:, 0], pos_after[:, 1], 'g-', alpha=0.5, label='After')
    ax2.scatter(pos_after[0, 0], pos_after[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(pos_after[-1, 0], pos_after[-1, 1], c='r', s=100, marker='x', label='End')

    # Draw loop closures
    for lc in loop_closures:
        i, j = lc.frame_i, lc.frame_j
        ax2.plot([pos_after[i, 0], pos_after[j, 0]],
                [pos_after[i, 1], pos_after[j, 1]],
                'r--', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Trajectory After Optimization')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # 3D comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(pos_before[:, 0], pos_before[:, 1], pos_before[:, 2],
             'b-', alpha=0.5, label='Before')
    ax3.plot(pos_after[:, 0], pos_after[:, 1], pos_after[:, 2],
             'g-', alpha=0.5, label='After')
    ax3.scatter(pos_after[0, 0], pos_after[0, 1], pos_after[0, 2],
               c='g', s=100, marker='o', label='Start')
    ax3.scatter(pos_after[-1, 0], pos_after[-1, 1], pos_after[-1, 2],
               c='r', s=100, marker='x', label='End')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D Trajectory Comparison')
    ax3.legend()

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, "trajectory_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    logging.info(f"Saved trajectory visualization: {output_path}")
