# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Memory-Efficient Batched Point Cloud Map Generation

This module provides functionality to generate dense point cloud maps from
long RGB videos (5000-10000+ frames) with minimal memory footprint.
Inspired by VGGT-Long for efficient long-sequence processing.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any, Optional

import numpy as np
import open3d as o3d
import torch

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .point_cloud import (
    create_intrinsics_matrix,
    depths_to_point_cloud,
    load_camera_params,
)


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def export_to_point_cloud_ply_batched(
    model: Any,
    image_paths: list[str],
    export_dir: str,
    camera_params: Optional[dict] = None,
    batch_size: int = 20,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    voxel_size: float = 0.01,
    conf_thresh: float = 1.0,
    conf_thresh_percentile: float = 40.0,
    depth_scale: float = 1.0,
    max_depth: float = 100.0,
    use_icp_alignment: bool = False,
    icp_voxel_size: float = 0.05,
    remove_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    export_per_frame: bool = False,
    merge_stride: int = 5,
    final_voxel_size: Optional[float] = None,
) -> str:
    """
    Export point cloud from video frames using memory-efficient batched processing.

    This function processes frames in small batches, progressively merging point clouds
    to handle very long videos (5000-10000+ frames) without running out of memory.

    Args:
        model: DepthAnything3 model instance.
        image_paths: List of paths to input frames.
        export_dir: Output directory for the PLY file.
        camera_params: Optional dictionary with camera intrinsics parameters.
        batch_size: Number of frames to process at once (default: 20).
        process_res: Processing resolution.
        process_res_method: Resize method for processing.
        voxel_size: Voxel size for intermediate downsampling (0 for no downsampling).
        conf_thresh: Base confidence threshold.
        conf_thresh_percentile: Percentile for adaptive confidence threshold.
        depth_scale: Scale factor for depth values.
        max_depth: Maximum depth value to consider.
        use_icp_alignment: Whether to use ICP for frame-to-frame alignment.
        icp_voxel_size: Voxel size for ICP alignment.
        remove_outliers: Whether to remove statistical outliers.
        nb_neighbors: Number of neighbors for outlier removal.
        std_ratio: Standard deviation ratio for outlier removal.
        export_per_frame: Whether to export individual frame point clouds.
        merge_stride: Merge accumulated point clouds every N batches (default: 5).
        final_voxel_size: Final voxel size for the merged point cloud (if None, uses voxel_size).

    Returns:
        Path to the exported PLY file.
    """
    N = len(image_paths)
    os.makedirs(export_dir, exist_ok=True)

    logger.info(f"Starting batched point cloud generation for {N} frames")
    logger.info(f"Batch size: {batch_size}, Merge stride: {merge_stride}")
    logger.info(f"Memory-efficient mode enabled for long video processing")

    # Determine final voxel size
    if final_voxel_size is None:
        final_voxel_size = voxel_size

    # Process first frame to get image dimensions
    logger.info("Processing first frame to determine parameters...")
    first_pred = model.inference(
        image=[image_paths[0]],
        process_res=process_res,
        process_res_method=process_res_method,
        export_format="",  # No export
    )

    H, W = first_pred.depth.shape[1:3]
    logger.info(f"Frame size: {W}x{H}")

    # Prepare intrinsics
    intrinsics_list = _prepare_intrinsics(camera_params, first_pred, N, W, H)

    # Compute adaptive confidence threshold from first batch
    conf_thresh = _compute_adaptive_conf_thresh(
        first_pred, conf_thresh, conf_thresh_percentile
    )
    logger.info(f"Adaptive confidence threshold: {conf_thresh:.4f}")

    # Clean up first prediction
    del first_pred
    clear_memory()

    # Initialize accumulated point cloud
    accumulated_pcd = None
    batch_count = 0

    # Process frames in batches
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N)
        batch_paths = image_paths[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx-1})")

        # Run inference on batch
        try:
            prediction = model.inference(
                image=batch_paths,
                process_res=process_res,
                process_res_method=process_res_method,
                export_format="",  # No export during batch processing
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx} inference failed: {e}")
            clear_memory()
            continue

        # Process each frame in the batch
        batch_points = []
        batch_colors = []

        for i in range(len(batch_paths)):
            frame_idx = start_idx + i

            depth_i = prediction.depth[i]
            colors_i = prediction.processed_images[i]
            conf_i = prediction.conf[i] if prediction.conf is not None else None

            # Get intrinsics for this frame
            if isinstance(intrinsics_list, np.ndarray):
                K_i = intrinsics_list[frame_idx]
            else:
                K_i = intrinsics_list[min(frame_idx, len(intrinsics_list) - 1)]

            # Get extrinsics if available
            ext_i = None
            if prediction.extrinsics is not None:
                ext_i = prediction.extrinsics[i]

            # Generate point cloud for this frame
            pts, cols = depths_to_point_cloud(
                depth_i, K_i, ext_i, colors_i, conf_i,
                conf_thresh, depth_scale, max_depth
            )

            if pts.shape[0] > 0:
                batch_points.append(pts)
                batch_colors.append(cols)

                # Export individual frame if requested
                if export_per_frame:
                    frame_pcd = o3d.geometry.PointCloud()
                    frame_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    frame_pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
                    frame_path = os.path.join(export_dir, f"frame_{frame_idx:04d}.ply")
                    o3d.io.write_point_cloud(frame_path, frame_pcd)

        # Merge batch point clouds
        if len(batch_points) > 0:
            batch_pcd = _merge_batch_points(
                batch_points, batch_colors, voxel_size, remove_outliers, nb_neighbors, std_ratio
            )

            # Accumulate with previous batches
            if accumulated_pcd is None:
                accumulated_pcd = batch_pcd
            else:
                accumulated_pcd = accumulated_pcd + batch_pcd

            batch_count += 1

            # Progressive merge: downsample accumulated point cloud periodically
            if batch_count % merge_stride == 0:
                logger.info(f"Progressive merge at batch {batch_idx + 1}")
                if voxel_size > 0:
                    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)
                if remove_outliers and len(accumulated_pcd.points) > nb_neighbors:
                    accumulated_pcd, _ = accumulated_pcd.remove_statistical_outlier(
                        nb_neighbors=nb_neighbors, std_ratio=std_ratio
                    )
                logger.info(f"Accumulated point cloud: {len(accumulated_pcd.points)} points")

        # Clean up batch data
        del prediction, batch_points, batch_colors
        if 'batch_pcd' in locals():
            del batch_pcd
        clear_memory()

        logger.info(f"Batch {batch_idx + 1} processed. Current accumulated points: "
                   f"{len(accumulated_pcd.points) if accumulated_pcd else 0}")

    # Final processing
    if accumulated_pcd is None or len(accumulated_pcd.points) == 0:
        logger.error("No valid points generated!")
        return ""

    logger.info("Performing final point cloud refinement...")

    # Final downsampling with potentially different voxel size
    if final_voxel_size > 0 and final_voxel_size != voxel_size:
        logger.info(f"Final voxel downsampling with voxel_size={final_voxel_size}")
        accumulated_pcd = accumulated_pcd.voxel_down_sample(final_voxel_size)

    # Final outlier removal
    if remove_outliers and len(accumulated_pcd.points) > nb_neighbors:
        logger.info(f"Final outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
        accumulated_pcd, _ = accumulated_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

    final_point_count = len(accumulated_pcd.points)
    logger.info(f"Final point cloud: {final_point_count} points")

    # Save to PLY
    out_path = os.path.join(export_dir, "point_cloud_map.ply")
    o3d.io.write_point_cloud(out_path, accumulated_pcd)
    logger.info(f"Point cloud map exported to: {out_path}")

    # Save metadata
    metadata = {
        "num_frames": N,
        "num_points": int(final_point_count),
        "batch_size": batch_size,
        "merge_stride": merge_stride,
        "voxel_size": voxel_size,
        "final_voxel_size": final_voxel_size,
        "conf_thresh": float(conf_thresh),
        "depth_scale": depth_scale,
        "max_depth": max_depth,
        "use_icp_alignment": use_icp_alignment,
        "processing_mode": "batched",
    }
    if camera_params is not None:
        metadata["camera_params"] = camera_params

    metadata_path = os.path.join(export_dir, "point_cloud_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    clear_memory()
    return out_path


def _prepare_intrinsics(
    camera_params: Optional[dict],
    first_pred: Prediction,
    N: int,
    W: int,
    H: int
) -> np.ndarray:
    """Prepare intrinsics matrix for all frames."""
    if camera_params is not None:
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        cx = camera_params.get("cx", W / 2)
        cy = camera_params.get("cy", H / 2)

        # Scale intrinsics if image was resized
        if "width" in camera_params and "height" in camera_params:
            orig_w = camera_params["width"]
            orig_h = camera_params["height"]
            scale_x = W / orig_w
            scale_y = H / orig_h
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y

        K = create_intrinsics_matrix(fx, fy, cx, cy)
        intrinsics_list = np.array([K] * N)
        logger.info(f"Using provided camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    elif first_pred.intrinsics is not None:
        intrinsics_list = first_pred.intrinsics
        if len(intrinsics_list) < N:
            # Repeat last intrinsics if needed
            intrinsics_list = np.array(list(intrinsics_list) + [intrinsics_list[-1]] * (N - len(intrinsics_list)))
        logger.info("Using predicted camera intrinsics")
    else:
        # Use default intrinsics
        fx = fy = max(W, H) * 0.8
        cx, cy = W / 2, H / 2
        K = create_intrinsics_matrix(fx, fy, cx, cy)
        intrinsics_list = np.array([K] * N)
        logger.warning(f"No intrinsics provided, using default: fx=fy={fx:.2f}")

    return intrinsics_list


def _compute_adaptive_conf_thresh(
    prediction: Prediction,
    conf_thresh: float,
    conf_thresh_percentile: float
) -> float:
    """Compute adaptive confidence threshold from prediction."""
    if prediction.conf is not None:
        conf_pixels = prediction.conf.flatten()
        lower = np.percentile(conf_pixels, conf_thresh_percentile)
        upper = np.percentile(conf_pixels, 90.0)
        conf_thresh = min(max(conf_thresh, lower), upper)
    return conf_thresh


def _merge_batch_points(
    points_list: list[np.ndarray],
    colors_list: list[np.ndarray],
    voxel_size: float,
    remove_outliers: bool,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    """Merge point clouds from a batch and optionally downsample."""
    if len(points_list) == 0:
        return o3d.geometry.PointCloud()

    # Concatenate all points
    all_points = np.concatenate(points_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)

    # Voxel downsampling
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Remove statistical outliers
    if remove_outliers and len(pcd.points) > nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return pcd
