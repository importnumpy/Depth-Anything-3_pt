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
Point Cloud Map Generation Module

This module provides functionality to generate dense point cloud maps from
RGB video with known camera parameters (intrinsics and optional extrinsics).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import open3d as o3d

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger


def load_camera_params(camera_params_path: str) -> dict:
    """Load camera parameters from a JSON file.

    Expected JSON format:
    {
        "fx": float,           # focal length x
        "fy": float,           # focal length y
        "cx": float,           # principal point x
        "cy": float,           # principal point y
        "width": int,          # image width (optional)
        "height": int,         # image height (optional)
        "depth_scale": float   # depth scale factor (optional, default 1.0)
    }

    Args:
        camera_params_path: Path to the JSON file containing camera parameters.

    Returns:
        Dictionary containing camera parameters.
    """
    with open(camera_params_path, "r") as f:
        params = json.load(f)

    required_keys = ["fx", "fy", "cx", "cy"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required camera parameter: {key}")

    # Set default values for optional parameters
    params.setdefault("depth_scale", 1.0)

    return params


def create_intrinsics_matrix(
    fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Create a 3x3 camera intrinsics matrix.

    Args:
        fx: Focal length in x direction.
        fy: Focal length in y direction.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        3x3 intrinsics matrix.
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)


def depths_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.0,
    depth_scale: float = 1.0,
    max_depth: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D point cloud.

    Args:
        depth: Depth map of shape (H, W).
        intrinsics: 3x3 camera intrinsics matrix.
        extrinsics: Optional 4x4 world-to-camera transformation matrix (w2c).
                    If None, identity is used (camera frame = world frame).
        colors: Optional RGB colors of shape (H, W, 3) in uint8.
        conf: Optional confidence map of shape (H, W).
        conf_thresh: Confidence threshold for filtering points.
        depth_scale: Scale factor for depth values.
        max_depth: Maximum depth value to consider.

    Returns:
        Tuple of (points, colors) where:
            - points: (N, 3) array of 3D points in world coordinates
            - colors: (N, 3) array of RGB colors in uint8
    """
    H, W = depth.shape

    # Create pixel coordinate grid
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)

    # Apply depth scale
    depth_scaled = depth * depth_scale

    # Create validity mask
    valid = np.isfinite(depth_scaled) & (depth_scaled > 0) & (depth_scaled < max_depth)

    if conf is not None:
        valid &= conf >= conf_thresh

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    d_flat = depth_scaled.reshape(-1)
    vidx = np.flatnonzero(valid.reshape(-1))

    # Backproject to camera coordinates
    K_inv = np.linalg.inv(intrinsics)
    rays = K_inv @ pix[vidx].T  # (3, M)
    Xc = rays * d_flat[vidx][None, :]  # (3, M)

    # Transform to world coordinates
    if extrinsics is not None:
        # Convert w2c to c2w
        if extrinsics.shape == (3, 4):
            w2c = np.eye(4, dtype=extrinsics.dtype)
            w2c[:3, :4] = extrinsics
        else:
            w2c = extrinsics
        c2w = np.linalg.inv(w2c)

        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3)
    else:
        Xw = Xc.T.astype(np.float32)  # (M, 3)

    # Extract colors
    if colors is not None:
        cols = colors.reshape(-1, 3)[vidx].astype(np.uint8)
    else:
        cols = np.full((Xw.shape[0], 3), 128, dtype=np.uint8)

    return Xw, cols


def merge_point_clouds(
    points_list: list[np.ndarray],
    colors_list: list[np.ndarray],
    voxel_size: float = 0.0,
    remove_statistical_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple point clouds and optionally downsample.

    Args:
        points_list: List of (N_i, 3) point arrays.
        colors_list: List of (N_i, 3) color arrays in uint8.
        voxel_size: Voxel size for downsampling. If 0, no downsampling is performed.
        remove_statistical_outliers: Whether to remove statistical outliers.
        nb_neighbors: Number of neighbors for outlier detection.
        std_ratio: Standard deviation ratio for outlier detection.

    Returns:
        Tuple of merged (points, colors).
    """
    if len(points_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Concatenate all point clouds
    all_points = np.concatenate(points_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)

    if all_points.shape[0] == 0:
        return all_points, all_colors

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float64) / 255.0)

    # Voxel downsampling
    if voxel_size > 0:
        logger.info(f"Voxel downsampling with voxel_size={voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)

    # Remove statistical outliers
    if remove_statistical_outliers and len(pcd.points) > nb_neighbors:
        logger.info(f"Removing statistical outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Convert back to numpy
    final_points = np.asarray(pcd.points, dtype=np.float32)
    final_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    return final_points, final_colors


def estimate_frame_to_frame_pose(
    pcd_source: o3d.geometry.PointCloud,
    pcd_target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray = np.eye(4),
    max_correspondence_distance: float = 0.1,
    estimation_method: str = "point_to_plane",
) -> np.ndarray:
    """Estimate transformation between two point clouds using ICP.

    Args:
        pcd_source: Source point cloud.
        pcd_target: Target point cloud.
        initial_transform: Initial transformation guess (4x4 matrix).
        max_correspondence_distance: Maximum correspondence distance for ICP.
        estimation_method: "point_to_point" or "point_to_plane".

    Returns:
        4x4 transformation matrix from source to target.
    """
    if len(pcd_source.points) < 10 or len(pcd_target.points) < 10:
        return initial_transform

    # Compute normals if using point-to-plane
    if estimation_method == "point_to_plane":
        if not pcd_source.has_normals():
            pcd_source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        if not pcd_target.has_normals():
            pcd_target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        pcd_source,
        pcd_target,
        max_correspondence_distance,
        initial_transform,
        method,
    )

    return result.transformation


def export_to_point_cloud_ply(
    prediction: Prediction,
    export_dir: str,
    camera_params: Optional[dict] = None,
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
) -> str:
    """Export prediction to a dense point cloud PLY file.

    This function creates a high-quality point cloud map from multiple depth
    predictions, optionally using ICP for frame alignment.

    Args:
        prediction: Model prediction containing depth, confidence, and optionally
                    intrinsics and extrinsics.
        export_dir: Output directory for the PLY file.
        camera_params: Optional dictionary with camera intrinsics parameters.
                       If provided, these override prediction.intrinsics.
        voxel_size: Voxel size for final downsampling (0 for no downsampling).
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

    Returns:
        Path to the exported PLY file.
    """
    assert prediction.depth is not None, "Depth data is required"
    assert prediction.processed_images is not None, "Processed images are required"

    N = prediction.depth.shape[0]
    H, W = prediction.depth.shape[1:3]

    logger.info(f"Generating point cloud map from {N} frames")
    logger.info(f"Frame size: {W}x{H}")

    # Prepare intrinsics
    if camera_params is not None:
        # Use provided camera parameters
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
        intrinsics_list = [K] * N
        logger.info(f"Using provided camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        depth_scale = camera_params.get("depth_scale", depth_scale)
    elif prediction.intrinsics is not None:
        intrinsics_list = prediction.intrinsics
        logger.info("Using predicted camera intrinsics")
    else:
        # Use default intrinsics (assume standard pinhole camera)
        fx = fy = max(W, H) * 0.8
        cx, cy = W / 2, H / 2
        K = create_intrinsics_matrix(fx, fy, cx, cy)
        intrinsics_list = [K] * N
        logger.warning(f"No intrinsics provided, using default: fx=fy={fx:.2f}")

    # Compute adaptive confidence threshold
    if prediction.conf is not None:
        conf_pixels = prediction.conf.flatten()
        lower = np.percentile(conf_pixels, conf_thresh_percentile)
        upper = np.percentile(conf_pixels, 90.0)
        conf_thresh = min(max(conf_thresh, lower), upper)
        logger.info(f"Adaptive confidence threshold: {conf_thresh:.4f}")

    # Generate point clouds for each frame
    points_list = []
    colors_list = []
    pcds_for_icp = []

    # Prepare extrinsics
    if prediction.extrinsics is not None:
        extrinsics_list = prediction.extrinsics
        logger.info("Using predicted camera extrinsics")
    else:
        # No extrinsics available - will use ICP or identity
        extrinsics_list = [np.eye(4, dtype=np.float64)] * N
        if use_icp_alignment:
            logger.info("No extrinsics provided, will use ICP for alignment")
        else:
            logger.warning("No extrinsics provided and ICP disabled, frames will overlap!")

    # Accumulated transformation for ICP
    accumulated_transform = np.eye(4, dtype=np.float64)

    os.makedirs(export_dir, exist_ok=True)

    for i in range(N):
        logger.info(f"Processing frame {i+1}/{N}")

        depth_i = prediction.depth[i]
        colors_i = prediction.processed_images[i]
        conf_i = prediction.conf[i] if prediction.conf is not None else None

        if isinstance(intrinsics_list, np.ndarray):
            K_i = intrinsics_list[i]
        else:
            K_i = intrinsics_list[i] if i < len(intrinsics_list) else intrinsics_list[-1]

        # Generate point cloud for this frame
        if prediction.extrinsics is not None:
            # Use provided extrinsics
            ext_i = extrinsics_list[i]
            pts, cols = depths_to_point_cloud(
                depth_i, K_i, ext_i, colors_i, conf_i,
                conf_thresh, depth_scale, max_depth
            )
        else:
            # No extrinsics - use camera frame
            pts, cols = depths_to_point_cloud(
                depth_i, K_i, None, colors_i, conf_i,
                conf_thresh, depth_scale, max_depth
            )

            # Apply accumulated transformation
            if use_icp_alignment and pts.shape[0] > 0:
                pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
                pts = (accumulated_transform @ pts_h.T)[:3].T.astype(np.float32)

        if pts.shape[0] > 0:
            points_list.append(pts)
            colors_list.append(cols)

            # Create Open3D point cloud for ICP
            if use_icp_alignment and prediction.extrinsics is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

                if icp_voxel_size > 0:
                    pcd = pcd.voxel_down_sample(icp_voxel_size)

                pcds_for_icp.append(pcd)

                # Estimate transformation to next frame using ICP
                if len(pcds_for_icp) > 1:
                    transform = estimate_frame_to_frame_pose(
                        pcds_for_icp[-1],
                        pcds_for_icp[-2],
                        np.eye(4),
                        max_correspondence_distance=icp_voxel_size * 2,
                        estimation_method="point_to_plane"
                    )
                    # Accumulate transformation for next frame
                    # Note: We're computing frame i-1 to frame i,
                    # so we need the inverse for the next frame
                    accumulated_transform = accumulated_transform @ np.linalg.inv(transform)

            # Export individual frame point cloud if requested
            if export_per_frame:
                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                frame_pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
                frame_path = os.path.join(export_dir, f"frame_{i:04d}.ply")
                o3d.io.write_point_cloud(frame_path, frame_pcd)
                logger.info(f"Exported frame {i} point cloud: {frame_path}")
        else:
            logger.warning(f"Frame {i} produced no valid points")

    # Merge all point clouds
    logger.info("Merging point clouds...")
    final_points, final_colors = merge_point_clouds(
        points_list,
        colors_list,
        voxel_size=voxel_size,
        remove_statistical_outliers=remove_outliers,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    logger.info(f"Final point cloud: {final_points.shape[0]} points")

    # Save to PLY
    if final_points.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(final_colors.astype(np.float64) / 255.0)

        out_path = os.path.join(export_dir, "point_cloud_map.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        logger.info(f"Point cloud map exported to: {out_path}")

        # Also save metadata
        metadata = {
            "num_frames": N,
            "num_points": int(final_points.shape[0]),
            "voxel_size": voxel_size,
            "conf_thresh": float(conf_thresh),
            "depth_scale": depth_scale,
            "max_depth": max_depth,
            "use_icp_alignment": use_icp_alignment,
        }
        if camera_params is not None:
            metadata["camera_params"] = camera_params

        metadata_path = os.path.join(export_dir, "point_cloud_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return out_path
    else:
        logger.error("No valid points generated!")
        return ""
