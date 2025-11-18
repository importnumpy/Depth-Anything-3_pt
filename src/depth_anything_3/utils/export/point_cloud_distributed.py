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
Multi-GPU Distributed Point Cloud Generation

This module implements distributed point cloud generation across multiple GPUs
for maximum efficiency when processing very long videos (10,000+ frames).

Key features:
- Data-parallel processing across multiple GPUs
- Each GPU processes its partition independently
- Efficient point cloud gathering and merging
- Near-linear scaling with GPU count
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from typing import Any, Optional

import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from depth_anything_3.utils.logger import logger

from .point_cloud_batched import (
    _compute_adaptive_conf_thresh,
    _merge_batch_points,
    _prepare_intrinsics,
    clear_memory,
    depths_to_point_cloud,
)


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed processing group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed processing group."""
    dist.destroy_process_group()


def process_partition(
    rank: int,
    world_size: int,
    model_dir: str,
    image_paths: list[str],
    export_dir: str,
    camera_params: Optional[dict],
    batch_size: int,
    process_res: int,
    process_res_method: str,
    voxel_size: float,
    conf_thresh: float,
    conf_thresh_percentile: float,
    depth_scale: float,
    max_depth: float,
    remove_outliers: bool,
    nb_neighbors: int,
    std_ratio: float,
    temp_dir: str,
):
    """
    Process a partition of frames on a specific GPU.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        model_dir: Path to model directory
        image_paths: Full list of image paths
        export_dir: Output directory
        camera_params: Camera intrinsics parameters
        batch_size: Batch size for processing
        process_res: Processing resolution
        process_res_method: Resize method
        voxel_size: Voxel size for downsampling
        conf_thresh: Confidence threshold
        conf_thresh_percentile: Percentile for adaptive threshold
        depth_scale: Depth scale factor
        max_depth: Maximum depth value
        remove_outliers: Whether to remove outliers
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio
        temp_dir: Temporary directory for intermediate files
    """
    # Setup distributed processing
    setup_distributed(rank, world_size)

    try:
        from depth_anything_3.api import DepthAnything3

        # Partition data for this GPU
        total_frames = len(image_paths)
        frames_per_gpu = (total_frames + world_size - 1) // world_size

        start_idx = rank * frames_per_gpu
        end_idx = min(start_idx + frames_per_gpu, total_frames)

        local_image_paths = image_paths[start_idx:end_idx]
        local_frame_count = len(local_image_paths)

        logger.info(
            f"[GPU {rank}] Processing {local_frame_count} frames "
            f"({start_idx} to {end_idx-1})"
        )

        # Load model on this GPU
        device = f"cuda:{rank}"
        logger.info(f"[GPU {rank}] Loading model to {device}...")
        model = DepthAnything3.from_pretrained(model_dir).to(device)

        # Process first frame to get parameters
        first_pred = model.inference(
            image=[local_image_paths[0]],
            process_res=process_res,
            process_res_method=process_res_method,
            export_format="",
        )

        H, W = first_pred.depth.shape[1:3]

        # Prepare intrinsics
        intrinsics_list = _prepare_intrinsics(
            camera_params, first_pred, local_frame_count, W, H
        )

        # Compute adaptive confidence threshold
        conf_thresh = _compute_adaptive_conf_thresh(
            first_pred, conf_thresh, conf_thresh_percentile
        )

        del first_pred
        clear_memory()

        # Process batches on this GPU
        accumulated_pcd = None
        num_batches = (local_frame_count + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, local_frame_count)
            batch_paths = local_image_paths[batch_start:batch_end]

            logger.info(
                f"[GPU {rank}] Batch {batch_idx + 1}/{num_batches} "
                f"(frames {start_idx + batch_start}-{start_idx + batch_end - 1})"
            )

            # Run inference
            try:
                prediction = model.inference(
                    image=batch_paths,
                    process_res=process_res,
                    process_res_method=process_res_method,
                    export_format="",
                )
            except Exception as e:
                logger.error(f"[GPU {rank}] Batch {batch_idx} failed: {e}")
                clear_memory()
                continue

            # Convert to point clouds
            batch_points = []
            batch_colors = []

            for i in range(len(batch_paths)):
                frame_idx = batch_start + i

                depth_i = prediction.depth[i]
                colors_i = prediction.processed_images[i]
                conf_i = prediction.conf[i] if prediction.conf is not None else None

                if isinstance(intrinsics_list, np.ndarray):
                    K_i = intrinsics_list[frame_idx]
                else:
                    K_i = intrinsics_list[min(frame_idx, len(intrinsics_list) - 1)]

                ext_i = None
                if prediction.extrinsics is not None:
                    ext_i = prediction.extrinsics[i]

                pts, cols = depths_to_point_cloud(
                    depth_i, K_i, ext_i, colors_i, conf_i,
                    conf_thresh, depth_scale, max_depth
                )

                if pts.shape[0] > 0:
                    batch_points.append(pts)
                    batch_colors.append(cols)

            # Merge batch
            if len(batch_points) > 0:
                batch_pcd = _merge_batch_points(
                    batch_points, batch_colors, voxel_size,
                    remove_outliers, nb_neighbors, std_ratio
                )

                if accumulated_pcd is None:
                    accumulated_pcd = batch_pcd
                else:
                    accumulated_pcd = accumulated_pcd + batch_pcd

                # Downsample periodically
                if (batch_idx + 1) % 3 == 0 and voxel_size > 0:
                    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)

            del prediction, batch_points, batch_colors, batch_pcd
            clear_memory()

        # Save local point cloud to temporary file
        if accumulated_pcd is not None and len(accumulated_pcd.points) > 0:
            temp_file = os.path.join(temp_dir, f"partition_{rank:02d}.ply")
            o3d.io.write_point_cloud(temp_file, accumulated_pcd)
            logger.info(
                f"[GPU {rank}] Saved {len(accumulated_pcd.points)} points to {temp_file}"
            )

            # Send point count to rank 0
            point_count = len(accumulated_pcd.points)
        else:
            point_count = 0

        # Synchronize all GPUs
        dist.barrier()

        # Gather point counts on rank 0
        if rank == 0:
            point_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.gather(torch.tensor([point_count], dtype=torch.long, device=device), gather_list=point_counts)
            total_points = sum([pc.item() for pc in point_counts])
            logger.info(f"[GPU 0] Total points from all GPUs: {total_points}")
        else:
            dist.gather(torch.tensor([point_count], dtype=torch.long, device=device))

        # Clean up
        del model, accumulated_pcd
        clear_memory()

    finally:
        cleanup_distributed()


def merge_partitions(
    temp_dir: str,
    world_size: int,
    voxel_size: float,
    remove_outliers: bool,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    """
    Merge point cloud partitions from all GPUs.

    Args:
        temp_dir: Directory containing partition files
        world_size: Number of GPUs
        voxel_size: Final voxel size for downsampling
        remove_outliers: Whether to remove outliers
        nb_neighbors: Number of neighbors
        std_ratio: Standard deviation ratio

    Returns:
        Merged point cloud
    """
    logger.info("Merging point cloud partitions from all GPUs...")

    merged_pcd = o3d.geometry.PointCloud()

    for rank in range(world_size):
        temp_file = os.path.join(temp_dir, f"partition_{rank:02d}.ply")

        if os.path.exists(temp_file):
            logger.info(f"Loading partition {rank} from {temp_file}")
            partition_pcd = o3d.io.read_point_cloud(temp_file)
            merged_pcd = merged_pcd + partition_pcd
            logger.info(f"Merged {len(partition_pcd.points)} points from GPU {rank}")
            del partition_pcd
        else:
            logger.warning(f"Partition file not found: {temp_file}")

    # Final processing
    if len(merged_pcd.points) > 0:
        logger.info(f"Total merged points: {len(merged_pcd.points)}")

        if voxel_size > 0:
            logger.info(f"Final voxel downsampling with voxel_size={voxel_size}")
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size)

        if remove_outliers and len(merged_pcd.points) > nb_neighbors:
            logger.info("Final outlier removal")
            merged_pcd, _ = merged_pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )

        logger.info(f"Final point cloud: {len(merged_pcd.points)} points")

    return merged_pcd


def export_to_point_cloud_ply_distributed(
    model_dir: str,
    image_paths: list[str],
    export_dir: str,
    camera_params: Optional[dict] = None,
    world_size: int = 4,
    batch_size: int = 20,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    voxel_size: float = 0.01,
    conf_thresh: float = 1.0,
    conf_thresh_percentile: float = 40.0,
    depth_scale: float = 1.0,
    max_depth: float = 100.0,
    remove_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    final_voxel_size: Optional[float] = None,
) -> str:
    """
    Export point cloud using distributed multi-GPU processing.

    This function distributes the workload across multiple GPUs, with each GPU
    processing a partition of the frames independently. The results are then
    merged into a single point cloud.

    Args:
        model_dir: Path to model directory
        image_paths: List of image file paths
        export_dir: Output directory
        camera_params: Optional camera intrinsics
        world_size: Number of GPUs to use
        batch_size: Batch size per GPU
        process_res: Processing resolution
        process_res_method: Resize method
        voxel_size: Intermediate voxel size
        conf_thresh: Base confidence threshold
        conf_thresh_percentile: Adaptive threshold percentile
        depth_scale: Depth scale factor
        max_depth: Maximum depth value
        remove_outliers: Whether to remove outliers
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio
        final_voxel_size: Final voxel size (if None, uses voxel_size)

    Returns:
        Path to the exported PLY file
    """
    N = len(image_paths)

    if world_size > torch.cuda.device_count():
        world_size = torch.cuda.device_count()
        logger.warning(
            f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available. "
            f"Using {world_size} GPUs."
        )

    logger.info("=" * 70)
    logger.info("DISTRIBUTED MULTI-GPU POINT CLOUD GENERATION")
    logger.info("=" * 70)
    logger.info(f"Total frames: {N}")
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Frames per GPU: ~{N // world_size}")
    logger.info(f"Batch size per GPU: {batch_size}")

    os.makedirs(export_dir, exist_ok=True)

    # Create temporary directory for partition files
    temp_dir = tempfile.mkdtemp(prefix="pointcloud_partitions_")
    logger.info(f"Temporary directory: {temp_dir}")

    if final_voxel_size is None:
        final_voxel_size = voxel_size

    try:
        # Spawn processes for each GPU
        mp.spawn(
            process_partition,
            args=(
                world_size,
                model_dir,
                image_paths,
                export_dir,
                camera_params,
                batch_size,
                process_res,
                process_res_method,
                voxel_size,
                conf_thresh,
                conf_thresh_percentile,
                depth_scale,
                max_depth,
                remove_outliers,
                nb_neighbors,
                std_ratio,
                temp_dir,
            ),
            nprocs=world_size,
            join=True,
        )

        logger.info("All GPU processes completed")

        # Merge partitions on main process
        final_pcd = merge_partitions(
            temp_dir, world_size, final_voxel_size,
            remove_outliers, nb_neighbors, std_ratio
        )

        # Save final point cloud
        if len(final_pcd.points) > 0:
            out_path = os.path.join(export_dir, "point_cloud_map.ply")
            o3d.io.write_point_cloud(out_path, final_pcd)
            logger.info(f"Point cloud map exported to: {out_path}")

            # Save metadata
            metadata = {
                "num_frames": N,
                "num_points": int(len(final_pcd.points)),
                "num_gpus": world_size,
                "batch_size": batch_size,
                "voxel_size": voxel_size,
                "final_voxel_size": final_voxel_size,
                "conf_thresh": float(conf_thresh),
                "depth_scale": depth_scale,
                "max_depth": max_depth,
                "processing_mode": "distributed_multi_gpu",
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

    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
