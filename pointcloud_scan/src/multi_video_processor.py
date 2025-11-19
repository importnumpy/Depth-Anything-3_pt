"""
Multi-Video Point Cloud Processing Pipeline

Processes multiple MP4 videos with loop closure for 360° object scanning.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.point_cloud import load_camera_params, depths_to_point_cloud
from depth_anything_3.services.input_handlers import VideoHandler

from .loop_closure import LoopClosureDetector, LoopClosure
from .pose_optimization import PoseGraphOptimizer, compute_trajectory_error
from .mesh_reconstruction import MeshReconstructor
from .utils import (
    save_trajectory_visualization,
    save_loop_closures_json,
    merge_point_clouds,
    setup_logging,
)

logger = logging.getLogger(__name__)


class MultiVideoPointCloudProcessor:
    """Process multiple videos with loop closure optimization"""

    def __init__(
        self,
        model_dir: str,
        camera_params_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 20,
    ):
        """
        Initialize processor.

        Args:
            model_dir: Path to Depth-Anything-3 model
            camera_params_path: Path to camera parameters JSON
            device: Device for inference
            batch_size: Batch size for processing
        """
        self.model_dir = model_dir
        self.camera_params_path = camera_params_path
        self.device = device
        self.batch_size = batch_size

        # Load model
        logger.info(f"Loading model from {model_dir}...")
        self.model = DepthAnything3.from_pretrained(model_dir).to(device)
        logger.info("Model loaded successfully")

        # Load camera parameters
        self.camera_params = None
        if camera_params_path:
            self.camera_params = load_camera_params(camera_params_path)
            logger.info(f"Loaded camera parameters: fx={self.camera_params['fx']:.2f}, "
                       f"fy={self.camera_params['fy']:.2f}")

    def process_videos(
        self,
        video_paths: List[str],
        output_dir: str,
        fps: float = 3.0,
        process_res: int = 504,
        enable_loop_closure: bool = True,
        loop_threshold: float = 0.75,
        generate_mesh: bool = True,
        mesh_method: str = "poisson",
        voxel_size: float = 0.005,
    ) -> dict:
        """
        Process multiple videos with loop closure.

        Args:
            video_paths: List of video file paths
            output_dir: Output directory
            fps: Frame sampling rate
            process_res: Processing resolution
            enable_loop_closure: Enable loop closure detection
            loop_threshold: Loop closure detection threshold
            generate_mesh: Generate mesh from point cloud
            mesh_method: Mesh reconstruction method
            voxel_size: Voxel size for point cloud downsampling

        Returns:
            Dictionary with processing results
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 70)
        logger.info("MULTI-VIDEO POINT CLOUD PROCESSING WITH LOOP CLOSURE")
        logger.info("=" * 70)
        logger.info(f"Input videos: {len(video_paths)}")
        for i, vp in enumerate(video_paths):
            logger.info(f"  [{i+1}] {vp}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Loop closure: {enable_loop_closure}")
        logger.info("")

        # Step 1: Extract frames from all videos
        logger.info("Step 1: Extracting frames from videos...")
        all_frames, video_frame_ranges = self._extract_frames(
            video_paths, output_dir, fps
        )
        logger.info(f"Extracted {len(all_frames)} total frames")

        # Step 2: Run depth inference
        logger.info("\nStep 2: Running depth inference...")
        depths, confs, intrinsics_list, images = self._run_inference(
            all_frames, process_res
        )

        # Step 3: Generate initial point clouds
        logger.info("\nStep 3: Generating initial point clouds...")
        point_clouds, poses_initial = self._generate_point_clouds(
            depths, confs, intrinsics_list, images
        )

        # Save raw point cloud
        raw_pcd = merge_point_clouds(point_clouds, voxel_size=voxel_size)
        raw_path = os.path.join(output_dir, "point_cloud_raw.ply")
        o3d.io.write_point_cloud(raw_path, raw_pcd)
        logger.info(f"Saved raw point cloud: {raw_path} ({len(raw_pcd.points)} points)")

        # Step 4: Loop closure detection
        loop_closures = []
        poses_optimized = poses_initial
        optimization_info = {}

        if enable_loop_closure:
            logger.info("\nStep 4: Detecting loop closures...")
            detector = LoopClosureDetector(
                method="hybrid" if len(images) < 1000 else "orb",
                combined_threshold=loop_threshold,
                temporal_gap=30,
            )

            loop_closures = detector.detect_loops(images, point_clouds)
            logger.info(f"Detected {len(loop_closures)} loop closures")

            # Save loop closures
            loops_path = os.path.join(output_dir, "loop_closures.json")
            save_loop_closures_json(loop_closures, loops_path)

            if len(loop_closures) > 0:
                # Step 5: Pose graph optimization
                logger.info("\nStep 5: Optimizing pose graph...")
                optimizer = PoseGraphOptimizer(use_g2o=None, max_iterations=50)

                poses_optimized, optimization_info = optimizer.optimize(
                    poses_initial, loop_closures
                )

                # Compute improvement
                error_metrics = compute_trajectory_error(poses_initial, poses_optimized)
                logger.info(f"Optimization results:")
                logger.info(f"  ATE translation: {error_metrics['ate_translation_mean']:.4f}m "
                           f"(±{error_metrics['ate_translation_std']:.4f})")
                logger.info(f"  ATE rotation: {error_metrics['ate_rotation_mean']:.4f}rad "
                           f"(±{error_metrics['ate_rotation_std']:.4f})")

                # Save trajectory visualization
                viz_dir = os.path.join(output_dir, "visualization")
                os.makedirs(viz_dir, exist_ok=True)
                save_trajectory_visualization(
                    poses_initial, poses_optimized, loop_closures, viz_dir
                )
            else:
                logger.warning("No loop closures detected, skipping optimization")

        # Step 6: Generate optimized point cloud
        logger.info("\nStep 6: Generating optimized point cloud...")
        final_pcd = self._apply_optimized_poses(
            point_clouds, poses_initial, poses_optimized, voxel_size
        )

        optimized_path = os.path.join(output_dir, "point_cloud_optimized.ply")
        o3d.io.write_point_cloud(optimized_path, final_pcd)
        logger.info(f"Saved optimized point cloud: {optimized_path} "
                   f"({len(final_pcd.points)} points)")

        # Step 7: Mesh reconstruction (optional)
        mesh_path = None
        if generate_mesh and len(final_pcd.points) > 1000:
            logger.info(f"\nStep 7: Generating mesh ({mesh_method})...")
            mesh_reconstructor = MeshReconstructor()

            mesh = mesh_reconstructor.reconstruct(
                final_pcd, method=mesh_method, depth=9
            )

            if mesh is not None:
                mesh_path = os.path.join(output_dir, "mesh.ply")
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                logger.info(f"Saved mesh: {mesh_path} "
                           f"({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")

                # Also save as OBJ
                obj_path = os.path.join(output_dir, "mesh.obj")
                o3d.io.write_triangle_mesh(obj_path, mesh)
                logger.info(f"Saved mesh (OBJ): {obj_path}")

        # Save metadata
        metadata = {
            "num_videos": len(video_paths),
            "video_paths": video_paths,
            "total_frames": len(all_frames),
            "video_frame_ranges": video_frame_ranges,
            "fps": fps,
            "voxel_size": voxel_size,
            "loop_closure_enabled": enable_loop_closure,
            "num_loop_closures": len(loop_closures),
            "optimization_info": optimization_info,
            "output_files": {
                "raw_point_cloud": raw_path,
                "optimized_point_cloud": optimized_path,
                "mesh": mesh_path,
                "loop_closures": loops_path if enable_loop_closure else None,
            },
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Files generated:")
        logger.info(f"  - point_cloud_raw.ply")
        logger.info(f"  - point_cloud_optimized.ply")
        if mesh_path:
            logger.info(f"  - mesh.ply / mesh.obj")
        if enable_loop_closure:
            logger.info(f"  - loop_closures.json")
            logger.info(f"  - visualization/trajectory_*.png")
        logger.info(f"  - metadata.json")

        return metadata

    def _extract_frames(
        self,
        video_paths: List[str],
        output_dir: str,
        fps: float,
    ) -> Tuple[List[str], dict]:
        """Extract frames from all videos"""
        all_frames = []
        video_frame_ranges = {}

        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, video_path in enumerate(video_paths):
            logger.info(f"  Processing video {i+1}/{len(video_paths)}: {video_path}")

            # Extract frames for this video
            video_frames_dir = os.path.join(frames_dir, f"video_{i:02d}")
            frames = VideoHandler.process(video_path, video_frames_dir, fps)

            start_idx = len(all_frames)
            all_frames.extend(frames)
            end_idx = len(all_frames)

            video_frame_ranges[f"video_{i}"] = {
                "path": video_path,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "num_frames": len(frames),
            }

            logger.info(f"    Extracted {len(frames)} frames "
                       f"(global indices: {start_idx}-{end_idx-1})")

        return all_frames, video_frame_ranges

    def _run_inference(
        self,
        frame_paths: List[str],
        process_res: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Run depth inference on all frames"""
        n = len(frame_paths)
        depths_list = []
        confs_list = []
        images_list = []
        intrinsics = None

        # Process in batches
        for i in range(0, n, self.batch_size):
            end_idx = min(i + self.batch_size, n)
            batch_paths = frame_paths[i:end_idx]

            logger.info(f"  Processing frames {i}-{end_idx-1}/{n}")

            prediction = self.model.inference(
                image=batch_paths,
                process_res=process_res,
                export_format="",
            )

            depths_list.append(prediction.depth)
            if prediction.conf is not None:
                confs_list.append(prediction.conf)
            images_list.append(prediction.processed_images)

            if intrinsics is None and prediction.intrinsics is not None:
                intrinsics = prediction.intrinsics

            del prediction

        # Concatenate
        depths = np.concatenate(depths_list, axis=0)
        confs = np.concatenate(confs_list, axis=0) if confs_list else None
        images = np.concatenate(images_list, axis=0)

        logger.info(f"  Inference complete: {depths.shape}")

        return depths, confs, intrinsics, images

    def _generate_point_clouds(
        self,
        depths: np.ndarray,
        confs: Optional[np.ndarray],
        intrinsics_list: Optional[List[np.ndarray]],
        images: np.ndarray,
    ) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
        """Generate point clouds from depth maps"""
        n = depths.shape[0]
        H, W = depths.shape[1:3]

        # Prepare intrinsics
        if self.camera_params:
            from depth_anything_3.utils.export.point_cloud import create_intrinsics_matrix
            fx = self.camera_params["fx"]
            fy = self.camera_params["fy"]
            cx = self.camera_params.get("cx", W / 2)
            cy = self.camera_params.get("cy", H / 2)

            # Scale if needed
            if "width" in self.camera_params:
                scale_x = W / self.camera_params["width"]
                scale_y = H / self.camera_params["height"]
                fx *= scale_x
                fy *= scale_y
                cx *= scale_x
                cy *= scale_y

            K = create_intrinsics_matrix(fx, fy, cx, cy)
            intrinsics_array = np.array([K] * n)
        elif intrinsics_list is not None:
            intrinsics_array = intrinsics_list
        else:
            # Default intrinsics
            fx = fy = max(W, H) * 0.8
            cx, cy = W / 2, H / 2
            from depth_anything_3.utils.export.point_cloud import create_intrinsics_matrix
            K = create_intrinsics_matrix(fx, fy, cx, cy)
            intrinsics_array = np.array([K] * n)

        # Generate point clouds
        point_clouds = []
        poses = []

        # Adaptive confidence threshold
        if confs is not None:
            conf_thresh = np.percentile(confs.flatten(), 40.0)
        else:
            conf_thresh = 0.0

        for i in range(n):
            if i % 100 == 0:
                logger.info(f"  Generating point clouds: {i}/{n}")

            depth_i = depths[i]
            conf_i = confs[i] if confs is not None else None
            colors_i = images[i]
            K_i = intrinsics_array[i]

            # Initial pose (identity for now, will be optimized)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = [0, 0, i * 0.1]  # Small offset for visualization

            # Generate point cloud
            pts, cols = depths_to_point_cloud(
                depth_i, K_i, None, colors_i, conf_i,
                conf_thresh, 1.0, 100.0
            )

            if pts.shape[0] > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
                point_clouds.append(pcd)
            else:
                # Empty point cloud
                point_clouds.append(o3d.geometry.PointCloud())

            poses.append(pose)

        logger.info(f"  Generated {len(point_clouds)} point clouds")

        return point_clouds, poses

    def _apply_optimized_poses(
        self,
        point_clouds: List[o3d.geometry.PointCloud],
        poses_initial: List[np.ndarray],
        poses_optimized: List[np.ndarray],
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        """Apply optimized poses and merge point clouds"""
        merged = o3d.geometry.PointCloud()

        for i, pcd in enumerate(point_clouds):
            if len(pcd.points) == 0:
                continue

            # Transform from initial to optimized pose
            T_correction = poses_optimized[i] @ np.linalg.inv(poses_initial[i])

            # Apply transformation
            pcd_transformed = pcd.transform(T_correction)

            # Merge
            merged += pcd_transformed

        # Downsample
        if voxel_size > 0:
            logger.info(f"  Downsampling with voxel_size={voxel_size}")
            merged = merged.voxel_down_sample(voxel_size)

        # Remove outliers
        if len(merged.points) > 100:
            logger.info("  Removing statistical outliers")
            merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        return merged


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-video point cloud processing with loop closure"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="List of video files to process",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--camera-params",
        help="Path to camera parameters JSON file",
    )
    parser.add_argument(
        "--model-dir",
        default="depth-anything/Depth-Anything-V2-Large",
        help="Model directory",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frame sampling rate",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--enable-loop-closure",
        action="store_true",
        default=True,
        help="Enable loop closure detection",
    )
    parser.add_argument(
        "--no-loop-closure",
        dest="enable_loop_closure",
        action="store_false",
        help="Disable loop closure",
    )
    parser.add_argument(
        "--loop-threshold",
        type=float,
        default=0.75,
        help="Loop closure detection threshold",
    )
    parser.add_argument(
        "--generate-mesh",
        action="store_true",
        default=True,
        help="Generate mesh from point cloud",
    )
    parser.add_argument(
        "--no-mesh",
        dest="generate_mesh",
        action="store_false",
        help="Skip mesh generation",
    )
    parser.add_argument(
        "--mesh-method",
        choices=["poisson", "ball_pivoting"],
        default="poisson",
        help="Mesh reconstruction method",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel size for downsampling",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)

    # Process
    processor = MultiVideoPointCloudProcessor(
        model_dir=args.model_dir,
        camera_params_path=args.camera_params,
        device=args.device,
        batch_size=args.batch_size,
    )

    processor.process_videos(
        video_paths=args.videos,
        output_dir=args.output_dir,
        fps=args.fps,
        enable_loop_closure=args.enable_loop_closure,
        loop_threshold=args.loop_threshold,
        generate_mesh=args.generate_mesh,
        mesh_method=args.mesh_method,
        voxel_size=args.voxel_size,
    )


if __name__ == "__main__":
    main()
