"""
Mesh Reconstruction Module

Generates high-quality mesh from point cloud using various algorithms.
"""

import logging
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class MeshReconstructor:
    """Reconstruct mesh from point cloud"""

    def __init__(self):
        pass

    def reconstruct(
        self,
        point_cloud: o3d.geometry.PointCloud,
        method: str = "poisson",
        depth: int = 9,
        density_threshold: float = 0.01,
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Reconstruct mesh from point cloud.

        Args:
            point_cloud: Input point cloud
            method: Reconstruction method ("poisson" or "ball_pivoting")
            depth: Octree depth for Poisson (higher = more detail)
            density_threshold: Density threshold for Poisson

        Returns:
            Triangle mesh or None if reconstruction fails
        """
        if len(point_cloud.points) < 100:
            logger.error("Too few points for mesh reconstruction")
            return None

        logger.info(f"Reconstructing mesh using {method} method...")
        logger.info(f"Input point cloud: {len(point_cloud.points)} points")

        # Ensure normals are estimated
        if not point_cloud.has_normals():
            logger.info("Estimating normals...")
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            point_cloud.orient_normals_consistent_tangent_plane(30)

        if method == "poisson":
            mesh = self._poisson_reconstruction(
                point_cloud, depth, density_threshold
            )
        elif method == "ball_pivoting":
            mesh = self._ball_pivoting_reconstruction(point_cloud)
        else:
            raise ValueError(f"Unknown method: {method}")

        if mesh is not None:
            logger.info(f"Mesh generated: {len(mesh.vertices)} vertices, "
                       f"{len(mesh.triangles)} triangles")

            # Clean up mesh
            mesh = self._clean_mesh(mesh)

        return mesh

    def _poisson_reconstruction(
        self,
        point_cloud: o3d.geometry.PointCloud,
        depth: int,
        density_threshold: float,
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """Poisson surface reconstruction"""
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=depth
            )

            # Remove low-density vertices
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, density_threshold)

            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            return mesh

        except Exception as e:
            logger.error(f"Poisson reconstruction failed: {e}")
            return None

    def _ball_pivoting_reconstruction(
        self,
        point_cloud: o3d.geometry.PointCloud,
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """Ball pivoting reconstruction"""
        try:
            # Estimate point cloud density
            distances = point_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)

            # Set radii for ball pivoting
            radii = o3d.utility.DoubleVector([
                avg_dist * 1.0,
                avg_dist * 2.0,
                avg_dist * 4.0,
            ])

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, radii
            )

            return mesh

        except Exception as e:
            logger.error(f"Ball pivoting reconstruction failed: {e}")
            return None

    def _clean_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:
        """Clean and smooth mesh"""
        # Remove duplicated vertices and triangles
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # Remove non-manifold edges
        mesh.remove_non_manifold_edges()

        # Smooth mesh
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)

        # Compute vertex normals
        mesh.compute_vertex_normals()

        return mesh
