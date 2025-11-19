"""
Pose Graph Optimization Module

Optimizes camera poses using loop closure constraints.
Supports both g2opy and scipy-based optimization.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass

try:
    import g2o
    HAS_G2O = True
except ImportError:
    HAS_G2O = False

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .loop_closure import LoopClosure

logger = logging.getLogger(__name__)


@dataclass
class PoseConstraint:
    """Represents a pose constraint (odometry or loop closure)"""
    from_frame: int
    to_frame: int
    transformation: np.ndarray  # 4x4 matrix
    information: np.ndarray  # 6x6 information matrix
    is_loop_closure: bool = False


class PoseGraphOptimizer:
    """Optimizes pose graph with loop closure constraints"""

    def __init__(
        self,
        use_g2o: bool = None,
        max_iterations: int = 50,
        verbose: bool = True,
    ):
        """
        Initialize pose graph optimizer.

        Args:
            use_g2o: Use g2o if available (None = auto-detect)
            max_iterations: Maximum optimization iterations
            verbose: Print optimization progress
        """
        if use_g2o is None:
            use_g2o = HAS_G2O

        self.use_g2o = use_g2o and HAS_G2O
        self.max_iterations = max_iterations
        self.verbose = verbose

        if self.use_g2o:
            logger.info("Using g2o for pose graph optimization")
        else:
            logger.info("Using scipy for pose graph optimization (slower but works everywhere)")

    def optimize(
        self,
        poses: List[np.ndarray],
        loop_closures: List[LoopClosure],
        odometry_sigma: Tuple[float, float] = (0.01, 0.01),  # (translation, rotation)
        loop_sigma: Tuple[float, float] = (0.005, 0.005),
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Optimize poses using loop closure constraints.

        Args:
            poses: Initial poses (list of 4x4 matrices)
            loop_closures: Detected loop closures
            odometry_sigma: Odometry constraint sigma (m, rad)
            loop_sigma: Loop closure constraint sigma (m, rad)

        Returns:
            optimized_poses: Optimized poses
            info: Optimization info dict
        """
        n_poses = len(poses)
        n_loops = len(loop_closures)

        logger.info(f"Optimizing {n_poses} poses with {n_loops} loop closures...")

        # Build constraints
        constraints = self._build_constraints(
            poses, loop_closures, odometry_sigma, loop_sigma
        )

        # Optimize
        if self.use_g2o:
            optimized_poses, info = self._optimize_g2o(poses, constraints)
        else:
            optimized_poses, info = self._optimize_scipy(poses, constraints)

        logger.info(f"Optimization complete: {info['iterations']} iterations, "
                   f"final error = {info['final_error']:.6f}")

        return optimized_poses, info

    def _build_constraints(
        self,
        poses: List[np.ndarray],
        loop_closures: List[LoopClosure],
        odometry_sigma: Tuple[float, float],
        loop_sigma: Tuple[float, float],
    ) -> List[PoseConstraint]:
        """Build list of pose constraints"""
        constraints = []

        # Odometry constraints (sequential)
        for i in range(len(poses) - 1):
            # Relative transformation
            T_rel = np.linalg.inv(poses[i]) @ poses[i + 1]

            # Information matrix
            info = self._build_information_matrix(*odometry_sigma)

            constraint = PoseConstraint(
                from_frame=i,
                to_frame=i + 1,
                transformation=T_rel,
                information=info,
                is_loop_closure=False,
            )
            constraints.append(constraint)

        # Loop closure constraints
        for lc in loop_closures:
            info = self._build_information_matrix(*loop_sigma)

            constraint = PoseConstraint(
                from_frame=lc.frame_i,
                to_frame=lc.frame_j,
                transformation=lc.transformation,
                information=info,
                is_loop_closure=True,
            )
            constraints.append(constraint)

        logger.info(f"Built {len(constraints)} constraints "
                   f"({len(poses)-1} odometry + {len(loop_closures)} loops)")

        return constraints

    def _build_information_matrix(
        self,
        sigma_translation: float,
        sigma_rotation: float,
    ) -> np.ndarray:
        """Build 6x6 information matrix"""
        # Information = inverse of covariance
        info = np.eye(6)
        info[:3, :3] *= 1.0 / (sigma_translation ** 2)  # Translation
        info[3:, 3:] *= 1.0 / (sigma_rotation ** 2)  # Rotation
        return info

    def _optimize_g2o(
        self,
        initial_poses: List[np.ndarray],
        constraints: List[PoseConstraint],
    ) -> Tuple[List[np.ndarray], dict]:
        """Optimize using g2o (fast, requires g2opy)"""
        # Initialize optimizer
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)

        # Add vertices (poses)
        for i, pose in enumerate(initial_poses):
            v = g2o.VertexSE3()
            v.set_id(i)

            # Convert to g2o format
            R = pose[:3, :3]
            t = pose[:3, 3]
            se3_quat = g2o.SE3Quat(R, t)
            v.set_estimate(se3_quat)

            # Fix first pose
            if i == 0:
                v.set_fixed(True)

            optimizer.add_vertex(v)

        # Add edges (constraints)
        for constraint in constraints:
            edge = g2o.EdgeSE3()
            edge.set_vertex(0, optimizer.vertex(constraint.from_frame))
            edge.set_vertex(1, optimizer.vertex(constraint.to_frame))

            # Set measurement
            T = constraint.transformation
            R = T[:3, :3]
            t = T[:3, 3]
            se3_quat = g2o.SE3Quat(R, t)
            edge.set_measurement(se3_quat)

            # Set information
            edge.set_information(constraint.information)

            optimizer.add_edge(edge)

        # Optimize
        if self.verbose:
            optimizer.set_verbose(True)

        optimizer.initialize_optimization()
        optimizer.optimize(self.max_iterations)

        # Extract optimized poses
        optimized_poses = []
        for i in range(len(initial_poses)):
            v = optimizer.vertex(i)
            est = v.estimate()

            # Convert back to 4x4 matrix
            pose = np.eye(4)
            pose[:3, :3] = est.rotation().matrix()
            pose[:3, 3] = est.translation()

            optimized_poses.append(pose)

        # Get optimization info
        info = {
            'iterations': self.max_iterations,
            'final_error': optimizer.chi2(),
            'method': 'g2o',
        }

        return optimized_poses, info

    def _optimize_scipy(
        self,
        initial_poses: List[np.ndarray],
        constraints: List[PoseConstraint],
    ) -> Tuple[List[np.ndarray], dict]:
        """Optimize using scipy (slower but no dependencies)"""
        n_poses = len(initial_poses)

        # Convert poses to parameter vector
        # Each pose: 3 translation + 4 quaternion = 7 parameters
        x0 = np.zeros(n_poses * 7)
        for i, pose in enumerate(initial_poses):
            idx = i * 7
            x0[idx:idx+3] = pose[:3, 3]  # Translation
            R = Rotation.from_matrix(pose[:3, :3])
            x0[idx+3:idx+7] = R.as_quat()  # Quaternion (xyzw)

        # Define residual function
        def residuals(x):
            """Compute all constraint residuals"""
            res = []

            for constraint in constraints:
                i = constraint.from_frame
                j = constraint.to_frame

                # Extract poses
                idx_i = i * 7
                idx_j = j * 7

                t_i = x[idx_i:idx_i+3]
                q_i = x[idx_i+3:idx_i+7]
                R_i = Rotation.from_quat(q_i).as_matrix()

                t_j = x[idx_j:idx_j+3]
                q_j = x[idx_j+3:idx_j+7]
                R_j = Rotation.from_quat(q_j).as_matrix()

                # Compute relative transformation
                T_ij_est = np.eye(4)
                T_ij_est[:3, :3] = R_i.T @ R_j
                T_ij_est[:3, 3] = R_i.T @ (t_j - t_i)

                # Measurement
                T_ij_meas = constraint.transformation

                # Error
                error = self._pose_error(T_ij_est, T_ij_meas)

                # Weighted by information
                L = np.linalg.cholesky(constraint.information)
                weighted_error = L @ error

                res.extend(weighted_error.tolist())

            return np.array(res)

        # Optimize
        if self.verbose:
            logger.info("Starting scipy optimization...")

        result = least_squares(
            residuals,
            x0,
            method='lm',
            max_nfev=self.max_iterations * 10,
            verbose=2 if self.verbose else 0,
        )

        # Extract optimized poses
        optimized_poses = []
        for i in range(n_poses):
            idx = i * 7
            t = result.x[idx:idx+3]
            q = result.x[idx+3:idx+7]
            R = Rotation.from_quat(q).as_matrix()

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            optimized_poses.append(pose)

        # Info
        info = {
            'iterations': result.nfev,
            'final_error': result.cost,
            'method': 'scipy',
            'success': result.success,
        }

        return optimized_poses, info

    @staticmethod
    def _pose_error(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Compute error between two poses in se(3).

        Returns 6-vector: [translation_error, rotation_error]
        """
        T_diff = np.linalg.inv(T1) @ T2

        # Translation error
        t_error = T_diff[:3, 3]

        # Rotation error (axis-angle)
        R_diff = T_diff[:3, :3]
        rot = Rotation.from_matrix(R_diff)
        r_error = rot.as_rotvec()

        return np.concatenate([t_error, r_error])


def compute_trajectory_error(
    poses_before: List[np.ndarray],
    poses_after: List[np.ndarray],
) -> dict:
    """
    Compute trajectory error metrics before/after optimization.

    Returns dict with ATE, RPE metrics.
    """
    n = min(len(poses_before), len(poses_after))

    # Absolute Trajectory Error (ATE)
    ate_translations = []
    ate_rotations = []

    for i in range(n):
        T_diff = np.linalg.inv(poses_before[i]) @ poses_after[i]

        # Translation error
        t_error = np.linalg.norm(T_diff[:3, 3])
        ate_translations.append(t_error)

        # Rotation error
        R_diff = T_diff[:3, :3]
        rot = Rotation.from_matrix(R_diff)
        r_error = np.linalg.norm(rot.as_rotvec())
        ate_rotations.append(r_error)

    # Relative Pose Error (RPE)
    rpe_translations = []
    rpe_rotations = []

    for i in range(n - 1):
        # Relative poses
        T_rel_before = np.linalg.inv(poses_before[i]) @ poses_before[i + 1]
        T_rel_after = np.linalg.inv(poses_after[i]) @ poses_after[i + 1]

        # Error in relative pose
        T_diff = np.linalg.inv(T_rel_before) @ T_rel_after

        t_error = np.linalg.norm(T_diff[:3, 3])
        rpe_translations.append(t_error)

        R_diff = T_diff[:3, :3]
        rot = Rotation.from_matrix(R_diff)
        r_error = np.linalg.norm(rot.as_rotvec())
        rpe_rotations.append(r_error)

    return {
        'ate_translation_mean': np.mean(ate_translations),
        'ate_translation_std': np.std(ate_translations),
        'ate_rotation_mean': np.mean(ate_rotations),
        'ate_rotation_std': np.std(ate_rotations),
        'rpe_translation_mean': np.mean(rpe_translations),
        'rpe_translation_std': np.std(rpe_translations),
        'rpe_rotation_mean': np.mean(rpe_rotations),
        'rpe_rotation_std': np.std(rpe_rotations),
    }
