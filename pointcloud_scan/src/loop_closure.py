"""
Loop Closure Detection Module

Detects when the camera revisits a location during 360° scanning.
Uses deep features and geometric verification.
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
import logging

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoopClosure:
    """Represents a detected loop closure"""
    frame_i: int
    frame_j: int
    transformation: np.ndarray  # 4x4 homogeneous matrix
    score: float
    visual_score: float
    geometric_score: float
    inlier_rmse: float


class LoopClosureDetector:
    """Detects loop closures using visual and geometric features"""

    def __init__(
        self,
        method: str = "deep",  # "deep", "orb", or "hybrid"
        visual_threshold: float = 0.7,
        geometric_threshold: float = 0.6,
        combined_threshold: float = 0.75,
        temporal_gap: int = 30,  # Minimum frames between loop closure candidates
        use_gpu: bool = True,
    ):
        """
        Initialize loop closure detector.

        Args:
            method: Detection method ("deep", "orb", "hybrid")
            visual_threshold: Minimum visual similarity score
            geometric_threshold: Minimum geometric consistency score
            combined_threshold: Minimum combined score
            temporal_gap: Minimum frame gap for loop candidates
            use_gpu: Use GPU for deep features if available
        """
        self.method = method
        self.visual_threshold = visual_threshold
        self.geometric_threshold = geometric_threshold
        self.combined_threshold = combined_threshold
        self.temporal_gap = temporal_gap
        self.use_gpu = use_gpu and torch.cuda.is_available() if HAS_TORCH else False

        # Initialize feature extractor
        if method in ["deep", "hybrid"] and HAS_TORCH:
            self._init_deep_extractor()
        elif method in ["orb", "hybrid"] and HAS_CV2:
            self._init_orb_extractor()
        else:
            raise ValueError(
                f"Method '{method}' requires cv2 or torch. "
                f"HAS_CV2={HAS_CV2}, HAS_TORCH={HAS_TORCH}"
            )

        self.feature_cache = {}

    def _init_deep_extractor(self):
        """Initialize deep feature extractor (ResNet)"""
        self.deep_model = models.resnet50(pretrained=True)
        # Remove classification layer
        self.deep_model = torch.nn.Sequential(*list(self.deep_model.children())[:-1])
        self.deep_model.eval()

        if self.use_gpu:
            self.deep_model = self.deep_model.cuda()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Initialized deep feature extractor (GPU={self.use_gpu})")

    def _init_orb_extractor(self):
        """Initialize ORB feature extractor"""
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        logger.info("Initialized ORB feature extractor")

    def extract_visual_features(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        """Extract visual features from image"""
        # Check cache
        if frame_id in self.feature_cache:
            return self.feature_cache[frame_id]

        if self.method == "deep" or self.method == "hybrid":
            features = self._extract_deep_features(image)
        elif self.method == "orb":
            features = self._extract_orb_features(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Cache features
        self.feature_cache[frame_id] = features
        return features

    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep CNN features"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Preprocess and extract
        img_tensor = self.transform(image).unsqueeze(0)

        if self.use_gpu:
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            features = self.deep_model(img_tensor)
            features = features.squeeze().cpu().numpy()

        return features / np.linalg.norm(features)  # Normalize

    def _extract_orb_features(self, image: np.ndarray) -> Tuple:
        """Extract ORB features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kp, des = self.orb.detectAndCompute(gray, None)
        return (kp, des)

    def compute_visual_similarity(
        self,
        features_i,
        features_j
    ) -> float:
        """Compute visual similarity score"""
        if self.method == "deep" or self.method == "hybrid":
            # Cosine similarity
            return float(np.dot(features_i, features_j))
        elif self.method == "orb":
            kp_i, des_i = features_i
            kp_j, des_j = features_j

            if des_i is None or des_j is None:
                return 0.0

            # Match features
            matches = self.matcher.match(des_i, des_j)

            # Similarity based on match ratio
            similarity = len(matches) / max(len(kp_i), len(kp_j), 1)
            return similarity
        else:
            return 0.0

    def compute_geometric_consistency(
        self,
        pcd_i: o3d.geometry.PointCloud,
        pcd_j: o3d.geometry.PointCloud,
        max_correspondence_distance: float = 0.05,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute geometric consistency using ICP.

        Returns:
            transformation: 4x4 transformation matrix
            fitness: ICP fitness score (0-1)
            inlier_rmse: RMSE of inliers
        """
        # Downsample for speed
        voxel_size = 0.02
        pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
        pcd_j_down = pcd_j.voxel_down_sample(voxel_size)

        # Estimate normals
        pcd_i_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
        pcd_j_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )

        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            pcd_j_down,
            pcd_i_down,
            max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50
            )
        )

        return result.transformation, result.fitness, result.inlier_rmse

    def detect_loops(
        self,
        images: List[np.ndarray],
        point_clouds: List[o3d.geometry.PointCloud],
        show_progress: bool = True,
    ) -> List[LoopClosure]:
        """
        Detect loop closures in a sequence.

        Args:
            images: List of images (RGB uint8)
            point_clouds: List of point clouds
            show_progress: Show progress bar

        Returns:
            List of detected loop closures
        """
        n = len(images)
        loop_closures = []

        logger.info(f"Detecting loop closures in {n} frames...")
        logger.info(f"Method: {self.method}, Temporal gap: {self.temporal_gap}")

        # Extract all features first
        logger.info("Extracting visual features...")
        features = []
        for i, img in enumerate(images):
            if show_progress and i % 50 == 0:
                logger.info(f"  Extracting features: {i}/{n}")
            features.append(self.extract_visual_features(img, i))

        # Compare all pairs
        logger.info("Comparing frame pairs...")
        comparisons = 0
        for i in range(n):
            if show_progress and i % 50 == 0:
                logger.info(f"  Processing frame {i}/{n}, found {len(loop_closures)} loops")

            # Only compare with frames far enough in time
            for j in range(i + self.temporal_gap, n):
                comparisons += 1

                # Visual similarity
                visual_score = self.compute_visual_similarity(features[i], features[j])

                if visual_score < self.visual_threshold:
                    continue

                # Geometric consistency
                T, fitness, rmse = self.compute_geometric_consistency(
                    point_clouds[i], point_clouds[j]
                )

                geometric_score = fitness

                if geometric_score < self.geometric_threshold:
                    continue

                # Combined score
                combined_score = 0.4 * visual_score + 0.6 * geometric_score

                if combined_score >= self.combined_threshold:
                    loop_closure = LoopClosure(
                        frame_i=i,
                        frame_j=j,
                        transformation=T,
                        score=combined_score,
                        visual_score=visual_score,
                        geometric_score=geometric_score,
                        inlier_rmse=rmse,
                    )
                    loop_closures.append(loop_closure)
                    logger.info(
                        f"  Loop detected: {i} ↔ {j} "
                        f"(score={combined_score:.3f}, visual={visual_score:.3f}, "
                        f"geom={geometric_score:.3f}, rmse={rmse:.4f})"
                    )

        logger.info(
            f"Loop closure detection complete: {len(loop_closures)} loops found "
            f"from {comparisons} comparisons"
        )

        return loop_closures

    def clear_cache(self):
        """Clear feature cache to free memory"""
        self.feature_cache.clear()
