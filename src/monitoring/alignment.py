"""Align the current camera frame to a stored reference frame.

Zones are defined in the coordinate system of a reference frame. Cameras on
shelves drift, and they get re-aimed by hand, so every sample maps the cat's
anchor point back into reference coordinates before the zone lookup. Only one
point is transformed, so the polygons themselves never need to move.

All geometry here works in normalized coordinates (0-1) relative to each frame,
which keeps it independent of stream resolution and of the matching downscale.
"""

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

GOOD = "good"
DEGRADED = "degraded"
FAILED = "failed"

RATIO_TEST = 0.75
RANSAC_REPROJECTION_THRESHOLD = 0.004
DEFAULT_WORK_WIDTH = 640


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ReferenceFeatures:
    """ORB features of the reference frame, plus the frame itself for overlays."""

    frame: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    width: int
    height: int

    @property
    def usable(self) -> bool:
        return self.descriptors is not None and len(self.keypoints) >= 8


@dataclass
class AlignmentResult:
    quality: str
    matrix: np.ndarray | None
    inliers: int
    matches: int
    residual: float
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.matrix is not None


@dataclass
class AlignmentState:
    matrix: np.ndarray | None
    quality: str
    note: str
    zone_lookup_allowed: bool


def _resize_for_matching(frame: np.ndarray, work_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= work_width:
        return frame
    scale = work_width / width
    return cv2.resize(frame, (work_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _detect(frame: np.ndarray, work_width: int) -> tuple[list, np.ndarray | None, tuple[int, int]]:
    resized = _resize_for_matching(frame, work_width)
    keypoints, descriptors = cv2.ORB_create(nfeatures=2000).detectAndCompute(resized, None)
    height, width = resized.shape[:2]
    return list(keypoints), descriptors, (width, height)


def build_reference_features(frame: np.ndarray, work_width: int = DEFAULT_WORK_WIDTH):
    resized = _resize_for_matching(frame, work_width)
    keypoints, descriptors = cv2.ORB_create(nfeatures=2000).detectAndCompute(resized, None)
    height, width = resized.shape[:2]
    return ReferenceFeatures(
        frame=resized,
        keypoints=list(keypoints),
        descriptors=descriptors,
        width=width,
        height=height,
    )


def _normalized_points(keypoints, indices, width: int, height: int) -> np.ndarray:
    return np.array(
        [[keypoints[index].pt[0] / width, keypoints[index].pt[1] / height] for index in indices],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def estimate_alignment(
    reference: ReferenceFeatures,
    frame: np.ndarray,
    work_width: int = DEFAULT_WORK_WIDTH,
    min_inliers: int = 15,
    good_inlier_ratio: float = 0.5,
    max_residual: float = 0.01,
) -> AlignmentResult:
    """Estimate the affine transform mapping the current frame onto the reference."""
    if not reference.usable:
        return AlignmentResult(FAILED, None, 0, 0, 0.0, "reference frame has too few features")

    keypoints, descriptors, (width, height) = _detect(frame, work_width)
    if descriptors is None or len(keypoints) < min_inliers:
        return AlignmentResult(FAILED, None, 0, 0, 0.0, "current frame has too few features")

    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, reference.descriptors, k=2)
    good_matches = [first for first, second in knn if first.distance < RATIO_TEST * second.distance]
    if len(good_matches) < min_inliers:
        return AlignmentResult(
            FAILED, None, 0, len(good_matches), 0.0, f"only {len(good_matches)} feature matches"
        )

    source = _normalized_points(keypoints, [m.queryIdx for m in good_matches], width, height)
    target = _normalized_points(
        reference.keypoints,
        [m.trainIdx for m in good_matches],
        reference.width,
        reference.height,
    )

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
    )
    if matrix is None or inlier_mask is None:
        return AlignmentResult(
            FAILED, None, 0, len(good_matches), 0.0, "transform could not be estimated"
        )

    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        return AlignmentResult(
            FAILED,
            None,
            inliers,
            len(good_matches),
            0.0,
            f"only {inliers} inliers after RANSAC",
        )

    projected = cv2.transform(source, matrix)
    residual = float(
        np.mean(
            np.linalg.norm(
                projected[inlier_mask.ravel() == 1] - target[inlier_mask.ravel() == 1], axis=1
            )
        )
    )
    inlier_ratio = inliers / len(good_matches)

    if inlier_ratio >= good_inlier_ratio and residual <= max_residual:
        return AlignmentResult(GOOD, matrix, inliers, len(good_matches), residual)
    return AlignmentResult(
        DEGRADED,
        matrix,
        inliers,
        len(good_matches),
        residual,
        f"weak match (inlier ratio {inlier_ratio:.2f}, residual {residual:.4f})",
    )


def invert(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(matrix)


def transform_points(
    matrix: np.ndarray, points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not points:
        return []
    source = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.transform(source, matrix).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in projected]


def transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    return transform_points(matrix, [point])[0]


def describe_transform(matrix: np.ndarray) -> str:
    scale = math.hypot(matrix[0][0], matrix[1][0])
    rotation = math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
    return (
        f"shift=({matrix[0][2]:+.3f}, {matrix[1][2]:+.3f}) "
        f"rotation={rotation:+.2f}deg scale={scale:.3f}"
    )


@dataclass
class AlignmentTracker:
    """Per-camera alignment with a fallback window for temporarily bad matches."""

    camera: str
    reference: ReferenceFeatures | None = None
    work_width: int = DEFAULT_WORK_WIDTH
    min_inliers: int = 15
    good_inlier_ratio: float = 0.5
    max_residual: float = 0.01
    trust_last_good_seconds: float = 60.0
    last_good_matrix: np.ndarray | None = None
    last_good_at: float = 0.0
    consecutive_failures: int = 0
    quality: str = GOOD
    note: str = ""
    history: list = field(default_factory=list)

    def set_reference(self, reference: ReferenceFeatures | None) -> None:
        self.reference = reference
        self.last_good_matrix = None
        self.last_good_at = 0.0
        self.consecutive_failures = 0
        self.quality = GOOD if reference is not None else DEGRADED
        self.note = "" if reference is not None else "no reference frame"

    def resolve(self, frame: np.ndarray, now: float | None = None) -> AlignmentState:
        now = time.time() if now is None else now
        if self.reference is None:
            # Legacy camera without a calibration: use the frame as captured.
            self.quality = DEGRADED
            self.note = "no reference frame; zones used as captured"
            return AlignmentState(None, DEGRADED, self.note, True)

        result = estimate_alignment(
            self.reference,
            frame,
            work_width=self.work_width,
            min_inliers=self.min_inliers,
            good_inlier_ratio=self.good_inlier_ratio,
            max_residual=self.max_residual,
        )

        if result.quality == FAILED:
            self.consecutive_failures += 1
            if (
                self.last_good_matrix is not None
                and now - self.last_good_at <= self.trust_last_good_seconds
            ):
                self.quality = DEGRADED
                self.note = f"{result.reason}; reusing the last good alignment"
                return AlignmentState(self.last_good_matrix, DEGRADED, self.note, True)
            self.quality = FAILED
            self.note = result.reason
            return AlignmentState(None, FAILED, self.note, False)

        self.consecutive_failures = 0
        self.quality = result.quality
        self.note = result.reason
        if result.quality == GOOD:
            self.last_good_matrix = result.matrix
            self.last_good_at = now
        self.history.append(describe_transform(result.matrix))
        del self.history[:-50]
        return AlignmentState(result.matrix, result.quality, self.note, True)
