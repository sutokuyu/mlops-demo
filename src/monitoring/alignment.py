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

# What to do when matching fails outright, i.e. past the trust-last-good window.
# ``USE_FRAME`` resolves zones from the frame as captured, ``SKIP`` leaves the zone
# unknown. See ``AlignmentTracker.on_failure``.
USE_FRAME = "use_frame"
SKIP = "skip"

RATIO_TEST = 0.75
RANSAC_REPROJECTION_THRESHOLD = 0.004
DEFAULT_WORK_WIDTH = 640

# How big a usable transform has to get before it counts as "the camera moved".
# Measured on these three cameras by matching every stored reference frame of a
# camera against every other one (28 / 21 / 36 pairs, spanning 22:00 night frames
# and 11:00 daylight frames of cameras nobody touched): a pure illumination change
# never produced more than shift 0.019, rotation 0.47deg or scale 0.005, and often
# produced no transform at all. The thresholds sit above that noise floor, and a
# genuine knock is an order of magnitude larger than either.
DEFAULT_MIN_SHIFT = 0.04
DEFAULT_MIN_ROTATION_DEG = 1.0
DEFAULT_MIN_SCALE_DELTA = 0.02


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


def transform_magnitude(matrix: np.ndarray) -> tuple[float, float, float]:
    """``(shift, |rotation in degrees|, |scale - 1|)`` of an affine transform.

    This is what separates "the camera was knocked" from "the light changed". ORB
    is rotation-invariant, so a camera that actually moved still matches plenty of
    keypoints and hands back a transform with a visible shift or rotation, whereas
    an illumination change collapses the match count and returns no transform at
    all. So a large transform is evidence of movement and a failure is not, which is
    the opposite of what the re-anchor trigger used to assume.
    """
    return (
        math.hypot(matrix[0][2], matrix[1][2]),
        abs(math.degrees(math.atan2(matrix[1][0], matrix[0][0]))),
        abs(math.hypot(matrix[0][0], matrix[1][0]) - 1.0),
    )


@dataclass
class AlignmentTracker:
    """Per-camera alignment with a fallback window for temporarily bad matches.

    ``on_failure`` decides what a hard failure means for the zone lookup. The
    default (``USE_FRAME``) resolves zones from the frame as captured, on the
    assumption that a fixed camera plus a lighting change is far more likely than
    a camera that moved and stopped matching. It is not free: a genuinely moved
    camera then yields wrong zones instead of missing ones. Those samples are
    still marked ``alignment_quality == FAILED`` in the database, so they can be
    filtered out afterwards - which is exactly why recording the quality matters
    more than discarding the row. ``SKIP`` keeps the older, stricter behaviour of
    leaving the zone unknown.
    """

    camera: str
    reference: ReferenceFeatures | None = None
    work_width: int = DEFAULT_WORK_WIDTH
    min_inliers: int = 15
    good_inlier_ratio: float = 0.5
    max_residual: float = 0.01
    trust_last_good_seconds: float = 60.0
    on_failure: str = USE_FRAME
    # Above these, a usable transform means the camera moved rather than that the
    # light changed. See transform_magnitude() for where the numbers come from.
    min_shift: float = DEFAULT_MIN_SHIFT
    min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG
    min_scale_delta: float = DEFAULT_MIN_SCALE_DELTA
    last_good_matrix: np.ndarray | None = None
    last_good_at: float = 0.0
    consecutive_failures: int = 0
    # Consecutive samples whose transform was large. This is what the re-anchor
    # trigger watches by default: it acts while matching still works, which is the
    # only moment projecting the zones onto a new frame can succeed.
    displacement_streak: int = 0
    last_magnitude: tuple[float, float, float] | None = None
    quality: str = GOOD
    note: str = ""
    history: list = field(default_factory=list)

    def _record_displacement(self, matrix: np.ndarray) -> None:
        """Update the displacement streak from a usable transform."""
        magnitude = transform_magnitude(matrix)
        self.last_magnitude = magnitude
        shift, rotation, scale_delta = magnitude
        if (
            shift >= self.min_shift
            or rotation >= self.min_rotation_deg
            or scale_delta >= self.min_scale_delta
        ):
            self.displacement_streak += 1
        else:
            self.displacement_streak = 0

    def set_reference(self, reference: ReferenceFeatures | None) -> None:
        self.reference = reference
        self.last_good_matrix = None
        self.last_good_at = 0.0
        self.consecutive_failures = 0
        self.displacement_streak = 0
        self.last_magnitude = None
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
            # A failure carries no transform, so it is no evidence of movement:
            # break the streak rather than let it keep building up on samples that
            # say nothing about where the camera is pointing.
            self.displacement_streak = 0
            if (
                self.last_good_matrix is not None
                and now - self.last_good_at <= self.trust_last_good_seconds
            ):
                self.quality = DEGRADED
                self.note = f"{result.reason}; reusing the last good alignment"
                return AlignmentState(self.last_good_matrix, DEGRADED, self.note, True)
            self.quality = FAILED
            self.note = result.reason
            # No matrix: the caller uses the anchor point as captured, which is
            # the same path an uncalibrated camera takes. Applying the stale
            # last_good_matrix here would stack one unverified guess on another.
            return AlignmentState(None, FAILED, self.note, self.on_failure == USE_FRAME)

        self.consecutive_failures = 0
        self.quality = result.quality
        self.note = result.reason
        self._record_displacement(result.matrix)
        if result.quality == GOOD:
            self.last_good_matrix = result.matrix
            self.last_good_at = now
        self.history.append(describe_transform(result.matrix))
        del self.history[:-50]
        return AlignmentState(result.matrix, result.quality, self.note, True)
