"""Pure aggregation helpers for evaluating pose-estimation runs."""

import math
from dataclasses import asdict, dataclass, field

from backend.pose_estimator import PoseFrame, PoseLandmark

REQUIRED_LANDMARKS = (
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
)
REQUIRED_LANDMARK_INDICES = tuple(index for _, index in REQUIRED_LANDMARKS)


@dataclass(frozen=True)
class JointFailureCounts:
    """Diagnostic reason counts; one unusable observation may have several."""

    missing_pose_count: int = 0
    missing_landmark_count: int = 0
    invalid_coordinate_count: int = 0
    low_visibility_count: int = 0
    low_presence_count: int = 0


@dataclass(frozen=True)
class JointCoverage:
    """Usability counts for one required landmark across sampled frames."""

    landmark_index: int
    landmark_name: str
    usable_count: int
    unusable_count: int
    coverage_percentage: float
    failure_counts: JointFailureCounts = field(default_factory=JointFailureCounts)


@dataclass(frozen=True)
class PoseBenchmarkSummary:
    """Serializable automated metrics from one video/model run."""

    video_name: str
    model_name: str
    target_fps: float
    confidence_threshold: float
    sampled_frame_count: int
    detected_pose_count: int
    detection_percentage: float
    usable_frame_count: int
    usable_frame_percentage: float
    required_joint_coverage_percentage: float
    required_joint_coverages: tuple[JointCoverage, ...]
    missing_pose_frame_count: int
    longest_missing_pose_run: int
    analyzed_video_seconds: float
    processing_seconds: float
    processed_fps: float
    processing_time_ratio: float

    def as_dict(self) -> dict[str, object]:
        """Return fields in a form that can be written directly as JSON."""

        return asdict(self)


def landmark_is_usable(
    landmark: PoseLandmark,
    confidence_threshold: float,
) -> bool:
    """Return whether one landmark has valid coordinates and confidence.

    A usable landmark has x/y inside the normalized image, and both its
    visibility and presence meet the threshold. Missing confidence fails.
    """

    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    return (landmark.x >= 0.0 and landmark.x <= 1.0 and
            landmark.y >= 0.0 and landmark.y <= 1.0 and
            (landmark.visibility is not None and landmark.visibility >= confidence_threshold) and
            (landmark.presence is not None and landmark.presence >= confidence_threshold))


def calculate_required_joint_coverage(
    pose_frames: tuple[PoseFrame, ...],
    confidence_threshold: float,
) -> float:
    """Return the percentage of required joint observations that are usable.

    The denominator is every required joint in every sampled frame. A frame
    with no pose therefore contributes zero usable joints.
    """

    total_required_joints = len(pose_frames) * len(REQUIRED_LANDMARK_INDICES)
    usable_required_joints = 0

    for frame in pose_frames:
        if frame.landmarks is None:
            continue
        for index in REQUIRED_LANDMARK_INDICES:
            if index < len(frame.landmarks):
                landmark = frame.landmarks[index]
                if landmark_is_usable(landmark, confidence_threshold):
                    usable_required_joints += 1

    return (usable_required_joints / total_required_joints) * 100 if total_required_joints > 0 else 0.0


def calculate_per_joint_coverage(
    pose_frames: tuple[PoseFrame, ...],
    confidence_threshold: float,
) -> tuple[JointCoverage, ...]:
    """Return ordered usability metrics for every required landmark.

    Every sampled frame contributes one observation per joint. A missing pose
    or missing landmark therefore counts as one unusable observation.
    """

    coverages = []
    for landmark_name, landmark_index in REQUIRED_LANDMARKS:
        usable_count = 0
        unusable_count = 0
        missing_pose_count = 0
        missing_landmark_count = 0
        invalid_coordinate_count = 0
        low_visibility_count = 0
        low_presence_count = 0

        # TODO 5: increment the appropriate failure counters in this loop.
        for frame in pose_frames:
            if frame.landmarks is None:
                missing_pose_count += 1
                unusable_count += 1
                continue

            if landmark_index >= len(frame.landmarks):
                missing_landmark_count += 1
                unusable_count += 1
                continue

            landmark = frame.landmarks[landmark_index]
            if landmark_is_usable(landmark, confidence_threshold):
                usable_count += 1

            else:
                unusable_count += 1

                if not (landmark.x >= 0.0 and landmark.x <= 1.0 and landmark.y >= 0.0 and landmark.y <= 1.0):
                    invalid_coordinate_count += 1

                if landmark.visibility is None or landmark.visibility < confidence_threshold:
                    low_visibility_count += 1

                if landmark.presence is None or landmark.presence < confidence_threshold:
                    low_presence_count += 1

        coverage_percentage = (usable_count / (usable_count + unusable_count)) * 100 if (usable_count + unusable_count) > 0 else 0.0

        coverages.append(
            JointCoverage(
                landmark_index=landmark_index,
                landmark_name=landmark_name,
                usable_count=usable_count,
                unusable_count=unusable_count,
                coverage_percentage=coverage_percentage,
                failure_counts=JointFailureCounts(
                    missing_pose_count=missing_pose_count,
                    missing_landmark_count=missing_landmark_count,
                    invalid_coordinate_count=invalid_coordinate_count,
                    low_visibility_count=low_visibility_count,
                    low_presence_count=low_presence_count,
                ),
            )
        )

    return tuple(coverages)


def calculate_longest_missing_pose_run(
    pose_frames: tuple[PoseFrame, ...],
) -> int:
    """Return the longest consecutive run whose landmarks are missing."""

    longest_run = 0
    current_run = 0

    for frame in pose_frames:
        if frame.landmarks is None:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return longest_run


def _frame_has_usable_required_joints(
    pose_frame: PoseFrame,
    confidence_threshold: float,
) -> bool:
    if pose_frame.landmarks is None:
        return False
    if len(pose_frame.landmarks) <= max(REQUIRED_LANDMARK_INDICES):
        return False
    return all(
        landmark_is_usable(
            pose_frame.landmarks[index],
            confidence_threshold,
        )
        for index in REQUIRED_LANDMARK_INDICES
    )


def build_pose_benchmark_summary(
    *,
    video_name: str,
    model_name: str,
    target_fps: float,
    confidence_threshold: float,
    pose_frames: tuple[PoseFrame, ...],
    processing_seconds: float,
) -> PoseBenchmarkSummary:
    """Aggregate one completed inference run into automated metrics."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    if not math.isfinite(processing_seconds) or processing_seconds <= 0:
        raise ValueError("processing_seconds must be finite and greater than zero")
    if not pose_frames:
        raise ValueError("At least one sampled pose frame is required")

    sampled_count = len(pose_frames)
    detected_count = sum(frame.landmarks is not None for frame in pose_frames)
    usable_count = sum(
        _frame_has_usable_required_joints(frame, confidence_threshold)
        for frame in pose_frames
    )
    analyzed_seconds = (
        (pose_frames[-1].timestamp_ms - pose_frames[0].timestamp_ms) / 1_000
        + 1 / target_fps
    )

    return PoseBenchmarkSummary(
        video_name=video_name,
        model_name=model_name,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold,
        sampled_frame_count=sampled_count,
        detected_pose_count=detected_count,
        detection_percentage=100 * detected_count / sampled_count,
        usable_frame_count=usable_count,
        usable_frame_percentage=100 * usable_count / sampled_count,
        required_joint_coverage_percentage=calculate_required_joint_coverage(
            pose_frames,
            confidence_threshold,
        ),
        required_joint_coverages=calculate_per_joint_coverage(
            pose_frames,
            confidence_threshold,
        ),
        missing_pose_frame_count=sampled_count - detected_count,
        longest_missing_pose_run=calculate_longest_missing_pose_run(pose_frames),
        analyzed_video_seconds=analyzed_seconds,
        processing_seconds=processing_seconds,
        processed_fps=sampled_count / processing_seconds,
        processing_time_ratio=processing_seconds / analyzed_seconds,
    )
