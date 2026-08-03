import pytest

from backend.pose_benchmark import (
    JointCoverage,
    JointFailureCounts,
    calculate_longest_missing_pose_run,
    calculate_per_joint_coverage,
    calculate_required_joint_coverage,
    landmark_is_usable,
)
from backend.pose_estimator import PoseFrame, PoseLandmark


def make_landmark(
    *,
    x: float = 0.5,
    y: float = 0.5,
    visibility: float | None = 0.9,
    presence: float | None = 0.9,
) -> PoseLandmark:
    return PoseLandmark(x, y, 0.0, visibility, presence)


def make_pose(timestamp_ms: int, landmark: PoseLandmark | None) -> PoseFrame:
    landmarks = None if landmark is None else (landmark,) * 33
    return PoseFrame(timestamp_ms, timestamp_ms, landmarks)


@pytest.mark.parametrize(
    ("landmark", "expected"),
    [
        (make_landmark(), True),
        (make_landmark(x=-0.1), False),
        (make_landmark(y=1.1), False),
        (make_landmark(visibility=0.4), False),
        (make_landmark(presence=0.4), False),
        (make_landmark(visibility=None), False),
    ],
)
def test_landmark_is_usable(landmark: PoseLandmark, expected: bool) -> None:
    assert landmark_is_usable(landmark, confidence_threshold=0.5) is expected


@pytest.mark.parametrize("confidence_threshold", [0.0, 1.0])
def test_landmark_is_usable_accepts_threshold_endpoints(
    confidence_threshold: float,
) -> None:
    landmark = make_landmark(visibility=1.0, presence=1.0)

    assert landmark_is_usable(landmark, confidence_threshold) is True


@pytest.mark.parametrize(
    "confidence_threshold",
    [-0.1, 1.1, float("inf"), float("-inf"), float("nan")],
)
def test_landmark_is_usable_rejects_invalid_threshold(
    confidence_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        landmark_is_usable(make_landmark(), confidence_threshold)


def test_required_joint_coverage_counts_missing_and_low_confidence() -> None:
    pose_frames = (
        make_pose(0, make_landmark()),
        make_pose(100, make_landmark(visibility=0.4)),
        make_pose(200, None),
        make_pose(300, make_landmark()),
    )

    coverage = calculate_required_joint_coverage(pose_frames, 0.5)

    assert coverage == pytest.approx(50.0)


def test_per_joint_coverage_identifies_the_low_confidence_joint() -> None:
    valid_landmarks = [make_landmark() for _ in range(33)]
    low_wrist_landmarks = valid_landmarks.copy()
    low_wrist_landmarks[15] = make_landmark(visibility=0.4)
    pose_frames = (
        PoseFrame(0, 0, tuple(valid_landmarks)),
        PoseFrame(1, 100, tuple(low_wrist_landmarks)),
        PoseFrame(2, 200, None),
    )

    coverages = calculate_per_joint_coverage(pose_frames, 0.5)

    assert len(coverages) == 12
    assert coverages[0] == JointCoverage(
        landmark_index=11,
        landmark_name="left_shoulder",
        usable_count=2,
        unusable_count=1,
        coverage_percentage=pytest.approx(100 * 2 / 3),
        failure_counts=JointFailureCounts(
            missing_pose_count=1,
        ),
    )
    assert coverages[4] == JointCoverage(
        landmark_index=15,
        landmark_name="left_wrist",
        usable_count=1,
        unusable_count=2,
        coverage_percentage=pytest.approx(100 / 3),
        failure_counts=JointFailureCounts(
            missing_pose_count=1,
            low_visibility_count=1,
        ),
    )


def test_per_joint_coverage_reports_failure_reasons() -> None:
    valid_landmarks = [make_landmark() for _ in range(33)]

    missing_wrist_landmarks = tuple(valid_landmarks[:15])

    invalid_coordinate_landmarks = valid_landmarks.copy()
    invalid_coordinate_landmarks[15] = make_landmark(x=-0.1)

    low_visibility_landmarks = valid_landmarks.copy()
    low_visibility_landmarks[15] = make_landmark(visibility=0.4)

    low_presence_landmarks = valid_landmarks.copy()
    low_presence_landmarks[15] = make_landmark(presence=0.4)

    pose_frames = (
        PoseFrame(0, 0, tuple(valid_landmarks)),
        PoseFrame(1, 100, None),
        PoseFrame(2, 200, missing_wrist_landmarks),
        PoseFrame(3, 300, tuple(invalid_coordinate_landmarks)),
        PoseFrame(4, 400, tuple(low_visibility_landmarks)),
        PoseFrame(5, 500, tuple(low_presence_landmarks)),
    )

    left_wrist = calculate_per_joint_coverage(pose_frames, 0.5)[4]

    assert left_wrist.usable_count == 1
    assert left_wrist.unusable_count == 5
    assert left_wrist.failure_counts == JointFailureCounts(
        missing_pose_count=1,
        missing_landmark_count=1,
        invalid_coordinate_count=1,
        low_visibility_count=1,
        low_presence_count=1,
    )


def test_longest_missing_pose_run_includes_run_at_end() -> None:
    pose_frames = (
        make_pose(0, None),
        make_pose(100, make_landmark()),
        make_pose(200, None),
        make_pose(300, None),
    )

    assert calculate_longest_missing_pose_run(pose_frames) == 2
