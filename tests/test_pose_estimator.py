from types import SimpleNamespace

import pytest

from backend.pose_estimator import pose_frame_from_result


def fake_pose_result(landmark_count: int = 33) -> SimpleNamespace:
    """Build a small MediaPipe-shaped result without loading a model."""

    landmarks = [
        SimpleNamespace(
            x=index / 100,
            y=index / 100 + 0.1,
            z=-index / 100,
            visibility=0.9,
            presence=0.8,
        )
        for index in range(landmark_count)
    ]
    return SimpleNamespace(pose_landmarks=[landmarks])


@pytest.mark.skip(reason="Remove after implementing pose_frame_from_result")
def test_pose_frame_from_result_maps_landmarks() -> None:
    pose_frame = pose_frame_from_result(
        source_frame_index=12,
        timestamp_ms=500,
        result=fake_pose_result(),
    )

    assert pose_frame.source_frame_index == 12
    assert pose_frame.timestamp_ms == 500
    assert pose_frame.landmarks is not None
    assert len(pose_frame.landmarks) == 33
    assert pose_frame.landmarks[0].x == 0.0
    assert pose_frame.landmarks[32].x == 0.32


@pytest.mark.skip(reason="Remove after implementing pose_frame_from_result")
def test_pose_frame_from_result_preserves_missing_pose() -> None:
    result = SimpleNamespace(pose_landmarks=[])

    pose_frame = pose_frame_from_result(3, 250, result)

    assert pose_frame.source_frame_index == 3
    assert pose_frame.timestamp_ms == 250
    assert pose_frame.landmarks is None


@pytest.mark.skip(reason="Remove after implementing pose_frame_from_result")
def test_pose_frame_from_result_rejects_wrong_landmark_count() -> None:
    with pytest.raises(ValueError, match="33"):
        pose_frame_from_result(0, 0, fake_pose_result(landmark_count=32))
