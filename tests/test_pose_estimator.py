from types import SimpleNamespace

import av
import mediapipe as mp
import pytest

from backend.frame_sampler import TimestampedFrame
from backend.pose_estimator import (
    create_pose_landmarker,
    estimate_pose_frames,
    pose_frame_from_result,
)


class FakeLandmarker:
    """Record video calls and return a configured MediaPipe-shaped result."""

    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[tuple[mp.Image, int]] = []

    def detect_for_video(
        self,
        image: mp.Image,
        timestamp_ms: int,
    ) -> SimpleNamespace:
        self.calls.append((image, timestamp_ms))
        return self.result


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


def fake_timestamped_frame(
    source_frame_index: int,
    timestamp_ms: int,
) -> TimestampedFrame:
    """Build a small RGB frame for adapter tests."""

    return TimestampedFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        image=av.VideoFrame(4, 3, "rgb24"),
    )


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


def test_pose_frame_from_result_preserves_missing_pose() -> None:
    result = SimpleNamespace(pose_landmarks=[])

    pose_frame = pose_frame_from_result(3, 250, result)

    assert pose_frame.source_frame_index == 3
    assert pose_frame.timestamp_ms == 250
    assert pose_frame.landmarks is None


def test_pose_frame_from_result_rejects_wrong_landmark_count() -> None:
    with pytest.raises(ValueError, match="33"):
        pose_frame_from_result(0, 0, fake_pose_result(landmark_count=32))


def test_pose_frame_from_result_rejects_malformed_landmark() -> None:
    malformed_landmark = SimpleNamespace(x=None, y=0.1, z=0.2)
    result = SimpleNamespace(pose_landmarks=[[malformed_landmark] * 33])

    with pytest.raises(ValueError, match="Malformed"):
        pose_frame_from_result(0, 0, result)


def test_pose_frame_from_result_preserves_missing_visibility_and_presence() -> None:
    landmark = SimpleNamespace(x=0.1, y=0.2, z=0.3)
    result = SimpleNamespace(pose_landmarks=[[landmark] * 33])

    pose_frame = pose_frame_from_result(0, 0, result)

    assert pose_frame.landmarks is not None
    for lm in pose_frame.landmarks:
        assert lm.visibility is None
        assert lm.presence is None


def test_estimate_pose_frames_calls_landmarker_and_preserves_metadata() -> None:
    landmarker = FakeLandmarker(fake_pose_result())
    frames = [
        fake_timestamped_frame(4, 100),
        fake_timestamped_frame(8, 200),
    ]

    pose_frames = list(estimate_pose_frames(frames, landmarker))

    assert [call[1] for call in landmarker.calls] == [100, 200]
    assert landmarker.calls[0][0].image_format == mp.ImageFormat.SRGB
    assert landmarker.calls[0][0].width == 4
    assert landmarker.calls[0][0].height == 3
    assert [pose.source_frame_index for pose in pose_frames] == [4, 8]
    assert [pose.timestamp_ms for pose in pose_frames] == [100, 200]


def test_estimate_pose_frames_rejects_non_increasing_timestamps() -> None:
    landmarker = FakeLandmarker(fake_pose_result())
    frames = [
        fake_timestamped_frame(0, 100),
        fake_timestamped_frame(1, 100),
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        list(estimate_pose_frames(frames, landmarker))


def test_create_pose_landmarker_rejects_missing_model(tmp_path) -> None:
    missing_model = tmp_path / "missing.task"

    with pytest.raises(FileNotFoundError, match="Pose model not found"):
        create_pose_landmarker(missing_model)
