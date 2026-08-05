"""Pose data structures and MediaPipe video inference adapter."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp

from backend.frame_sampler import TimestampedFrame

EXPECTED_LANDMARK_COUNT = 33


def create_pose_landmarker(
    model_path: str | Path,
) -> "mp.tasks.vision.PoseLandmarker":
    """Create a single-person MediaPipe landmarker for video inference."""

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


@dataclass(frozen=True)
class PoseLandmark:
    """One normalized body landmark returned by pose estimation."""

    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None


@dataclass(frozen=True)
class WorldPoseLandmark:
    """One hip-centered 3D body landmark measured in meters."""

    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None


@dataclass(frozen=True)
class PoseFrame:
    """Image and world landmarks associated with one source frame."""

    source_frame_index: int
    timestamp_ms: int
    landmarks: tuple[PoseLandmark, ...] | None
    world_landmarks: tuple[WorldPoseLandmark, ...] | None = None


def _world_landmarks_from_result(
    result: "mp.tasks.vision.PoseLandmarkerResult",
) -> tuple[WorldPoseLandmark, ...] | None:
    """Convert the first optional MediaPipe world pose into project data."""

    world_poses = getattr(result, "pose_world_landmarks", None)

    if not world_poses:
        return None

    world_pose = world_poses[0]

    if len(world_pose) != EXPECTED_LANDMARK_COUNT:
        raise ValueError("Expected 33 world landmarks")

    world_landmarks = []
    for lm in world_pose:
        if lm.x is None or lm.y is None or lm.z is None:
            raise ValueError("Malformed world landmark data")
        world_landmarks.append(
            WorldPoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, "visibility", None),
                presence=getattr(lm, "presence", None),
            )
        )

    return tuple(world_landmarks)


def pose_frame_from_result(
    source_frame_index: int,
    timestamp_ms: int,
    result: "mp.tasks.vision.PoseLandmarkerResult",
) -> PoseFrame:
    """Convert one MediaPipe result into the project's pose schema."""

    if result.pose_landmarks is None or len(result.pose_landmarks) == 0:
        return PoseFrame(
            source_frame_index=source_frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=None,
        )

    pose = result.pose_landmarks[0]
    if len(pose) != EXPECTED_LANDMARK_COUNT:
        raise ValueError("Expected 33 landmarks")

    landmarks = []
    for lm in pose:
        if lm.x is None or lm.y is None or lm.z is None:
            raise ValueError("Malformed landmark data")
        landmarks.append(
            PoseLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, "visibility", None),
                presence=getattr(lm, "presence", None),
            )
        )

    return PoseFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        landmarks=tuple(landmarks),
        world_landmarks=_world_landmarks_from_result(result),
    )


def estimate_pose_frames_with_source(
    frames: Iterable[TimestampedFrame],
    landmarker: "mp.tasks.vision.PoseLandmarker",
) -> Iterator[tuple[TimestampedFrame, PoseFrame]]:
    """Yield each source frame paired with its MediaPipe pose result."""

    previous_timestamp_ms: int | None = None

    for frame in frames:
        if previous_timestamp_ms is not None and frame.timestamp_ms <= previous_timestamp_ms:
            raise ValueError(
                f"Frame timestamps must be strictly increasing, "
                f"but {frame.timestamp_ms} <= {previous_timestamp_ms}"
            )

        rgb_array = frame.image.to_ndarray(format="rgb24")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result = landmarker.detect_for_video(image, frame.timestamp_ms)

        pose_frame = pose_frame_from_result(
            source_frame_index=frame.source_frame_index,
            timestamp_ms=frame.timestamp_ms,
            result=result,
        )

        previous_timestamp_ms = frame.timestamp_ms
        yield frame, pose_frame


def estimate_pose_frames(
    frames: Iterable[TimestampedFrame],
    landmarker: "mp.tasks.vision.PoseLandmarker",
) -> Iterator[PoseFrame]:
    """Run an initialized MediaPipe landmarker on timestamped RGB frames."""

    for _, pose_frame in estimate_pose_frames_with_source(frames, landmarker):
        yield pose_frame
