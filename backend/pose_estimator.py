"""Pose data structures and the future MediaPipe adapter."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mediapipe as mp


EXPECTED_LANDMARK_COUNT = 33


@dataclass(frozen=True)
class PoseLandmark:
    """One normalized body landmark returned by pose estimation."""

    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None


@dataclass(frozen=True)
class PoseFrame:
    """Pose landmarks associated with one timestamped source frame."""

    source_frame_index: int
    timestamp_ms: int
    landmarks: tuple[PoseLandmark, ...] | None


def pose_frame_from_result(
    source_frame_index: int,
    timestamp_ms: int,
    result: "mp.tasks.vision.PoseLandmarkerResult",
) -> PoseFrame:
    """Convert one MediaPipe result into the project's pose schema."""

    # TODO 1: Return landmarks=None when result.pose_landmarks is empty.

    # TODO 2: Select the first detected pose.

    # TODO 3: Require exactly EXPECTED_LANDMARK_COUNT raw landmarks.

    # TODO 4: Map each raw landmark into PoseLandmark.
    # Treat missing x, y, or z as malformed data; preserve missing
    # visibility and presence values as None.

    # TODO 5: Return a PoseFrame with an immutable landmark tuple.
    raise NotImplementedError("Complete the MediaPipe result mapping")
