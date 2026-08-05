"""Convert timestamped pose results into ML-ready numerical sequences."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.pose_estimator import EXPECTED_LANDMARK_COUNT, PoseFrame


@dataclass(frozen=True)
class PoseSequence:
    """Fixed-shape pose arrays preserving missing data and confidence."""

    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    image_xyz: np.ndarray
    image_visibility: np.ndarray
    image_presence: np.ndarray
    world_xyz: np.ndarray
    world_visibility: np.ndarray
    world_presence: np.ndarray
    image_pose_available: np.ndarray
    world_pose_available: np.ndarray


def build_pose_sequence(pose_frames: Iterable[PoseFrame]) -> PoseSequence:
    """Convert ordered pose frames without filtering or interpolation."""

    frames = tuple(pose_frames)
    if not frames:
        raise ValueError("At least one pose frame is required")

    for previous, current in zip(frames, frames[1:], strict=False):
        if current.timestamp_ms <= previous.timestamp_ms:
            raise ValueError("Pose frame timestamps must be strictly increasing")

    frame_count = len(frames)
    coordinate_shape = (frame_count, EXPECTED_LANDMARK_COUNT, 3)
    confidence_shape = (frame_count, EXPECTED_LANDMARK_COUNT)

    source_frame_indices = np.asarray(
        [frame.source_frame_index for frame in frames],
        dtype=np.int64,
    )
    timestamps_ms = np.asarray(
        [frame.timestamp_ms for frame in frames],
        dtype=np.int64,
    )
    image_xyz = np.full(coordinate_shape, np.nan, dtype=np.float32)
    image_visibility = np.full(confidence_shape, np.nan, dtype=np.float32)
    image_presence = np.full(confidence_shape, np.nan, dtype=np.float32)
    world_xyz = np.full(coordinate_shape, np.nan, dtype=np.float32)
    world_visibility = np.full(confidence_shape, np.nan, dtype=np.float32)
    world_presence = np.full(confidence_shape, np.nan, dtype=np.float32)
    image_pose_available = np.zeros(frame_count, dtype=np.bool_)
    world_pose_available = np.zeros(frame_count, dtype=np.bool_)

    for frame_index, frame in enumerate(frames):
        if frame.landmarks is not None:
            if len(frame.landmarks) != EXPECTED_LANDMARK_COUNT:
                raise ValueError("Expected 33 image landmarks")
            image_pose_available[frame_index] = True
            for landmark_index, landmark in enumerate(frame.landmarks):
                image_xyz[frame_index, landmark_index] = (landmark.x, landmark.y, landmark.z)
                if landmark.visibility is not None:
                    image_visibility[frame_index, landmark_index] = landmark.visibility
                if landmark.presence is not None:
                    image_presence[frame_index, landmark_index] = landmark.presence

        if frame.world_landmarks is not None:
            if len(frame.world_landmarks) != EXPECTED_LANDMARK_COUNT:
                raise ValueError("Expected 33 world landmarks")
            world_pose_available[frame_index] = True
            for landmark_index, landmark in enumerate(frame.world_landmarks):
                world_xyz[frame_index, landmark_index] = (landmark.x, landmark.y, landmark.z)
                if landmark.visibility is not None:
                    world_visibility[frame_index, landmark_index] = landmark.visibility
                if landmark.presence is not None:
                    world_presence[frame_index, landmark_index] = landmark.presence

    return PoseSequence(
        source_frame_indices=source_frame_indices,
        timestamps_ms=timestamps_ms,
        image_xyz=image_xyz,
        image_visibility=image_visibility,
        image_presence=image_presence,
        world_xyz=world_xyz,
        world_visibility=world_visibility,
        world_presence=world_presence,
        image_pose_available=image_pose_available,
        world_pose_available=world_pose_available,
    )


def save_pose_sequence(sequence: PoseSequence, output_path: Path) -> Path:
    """Write a pose sequence to one compressed NumPy archive."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            source_frame_indices=sequence.source_frame_indices,
            timestamps_ms=sequence.timestamps_ms,
            image_xyz=sequence.image_xyz,
            image_visibility=sequence.image_visibility,
            image_presence=sequence.image_presence,
            world_xyz=sequence.world_xyz,
            world_visibility=sequence.world_visibility,
            world_presence=sequence.world_presence,
            image_pose_available=sequence.image_pose_available,
            world_pose_available=sequence.world_pose_available,
        )
    return output_path


def load_pose_sequence(input_path: Path) -> PoseSequence:
    """Load a pose sequence without allowing pickled objects."""

    with np.load(input_path, allow_pickle=False) as archive:
        return PoseSequence(
            source_frame_indices=archive["source_frame_indices"].copy(),
            timestamps_ms=archive["timestamps_ms"].copy(),
            image_xyz=archive["image_xyz"].copy(),
            image_visibility=archive["image_visibility"].copy(),
            image_presence=archive["image_presence"].copy(),
            world_xyz=archive["world_xyz"].copy(),
            world_visibility=archive["world_visibility"].copy(),
            world_presence=archive["world_presence"].copy(),
            image_pose_available=archive["image_pose_available"].copy(),
            world_pose_available=archive["world_pose_available"].copy(),
        )
