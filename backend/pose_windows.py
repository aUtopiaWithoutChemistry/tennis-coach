"""Align stroke labels and slice pose sequences into fixed-length windows."""

from dataclasses import dataclass

import numpy as np

from backend.pose_sequence import PoseSequence
from backend.stroke_annotation import StrokeAnnotationFile


@dataclass(frozen=True)
class TimestampAlignment:
    """Nearest sampled pose timestamp for one annotated video timestamp."""

    target_timestamp_ms: int
    sequence_index: int
    pose_timestamp_ms: int
    error_ms: int


@dataclass(frozen=True)
class AlignedStroke:
    """One stroke whose annotated times are mapped onto pose frames."""

    stroke_type: str
    start: TimestampAlignment
    contact: TimestampAlignment | None
    end: TimestampAlignment


@dataclass(frozen=True)
class WorldPoseWindows:
    """Fixed-length world-pose samples with their original timing data."""

    start_sequence_indices: np.ndarray
    source_frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    world_xyz: np.ndarray
    world_visibility: np.ndarray
    world_presence: np.ndarray
    world_pose_available: np.ndarray


@dataclass(frozen=True)
class StrokeWindowMatch:
    """The best full window for a stroke, or None when none contains it."""

    stroke: AlignedStroke
    window_index: int | None


def nearest_timestamp_index(
    timestamps_ms: np.ndarray,
    target_timestamp_ms: int,
) -> int:
    """Return the earlier index when two timestamps are equally near."""

    if timestamps_ms.ndim != 1 or len(timestamps_ms) == 0:
        raise ValueError("timestamps_ms must be a nonempty one-dimensional array")
    if np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError("timestamps_ms must be strictly increasing")
    if type(target_timestamp_ms) is not int:
        raise ValueError("target_timestamp_ms must be an integer")
    if (
        target_timestamp_ms < timestamps_ms[0]
        or target_timestamp_ms > timestamps_ms[-1]
    ):
        raise ValueError("target timestamp falls outside the pose sequence")

    distances = np.abs(timestamps_ms - target_timestamp_ms)
    return int(np.argmin(distances))


def _align_timestamp(
    timestamps_ms: np.ndarray,
    target_timestamp_ms: int,
) -> TimestampAlignment:
    sequence_index = nearest_timestamp_index(timestamps_ms, target_timestamp_ms)
    pose_timestamp_ms = int(timestamps_ms[sequence_index])
    return TimestampAlignment(
        target_timestamp_ms=target_timestamp_ms,
        sequence_index=sequence_index,
        pose_timestamp_ms=pose_timestamp_ms,
        error_ms=pose_timestamp_ms - target_timestamp_ms,
    )


def align_strokes_to_pose(
    sequence: PoseSequence,
    annotations: StrokeAnnotationFile,
) -> tuple[AlignedStroke, ...]:
    """Map every annotated boundary to the nearest pose-sequence frame."""

    aligned_strokes = []
    for stroke in annotations.strokes:
        contact = None
        if stroke.contact_ms is not None:
            contact = _align_timestamp(
                sequence.timestamps_ms,
                stroke.contact_ms,
            )

        aligned_strokes.append(
            AlignedStroke(
                stroke_type=stroke.stroke_type,
                start=_align_timestamp(sequence.timestamps_ms, stroke.start_ms),
                contact=contact,
                end=_align_timestamp(sequence.timestamps_ms, stroke.end_ms),
            )
        )
    return tuple(aligned_strokes)


def window_start_indices(
    frame_count: int,
    window_size_frames: int,
    stride_frames: int,
) -> np.ndarray:
    """Return starts for every complete fixed-length sliding window."""

    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    if type(window_size_frames) is not int or window_size_frames <= 0:
        raise ValueError("window_size_frames must be a positive integer")
    if type(stride_frames) is not int or stride_frames <= 0:
        raise ValueError("stride_frames must be a positive integer")

    if frame_count < window_size_frames:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, frame_count - window_size_frames + 1, stride_frames, dtype=np.int64)


def _slice_windows(
    values: np.ndarray,
    starts: np.ndarray,
    window_size_frames: int,
) -> np.ndarray:
    """Copy matching slices from one pose-sequence array."""

    if len(starts) == 0:
        shape = (0, window_size_frames, *values.shape[1:])
        return np.empty(shape, dtype=values.dtype)
    return np.stack(
        [values[start : start + window_size_frames] for start in starts]
    )


def extract_world_pose_windows(
    sequence: PoseSequence,
    window_size_frames: int = 45,
    stride_frames: int = 8,
) -> WorldPoseWindows:
    """Extract complete overlapping windows without filling missing poses."""

    starts = window_start_indices(
        frame_count=len(sequence.timestamps_ms),
        window_size_frames=window_size_frames,
        stride_frames=stride_frames,
    )
    return WorldPoseWindows(
        start_sequence_indices=starts,
        source_frame_indices=_slice_windows(
            sequence.source_frame_indices,
            starts,
            window_size_frames,
        ),
        timestamps_ms=_slice_windows(
            sequence.timestamps_ms,
            starts,
            window_size_frames,
        ),
        world_xyz=_slice_windows(
            sequence.world_xyz,
            starts,
            window_size_frames,
        ),
        world_visibility=_slice_windows(
            sequence.world_visibility,
            starts,
            window_size_frames,
        ),
        world_presence=_slice_windows(
            sequence.world_presence,
            starts,
            window_size_frames,
        ),
        world_pose_available=_slice_windows(
            sequence.world_pose_available,
            starts,
            window_size_frames,
        ),
    )


def match_strokes_to_full_windows(
    strokes: tuple[AlignedStroke, ...],
    windows: WorldPoseWindows,
) -> tuple[StrokeWindowMatch, ...]:
    """Choose a containing window with contact nearest its temporal center."""

    if windows.timestamps_ms.ndim != 2:
        raise ValueError("window timestamps must be a two-dimensional array")

    matches = []
    for stroke in strokes:
        containing_indices = []
        for window_index, timestamps_ms in enumerate(windows.timestamps_ms):
            if (
                timestamps_ms[0] <= stroke.start.pose_timestamp_ms
                and timestamps_ms[-1] >= stroke.end.pose_timestamp_ms
            ):
                containing_indices.append(window_index)

        if not containing_indices:
            matches.append(StrokeWindowMatch(stroke=stroke, window_index=None))
            continue

        if stroke.contact is None:
            target_timestamp_ms = (
                stroke.start.pose_timestamp_ms + stroke.end.pose_timestamp_ms
            ) / 2
        else:
            target_timestamp_ms = stroke.contact.pose_timestamp_ms

        best_window_index = min(
            containing_indices,
            key=lambda index: abs(
                float(np.mean(windows.timestamps_ms[index])) - target_timestamp_ms
            ),
        )
        matches.append(
            StrokeWindowMatch(
                stroke=stroke,
                window_index=best_window_index,
            )
        )
    return tuple(matches)
