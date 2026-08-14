import numpy as np
import pytest

from backend.pose_sequence import PoseSequence
from backend.pose_windows import (
    align_strokes_to_pose,
    extract_world_pose_windows,
    match_strokes_to_full_windows,
    nearest_timestamp_index,
    window_start_indices,
)
from backend.stroke_annotation import StrokeAnnotation, StrokeAnnotationFile


def make_sequence(frame_count: int = 10) -> PoseSequence:
    coordinate_shape = (frame_count, 33, 3)
    confidence_shape = (frame_count, 33)
    world_xyz = np.arange(
        np.prod(coordinate_shape),
        dtype=np.float32,
    ).reshape(coordinate_shape)
    available = np.ones(frame_count, dtype=np.bool_)
    if frame_count >= 4:
        available[3] = False
        world_xyz[3] = np.nan

    return PoseSequence(
        source_frame_indices=np.arange(frame_count, dtype=np.int64) * 2,
        timestamps_ms=np.arange(frame_count, dtype=np.int64) * 100,
        image_xyz=np.zeros(coordinate_shape, dtype=np.float32),
        image_visibility=np.ones(confidence_shape, dtype=np.float32),
        image_presence=np.ones(confidence_shape, dtype=np.float32),
        world_xyz=world_xyz,
        world_visibility=np.ones(confidence_shape, dtype=np.float32),
        world_presence=np.ones(confidence_shape, dtype=np.float32),
        image_pose_available=np.ones(frame_count, dtype=np.bool_),
        world_pose_available=available,
    )


def test_nearest_timestamp_index_finds_exact_near_and_tied_values() -> None:
    timestamps_ms = np.array([0, 100, 200, 300], dtype=np.int64)

    assert nearest_timestamp_index(timestamps_ms, 200) == 2
    assert nearest_timestamp_index(timestamps_ms, 176) == 2
    assert nearest_timestamp_index(timestamps_ms, 150) == 1


def test_nearest_timestamp_index_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        nearest_timestamp_index(np.array([], dtype=np.int64), 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        nearest_timestamp_index(np.array([0, 100, 100]), 100)
    with pytest.raises(ValueError, match="outside"):
        nearest_timestamp_index(np.array([100, 200]), 50)


def test_align_strokes_maps_boundaries_and_preserves_optional_contact() -> None:
    sequence = make_sequence()
    annotations = StrokeAnnotationFile(
        source_video="video.mp4",
        strokes=(
            StrokeAnnotation(90, 260, 410, "forehand"),
            StrokeAnnotation(500, None, 700, "backhand"),
        ),
    )

    aligned = align_strokes_to_pose(sequence, annotations)

    assert aligned[0].start.sequence_index == 1
    assert aligned[0].start.error_ms == 10
    assert aligned[0].contact is not None
    assert aligned[0].contact.sequence_index == 3
    assert aligned[0].end.sequence_index == 4
    assert aligned[1].contact is None


def test_window_start_indices_includes_only_complete_windows() -> None:
    assert np.array_equal(
        window_start_indices(10, 4, 3),
        np.array([0, 3, 6], dtype=np.int64),
    )
    assert np.array_equal(
        window_start_indices(3, 4, 1),
        np.array([], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("frame_count", "window_size", "stride"),
    [(-1, 4, 1), (10, 0, 1), (10, 4, 0), (10, 4.5, 1)],
)
def test_window_start_indices_rejects_invalid_settings(
    frame_count: int,
    window_size: int,
    stride: int,
) -> None:
    with pytest.raises(ValueError):
        window_start_indices(frame_count, window_size, stride)


def test_extract_world_pose_windows_preserves_values_and_missing_mask() -> None:
    sequence = make_sequence()

    windows = extract_world_pose_windows(
        sequence,
        window_size_frames=4,
        stride_frames=3,
    )

    assert windows.start_sequence_indices.tolist() == [0, 3, 6]
    assert windows.source_frame_indices.shape == (3, 4)
    assert windows.timestamps_ms.shape == (3, 4)
    assert windows.world_xyz.shape == (3, 4, 33, 3)
    assert windows.world_visibility.shape == (3, 4, 33)
    assert windows.world_presence.shape == (3, 4, 33)
    assert windows.world_pose_available.shape == (3, 4)
    assert windows.timestamps_ms[1].tolist() == [300, 400, 500, 600]
    assert not windows.world_pose_available[0, 3]
    assert np.isnan(windows.world_xyz[0, 3]).all()


def test_extract_world_pose_windows_returns_shaped_empty_arrays() -> None:
    windows = extract_world_pose_windows(
        make_sequence(frame_count=3),
        window_size_frames=4,
        stride_frames=1,
    )

    assert windows.timestamps_ms.shape == (0, 4)
    assert windows.world_xyz.shape == (0, 4, 33, 3)
    assert windows.world_pose_available.shape == (0, 4)


def test_match_strokes_selects_centered_full_window_or_none() -> None:
    sequence = make_sequence()
    annotations = StrokeAnnotationFile(
        source_video="video.mp4",
        strokes=(
            StrokeAnnotation(400, 500, 500, "forehand"),
        ),
    )
    aligned = align_strokes_to_pose(sequence, annotations)
    windows = extract_world_pose_windows(
        sequence,
        window_size_frames=6,
        stride_frames=2,
    )

    matches = match_strokes_to_full_windows(aligned, windows)

    assert matches[0].window_index == 1

    long_stroke = align_strokes_to_pose(
        sequence,
        StrokeAnnotationFile(
            source_video="video.mp4",
            strokes=(StrokeAnnotation(0, 500, 900, "backhand"),),
        ),
    )
    assert match_strokes_to_full_windows(long_stroke, windows)[0].window_index is None
