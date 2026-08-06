from pathlib import Path

import numpy as np
import pytest

from backend.pose_estimator import PoseFrame, PoseLandmark, WorldPoseLandmark
from backend.pose_sequence import (
    PoseSequence,
    build_pose_sequence,
    load_pose_sequence,
    save_pose_sequence,
)


def make_image_landmarks(
    count: int = 33,
    *,
    visibility: float | None = 0.9,
    presence: float | None = 0.8,
) -> tuple[PoseLandmark, ...]:
    return tuple(
        PoseLandmark(
            x=index / 100,
            y=index / 100 + 0.1,
            z=-index / 100,
            visibility=visibility,
            presence=presence,
        )
        for index in range(count)
    )


def make_world_landmarks(
    count: int = 33,
    *,
    visibility: float | None = 0.7,
    presence: float | None = 0.6,
) -> tuple[WorldPoseLandmark, ...]:
    return tuple(
        WorldPoseLandmark(
            x=index / 10,
            y=index / 10 + 0.2,
            z=-index / 10,
            visibility=visibility,
            presence=presence,
        )
        for index in range(count)
    )


def test_build_pose_sequence_maps_fixed_shape_arrays() -> None:
    frames = (
        PoseFrame(4, 100, make_image_landmarks(), make_world_landmarks()),
        PoseFrame(8, 200, make_image_landmarks(), make_world_landmarks()),
    )

    sequence = build_pose_sequence(frames)

    np.testing.assert_array_equal(sequence.source_frame_indices, [4, 8])
    np.testing.assert_array_equal(sequence.timestamps_ms, [100, 200])
    assert sequence.image_xyz.shape == (2, 33, 3)
    assert sequence.world_xyz.shape == (2, 33, 3)
    np.testing.assert_allclose(sequence.image_xyz[0, 16], [0.16, 0.26, -0.16])
    np.testing.assert_allclose(sequence.world_xyz[1, 16], [1.6, 1.8, -1.6])
    np.testing.assert_array_equal(sequence.image_pose_available, [True, True])
    np.testing.assert_array_equal(sequence.world_pose_available, [True, True])


def test_build_pose_sequence_preserves_missing_data_as_nan() -> None:
    frames = (
        PoseFrame(0, 0, make_image_landmarks(), None),
        PoseFrame(1, 100, None, None),
    )

    sequence = build_pose_sequence(frames)

    np.testing.assert_array_equal(sequence.image_pose_available, [True, False])
    np.testing.assert_array_equal(sequence.world_pose_available, [False, False])
    assert np.isnan(sequence.image_xyz[1]).all()
    assert np.isnan(sequence.world_xyz).all()


def test_build_pose_sequence_preserves_low_and_missing_confidence() -> None:
    frame = PoseFrame(
        0,
        0,
        make_image_landmarks(visibility=0.1, presence=None),
        make_world_landmarks(visibility=0.2, presence=None),
    )

    sequence = build_pose_sequence((frame,))

    np.testing.assert_allclose(sequence.image_visibility[0], 0.1)
    np.testing.assert_allclose(sequence.world_visibility[0], 0.2)
    assert np.isnan(sequence.image_presence[0]).all()
    assert np.isnan(sequence.world_presence[0]).all()


def test_build_pose_sequence_rejects_wrong_image_landmark_count() -> None:
    frame = PoseFrame(0, 0, make_image_landmarks(count=32))

    with pytest.raises(ValueError, match="33 image landmarks"):
        build_pose_sequence((frame,))


def test_build_pose_sequence_rejects_wrong_world_landmark_count() -> None:
    frame = PoseFrame(
        0,
        0,
        make_image_landmarks(),
        make_world_landmarks(count=32),
    )

    with pytest.raises(ValueError, match="33 world landmarks"):
        build_pose_sequence((frame,))


def test_build_pose_sequence_rejects_empty_or_unordered_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        build_pose_sequence(())

    frames = (
        PoseFrame(0, 100, None),
        PoseFrame(1, 100, None),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_pose_sequence(frames)


def test_save_and_load_pose_sequence_round_trip(tmp_path: Path) -> None:
    sequence = PoseSequence(
        source_frame_indices=np.asarray([4], dtype=np.int64),
        timestamps_ms=np.asarray([100], dtype=np.int64),
        image_xyz=np.full((1, 33, 3), 0.5, dtype=np.float32),
        image_visibility=np.full((1, 33), 0.9, dtype=np.float32),
        image_presence=np.full((1, 33), 0.8, dtype=np.float32),
        world_xyz=np.full((1, 33, 3), np.nan, dtype=np.float32),
        world_visibility=np.full((1, 33), np.nan, dtype=np.float32),
        world_presence=np.full((1, 33), np.nan, dtype=np.float32),
        image_pose_available=np.asarray([True]),
        world_pose_available=np.asarray([False]),
    )
    output_path = tmp_path / "pose-sequence.npz"

    saved_path = save_pose_sequence(sequence, output_path)
    loaded = load_pose_sequence(output_path)

    assert saved_path == output_path
    assert output_path.is_file()
    np.testing.assert_array_equal(
        loaded.source_frame_indices,
        sequence.source_frame_indices,
    )
    np.testing.assert_array_equal(loaded.timestamps_ms, sequence.timestamps_ms)
    np.testing.assert_allclose(loaded.image_xyz, sequence.image_xyz)
    np.testing.assert_allclose(loaded.world_xyz, sequence.world_xyz, equal_nan=True)
    np.testing.assert_array_equal(
        loaded.image_pose_available,
        sequence.image_pose_available,
    )
    np.testing.assert_array_equal(
        loaded.world_pose_available,
        sequence.world_pose_available,
    )
