from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from backend.pose_sequence import PoseSequence
from scripts import export_pose_sequence


def make_sequence() -> PoseSequence:
    return PoseSequence(
        source_frame_indices=np.asarray([0, 3], dtype=np.int64),
        timestamps_ms=np.asarray([0, 100], dtype=np.int64),
        image_xyz=np.full((2, 33, 3), np.nan, dtype=np.float32),
        image_visibility=np.full((2, 33), np.nan, dtype=np.float32),
        image_presence=np.full((2, 33), np.nan, dtype=np.float32),
        world_xyz=np.full((2, 33, 3), np.nan, dtype=np.float32),
        world_visibility=np.full((2, 33), np.nan, dtype=np.float32),
        world_presence=np.full((2, 33), np.nan, dtype=np.float32),
        image_pose_available=np.asarray([True, False]),
        world_pose_available=np.asarray([True, False]),
    )


def test_export_pose_sequence_connects_pipeline_and_returns_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_frames = ("frame-0", "frame-1", "frame-2")
    bounded_pose_frames = ("pose-0", "pose-1")
    sequence = make_sequence()
    calls: list[object] = []

    def fake_sample(video_path: str, target_fps: float):
        calls.append(("sample", video_path, target_fps))
        return iter(sampled_frames)

    def fake_estimate(frames, landmarker):
        calls.append(("estimate", tuple(frames), landmarker))
        return iter(bounded_pose_frames)

    def fake_build(pose_frames):
        calls.append(("build", tuple(pose_frames)))
        return sequence

    def fake_save(received_sequence, output_path: Path) -> Path:
        calls.append(("save", received_sequence, output_path))
        return output_path

    landmarker = object()
    monkeypatch.setattr(export_pose_sequence, "sample_video_frames", fake_sample)
    monkeypatch.setattr(
        export_pose_sequence,
        "create_pose_landmarker",
        lambda model_path: calls.append(("model", model_path))
        or nullcontext(landmarker),
    )
    monkeypatch.setattr(export_pose_sequence, "estimate_pose_frames", fake_estimate)
    monkeypatch.setattr(export_pose_sequence, "build_pose_sequence", fake_build)
    monkeypatch.setattr(export_pose_sequence, "save_pose_sequence", fake_save)

    output_path = tmp_path / "video1.npz"
    summary = export_pose_sequence.export_pose_sequence(
        video_path=Path("video1.mp4"),
        model_path=Path("full.task"),
        output_path=output_path,
        target_fps=20.0,
        max_frames=2,
    )

    assert calls == [
        ("sample", "video1.mp4", 20.0),
        ("model", Path("full.task")),
        ("estimate", sampled_frames[:2], landmarker),
        ("build", bounded_pose_frames),
        ("save", sequence, output_path),
    ]
    assert summary.output_path == output_path
    assert summary.frame_count == 2
    assert summary.image_pose_count == 1
    assert summary.world_pose_count == 1
    assert summary.image_xyz_shape == (2, 33, 3)
    assert summary.world_xyz_shape == (2, 33, 3)


@pytest.mark.parametrize("target_fps", [0.0, -1.0, float("inf")])
def test_export_pose_sequence_rejects_invalid_target_fps(target_fps: float) -> None:
    with pytest.raises(ValueError, match="target_fps"):
        export_pose_sequence.export_pose_sequence(
            Path("video.mp4"),
            Path("model.task"),
            Path("output.npz"),
            target_fps=target_fps,
        )


def test_export_pose_sequence_rejects_invalid_max_frames() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        export_pose_sequence.export_pose_sequence(
            Path("video.mp4"),
            Path("model.task"),
            Path("output.npz"),
            max_frames=0,
        )
