from contextlib import nullcontext
from pathlib import Path

import av
import pytest

from backend.frame_sampler import TimestampedFrame
from backend.pose_estimator import PoseFrame, PoseLandmark
from scripts import smoke_test_pose


def make_frame(source_frame_index: int, timestamp_ms: int) -> TimestampedFrame:
    return TimestampedFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        image=av.VideoFrame(40, 30, "rgb24"),
    )


def make_pose(source_frame_index: int, timestamp_ms: int) -> PoseFrame:
    landmark = PoseLandmark(0.5, 0.5, 0.0, 0.9, 0.9)
    return PoseFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        landmarks=(landmark,) * 33,
    )


def test_save_pose_overlay_uses_source_metadata(tmp_path: Path) -> None:
    source_frame = make_frame(7, 250)
    pose_frame = PoseFrame(7, 250, landmarks=None)

    output_path = smoke_test_pose.save_pose_overlay(
        source_frame,
        pose_frame,
        tmp_path,
    )

    assert output_path.name == "pose_frame_000007_00000250ms.png"
    assert output_path.is_file()


def test_run_smoke_test_saves_only_requested_detected_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [make_frame(1, 100), make_frame(2, 200), make_frame(3, 300)]
    poses = [
        make_pose(1, 100),
        PoseFrame(2, 200, landmarks=None),
        make_pose(3, 300),
    ]
    saved_pairs: list[tuple[int, int]] = []

    monkeypatch.setattr(
        smoke_test_pose,
        "sample_video_frames",
        lambda *_: iter(frames),
    )
    monkeypatch.setattr(
        smoke_test_pose,
        "create_pose_landmarker",
        lambda *_: nullcontext(object()),
    )
    monkeypatch.setattr(
        smoke_test_pose,
        "estimate_pose_frames",
        lambda *_: iter(poses),
    )

    def fake_save_overlay(
        source_frame: TimestampedFrame,
        pose_frame: PoseFrame,
        output_dir: Path,
        diagnostic: bool = False,
    ) -> Path:
        assert diagnostic is False
        saved_pairs.append(
            (source_frame.source_frame_index, pose_frame.source_frame_index)
        )
        return output_dir / f"{source_frame.source_frame_index}.png"

    monkeypatch.setattr(smoke_test_pose, "save_pose_overlay", fake_save_overlay)

    summary = smoke_test_pose.run_smoke_test(
        video_path=Path("unused.mp4"),
        model_path=Path("unused.task"),
        target_fps=2.0,
        max_frames=3,
        overlay_dir=tmp_path / "overlays",
        max_overlays=1,
    )

    assert summary.sampled_frame_count == 3
    assert summary.detected_pose_count == 2
    assert summary.detected_timestamps_ms == (100, 300)
    assert saved_pairs == [(1, 1)]
    assert summary.saved_overlay_paths == (tmp_path / "overlays/1.png",)
