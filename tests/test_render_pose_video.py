from contextlib import nullcontext
from pathlib import Path

import av
import pytest

from backend.frame_sampler import TimestampedFrame
from backend.pose_estimator import PoseFrame, PoseLandmark
from scripts import render_pose_video


def make_frame(source_frame_index: int, timestamp_ms: int) -> TimestampedFrame:
    return TimestampedFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        image=av.VideoFrame(64, 48, "rgb24"),
    )


def make_pose(source_frame_index: int, timestamp_ms: int) -> PoseFrame:
    landmark = PoseLandmark(0.5, 0.5, 0.0, 0.9, 0.9)
    return PoseFrame(
        source_frame_index=source_frame_index,
        timestamp_ms=timestamp_ms,
        landmarks=(landmark,) * 33,
    )


def test_render_pose_video_writes_bounded_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [make_frame(0, 0), make_frame(3, 100), make_frame(6, 200)]
    poses = [
        make_pose(0, 0),
        PoseFrame(3, 100, landmarks=None),
        make_pose(6, 200),
    ]

    monkeypatch.setattr(
        render_pose_video,
        "sample_video_frames",
        lambda *_: iter(frames),
    )
    monkeypatch.setattr(
        render_pose_video,
        "create_pose_landmarker",
        lambda *_: nullcontext(object()),
    )
    monkeypatch.setattr(
        render_pose_video,
        "estimate_pose_frames_with_source",
        lambda *_: iter(zip(frames, poses, strict=True)),
    )

    output_path = tmp_path / "pose-overlay.mp4"
    summary = render_pose_video.render_pose_video(
        video_path=Path("unused.mp4"),
        output_path=output_path,
        model_path=Path("unused.task"),
        target_fps=10.0,
        max_frames=3,
    )

    with av.open(output_path) as container:
        decoded_frames = list(container.decode(container.streams.video[0]))

    assert summary.output_path == output_path
    assert summary.rendered_frame_count == 3
    assert summary.detected_pose_count == 2
    assert len(decoded_frames) == 3


def test_render_pose_video_uses_diagnostic_overlay_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_frame(0, 0)
    pose = make_pose(0, 0)
    overlay_calls: list[str] = []

    monkeypatch.setattr(
        render_pose_video,
        "sample_video_frames",
        lambda *_: iter((frame,)),
    )
    monkeypatch.setattr(
        render_pose_video,
        "create_pose_landmarker",
        lambda *_: nullcontext(object()),
    )
    monkeypatch.setattr(
        render_pose_video,
        "estimate_pose_frames_with_source",
        lambda *_: iter(((frame, pose),)),
    )
    monkeypatch.setattr(
        render_pose_video,
        "draw_pose_overlay",
        lambda image, _: overlay_calls.append("standard") or image,
    )
    monkeypatch.setattr(
        render_pose_video,
        "draw_pose_diagnostic_overlay",
        lambda image, _: overlay_calls.append("diagnostic") or image,
    )

    render_pose_video.render_pose_video(
        video_path=Path("unused.mp4"),
        output_path=tmp_path / "diagnostic.mp4",
        model_path=Path("unused.task"),
        target_fps=10.0,
        max_frames=1,
        diagnostic_overlays=True,
    )

    assert overlay_calls == ["diagnostic"]
