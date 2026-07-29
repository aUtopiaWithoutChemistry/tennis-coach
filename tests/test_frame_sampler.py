from pathlib import Path

import pytest

from backend.frame_sampler import sample_video_frames


def test_sample_video_frames_returns_timestamped_rgb_frames(
    sample_video: Path,
) -> None:
    frames = list(sample_video_frames(str(sample_video), target_fps=6))

    assert len(frames) == 6
    assert [frame.timestamp_ms for frame in frames] == pytest.approx(
        [0, 167, 333, 500, 667, 833],
        abs=1,
    )
    assert frames[0].source_frame_index == 0
    assert frames[0].image.width == 320
    assert frames[0].image.height == 240
    assert frames[0].image.format.name == "rgb24"


def test_sample_video_frames_zero_fps_raises(sample_video: Path) -> None:
    with pytest.raises(ValueError, match="target_fps must be a finite number greater than zero"):
        list(sample_video_frames(str(sample_video), target_fps=0))


def test_sample_video_frames_negative_fps_raises(sample_video: Path) -> None:
    with pytest.raises(ValueError, match="target_fps must be a finite number greater than zero"):
        list(sample_video_frames(str(sample_video), target_fps=-1.0))


def test_sample_video_frames_infinite_fps_raises(sample_video: Path) -> None:
    with pytest.raises(ValueError, match="target_fps must be a finite number greater than zero"):
        list(sample_video_frames(str(sample_video), target_fps=float("inf")))


def test_sample_video_frames_nan_fps_raises(sample_video: Path) -> None:
    with pytest.raises(ValueError, match="target_fps must be a finite number greater than zero"):
        list(sample_video_frames(str(sample_video), target_fps=float("nan")))


def test_sample_video_frames_audio_only_file_raises(audio_only_file: Path) -> None:
    with pytest.raises(ValueError, match="No video streams found"):
        list(sample_video_frames(str(audio_only_file), target_fps=6))