import pytest
from pathlib import Path

from backend.video_inspector import inspect_video


def test_inspect_video_reads_expected_metadata(sample_video: Path) -> None:
    metadata = inspect_video(str(sample_video))

    # Add assertions here.
    assert metadata.source_path         == str(sample_video)
    assert metadata.source_width        == 320
    assert metadata.source_height       == 240
    assert metadata.source_fps          == 24.0
    assert metadata.source_duration_ms  == 1000
    assert metadata.frame_count         == 24
    assert metadata.codec               == "mpeg4"


def test_inspect_video_raises_for_missing_file(tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        inspect_video(str(missing_video))


def test_inspect_video_rejects_audio_only_file(audio_only_file: Path) -> None:
    with pytest.raises(ValueError, match="No video streams found"):
        inspect_video(str(audio_only_file))


