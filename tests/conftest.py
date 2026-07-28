from fractions import Fraction
from pathlib import Path

import av
import pytest


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a one-second, 320x240, 24 FPS video for tests."""

    path = tmp_path / "sample.mp4"
    container = av.open(path, mode="w")
    stream = container.add_stream("mpeg4", rate=Fraction(24, 1))
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"

    for _ in range(24):
        frame = av.VideoFrame(320, 240, "yuv420p")
        container.mux(stream.encode(frame))

    container.mux(stream.encode())
    container.close()
    return path


@pytest.fixture
def audio_only_file(tmp_path: Path) -> Path:
    """Create a valid media file containing audio but no video stream."""

    path = tmp_path / "audio-only.wav"
    container = av.open(path, mode="w")
    stream = container.add_stream("pcm_s16le", rate=48_000)
    frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
    frame.sample_rate = 48_000

    container.mux(stream.encode(frame))
    container.mux(stream.encode())
    container.close()
    return path
