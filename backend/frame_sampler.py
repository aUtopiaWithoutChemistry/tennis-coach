"""Decode timestamped frames from a video at a requested sampling rate."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import av


@dataclass(frozen=True)
class TimestampedFrame:
    """One decoded RGB frame and its position in the source video."""

    source_frame_index: int
    timestamp_ms: int
    image: av.VideoFrame


def sample_video_frames(
    video_path: str,
    target_fps: float,
) -> Iterator[TimestampedFrame]:
    """Yield RGB frames at approximately ``target_fps`` with timestamps."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")

    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")

        stream = container.streams.video[0]
        sample_interval_ms = 1_000 / target_fps
        next_sample_ms = 0.0

        for source_frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded frame does not contain timestamp information.")
            timestamp_ms = frame.pts * frame.time_base * 1_000

            if timestamp_ms < next_sample_ms:
                continue

            rgb_frame = frame.reformat(format="rgb24")

            yield TimestampedFrame(
                source_frame_index=source_frame_index,
                timestamp_ms=round(timestamp_ms),
                image=rgb_frame,
            )

            while next_sample_ms <= timestamp_ms:
                next_sample_ms += sample_interval_ms
