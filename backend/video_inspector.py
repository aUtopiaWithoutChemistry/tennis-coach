import av
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    source_path: str
    source_width: int
    source_height: int
    source_fps: float | None
    source_duration_ms: int | None
    frame_count: int | None
    codec: str | None

def inspect_video(video_path: str) -> VideoMetadata:
    """
    Inspect a video file and return its metadata.

    Args:
        video_path (str): The path to the video file.

    Returns:
        VideoMetadata: A dataclass containing the video's metadata.
    """
    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {video_path}")

        stream = container.streams.video[0]
        frame_count = stream.frames or None
        fps = (
            float(stream.average_rate) 
            if stream.average_rate is not None
            else None
        )

        first_frame = next(container.decode(stream), None)
        if first_frame is None:
            raise ValueError("Video does not contain a decodable frame.")

        width = first_frame.width
        height = first_frame.height
        codec = stream.codec_context.name or None

        if stream.duration is not None and stream.time_base is not None:
            duration_ms = round(stream.duration * stream.time_base * 1000)
        elif container.duration is not None:
            duration_ms = container.duration // 1000
        else:
            duration_ms = None

        return VideoMetadata(
            source_path=video_path,
            source_width=width,
            source_height=height,
            source_fps=fps,
            source_duration_ms=duration_ms,
            frame_count=frame_count,
            codec=codec,
        )