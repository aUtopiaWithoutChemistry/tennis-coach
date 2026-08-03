"""Render a bounded, silent pose-overlay video from one local tennis clip."""

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, islice
from pathlib import Path

import av

from backend.frame_sampler import sample_video_frames
from backend.pose_estimator import (
    create_pose_landmarker,
    estimate_pose_frames_with_source,
)
from backend.pose_overlay import draw_pose_diagnostic_overlay, draw_pose_overlay

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_lite.task")


@dataclass(frozen=True)
class PoseVideoSummary:
    output_path: Path
    rendered_frame_count: int
    detected_pose_count: int


def render_pose_video(
    video_path: Path,
    output_path: Path,
    model_path: Path,
    target_fps: float = 10.0,
    max_frames: int = 100,
    diagnostic_overlays: bool = False,
) -> PoseVideoSummary:
    """Stream sampled pose overlays into a constant-frame-rate H.264 MP4."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    sampled_frames = islice(
        sample_video_frames(str(video_path), target_fps),
        max_frames,
    )
    output_rate = Fraction(str(target_fps)).limit_denominator(1_000)

    if diagnostic_overlays:
        draw_overlay = draw_pose_diagnostic_overlay
    else:
        draw_overlay = draw_pose_overlay

    with create_pose_landmarker(model_path) as landmarker:
        paired_frames = estimate_pose_frames_with_source(
            sampled_frames,
            landmarker,
        )
        first_pair = next(paired_frames, None)
        if first_pair is None:
            raise ValueError("No sampled frames were available to render")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_count = 0
        detected_count = 0

        with av.open(output_path, mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=output_rate)
            output_stream.width = first_pair[0].image.width
            output_stream.height = first_pair[0].image.height
            output_stream.pix_fmt = "yuv420p"

            for source_frame, pose_frame in chain((first_pair,), paired_frames):
                overlay = draw_overlay(
                    source_frame.image.to_image(),
                    pose_frame,
                )
                output_frame = av.VideoFrame.from_image(overlay)
                output_frame.pts = rendered_count
                output_frame.time_base = Fraction(
                    output_rate.denominator,
                    output_rate.numerator,
                )

                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)

                rendered_count += 1
                if pose_frame.landmarks is not None:
                    detected_count += 1

            for packet in output_stream.encode():
                output_container.mux(packet)

    return PoseVideoSummary(
        output_path=output_path,
        rendered_frame_count=rendered_count,
        detected_pose_count=detected_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--diagnostic-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_pose_video(
        video_path=args.video_path,
        output_path=args.output_path,
        model_path=args.model_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        diagnostic_overlays=args.diagnostic_overlays,
    )
    print(f"Rendered frames: {summary.rendered_frame_count}")
    print(f"Frames with a pose: {summary.detected_pose_count}")
    print(f"Saved video: {summary.output_path}")


if __name__ == "__main__":
    main()
