"""Benchmark pose inference on one local video and save automated metrics."""

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path

from backend.frame_sampler import sample_video_frames
from backend.pose_benchmark import (
    PoseBenchmarkSummary,
    build_pose_benchmark_summary,
)
from backend.pose_estimator import create_pose_landmarker, estimate_pose_frames

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_lite.task")


def run_pose_benchmark(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    target_fps: float = 30.0,
    max_frames: int = 900,
    confidence_threshold: float = 0.5,
) -> PoseBenchmarkSummary:
    """Run a bounded benchmark and write its summary as JSON."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    sampled_frames = islice(
        sample_video_frames(str(video_path), target_fps),
        max_frames,
    )

    start_time = time.perf_counter()
    with create_pose_landmarker(model_path) as landmarker:
        pose_frames = tuple(estimate_pose_frames(sampled_frames, landmarker))
    processing_seconds = time.perf_counter() - start_time

    summary = build_pose_benchmark_summary(
        video_name=video_path.name,
        model_name=model_path.stem,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold,
        pose_frames=pose_frames,
        processing_seconds=processing_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pose_benchmark(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"Sampled frames: {summary.sampled_frame_count}")
    print(f"Frames with a pose: {summary.detected_pose_count}")
    print(f"Usable frames: {summary.usable_frame_count}")
    print(f"Required-joint coverage: {summary.required_joint_coverage_percentage:.1f}%")
    print(f"Processed FPS: {summary.processed_fps:.1f}")
    print(f"Processing/video ratio: {summary.processing_time_ratio:.2f}")
    print(f"Saved benchmark: {args.output_path}")


if __name__ == "__main__":
    main()
