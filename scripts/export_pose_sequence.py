"""Export one bounded video as an ML-ready compressed pose sequence."""

import argparse
import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from backend.frame_sampler import sample_video_frames
from backend.pose_estimator import create_pose_landmarker, estimate_pose_frames
from backend.pose_sequence import build_pose_sequence, save_pose_sequence

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_full.task")


@dataclass(frozen=True)
class PoseSequenceExportSummary:
    """Small verification summary for one exported video."""

    output_path: Path
    frame_count: int
    image_pose_count: int
    world_pose_count: int
    image_xyz_shape: tuple[int, ...]
    world_xyz_shape: tuple[int, ...]


def export_pose_sequence(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    target_fps: float = 30.0,
    max_frames: int = 900,
) -> PoseSequenceExportSummary:
    """Run bounded pose inference and save one numerical sequence."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    sampled_frames = islice(
        sample_video_frames(str(video_path), target_fps),
        max_frames,
    )

    with create_pose_landmarker(model_path) as landmarker:
        bounded_pose_frames = tuple(
            estimate_pose_frames(sampled_frames, landmarker)
        )

    sequence = build_pose_sequence(bounded_pose_frames)
    saved_path = save_pose_sequence(sequence, output_path)

    return PoseSequenceExportSummary(
        output_path=saved_path,
        frame_count=len(sequence.timestamps_ms),
        image_pose_count=int(sequence.image_pose_available.sum()),
        world_pose_count=int(sequence.world_pose_available.sum()),
        image_xyz_shape=tuple(sequence.image_xyz.shape),
        world_xyz_shape=tuple(sequence.world_xyz.shape),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_pose_sequence(
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )
    print(f"Frames: {summary.frame_count}")
    print(f"Frames with image pose: {summary.image_pose_count}")
    print(f"Frames with world pose: {summary.world_pose_count}")
    print(f"Image XYZ shape: {summary.image_xyz_shape}")
    print(f"World XYZ shape: {summary.world_xyz_shape}")
    print(f"Saved sequence: {summary.output_path}")


if __name__ == "__main__":
    main()
