"""Run a bounded MediaPipe pose smoke test on one local video."""

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from backend.frame_sampler import TimestampedFrame, sample_video_frames
from backend.pose_estimator import (
    PoseFrame,
    create_pose_landmarker,
    estimate_pose_frames,
)
from backend.pose_overlay import draw_pose_overlay

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_lite.task")


@dataclass(frozen=True)
class PoseSmokeSummary:
    """Small result printed after a real-video smoke test."""

    sampled_frame_count: int
    detected_pose_count: int
    detected_timestamps_ms: tuple[int, ...]
    saved_overlay_paths: tuple[Path, ...]

    @property
    def detection_percentage(self) -> float:
        if self.sampled_frame_count == 0:
            return 0.0
        return 100 * self.detected_pose_count / self.sampled_frame_count


def save_pose_overlay(
    source_frame: TimestampedFrame,
    pose_frame: PoseFrame,
    output_dir: Path,
) -> Path:
    """Render and save one pose overlay using a deterministic filename."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"pose_frame_{source_frame.source_frame_index:06d}_"
        f"{source_frame.timestamp_ms:08d}ms.png"
    )
    overlay = draw_pose_overlay(source_frame.image.to_image(), pose_frame)
    overlay.save(output_path)
    return output_path


def run_smoke_test(
    video_path: Path,
    model_path: Path,
    target_fps: float,
    max_frames: int,
    overlay_dir: Path | None = None,
    max_overlays: int = 5,
) -> PoseSmokeSummary:
    """Run pose inference on at most ``max_frames`` sampled video frames."""

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if max_overlays < 0:
        raise ValueError("max_overlays must not be negative")

    sampled_frames = list(
        islice(
            sample_video_frames(str(video_path), target_fps),
            max_frames,
        )
    )

    with create_pose_landmarker(model_path) as landmarker:
        estimated_frames = estimate_pose_frames(sampled_frames, landmarker)

        sampled_count = 0
        detected_count = 0
        detected_timestamps = []
        saved_overlay_paths = []

        paired_frames = zip(sampled_frames, estimated_frames, strict=True)

        for sampled_frame, estimated_frame in paired_frames:
            sampled_count += 1
            if estimated_frame.landmarks is not None:
                detected_count += 1
                detected_timestamps.append(estimated_frame.timestamp_ms)

                if overlay_dir is not None and len(saved_overlay_paths) < max_overlays:
                    output_path = save_pose_overlay(
                        sampled_frame, estimated_frame, overlay_dir
                    )
                    saved_overlay_paths.append(output_path)

        return PoseSmokeSummary(
            sampled_frame_count=sampled_count,
            detected_pose_count=detected_count,
            detected_timestamps_ms=tuple(detected_timestamps),
            saved_overlay_paths=tuple(saved_overlay_paths),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--max-overlays", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke_test(
        video_path=args.video_path,
        model_path=args.model_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        overlay_dir=args.overlay_dir,
        max_overlays=args.max_overlays,
    )

    print(f"Sampled frames: {summary.sampled_frame_count}")
    print(f"Frames with a pose: {summary.detected_pose_count}")
    print(f"Detection percentage: {summary.detection_percentage:.1f}%")
    print(f"Detected timestamps (ms): {summary.detected_timestamps_ms}")
    for output_path in summary.saved_overlay_paths:
        print(f"Saved overlay: {output_path}")


if __name__ == "__main__":
    main()
