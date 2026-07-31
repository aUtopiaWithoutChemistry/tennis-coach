"""Run a bounded MediaPipe pose smoke test on one local video."""

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from backend.frame_sampler import sample_video_frames
from backend.pose_estimator import create_pose_landmarker, estimate_pose_frames

DEFAULT_MODEL_PATH = Path(".models/pose_landmarker_lite.task")


@dataclass(frozen=True)
class PoseSmokeSummary:
    """Small result printed after a real-video smoke test."""

    sampled_frame_count: int
    detected_pose_count: int
    detected_timestamps_ms: tuple[int, ...]

    @property
    def detection_percentage(self) -> float:
        if self.sampled_frame_count == 0:
            return 0.0
        return 100 * self.detected_pose_count / self.sampled_frame_count


def run_smoke_test(
    video_path: Path,
    model_path: Path,
    target_fps: float,
    max_frames: int,
) -> PoseSmokeSummary:
    """Run pose inference on at most ``max_frames`` sampled video frames."""

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    sampled_frames = islice(
        sample_video_frames(str(video_path), target_fps),
        max_frames,
    )

    with create_pose_landmarker(model_path) as landmarker:
        estimated_frames = estimate_pose_frames(sampled_frames, landmarker)

        sampled_count = 0
        detected_count = 0
        detected_timestamps = []

        for frame in estimated_frames:
            sampled_count += 1
            if frame.landmarks is not None:
                detected_count += 1
                detected_timestamps.append(frame.timestamp_ms)

        return PoseSmokeSummary(
            sampled_frame_count=sampled_count,
            detected_pose_count=detected_count,
            detected_timestamps_ms=tuple(detected_timestamps),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke_test(
        video_path=args.video_path,
        model_path=args.model_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )

    print(f"Sampled frames: {summary.sampled_frame_count}")
    print(f"Frames with a pose: {summary.detected_pose_count}")
    print(f"Detection percentage: {summary.detection_percentage:.1f}%")
    print(f"Detected timestamps (ms): {summary.detected_timestamps_ms}")


if __name__ == "__main__":
    main()
