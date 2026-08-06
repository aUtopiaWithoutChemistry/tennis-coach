"""Export a directory of tennis videos into per-video pose sequences."""

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.pose_sequence import load_pose_sequence
from scripts.export_pose_sequence import (
    DEFAULT_MODEL_PATH,
    PoseSequenceExportSummary,
    export_pose_sequence,
)


@dataclass(frozen=True)
class PoseDatasetEntry:
    """Manifest entry for one exported source video."""

    video_name: str
    sequence_file: str
    frame_count: int
    image_pose_count: int
    world_pose_count: int
    image_pose_percentage: float
    world_pose_percentage: float
    first_timestamp_ms: int
    last_timestamp_ms: int
    image_xyz_shape: tuple[int, ...]
    world_xyz_shape: tuple[int, ...]


@dataclass(frozen=True)
class PoseDatasetExportSummary:
    """Aggregate result from one completed directory export."""

    manifest_path: Path
    video_count: int
    total_frame_count: int
    total_image_pose_count: int
    total_world_pose_count: int
    entries: tuple[PoseDatasetEntry, ...]


def _natural_path_key(path: Path) -> tuple[object, ...]:
    """Sort video2 before video10 while remaining deterministic."""

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _entry_from_export(
    video_path: Path,
    export_summary: PoseSequenceExportSummary,
) -> PoseDatasetEntry:
    sequence = load_pose_sequence(export_summary.output_path)
    if len(sequence.timestamps_ms) == 0:
        raise ValueError(f"Exported sequence is empty: {video_path.name}")

    frame_count = export_summary.frame_count
    return PoseDatasetEntry(
        video_name=video_path.name,
        sequence_file=export_summary.output_path.name,
        frame_count=frame_count,
        image_pose_count=export_summary.image_pose_count,
        world_pose_count=export_summary.world_pose_count,
        image_pose_percentage=100 * export_summary.image_pose_count / frame_count,
        world_pose_percentage=100 * export_summary.world_pose_count / frame_count,
        first_timestamp_ms=int(sequence.timestamps_ms[0]),
        last_timestamp_ms=int(sequence.timestamps_ms[-1]),
        image_xyz_shape=export_summary.image_xyz_shape,
        world_xyz_shape=export_summary.world_xyz_shape,
    )


def export_pose_dataset(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    target_fps: float = 30.0,
    max_frames: int = 900,
) -> PoseDatasetExportSummary:
    """Export all MP4 videos and write a manifest after every export succeeds."""

    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a finite number greater than zero")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    video_paths = tuple(sorted(input_dir.glob("*.mp4"), key=_natural_path_key))
    if not video_paths:
        raise ValueError(f"No MP4 videos found in {input_dir}")

    model_label = model_path.stem.removeprefix("pose_landmarker_")
    fps_label = f"{target_fps:g}".replace(".", "p")
    entries = []

    for video_path in video_paths:
        output_path = output_dir / (
            f"{video_path.stem}_{model_label}_{fps_label}fps.npz"
        )
        export_summary = export_pose_sequence(
            video_path=video_path,
            model_path=model_path,
            output_path=output_path,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        entries.append(_entry_from_export(video_path, export_summary))

    manifest = {
        "schema_version": 1,
        "source_dataset": input_dir.name,
        "model_name": model_path.stem,
        "target_fps": target_fps,
        "max_frames": max_frames,
        "videos": [asdict(entry) for entry in entries],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    frozen_entries = tuple(entries)
    return PoseDatasetExportSummary(
        manifest_path=manifest_path,
        video_count=len(frozen_entries),
        total_frame_count=sum(entry.frame_count for entry in frozen_entries),
        total_image_pose_count=sum(
            entry.image_pose_count for entry in frozen_entries
        ),
        total_world_pose_count=sum(
            entry.world_pose_count for entry in frozen_entries
        ),
        entries=frozen_entries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_pose_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )
    print(f"Videos: {summary.video_count}")
    print(f"Frames: {summary.total_frame_count}")
    print(f"Frames with image pose: {summary.total_image_pose_count}")
    print(f"Frames with world pose: {summary.total_world_pose_count}")
    print(f"Saved manifest: {summary.manifest_path}")


if __name__ == "__main__":
    main()
