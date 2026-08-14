"""Inspect fixed-length pose windows and optional stroke-label alignment."""

import argparse
from pathlib import Path

from backend.pose_sequence import load_pose_sequence
from backend.pose_windows import (
    align_strokes_to_pose,
    extract_world_pose_windows,
    match_strokes_to_full_windows,
)
from backend.stroke_annotation import load_stroke_annotations


def inspect_pose_windows(
    sequence_path: Path,
    annotation_path: Path | None = None,
    window_size_frames: int = 45,
    stride_frames: int = 8,
) -> None:
    """Print a compact report for one sequence and its optional labels."""

    sequence = load_pose_sequence(sequence_path)
    windows = extract_world_pose_windows(
        sequence,
        window_size_frames=window_size_frames,
        stride_frames=stride_frames,
    )
    window_count = len(windows.start_sequence_indices)
    available_count = int(windows.world_pose_available.sum())
    sampled_count = int(windows.world_pose_available.size)

    print(f"Pose frames: {len(sequence.timestamps_ms)}")
    print(f"Windows: {window_count}")
    print(f"World XYZ shape: {windows.world_xyz.shape}")
    if sampled_count:
        print(
            "World-pose availability inside windows: "
            f"{100 * available_count / sampled_count:.1f}%"
        )

    if annotation_path is None:
        return

    annotations = load_stroke_annotations(annotation_path)
    aligned_strokes = align_strokes_to_pose(sequence, annotations)
    matches = match_strokes_to_full_windows(aligned_strokes, windows)
    for stroke_number, match in enumerate(matches, start=1):
        stroke = match.stroke
        alignment_errors = [stroke.start.error_ms, stroke.end.error_ms]
        if stroke.contact is not None:
            alignment_errors.append(stroke.contact.error_ms)
        max_error_ms = max(abs(error) for error in alignment_errors)
        matched_window = (
            "none" if match.window_index is None else str(match.window_index)
        )
        print(
            f"Stroke {stroke_number}: {stroke.stroke_type}, "
            f"max alignment error={max_error_ms} ms, "
            f"full window={matched_window}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--window-size-frames", type=int, default=45)
    parser.add_argument("--stride-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_pose_windows(
        sequence_path=args.sequence_path,
        annotation_path=args.annotations,
        window_size_frames=args.window_size_frames,
        stride_frames=args.stride_frames,
    )


if __name__ == "__main__":
    main()
