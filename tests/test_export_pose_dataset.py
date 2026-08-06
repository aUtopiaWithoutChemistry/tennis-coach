import json
from pathlib import Path

import numpy as np
import pytest

from backend.pose_sequence import PoseSequence
from scripts import export_pose_dataset
from scripts.export_pose_sequence import PoseSequenceExportSummary


def make_sequence(frame_count: int, pose_count: int) -> PoseSequence:
    timestamps_ms = np.arange(frame_count, dtype=np.int64) * 100
    image_available = np.zeros(frame_count, dtype=np.bool_)
    image_available[:pose_count] = True
    return PoseSequence(
        source_frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_ms=timestamps_ms,
        image_xyz=np.full((frame_count, 33, 3), np.nan, dtype=np.float32),
        image_visibility=np.full((frame_count, 33), np.nan, dtype=np.float32),
        image_presence=np.full((frame_count, 33), np.nan, dtype=np.float32),
        world_xyz=np.full((frame_count, 33, 3), np.nan, dtype=np.float32),
        world_visibility=np.full((frame_count, 33), np.nan, dtype=np.float32),
        world_presence=np.full((frame_count, 33), np.nan, dtype=np.float32),
        image_pose_available=image_available,
        world_pose_available=image_available.copy(),
    )


def test_export_pose_dataset_sorts_videos_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "tenis-backview"
    output_dir = tmp_path / "sequences"
    input_dir.mkdir()
    (input_dir / "video10.mp4").touch()
    (input_dir / "video2.mp4").touch()
    calls: list[tuple[Path, Path]] = []
    sequences: dict[Path, PoseSequence] = {}

    def fake_export(
        *,
        video_path: Path,
        model_path: Path,
        output_path: Path,
        target_fps: float,
        max_frames: int,
    ) -> PoseSequenceExportSummary:
        calls.append((video_path, output_path))
        frame_count = 2 if video_path.stem == "video2" else 3
        pose_count = frame_count - 1
        sequences[output_path] = make_sequence(frame_count, pose_count)
        return PoseSequenceExportSummary(
            output_path=output_path,
            frame_count=frame_count,
            image_pose_count=pose_count,
            world_pose_count=pose_count,
            image_xyz_shape=(frame_count, 33, 3),
            world_xyz_shape=(frame_count, 33, 3),
        )

    monkeypatch.setattr(export_pose_dataset, "export_pose_sequence", fake_export)
    monkeypatch.setattr(
        export_pose_dataset,
        "load_pose_sequence",
        lambda path: sequences[path],
    )

    summary = export_pose_dataset.export_pose_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=Path("pose_landmarker_full.task"),
        target_fps=30.0,
        max_frames=900,
    )

    assert [call[0].name for call in calls] == ["video2.mp4", "video10.mp4"]
    assert [call[1].name for call in calls] == [
        "video2_full_30fps.npz",
        "video10_full_30fps.npz",
    ]
    assert summary.video_count == 2
    assert summary.total_frame_count == 5
    assert summary.total_image_pose_count == 3
    assert summary.total_world_pose_count == 3
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_dataset"] == "tenis-backview"
    assert manifest["model_name"] == "pose_landmarker_full"
    assert manifest["target_fps"] == 30.0
    assert [video["video_name"] for video in manifest["videos"]] == [
        "video2.mp4",
        "video10.mp4",
    ]
    assert manifest["videos"][0]["image_pose_percentage"] == 50.0
    assert manifest["videos"][1]["last_timestamp_ms"] == 200


def test_export_pose_dataset_rejects_missing_or_empty_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input directory"):
        export_pose_dataset.export_pose_dataset(
            input_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            model_path=Path("model.task"),
        )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No MP4 videos"):
        export_pose_dataset.export_pose_dataset(
            input_dir=empty_dir,
            output_dir=tmp_path / "output",
            model_path=Path("model.task"),
        )


@pytest.mark.parametrize("target_fps", [0.0, -1.0, float("inf")])
def test_export_pose_dataset_rejects_invalid_target_fps(
    tmp_path: Path,
    target_fps: float,
) -> None:
    with pytest.raises(ValueError, match="target_fps"):
        export_pose_dataset.export_pose_dataset(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            model_path=Path("model.task"),
            target_fps=target_fps,
        )


def test_export_pose_dataset_rejects_invalid_max_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_frames"):
        export_pose_dataset.export_pose_dataset(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
            model_path=Path("model.task"),
            max_frames=0,
        )


def test_export_pose_dataset_does_not_write_manifest_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "video1.mp4").touch()

    def fail_export(**kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(export_pose_dataset, "export_pose_sequence", fail_export)

    with pytest.raises(RuntimeError, match="inference failed"):
        export_pose_dataset.export_pose_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=Path("model.task"),
        )

    assert not (output_dir / "manifest.json").exists()
