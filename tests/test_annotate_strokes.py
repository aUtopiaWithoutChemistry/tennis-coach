from pathlib import Path

import pytest

from backend.annotation_session import AnnotationSession
from backend.stroke_annotation import StrokeAnnotation, save_stroke_annotations
from scripts.annotate_strokes import (
    frame_timestamp_ms,
    load_annotation_session,
    save_annotation_session,
)


def test_frame_timestamp_uses_frame_index_and_fps() -> None:
    assert frame_timestamp_ms(0, 30.0) == 0
    assert frame_timestamp_ms(1, 30.0) == 33
    assert frame_timestamp_ms(30, 30.0) == 1_000


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf")])
def test_frame_timestamp_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        frame_timestamp_ms(0, fps)


def test_frame_timestamp_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        frame_timestamp_ms(-1, 30.0)


def test_load_annotation_session_starts_empty(tmp_path: Path) -> None:
    session = load_annotation_session(
        Path("video1.mp4"),
        tmp_path / "video1.json",
    )

    assert session == AnnotationSession(source_video="video1.mp4")


def test_annotation_session_saves_and_resumes(tmp_path: Path) -> None:
    output_path = tmp_path / "video1.json"
    stroke = StrokeAnnotation(100, 150, 200, "forehand")
    session = AnnotationSession("video1.mp4", strokes=[stroke])

    save_annotation_session(session, output_path)
    resumed = load_annotation_session(Path("video1.mp4"), output_path)

    assert resumed.source_video == "video1.mp4"
    assert resumed.strokes == [stroke]


def test_load_annotation_session_rejects_different_video(tmp_path: Path) -> None:
    output_path = tmp_path / "labels.json"
    save_stroke_annotations(
        AnnotationSession("video1.mp4").to_annotation_file(),
        output_path,
    )

    with pytest.raises(ValueError, match="video1.mp4.*video2.mp4"):
        load_annotation_session(Path("video2.mp4"), output_path)
