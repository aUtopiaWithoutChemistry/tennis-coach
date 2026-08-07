import pytest

from backend.annotation_session import AnnotationSession
from backend.stroke_annotation import StrokeAnnotation, StrokeAnnotationFile


def make_stroke(
    start_ms: int,
    end_ms: int,
    stroke_type: str = "forehand",
) -> StrokeAnnotation:
    return StrokeAnnotation(start_ms, None, end_ms, stroke_type)


def test_session_starts_from_existing_annotations() -> None:
    annotations = StrokeAnnotationFile(
        source_video="video1.mp4",
        strokes=(make_stroke(100, 200),),
    )

    session = AnnotationSession.from_annotation_file(annotations)

    assert session.source_video == "video1.mp4"
    assert session.strokes == [make_stroke(100, 200)]
    assert session.start_ms is None


def test_session_sets_and_replaces_pending_marks() -> None:
    session = AnnotationSession("video1.mp4")

    session.set_mark("start_ms", 100)
    session.set_mark("contact_ms", 150)
    session.set_mark("end_ms", 200)
    session.set_mark("contact_ms", 160)

    assert session.start_ms == 100
    assert session.contact_ms == 160
    assert session.end_ms == 200


@pytest.mark.parametrize(
    ("mark_name", "timestamp_ms"),
    [("invalid", 100), ("start_ms", -1), ("start_ms", 1.5)],
)
def test_session_rejects_invalid_mark(mark_name: str, timestamp_ms: object) -> None:
    session = AnnotationSession("video1.mp4")

    with pytest.raises(ValueError):
        session.set_mark(mark_name, timestamp_ms)


def test_session_requires_start_and_end_before_completion() -> None:
    session = AnnotationSession("video1.mp4")
    session.set_mark("start_ms", 100)

    with pytest.raises(ValueError, match="start.*end"):
        session.complete_stroke("forehand")


def test_session_completes_sorts_and_clears_pending_stroke() -> None:
    session = AnnotationSession(
        "video1.mp4",
        strokes=[make_stroke(300, 400, "backhand")],
    )
    session.set_mark("start_ms", 100)
    session.set_mark("contact_ms", 150)
    session.set_mark("end_ms", 200)

    result = session.complete_stroke("forehand")

    assert result == StrokeAnnotation(100, 150, 200, "forehand")
    assert session.strokes == [result, make_stroke(300, 400, "backhand")]
    assert session.start_ms is None
    assert session.contact_ms is None
    assert session.end_ms is None


def test_failed_completion_does_not_mutate_session() -> None:
    original = make_stroke(100, 300)
    session = AnnotationSession("video1.mp4", strokes=[original])
    session.set_mark("start_ms", 200)
    session.set_mark("end_ms", 400)

    with pytest.raises(ValueError, match="overlap"):
        session.complete_stroke("backhand")

    assert session.strokes == [original]
    assert session.start_ms == 200
    assert session.end_ms == 400


def test_session_undoes_last_stroke() -> None:
    first = make_stroke(100, 200)
    second = make_stroke(300, 400, "backhand")
    session = AnnotationSession("video1.mp4", strokes=[first, second])

    assert session.undo_last() == second
    assert session.strokes == [first]
    assert session.undo_last() == first
    assert session.undo_last() is None


def test_session_converts_to_immutable_annotation_file() -> None:
    stroke = make_stroke(100, 200)
    session = AnnotationSession("video1.mp4", strokes=[stroke])

    assert session.to_annotation_file() == StrokeAnnotationFile(
        source_video="video1.mp4",
        strokes=(stroke,),
    )
