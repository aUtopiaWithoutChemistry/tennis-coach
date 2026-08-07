import json
from pathlib import Path

import pytest

from backend.stroke_annotation import (
    StrokeAnnotation,
    StrokeAnnotationFile,
    load_stroke_annotations,
    save_stroke_annotations,
)


def make_stroke(
    start_ms: int = 100,
    contact_ms: int | None = 150,
    end_ms: int = 200,
    stroke_type: str = "forehand",
) -> StrokeAnnotation:
    return StrokeAnnotation(
        start_ms=start_ms,
        contact_ms=contact_ms,
        end_ms=end_ms,
        stroke_type=stroke_type,
    )


def test_stroke_annotations_round_trip_through_json(tmp_path: Path) -> None:
    annotations = StrokeAnnotationFile(
        source_video="video1.mp4",
        strokes=(
            make_stroke(),
            make_stroke(300, None, 450, "backhand"),
        ),
    )
    output_path = tmp_path / "nested" / "video1.json"

    result = save_stroke_annotations(annotations, output_path)

    assert result == output_path
    assert load_stroke_annotations(output_path) == annotations
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "source_video": "video1.mp4",
        "strokes": [
            {
                "start_ms": 100,
                "contact_ms": 150,
                "end_ms": 200,
                "stroke_type": "forehand",
            },
            {
                "start_ms": 300,
                "contact_ms": None,
                "end_ms": 450,
                "stroke_type": "backhand",
            },
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_ms", -1),
        ("contact_ms", -1),
        ("end_ms", -1),
        ("start_ms", 1.5),
        ("contact_ms", 1.5),
        ("end_ms", 1.5),
    ],
)
def test_stroke_rejects_invalid_timestamp(field: str, value: object) -> None:
    values = {
        "start_ms": 100,
        "contact_ms": 150,
        "end_ms": 200,
        "stroke_type": "forehand",
    }
    values[field] = value

    with pytest.raises(ValueError, match="timestamp"):
        StrokeAnnotation(**values)


@pytest.mark.parametrize(("start_ms", "end_ms"), [(200, 200), (201, 200)])
def test_stroke_requires_start_before_end(start_ms: int, end_ms: int) -> None:
    with pytest.raises(ValueError, match="start_ms"):
        make_stroke(start_ms=start_ms, contact_ms=None, end_ms=end_ms)


@pytest.mark.parametrize("contact_ms", [99, 201])
def test_stroke_rejects_contact_outside_interval(contact_ms: int) -> None:
    with pytest.raises(ValueError, match="contact_ms"):
        make_stroke(contact_ms=contact_ms)


def test_stroke_allows_contact_at_interval_boundary() -> None:
    assert make_stroke(contact_ms=100).contact_ms == 100
    assert make_stroke(contact_ms=200).contact_ms == 200


def test_stroke_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="stroke_type"):
        make_stroke(stroke_type="drop_shot")


def test_annotation_file_rejects_unsorted_strokes() -> None:
    with pytest.raises(ValueError, match="ordered"):
        StrokeAnnotationFile(
            source_video="video1.mp4",
            strokes=(make_stroke(300, 350, 400), make_stroke()),
        )


def test_annotation_file_rejects_overlapping_strokes() -> None:
    with pytest.raises(ValueError, match="overlap"):
        StrokeAnnotationFile(
            source_video="video1.mp4",
            strokes=(
                make_stroke(100, 150, 300),
                make_stroke(250, 300, 400),
            ),
        )


def test_annotation_file_allows_touching_intervals() -> None:
    annotations = StrokeAnnotationFile(
        source_video="video1.mp4",
        strokes=(
            make_stroke(100, 150, 200),
            make_stroke(200, 250, 300),
        ),
    )

    assert len(annotations.strokes) == 2


def test_annotation_file_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="source_video"):
        StrokeAnnotationFile(source_video="  ", strokes=())

    with pytest.raises(ValueError, match="schema version"):
        StrokeAnnotationFile(
            source_video="video1.mp4",
            strokes=(),
            schema_version=2,
        )


def test_load_rejects_invalid_structure(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_stroke_annotations(input_path)
