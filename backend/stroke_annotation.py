"""Represent and persist manually labeled tennis strokes."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


SCHEMA_VERSION: Final = 1
STROKE_TYPES: Final = frozenset(
    {"forehand", "backhand", "serve", "unknown"}
)


@dataclass(frozen=True)
class StrokeAnnotation:
    """One labeled stroke interval on the source video's timeline."""

    start_ms: int
    contact_ms: int | None
    end_ms: int
    stroke_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.start_ms, int) or self.start_ms < 0:
            raise ValueError("timestamp start_ms must be a nonnegative integer")
        if not isinstance(self.end_ms, int) or self.end_ms < 0:
            raise ValueError("timestamp end_ms must be a nonnegative integer")
        if self.start_ms >= self.end_ms:
            raise ValueError("timestamp start_ms must be less than timestamp end_ms")
        if self.contact_ms is not None:
            if not isinstance(self.contact_ms, int) or self.contact_ms < 0:
                raise ValueError("timestamp contact_ms must be a nonnegative integer")
            if not (self.start_ms <= self.contact_ms <= self.end_ms):
                raise ValueError(
                    "timestamp contact_ms must be between timestamp start_ms and timestamp end_ms"
                )
        if self.stroke_type not in STROKE_TYPES:
            raise ValueError(
                f"stroke_type must be one of {sorted(STROKE_TYPES)}"
            )


@dataclass(frozen=True)
class StrokeAnnotationFile:
    """All stroke labels belonging to one source video."""

    source_video: str
    strokes: tuple[StrokeAnnotation, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported annotation schema version: {self.schema_version}"
            )
        if not self.source_video.strip():
            raise ValueError("source_video must not be empty")

        for previous, current in zip(self.strokes, self.strokes[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ValueError(
                    "Strokes must be ordered by timestamp start_ms"
                )
            if current.start_ms < previous.end_ms:
                raise ValueError(
                    "Consecutive strokes must not overlap"
                )


def save_stroke_annotations(
    annotations: StrokeAnnotationFile,
    output_path: Path,
) -> Path:
    """Save one annotation file as deterministic, readable JSON."""

    payload = {
        "schema_version": annotations.schema_version,
        "source_video": annotations.source_video,
        "strokes": [asdict(stroke) for stroke in annotations.strokes],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_stroke_annotations(input_path: Path) -> StrokeAnnotationFile:
    """Load and validate one annotation JSON file."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Annotation file must contain a JSON object")

    try:
        raw_strokes = payload["strokes"]
        if not isinstance(raw_strokes, list):
            raise ValueError("strokes must be a JSON array")

        strokes = tuple(
            StrokeAnnotation(
                start_ms=stroke["start_ms"],
                contact_ms=stroke["contact_ms"],
                end_ms=stroke["end_ms"],
                stroke_type=stroke["stroke_type"],
            )
            for stroke in raw_strokes
        )
        return StrokeAnnotationFile(
            schema_version=payload["schema_version"],
            source_video=payload["source_video"],
            strokes=strokes,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid annotation file structure") from error
