"""Manage the editable state of one stroke-annotation session."""

from dataclasses import dataclass, field
from typing import Final

from backend.stroke_annotation import StrokeAnnotation, StrokeAnnotationFile


MARK_NAMES: Final = frozenset({"start_ms", "contact_ms", "end_ms"})


@dataclass
class AnnotationSession:
    """Mutable marks and saved strokes for one video annotation session."""

    source_video: str
    strokes: list[StrokeAnnotation] = field(default_factory=list)
    start_ms: int | None = None
    contact_ms: int | None = None
    end_ms: int | None = None

    @classmethod
    def from_annotation_file(
        cls,
        annotations: StrokeAnnotationFile,
    ) -> "AnnotationSession":
        """Start an editable session from previously validated labels."""

        return cls(
            source_video=annotations.source_video,
            strokes=list(annotations.strokes),
        )

    def set_mark(self, mark_name: str, timestamp_ms: int) -> None:
        """Set or replace one pending timestamp."""

        if mark_name not in MARK_NAMES:
            raise ValueError(f"Unknown mark name: {mark_name}")
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError(f"Invalid timestamp: {timestamp_ms}")

        setattr(self, mark_name, timestamp_ms)

    def complete_stroke(self, stroke_type: str) -> StrokeAnnotation:
        """Validate, chronologically insert, and return the pending stroke."""

        if self.start_ms is None or self.end_ms is None:
            raise ValueError("Both start_ms and end_ms must be set")
        if type(self.start_ms) is not int or self.start_ms < 0:
            raise ValueError(f"Invalid start_ms: {self.start_ms}")
        if type(self.end_ms) is not int or self.end_ms < 0:
            raise ValueError(f"Invalid end_ms: {self.end_ms}")
        if self.contact_ms is not None and (type(self.contact_ms) is not int or self.contact_ms < 0):
            raise ValueError(f"Invalid contact_ms: {self.contact_ms}")

        stroke = StrokeAnnotation(
            stroke_type=stroke_type,
            start_ms=self.start_ms,
            contact_ms=self.contact_ms,
            end_ms=self.end_ms,
        )

        candidate_strokes = sorted(self.strokes + [stroke], key=lambda s: s.start_ms)

        StrokeAnnotationFile(
            source_video=self.source_video,
            strokes=tuple(candidate_strokes),
        )

        self.strokes = candidate_strokes
        self.clear_pending()

        return stroke

    def undo_last(self) -> StrokeAnnotation | None:
        """Remove and return the chronologically last saved stroke, if any."""

        if not self.strokes:
            return None

        return self.strokes.pop()

    def clear_pending(self) -> None:
        """Clear the current unsaved marks."""

        self.start_ms = None
        self.contact_ms = None
        self.end_ms = None

    def to_annotation_file(self) -> StrokeAnnotationFile:
        """Create the validated immutable form used for JSON persistence."""

        return StrokeAnnotationFile(
            source_video=self.source_video,
            strokes=tuple(self.strokes),
        )
