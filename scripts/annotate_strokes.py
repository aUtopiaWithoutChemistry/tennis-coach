"""Interactively label stroke intervals in one local tennis video."""

import argparse
import math
from pathlib import Path

import cv2

from backend.annotation_session import AnnotationSession
from backend.stroke_annotation import (
    load_stroke_annotations,
    save_stroke_annotations,
)


WINDOW_NAME = "Tennis Coach - Stroke Annotator"
LABEL_KEYS = {
    ord("f"): "forehand",
    ord("b"): "backhand",
    ord("s"): "serve",
    ord("u"): "unknown",
}
MARK_KEYS = {
    ord("i"): "start_ms",
    ord("c"): "contact_ms",
    ord("o"): "end_ms",
}


def frame_timestamp_ms(frame_index: int, fps: float) -> int:
    """Convert a zero-based constant-rate frame index to milliseconds."""

    if frame_index < 0:
        raise ValueError("frame_index must be nonnegative")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite number greater than zero")
    return round(1_000 * frame_index / fps)


def load_annotation_session(
    video_path: Path,
    output_path: Path,
) -> AnnotationSession:
    """Load existing labels or create an empty session for the video."""

    if not output_path.exists():
        return AnnotationSession(source_video=video_path.name)

    annotations = load_stroke_annotations(output_path)
    if annotations.source_video != video_path.name:
        raise ValueError(
            "Existing annotations belong to "
            f"{annotations.source_video}, not {video_path.name}"
        )
    return AnnotationSession.from_annotation_file(annotations)


def save_annotation_session(
    session: AnnotationSession,
    output_path: Path,
) -> Path:
    """Validate and save the session's current completed strokes."""

    return save_stroke_annotations(session.to_annotation_file(), output_path)


def _format_timestamp(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "--"
    minutes, remaining_ms = divmod(timestamp_ms, 60_000)
    seconds, milliseconds = divmod(remaining_ms, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _draw_status(
    frame,
    session: AnnotationSession,
    frame_index: int,
    timestamp_ms: int,
    playing: bool,
    message: str,
):
    """Draw annotation state and keyboard help without changing the source."""

    overlay = frame.copy()
    mode = "PLAYING" if playing else "PAUSED"
    lines = (
        f"{mode} | frame {frame_index} | {_format_timestamp(timestamp_ms)} "
        f"| saved strokes {len(session.strokes)}",
        "pending: "
        f"start={_format_timestamp(session.start_ms)}  "
        f"contact={_format_timestamp(session.contact_ms)}  "
        f"end={_format_timestamp(session.end_ms)}",
        "i=start  c=contact  o=end  |  f=forehand  b=backhand  "
        "s=serve  u=unknown",
        "space=play/pause  ,/.=one frame  j/l=one second  "
        "z=undo  q=save+quit",
        message,
    )
    line_height = 28
    box_height = 16 + line_height * len(lines)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], box_height), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        color = (0, 220, 255) if line_index == len(lines) - 1 else (255, 255, 255)
        cv2.putText(
            overlay,
            line,
            (12, 25 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def annotate_video(video_path: Path, output_path: Path) -> Path:
    """Open the interactive annotator and persist its completed labels."""

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Video must report a valid frame rate and frame count")

    session = load_annotation_session(video_path, output_path)
    frame_index = 0
    playing = False
    message = "Ready. Mark start, optional contact, and end."
    frame_delay_ms = max(1, round(1_000 / fps))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not decode frame {frame_index}")

            timestamp_ms = frame_timestamp_ms(frame_index, fps)
            display_frame = _draw_status(
                frame,
                session,
                frame_index,
                timestamp_ms,
                playing,
                message,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(frame_delay_ms if playing else 0)

            if key == -1:
                if playing and frame_index < frame_count - 1:
                    frame_index += 1
                elif playing:
                    playing = False
                    message = "Reached the end of the video."
                continue

            key &= 0xFF
            if key == ord("q"):
                save_annotation_session(session, output_path)
                break
            if key == ord(" "):
                playing = not playing
                message = "Playback resumed." if playing else "Playback paused."
                continue
            if key in (ord(","), ord("."), ord("j"), ord("l")):
                playing = False
                if key == ord(","):
                    offset = -1
                elif key == ord("."):
                    offset = 1
                elif key == ord("j"):
                    offset = -round(fps)
                else:
                    offset = round(fps)
                frame_index = min(max(frame_index + offset, 0), frame_count - 1)
                message = "Moved on the video timeline."
                continue

            try:
                if key in MARK_KEYS:
                    mark_name = MARK_KEYS[key]
                    session.set_mark(mark_name, timestamp_ms)
                    message = f"Set {mark_name} to {_format_timestamp(timestamp_ms)}."
                elif key in LABEL_KEYS:
                    stroke = session.complete_stroke(LABEL_KEYS[key])
                    save_annotation_session(session, output_path)
                    message = (
                        f"Saved {stroke.stroke_type} from "
                        f"{_format_timestamp(stroke.start_ms)} to "
                        f"{_format_timestamp(stroke.end_ms)}."
                    )
                elif key == ord("z"):
                    removed = session.undo_last()
                    save_annotation_session(session, output_path)
                    message = (
                        "No saved stroke to undo."
                        if removed is None
                        else f"Removed {removed.stroke_type} annotation."
                    )
            except ValueError as error:
                message = f"Cannot save annotation: {error}"
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = annotate_video(args.video_path, args.output_path)
    print(f"Saved annotations: {output_path}")


if __name__ == "__main__":
    main()
