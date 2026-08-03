"""Render project pose landmarks on copies of source images."""

import mediapipe as mp
from PIL import Image, ImageDraw

from backend.pose_estimator import PoseFrame, PoseLandmark

POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
)
FIRST_BODY_LANDMARK_INDEX = 11
DIAGNOSTIC_USABLE_COLOR = (0, 255, 0)
DIAGNOSTIC_LOW_CONFIDENCE_COLOR = (255, 165, 0)
DIAGNOSTIC_LANDMARKS = (
    ("left elbow", 13),
    ("right elbow", 14),
    ("left wrist", 15),
    ("right wrist", 16),
)


def _coordinate_to_pixel(
    landmark: PoseLandmark,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    """Convert in-bounds normalized coordinates without filtering confidence."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if landmark.x < 0.0 or landmark.x > 1.0 or landmark.y < 0.0 or landmark.y > 1.0:
        return None
    return (
        round(landmark.x * (image_width - 1)),
        round(landmark.y * (image_height - 1)),
    )


def landmark_to_pixel(
    landmark: PoseLandmark,
    image_width: int,
    image_height: int,
    confidence_threshold: float = 0.5,
) -> tuple[int, int] | None:
    """Convert one usable normalized landmark to an image pixel coordinate."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    if landmark.x < 0.0 or landmark.x > 1.0 or landmark.y < 0.0 or landmark.y > 1.0:
        return None

    if (
        (landmark.visibility is not None and landmark.visibility < confidence_threshold)
        or (landmark.presence is not None and landmark.presence < confidence_threshold)
    ):
        return None

    pixel_x = round(landmark.x * (image_width - 1))
    pixel_y = round(landmark.y * (image_height - 1))
    return (pixel_x, pixel_y)


def draw_pose_overlay(
    image: Image.Image,
    pose_frame: PoseFrame,
    confidence_threshold: float = 0.5,
) -> Image.Image:
    """Return an RGB copy with visible pose landmarks and connections drawn."""

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    if pose_frame.landmarks is None:
        draw.text(
            (8, 8),
            f"{pose_frame.timestamp_ms} ms | no pose",
            fill=(255, 255, 0),
        )
        return overlay

    points = tuple(
        landmark_to_pixel(
            landmark,
            overlay.width,
            overlay.height,
            confidence_threshold,
        )
        for landmark in pose_frame.landmarks
    )

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index < FIRST_BODY_LANDMARK_INDEX or end_index < FIRST_BODY_LANDMARK_INDEX:
            continue
        start = points[start_index]
        end = points[end_index]
        if start is not None and end is not None:
            draw.line((start, end), fill=(0, 255, 0), width=3)

    joint_radius = 4
    for landmark_index, point in enumerate(points):
        if landmark_index < FIRST_BODY_LANDMARK_INDEX:
            continue
        if point is None:
            continue
        x, y = point
        draw.ellipse(
            (
                x - joint_radius,
                y - joint_radius,
                x + joint_radius,
                y + joint_radius,
            ),
            fill=(255, 64, 64),
        )

    draw.text(
        (8, 8),
        f"{pose_frame.timestamp_ms} ms",
        fill=(255, 255, 0),
    )
    return overlay


def draw_pose_diagnostic_overlay(
    image: Image.Image,
    pose_frame: PoseFrame,
    confidence_threshold: float = 0.5,
) -> Image.Image:
    """Draw confident and low-confidence body landmarks in different colors."""

    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    if pose_frame.landmarks is None:
        draw.text(
            (8, 8),
            f"{pose_frame.timestamp_ms} ms | no pose",
            fill=(255, 255, 0),
        )
        return overlay

    raw_points = tuple(
        _coordinate_to_pixel(landmark, overlay.width, overlay.height)
        for landmark in pose_frame.landmarks
    )
    usable = tuple(
        point is not None
        and landmark.visibility is not None
        and landmark.visibility >= confidence_threshold
        and landmark.presence is not None
        and landmark.presence >= confidence_threshold
        for landmark, point in zip(pose_frame.landmarks, raw_points, strict=True)
    )

    for start_index, end_index in POSE_CONNECTIONS:
        if start_index < FIRST_BODY_LANDMARK_INDEX or end_index < FIRST_BODY_LANDMARK_INDEX:
            continue
        start = raw_points[start_index]
        end = raw_points[end_index]
        if start is not None and end is not None:
            draw.line((start, end), fill=(128, 128, 128), width=2)

    joint_radius = 4
    for landmark_index, point in enumerate(raw_points):
        if landmark_index < FIRST_BODY_LANDMARK_INDEX or point is None:
            continue
        color = (
            DIAGNOSTIC_USABLE_COLOR
            if usable[landmark_index]
            else DIAGNOSTIC_LOW_CONFIDENCE_COLOR
        )
        x, y = point
        draw.ellipse(
            (
                x - joint_radius,
                y - joint_radius,
                x + joint_radius,
                y + joint_radius,
            ),
            fill=color,
        )

    draw.text(
        (8, 8),
        f"{pose_frame.timestamp_ms} ms | green=usable orange=low confidence",
        fill=(255, 255, 0),
    )
    for line_index, (name, landmark_index) in enumerate(
        DIAGNOSTIC_LANDMARKS,
        start=1,
    ):
        landmark = pose_frame.landmarks[landmark_index]
        visibility = "?" if landmark.visibility is None else f"{landmark.visibility:.2f}"
        presence = "?" if landmark.presence is None else f"{landmark.presence:.2f}"
        draw.text(
            (8, 8 + 14 * line_index),
            f"{name}: visibility={visibility} presence={presence}",
            fill=(255, 255, 0),
        )

    return overlay
