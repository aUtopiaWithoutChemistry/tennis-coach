"""Render project pose landmarks on copies of source images."""

import mediapipe as mp
from PIL import Image, ImageDraw

from backend.pose_estimator import PoseFrame, PoseLandmark

POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
)
FIRST_BODY_LANDMARK_INDEX = 11


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
