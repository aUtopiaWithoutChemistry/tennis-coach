from PIL import Image
import pytest

from backend.pose_estimator import PoseFrame, PoseLandmark
from backend.pose_overlay import (
    DIAGNOSTIC_LOW_CONFIDENCE_COLOR,
    DIAGNOSTIC_USABLE_COLOR,
    draw_pose_diagnostic_overlay,
    draw_pose_overlay,
    landmark_to_pixel,
)


def make_landmark(
    x: float = 0.5,
    y: float = 0.5,
    visibility: float | None = 0.9,
    presence: float | None = 0.9,
) -> PoseLandmark:
    return PoseLandmark(
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
        presence=presence,
    )


def test_landmark_to_pixel_maps_normalized_coordinates() -> None:
    landmark = make_landmark(x=0.5, y=0.25)

    assert landmark_to_pixel(landmark, 101, 81) == (50, 20)
    assert landmark_to_pixel(make_landmark(x=0.0, y=0.0), 101, 81) == (0, 0)
    assert landmark_to_pixel(make_landmark(x=1.0, y=1.0), 101, 81) == (100, 80)


def test_landmark_to_pixel_rejects_unusable_landmarks() -> None:
    assert landmark_to_pixel(make_landmark(x=-0.01), 100, 100) is None
    assert landmark_to_pixel(make_landmark(y=1.01), 100, 100) is None
    assert landmark_to_pixel(make_landmark(visibility=0.4), 100, 100) is None
    assert landmark_to_pixel(make_landmark(presence=0.4), 100, 100) is None


def test_landmark_to_pixel_accepts_missing_confidence() -> None:
    landmark = make_landmark(visibility=None, presence=None)

    assert landmark_to_pixel(landmark, 100, 100) == (50, 50)


def test_draw_pose_overlay_does_not_modify_image_when_pose_is_missing() -> None:
    image = Image.new("RGB", (40, 30), color=(0, 0, 0))
    pose_frame = PoseFrame(
        source_frame_index=0,
        timestamp_ms=250,
        landmarks=None,
    )

    overlay = draw_pose_overlay(image, pose_frame)

    assert overlay is not image
    assert image.getpixel((8, 8)) == (0, 0, 0)


def test_landmark_to_pixel_rejects_invalid_configuration() -> None:
    landmark = make_landmark()

    with pytest.raises(ValueError, match="dimensions"):
        landmark_to_pixel(landmark, 0, 100)
    with pytest.raises(ValueError, match="threshold"):
        landmark_to_pixel(landmark, 100, 100, confidence_threshold=1.1)


def test_draw_pose_overlay_hides_face_landmarks() -> None:
    face_landmark = make_landmark(x=0.1, y=0.8)
    body_landmark = make_landmark(x=0.8, y=0.2)
    pose_frame = PoseFrame(
        source_frame_index=0,
        timestamp_ms=0,
        landmarks=(face_landmark,) * 11 + (body_landmark,) * 22,
    )
    image = Image.new("RGB", (101, 101), color=(0, 0, 0))

    overlay = draw_pose_overlay(image, pose_frame)

    assert overlay.getpixel((10, 80)) == (0, 0, 0)
    assert overlay.getpixel((80, 20)) == (255, 64, 64)


def test_diagnostic_overlay_distinguishes_low_confidence_landmarks() -> None:
    landmarks = [make_landmark(x=0.1, y=0.1) for _ in range(33)]
    landmarks[15] = make_landmark(x=0.4, y=0.5, visibility=0.4)
    landmarks[16] = make_landmark(x=0.7, y=0.5)
    pose_frame = PoseFrame(0, 100, tuple(landmarks))
    image = Image.new("RGB", (101, 101), color=(0, 0, 0))

    overlay = draw_pose_diagnostic_overlay(image, pose_frame)

    assert overlay.getpixel((40, 50)) == DIAGNOSTIC_LOW_CONFIDENCE_COLOR
    assert overlay.getpixel((70, 50)) == DIAGNOSTIC_USABLE_COLOR
