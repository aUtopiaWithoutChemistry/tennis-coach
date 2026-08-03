# Pose Model Benchmark

This note records why MediaPipe was tested, what the benchmark measured, and
what evidence supports the MVP pose-model decision. Local videos, model files,
and generated overlays are intentionally excluded from Git.

## Experiment Setup

- Date: 2026-08-03
- Video: `tenis-backview/video1.mp4`
- View: single tennis player filmed from behind the court
- Sample rate: 30 FPS
- Sampled frames: 345 (approximately 11.5 seconds)
- Landmark usability threshold: 0.5
- Hardware: local MacBook Pro
- Models: MediaPipe Pose Landmarker Lite, Full, and Heavy

The automated benchmark measures pose detection, required-joint usability,
missing-pose continuity, and processing speed. It does **not** measure spatial
keypoint accuracy against human-annotated ground truth.

## Automated Results

| Metric | Lite | Full | Heavy |
| --- | ---: | ---: | ---: |
| Pose detection | 98.0% | 98.0% | 98.0% |
| All-required-joints usable frames | 3.5% | 7.8% | 5.8% |
| Overall required-joint coverage | 81.5% | 81.2% | 79.8% |
| Left elbow coverage | 83.5% | 54.8% | 45.5% |
| Right elbow coverage | 65.2% | 58.6% | 54.8% |
| Left wrist coverage | 33.0% | 44.6% | 45.8% |
| Right wrist coverage | 13.6% | 33.3% | 33.3% |
| Processing FPS | 97.1 | 83.5 | 51.5 |

Full and Heavy improved wrist confidence but reduced elbow confidence, leaving
overall required-joint coverage nearly unchanged. All three models processed
this clip faster than its 30 FPS sampling rate.

## Domain Limitations

MediaPipe reports pose quality on Yoga, Dance, and HIIT validation images with
one person 2–4 meters from the camera. Its current documentation describes the
model as optimized for real-time fitness applications. Rear-court tennis adds
a more distant back view, rapid arm motion, self-occlusion, and a racket that
the pose model does not detect.

These differences do not prove that MediaPipe is unusable for tennis. They do
mean that published fitness results cannot establish accuracy for this project.

- [MediaPipe Pose quality evaluation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#pose-estimation-quality)
- [Current Pose Landmarker model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models)
- [BlazePose paper](https://arxiv.org/abs/2006.10204)

## Visual Diagnostic

Matching Lite and Full diagnostic videos displayed usable landmarks in green
and low-confidence landmarks in orange. Both models showed the same important
pattern:

- Elbows and wrists were commonly orange while hidden behind the player's body.
- The same joints became green when the swing moved them into view.
- The high low-visibility counts therefore mostly reflect expected
  self-occlusion rather than persistent pose-estimation failure.
- Neither diagnostic video established a clear visual winner during the swing.

This observation corrects an initially plausible but incomplete interpretation
of the aggregate counts. Whole-video joint coverage mixes task-relevant swings
with preparation and recovery frames in which an arm may be hidden. Future
quality measurements should calculate dominant-arm coverage specifically within
detected stroke phases.

## Provisional Decision

MediaPipe is adequate as the MVP pose baseline because both tested variants
produced confident arm landmarks during the swing and ran faster than real
time. Full remains the provisional default because it improved aggregate wrist
coverage while retaining substantial processing headroom. Heavy does not
currently justify its additional cost.

The result does not establish ground-truth keypoint accuracy, and MediaPipe's
fitness-oriented evaluation domain does not match rear-court tennis. The model
will remain behind the project-owned `PoseFrame` interface so a tennis-oriented
pose estimator can replace it later. Confidence coverage must not be presented
as keypoint accuracy.
