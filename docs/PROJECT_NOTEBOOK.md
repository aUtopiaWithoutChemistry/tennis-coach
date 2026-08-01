# Tennis Coach Engineering Notebook

This is a concise record of decisions, experiments, resolved failures, verified tests, and milestones. Raw terminal output does not belong here.

## Current Milestone

**M1 — Pose visualization complete**

The project can now turn a real tennis drill clip into a playable H.264 video with timestamped body-pose overlays. Pose inference and visualization are complete; performance and landmark-continuity benchmarking remain before the upload vertical slice.

## Decision Log

| ID | Date | Decision | Rationale |
| --- | --- | --- | --- |
| D-001 | 2026-07-25 | Build a local React/TypeScript application backed by a Python analysis service. | Provides a useful personal tool and demonstrates full-stack plus computer-vision skills. |
| D-002 | 2026-07-25 | Target fast post-clip analysis of 30–60 second videos, starting with forehands. | Produces a credible first version without live-streaming complexity. |
| D-003 | 2026-07-25 | Validate a stable rear-diagonal camera position. | Makes forehand rotation and contact-side motion more observable than a centered rear view. |
| D-004 | 2026-07-25 | Start with MediaPipe 2D landmarks and keep 3D/velocity experimental. | Explainable 2D measurements are a safer baseline for monocular sports footage. |
| D-005 | 2026-07-25 | Publish code while keeping personal videos and generated artifacts local. | Supports recruiter review without exposing private practice footage. |
| D-006 | 2026-07-25 | Use deterministic, evidence-linked feedback before training a quality model. | Professional clips alone do not provide trustworthy coaching labels. |
| D-007 | 2026-07-27 | Add a learner-led Video Inspector before pose estimation. | Establishes the video input boundary in a small feature that can be implemented and tested collaboratively. |
| D-008 | 2026-07-29 | Sample decoded frames using presentation timestamps and yield them as a generator. | Supports variable frame timing while limiting memory use and pose-processing cost. |
| D-009 | 2026-07-29 | Use MediaPipe as the MVP pose baseline and reserve YOLO26 Pose for a later benchmark. | MediaPipe provides 33 landmarks, timestamp-aware video processing, and a simpler license fit; the MVP needs landmarks rather than custom pose-model training. |
| D-010 | 2026-07-31 | Keep all MediaPipe landmarks in numerical pose data but render only body landmarks 11–32. | Facial landmarks add visual clutter to a full-body tennis overlay but may remain useful to later analysis. |
| D-011 | 2026-07-31 | Stream source frames and pose results together into a bounded, silent, constant-frame-rate H.264 preview. | Preserves the source pixels needed for drawing, limits memory use, and creates an easy-to-review MVP artifact. |

## Experiment and Debug Log

| ID | Date | Observation | Resolution |
| --- | --- | --- | --- |
| E-001 | 2026-07-27 | A one-frame AAC fixture raised `av.error.ArgumentError` while flushing the encoder, so the test errored before reaching the inspector. | Replaced compressed AAC/M4A with deterministic PCM/WAV audio and reran the full suite twice. |
| E-002 | 2026-07-29 | A mixed-stream Matroska fixture did not expose a separate video-stream duration, so the regression test exercised the container fallback. | Switched the fixture to MOV, which exposed the one-second video duration separately from its two-second audio and container duration. |
| E-003 | 2026-07-30 | Pose Landmarker Lite detected 0 of 14 frames in a wide broadcast clip; players occupied too few pixels in the full frame. A lower-court crop detected 5 of 14 frames, and a closer rear-court drill clip detected 14 of 20 frames without cropping. | Treat camera framing and player scale as input requirements. Keep wide broadcast footage as a later cropping benchmark rather than the primary MVP domain. |
| E-004 | 2026-07-31 | The first pose-overlay images showed several tightly grouped dots around the player's head. | Confirmed that these were MediaPipe facial landmarks 0–10 and excluded them from drawing while preserving them in pose results. |

## Verification Status

`python -m pytest -v`: **27 passed** on Python 3.12.13 with PyAV 18.0.0 and MediaPipe 1.0.0.

Coverage includes stream-specific duration, video inspection, timestamped RGB sampling, invalid media handling, MediaPipe result mapping and adapter behavior, landmark-to-pixel conversion, body-only overlay drawing, diagnostic image output, and decodable H.264 video rendering. Real-model inference and visual alignment are verified separately with local smoke tests and gitignored artifacts.

## Milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0 — Video inspector | Complete | Inspector implemented; 3 automated tests pass. |
| M1a — Pose inference baseline | Complete | Pose mapping and video adapter implemented; Lite model detected poses in 14 of 20 samples from a real rear-court drill clip. |
| M1b — Pose visualization | Complete | Body-only diagnostic images and a playable 100-frame H.264 overlay video generated; 27 automated tests pass. |
| M1c — Pose benchmark | Not started | Processing speed, landmark coverage, and missing-frame continuity still need measurement. |
| M2 — Upload vertical slice | Not started | — |
| M3 — Stroke detection | Not started | — |
| M4 — Explainable feedback | Not started | — |
| M5 — Product hardening | Not started | — |
| M6 — Portfolio release | Not started | — |

## Session Notes

### 2026-07-27

**Completed:** Built the learner-led Video Inspector using PyAV. It returns resolution, FPS, duration, frame-count metadata, codec, and source path; it also validates that a frame can be decoded. Added tests for a valid video, a missing file, and audio-only media. Final check: **3 passed**.

**Remember:**

- Container/stream metadata describes a video without decoding every frame, but decoding one frame confirms the content is readable.
- Metadata may be missing, so optional values use `None`; do not replace “unknown” with a misleading zero.
- Let library code raise exceptions so the future CLI or web layer can decide how to display errors.
- Pytest discovers `test_*` functions and injects fixtures by parameter name. `ERROR` means setup failed; `FAILED` means the test ran but behavior/assertions were wrong.
- Activate the environment with `source .venv/bin/activate`; run all tests from the project root with `python -m pytest -v`.

**Next session:** Define and approve a small Frame Sampler that yields timestamped frames at a target FPS. Do not begin pose estimation yet. After sampling is reliable, add a pose backend; MediaPipe remains the recommended first baseline, with YOLO Pose reserved for comparison if needed.

### 2026-07-29

**Completed:** Added a generator-based Frame Sampler that decodes a video in sequence, selects frames at a configurable target rate using presentation timestamps, converts selected frames to RGB, and preserves their source indexes and timestamps. Added success and failure-path tests. Final check: **9 passed**.

**Remember:**

- A frame timestamp is `PTS × time_base`; source frame indexes alone are unreliable for variable-frame-rate video.
- Decoding reconstructs available frames but cannot invent missing frames or remove timestamp gaps.
- Advancing the sampling boundary past the current timestamp prevents rapid catch-up after a large gap.
- Generator code runs when it is consumed, so tests use `list(...)` to trigger validation and decoding.

**Additional progress:** Corrected and merged the Video Inspector duration calculation so it prefers the selected video stream over a longer container stream. Selected MediaPipe for the MVP, installed MediaPipe 1.0.0, and added immutable `PoseLandmark` and `PoseFrame` schemas plus a learner-owned result-mapping scaffold. Private broadcast clips under `data/output_clips/` are available for a later smoke test and remain ignored by Git.

**Remember:** MediaPipe supplies body landmarks; our later code must calculate angles, rotations, stroke phases, and feedback. Professional clips may help build reference distributions, but they are not trustworthy quality labels by themselves.

**Next:** Complete `pose_frame_from_result`, enable its three fake-result tests, then connect the mapping to MediaPipe video inference before running a private smoke test.

### 2026-07-30

**Completed:** Connected timestamped RGB frames to MediaPipe video inference and mapped the results into immutable `PoseFrame` values. Added a bounded smoke-test command and verified real inference on a rear-court professional drill clip: Pose Landmarker Lite detected a pose in **14 of 20 samples (70%)**. Downloaded and validated the Full and Heavy bundles for later comparison; all model files remain under the ignored `.models/` directory.

**Remember:** Pose detection depends strongly on how much of the frame the player occupies. A larger pose model cannot recover visual detail that is absent from a distant subject. Use camera-matched practice footage for MVP development, and retain broadcast footage as a harder future test for player localization and cropping.

**Next:** Visually validate that detected landmarks align with the intended player before deriving joint angles or other coaching measurements.

### 2026-07-31

**Completed:** Added reusable pose-overlay drawing, extended the smoke test to save diagnostic PNGs, and built a streaming video renderer that pairs each sampled source frame with its `PoseFrame` and writes a bounded, silent H.264 MP4. Removed facial landmarks 0–10 from the drawing while retaining them in numerical pose data. Generated a visually plausible 1280×720, 10 FPS, 100-frame local artifact. Final check: **27 passed**.

**Remember:** Calling `next()` to inspect the first source/pose pair consumes it, so `chain((first_pair,), paired_frames)` places it back before the remaining iterator. H.264 encoding may buffer frames, so the encoder must be flushed after the loop. The renderer currently creates a constant-frame-rate visualization without source audio; it is a diagnostic artifact, not yet a full-fidelity export.

**Next:** Benchmark processing time, required-joint coverage, and missing-pose continuity on camera-matched clips before beginning the upload vertical slice.
