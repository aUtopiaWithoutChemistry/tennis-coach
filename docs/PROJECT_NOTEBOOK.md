# Tennis Coach Engineering Notebook

This is a concise record of decisions, experiments, resolved failures, verified tests, and milestones. Raw terminal output does not belong here.

## Current Milestone

**M1 — Pose baseline: MediaPipe mapping in progress**

Current goal: map MediaPipe's raw landmarks into the project's timestamped pose schema, then connect that mapping to sampled RGB frames.

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

## Experiment and Debug Log

| ID | Date | Observation | Resolution |
| --- | --- | --- | --- |
| E-001 | 2026-07-27 | A one-frame AAC fixture raised `av.error.ArgumentError` while flushing the encoder, so the test errored before reaching the inspector. | Replaced compressed AAC/M4A with deterministic PCM/WAV audio and reran the full suite twice. |
| E-002 | 2026-07-29 | A mixed-stream Matroska fixture did not expose a separate video-stream duration, so the regression test exercised the container fallback. | Switched the fixture to MOV, which exposed the one-second video duration separately from its two-second audio and container duration. |

## Verification Status

`python -m pytest -v`: **10 passed, 3 skipped** on Python 3.12.13 with PyAV 18.0.0 and MediaPipe 1.0.0.

The skipped tests describe the unfinished MediaPipe result mapping; completed coverage includes stream-specific duration, video inspection, timestamped RGB sampling, invalid sampling rates, and invalid media handling.

## Milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0 — Video inspector | Complete | Inspector implemented; 3 automated tests pass. |
| M1 — Pose baseline | In progress | MediaPipe installed; pose schema and fake-result mapping tests scaffolded; mapping and real-video inference remain unfinished. |
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
