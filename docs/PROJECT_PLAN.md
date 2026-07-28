# Tennis Coach: Six-Week Recruiting MVP

## Summary

Build a public, local-first tennis coaching application by September 6, 2026:

- React/TypeScript interface with a FastAPI/Python analysis service.
- Import 30–60 second iPhone videos manually after recording.
- Support one beginner-to-intermediate player, forehands, and a guided rear-diagonal camera position.
- Automatically suggest stroke clips, allow boundary correction, and produce evidence-linked feedback within 60 seconds.
- Keep recordings and generated artifacts local and excluded from Git.
- Defer live streaming, ball/racket tracking, learned scoring, and authoritative 3D biomechanics.

## Architecture and Product Behavior

- Pin Python 3.12 with `uv`; use React, TypeScript, Vite, FastAPI, PyAV, and SQLite. Avoid requiring a system FFmpeg installation.
- Process videos at 720p/30 FPS while retaining the original for playback.
- Use MediaPipe Pose Landmarker as the first backend. Keep it behind a `PoseEstimator` interface so YOLO26 or Apple Vision can be benchmarked later. Estimated world coordinates remain experimental and will not drive coaching feedback.
- Store session metadata and job status in SQLite; store original videos, pose artifacts, and overlays in a gitignored local data directory.
- Run one analysis job at a time in a separate local process. The UI polls job status and reports actionable failures.
- Use deterministic, versioned coaching rules rather than an opaque quality score. Every result includes the measured observation, relevant video interval, confidence, and a conservative practice suggestion.
- Support right- and left-handed players by mirroring canonical landmark roles.

Planned API:

- `POST /api/analyses`: accept a video and dominant hand; return an analysis ID.
- `GET /api/analyses/{id}`: return progress, detected strokes, metrics, and feedback.
- `PATCH /api/analyses/{id}/strokes/{stroke_id}`: confirm, reject, or adjust a proposed stroke.

## Six-Week Roadmap

1. **Pose baseline:** Repository/tooling, MediaPipe video processing, standard pose schema, overlay output, latency and continuity benchmark, and engineering notebook.
2. **Upload vertical slice:** Mac browser upload, local job processing, progress/error states, and analysis history.
3. **Stroke detection:** Propose forehand windows using dominant-wrist motion and pose sequencing; provide confirmation and boundary adjustment.
4. **Explainable feedback:** Detect preparation, loading, estimated-contact, and follow-through phases; add a small sourced rubric for stance/loading, balance, preparation, and follow-through.
5. **Product hardening:** Synchronized video overlay, confidence warnings, retry behavior, privacy controls, and performance optimization.
6. **Portfolio release:** Evaluation report, architecture diagram, demo recording, recruiter-focused README, automated tests, reproducible setup, and documented limitations.

After the recruiting MVP, evaluate ball/racket tracking, Apple Vision or YOLO26, monocular 3D, coach-labeled learning, additional strokes, and local-network iPhone upload.

## First Feature: Pose Baseline

Inputs:

- Three private 10–20 second iPhone clips: controlled shadow swings, normal hitting, and a mildly difficult visibility case.
- Landscape rear-diagonal view, full body visible, stable phone, and good lighting.

Outputs:

- Canonical timestamped pose JSON.
- Annotated overlay video.
- Benchmark JSON containing processed FPS, processing/video-time ratio, required-joint coverage, and missing-frame count.
- A short benchmark entry in `docs/PROJECT_NOTEBOOK.md`.

Acceptance criteria:

- Process a 30-second equivalent in at most 60 seconds on the M5 Mac, with a goal of faster than video duration.
- At least 95% of 60 evenly sampled review frames contain plausible shoulders, elbows, wrists, hips, knees, and ankles.
- Keep the overlay synchronized with the source video.
- Represent missing or low-confidence poses explicitly rather than silently interpolating them.
- Unit-test landmark mapping, timestamps, missing detections, benchmark aggregation, and malformed-video handling.
- Use a mocked estimator in automated tests; never commit private tennis recordings or downloaded models.

## Assumptions and Boundaries

- Initial feedback is experimental and educational, not medical advice or a replacement for a qualified coach.
- Manual AirDrop/file import is acceptable for the recruiting MVP.
- No coach validation is currently available; later coach review will use structured result data.
- Approval of the roadmap is not approval to implement every milestone. Work proceeds one approved feature at a time.

