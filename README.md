# Tennis Coach

A local-first application that analyzes tennis practice videos and aims to provide clear, evidence-linked feedback for recreational learners.

> **Project status:** Early development. The Video Inspector milestone is complete; pose estimation and stroke analysis have not been implemented yet.

## Why I'm Building This

I am a computer science sophomore and a recreational tennis learner. I want a way to review my practice between coaching sessions and understand which parts of my technique need attention. I am also using this project to develop practical experience in computer vision, machine learning, full-stack development, testing, and software design.

## Project Vision

A learner records practice with an iPhone positioned behind the court. The video is transferred to a Mac, analyzed locally, and presented through an annotated replay with actionable suggestions.

The planned pipeline is:

```text
Inspect video
    ↓
Decode and sample timestamped frames
    ↓
Estimate player pose
    ↓
Detect strokes and movement phases
    ↓
Calculate explainable measurements
    ↓
Generate coaching feedback
```

## Current Capabilities

The first completed component is the **Video Inspector**. It:

- Extracts video resolution, frame rate, duration, frame-count metadata, and codec.
- Confirms that the file contains a readable video stream.
- Reports missing files and audio-only media as errors.
- Preserves unavailable metadata as `None` instead of inventing values.
- Includes automated tests for successful and failure paths.

## Current Usage

```python
from backend.video_inspector import inspect_video

metadata = inspect_video("path/to/practice-video.mp4")
print(metadata)
```

## Development Setup

Requirements:

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

Clone and set up the project:

```bash
git clone https://github.com/aUtopiaWithoutChemistry/tennis-coach.git
cd tennis-coach
uv sync
source .venv/bin/activate
```

Run the tests:

```bash
python -m pytest -v
```

Current verified result: **3 tests passed**.

## Roadmap

- [x] Inspect and validate video inputs
- [ ] Decode and sample timestamped frames
- [ ] Extract body landmarks from sampled frames
- [ ] Render a synchronized pose overlay
- [ ] Detect forehands and movement phases
- [ ] Generate explainable coaching feedback
- [ ] Build the local React and Python application
- [ ] Evaluate accuracy and processing performance

See the detailed [project plan](docs/PROJECT_PLAN.md) and [engineering notebook](docs/PROJECT_NOTEBOOK.md).

## Privacy and Data

- Practice recordings are processed locally.
- Personal videos, datasets, generated artifacts, and model files are excluded from Git.
- This repository does not include the private training or evaluation videos used during development.

## Limitations

This project is experimental and under active development. It does not currently provide tennis coaching feedback, medical advice, or injury-prevention guidance, and it is not a replacement for a qualified coach.

## Feedback

Feedback and suggestions are welcome through [GitHub Issues](https://github.com/aUtopiaWithoutChemistry/tennis-coach/issues), especially about product usefulness, video-analysis architecture, testing, and computer-vision model choices.

## License

This project is available under the [MIT License](LICENSE).
