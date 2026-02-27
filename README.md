# Anime Segmentor

Anime Segmentor analyzes episode files and exports AMV-friendly clips using audio + visual scoring.

## What It Does
- Extracts **high-ranked** impactful clips (default: 20 per episode)
- Extracts **low-ranked** clips (default: 10 per episode)
- Uses dynamic clip lengths (default 1s to 11s)
- Writes all clips directly into the chosen output folder
- Supports episode-level parallel processing
- Uses GPU acceleration when available:
  - NVENC for clip encoding
  - CUDA hw decode path for analysis (with automatic CPU fallback)

## Project Files
- `amv_snippets.py`: main clip extraction pipeline
- `youtube_audio.py`: download audio from a YouTube URL into `output_song/`

## Requirements
- Python 3.10+
- `ffmpeg` and `ffprobe` installed and on PATH
- Python packages:
  - `opencv-python`
  - `librosa`
  - `numpy`
  - `yt-dlp` (for `youtube_audio.py`)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python librosa numpy yt-dlp
```

## Usage

### 1) Batch process episodes
Place videos in `input_episodes/` and run:

```bash
python3 amv_snippets.py
```

Default behavior:
- Input folder auto-detected as `input_episodes/` (if present)
- Output folder: `output_episodes/`
- All generated clips are written directly into `output_episodes/`

### 2) Custom input/output folders

```bash
python3 amv_snippets.py <input_dir> <output_dir>
```

Example:

```bash
python3 amv_snippets.py input_episodes output_episodes
```

### 3) Single episode

```bash
python3 amv_snippets.py /path/to/episode.mp4 output_episodes
```

## Tuning From Python
`amv_snippets.py` CLI is positional-only. For advanced controls, call from Python:

```python
from amv_snippets import extract_from_folder

clips = extract_from_folder(
    "input_episodes",
    "output_episodes",
    max_high_clips=20,
    max_low_clips=10,
    sample_fps=1.0,
    episode_workers=3,
    prefer_gpu_decode=True,
    progress=True,
)
print(f"Created {len(clips)} clips")
```

Useful parameters:
- `episode_workers`: number of parallel episode workers
- `prefer_gpu_decode`: try CUDA decode path first
- `sample_fps`: lower values reduce analysis cost
- `min_clip_duration` / `max_clip_duration`

## Faster Face Detection (Optional)
The pipeline prefers YuNet face detection when a model is available, then falls back to Haar cascade.

To enable YuNet, place this model at one of:
- `models/face_detection_yunet_2023mar.onnx`
- `./face_detection_yunet_2023mar.onnx`

If not present, processing still works with fallback detection.

## YouTube Audio Download

```bash
python3 youtube_audio.py "https://www.youtube.com/watch?v=<id>"
```

Output goes to `output_song/` by default.

## Notes
- GPU usage is automatic when supported; pipeline falls back to CPU when needed.
- Parallelism should be tuned to your hardware (CPU cores, disk speed, GPU session limits).
