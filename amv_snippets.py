from __future__ import annotations

import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import librosa
import numpy as np


@dataclass
class ScoredMoment:
    t: float
    score: float


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True, text=True).returncode != 0:
            raise RuntimeError(f"{tool} is required but not installed.")


@lru_cache(maxsize=1)
def _preferred_video_encoder() -> str:
    """
    Prefer NVIDIA NVENC when available, otherwise fall back to CPU x264.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return "libx264"

    return "h264_nvenc" if "h264_nvenc" in out else "libx264"


def _video_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def _extract_audio_envelope(video_path: str, sample_rate: int = 22050) -> np.ndarray:
    temp_wav = Path(video_path).with_suffix(".tmp_amv_audio.wav")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(temp_wav),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        y, sr = librosa.load(str(temp_wav), sr=sample_rate, mono=True)
        # 1-second envelope for easy alignment with frame sampling.
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=sample_rate)[0]
        return rms
    finally:
        if temp_wav.exists():
            temp_wav.unlink()


@lru_cache(maxsize=1)
def _ffmpeg_supports_cuda_hwaccel() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return False
    return "cuda" in out


def _video_dimensions(video_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    width_s, height_s = out.split("x")
    return int(width_s), int(height_s)


@lru_cache(maxsize=1)
def _has_cuda_device() -> bool:
    try:
        return bool(hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _default_yunet_model_path() -> str | None:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "models" / "face_detection_yunet_2023mar.onnx",
        base_dir / "face_detection_yunet_2023mar.onnx",
        Path("models") / "face_detection_yunet_2023mar.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


class _FaceDetector:
    """
    Prefer YuNet (OpenCV DNN) when model is available, then fall back to Haar.
    """

    def __init__(self) -> None:
        self.mode = "none"
        self.yunet = None
        self.haar = None

        model_path = _default_yunet_model_path()
        if model_path and hasattr(cv2, "FaceDetectorYN"):
            try:
                if _has_cuda_device():
                    backend = cv2.dnn.DNN_BACKEND_CUDA
                    target = cv2.dnn.DNN_TARGET_CUDA
                else:
                    backend = cv2.dnn.DNN_BACKEND_OPENCV
                    target = cv2.dnn.DNN_TARGET_CPU
                self.yunet = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (320, 320),
                    0.8,
                    0.3,
                    5000,
                    backend,
                    target,
                )
                self.mode = "yunet"
            except Exception:
                self.yunet = None

        if self.yunet is None:
            haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not haar.empty():
                self.haar = haar
                self.mode = "haar"

    def score(self, bgr: np.ndarray, gray: np.ndarray) -> float:
        frame_area = float(gray.shape[0] * gray.shape[1])
        if frame_area <= 0:
            return 0.0

        areas: List[float] = []
        if self.mode == "yunet" and self.yunet is not None:
            self.yunet.setInputSize((bgr.shape[1], bgr.shape[0]))
            _ok, faces = self.yunet.detect(bgr)
            if faces is not None and len(faces) > 0:
                for f in faces:
                    w = float(max(0.0, f[2]))
                    h = float(max(0.0, f[3]))
                    areas.append(w * h)
        elif self.mode == "haar" and self.haar is not None:
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
            if len(faces) > 0:
                areas = [float(w * h) for (_, _, w, h) in faces]

        if not areas:
            return 0.0
        return min(1.0, max(areas) / frame_area * 4.0)


def _analyze_frame(
    frame: np.ndarray,
    detector: _FaceDetector,
    prev_gray: np.ndarray | None,
    prev_hist: np.ndarray | None,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if prev_gray is None:
        motion = 0.0
    else:
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(np.mean(diff) / 255.0)

    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    if prev_hist is None:
        shot_change = 0.0
    else:
        corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        shot_change = float(max(0.0, 1.0 - corr))

    color = float(np.mean(hsv[:, :, 1]) / 255.0)
    face_score = detector.score(frame, gray)
    return motion, color, shot_change, face_score, gray, hist


def _scores_to_arrays(
    motion_scores: List[float],
    color_scores: List[float],
    shot_change_scores: List[float],
    face_scores: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(motion_scores, dtype=np.float32),
        np.array(color_scores, dtype=np.float32),
        np.array(shot_change_scores, dtype=np.float32),
        np.array(face_scores, dtype=np.float32),
    )


def _analyze_visuals_opencv(video_path: str, sample_fps: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_step = max(1, int(round(native_fps / sample_fps)))
    detector = _FaceDetector()

    prev_gray = None
    prev_hist = None
    motion_scores: List[float] = []
    color_scores: List[float] = []
    shot_change_scores: List[float] = []
    face_scores: List[float] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        small = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
        motion, color, shot_change, face_score, prev_gray, prev_hist = _analyze_frame(
            small, detector, prev_gray, prev_hist
        )
        motion_scores.append(motion)
        color_scores.append(color)
        shot_change_scores.append(shot_change)
        face_scores.append(face_score)
        frame_idx += 1

    cap.release()
    return _scores_to_arrays(motion_scores, color_scores, shot_change_scores, face_scores)


def _analyze_visuals_ffmpeg(
    video_path: str,
    sample_fps: float = 1.0,
    *,
    use_gpu_decode: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    in_w, in_h = _video_dimensions(video_path)
    target_w = 480
    target_h = max(2, int(round(in_h * target_w / in_w)))
    if target_h % 2 != 0:
        target_h += 1

    cmd = ["ffmpeg", "-v", "error"]
    if use_gpu_decode:
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    cmd.extend(
        [
            "-i",
            video_path,
            "-vf",
            f"fps={sample_fps},scale={target_w}:{target_h}:flags=lanczos,format=bgr24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to start ffmpeg frame extraction.")

    detector = _FaceDetector()
    prev_gray = None
    prev_hist = None
    motion_scores: List[float] = []
    color_scores: List[float] = []
    shot_change_scores: List[float] = []
    face_scores: List[float] = []

    frame_bytes = target_w * target_h * 3
    while True:
        raw = proc.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            proc.kill()
            raise RuntimeError("Incomplete frame read from ffmpeg stream.")

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 3))
        motion, color, shot_change, face_score, prev_gray, prev_hist = _analyze_frame(
            frame, detector, prev_gray, prev_hist
        )
        motion_scores.append(motion)
        color_scores.append(color)
        shot_change_scores.append(shot_change)
        face_scores.append(face_score)

    stderr_out = proc.stderr.read().decode("utf-8", errors="replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {stderr_out.strip()}")

    return _scores_to_arrays(motion_scores, color_scores, shot_change_scores, face_scores)


def _analyze_visuals(
    video_path: str,
    sample_fps: float = 1.0,
    *,
    prefer_gpu_decode: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if prefer_gpu_decode and _ffmpeg_supports_cuda_hwaccel():
        try:
            return _analyze_visuals_ffmpeg(video_path, sample_fps=sample_fps, use_gpu_decode=True)
        except Exception:
            pass

    try:
        return _analyze_visuals_ffmpeg(video_path, sample_fps=sample_fps, use_gpu_decode=False)
    except Exception:
        return _analyze_visuals_opencv(video_path, sample_fps=sample_fps)


def _normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if math.isclose(lo, hi):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _pick_non_overlapping_peaks(
    scores: np.ndarray,
    clip_len: float,
    max_clips: int,
    sample_period: float,
    duration: float,
    *,
    descending: bool = True,
    blocked_starts: Sequence[float] | None = None,
) -> List[float]:
    candidates = [ScoredMoment(i * sample_period, float(s)) for i, s in enumerate(scores)]
    candidates.sort(key=lambda m: m.score, reverse=descending)

    chosen: List[float] = []
    for c in candidates:
        if len(chosen) >= max_clips:
            break
        start = max(0.0, c.t - clip_len * 0.4)
        end = min(duration, start + clip_len)
        start = max(0.0, end - clip_len)

        overlap = False
        if blocked_starts:
            for s in blocked_starts:
                if not (end <= s or start >= s + clip_len):
                    overlap = True
                    break
            if overlap:
                continue
        for s in chosen:
            if not (end <= s or start >= s + clip_len):
                overlap = True
                break
        if not overlap:
            chosen.append(start)

    return sorted(chosen)


def _dynamic_duration_from_tone_change(
    scores: np.ndarray,
    start_seconds: float,
    sample_period: float,
    min_duration: float,
    max_duration: float,
) -> float:
    """
    Pick a clip duration based on the strongest tone change after the clip start.
    """
    if scores.size < 2:
        return min_duration

    start_idx = int(round(start_seconds / sample_period))
    start_idx = max(0, min(start_idx, scores.size - 2))
    max_idx = min(scores.size - 1, start_idx + int(round(max_duration / sample_period)))

    segment = scores[start_idx : max_idx + 1]
    if segment.size < 2:
        return min_duration

    tone_delta = np.abs(np.diff(segment))
    if tone_delta.size == 0:
        return min_duration

    strongest_change_idx = int(np.argmax(tone_delta)) + 1
    strongest_change_seconds = strongest_change_idx * sample_period
    return max(min_duration, min(max_duration, strongest_change_seconds))


def _cut_clip(video_path: str, start: float, duration: float, out_path: str) -> None:
    def _build_cmd(selected_encoder: str) -> List[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            video_path,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            selected_encoder,
        ]
        if selected_encoder == "h264_nvenc":
            cmd.extend(
                [
                    "-preset",
                    "p5",
                    "-cq",
                    "18",
                    "-b:v",
                    "0",
                ]
            )
        else:
            cmd.extend(
                [
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                ]
            )

        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                out_path,
            ]
        )
        return cmd

    encoder = _preferred_video_encoder()
    cmd = _build_cmd(encoder)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        if encoder == "libx264":
            raise
        # NVENC can be compiled in but unavailable at runtime; retry on CPU.
        fallback_cmd = _build_cmd("libx264")
        subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)


def extract_emotional_snippets(
    episode_path: str,
    output_dir: str,
    *,
    max_high_clips: int = 20,
    max_low_clips: int = 10,
    min_clip_duration: float = 1.0,
    max_clip_duration: float = 11.0,
    sample_fps: float = 1.0,
    prefer_gpu_decode: bool = True,
    audio_weight: float = 0.35,
    motion_weight: float = 0.25,
    shot_change_weight: float = 0.20,
    face_weight: float = 0.15,
    color_weight: float = 0.05,
    progress: bool = False,
) -> Sequence[str]:
    """
    Extract AMV-friendly snippets from an anime episode using heuristic scoring.

    Args:
        episode_path: Path to source episode file.
        output_dir: Folder where snippets will be written.
        max_high_clips: Number of highest scoring snippets to export.
        max_low_clips: Number of lowest scoring snippets to export.
        min_clip_duration: Minimum length (seconds) of a snippet.
        max_clip_duration: Maximum length (seconds) of a snippet.
        sample_fps: How many frames/sec to sample for analysis.
        prefer_gpu_decode: Prefer ffmpeg CUDA hw decode for frame analysis.
        *_weight: Weights of scoring components.

    Returns:
        List of output snippet file paths.
    """
    _require_ffmpeg()
    if min_clip_duration <= 0 or max_clip_duration < min_clip_duration:
        raise ValueError("Clip duration bounds must satisfy: 0 < min <= max.")

    episode = Path(episode_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"Analyzing episode: {episode.name}")

    duration = _video_duration_seconds(str(episode))
    if progress:
        print(f"Duration: {duration:.1f}s")

    motion, color, cuts, faces = _analyze_visuals(
        str(episode),
        sample_fps=sample_fps,
        prefer_gpu_decode=prefer_gpu_decode,
    )
    audio = _extract_audio_envelope(str(episode))

    # Align lengths to the shortest stream.
    n = min(len(motion), len(color), len(cuts), len(faces), len(audio))
    if n == 0:
        raise RuntimeError("No analyzable content found in the episode.")

    motion = _normalize(motion[:n])
    color = _normalize(color[:n])
    cuts = _normalize(cuts[:n])
    faces = _normalize(faces[:n])
    audio = _normalize(audio[:n])

    combined = (
        audio_weight * audio
        + motion_weight * motion
        + shot_change_weight * cuts
        + face_weight * faces
        + color_weight * color
    )

    high_starts = _pick_non_overlapping_peaks(
        combined,
        clip_len=max_clip_duration,
        max_clips=max_high_clips,
        sample_period=1.0 / sample_fps,
        duration=duration,
        descending=True,
    )
    low_starts = _pick_non_overlapping_peaks(
        combined,
        clip_len=max_clip_duration,
        max_clips=max_low_clips,
        sample_period=1.0 / sample_fps,
        duration=duration,
        descending=False,
        blocked_starts=high_starts,
    )

    outputs: List[str] = []
    stem = episode.stem
    if progress:
        print(f"Selected {len(high_starts)} highest and {len(low_starts)} lowest clip windows")
    sample_period = 1.0 / sample_fps
    for i, start in enumerate(high_starts, start=1):
        clip_duration = _dynamic_duration_from_tone_change(
            combined,
            start_seconds=start,
            sample_period=sample_period,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration,
        )
        out_path = out_dir / f"{stem}_highest_{i:02d}.mp4"
        _cut_clip(str(episode), start, clip_duration, str(out_path))
        outputs.append(str(out_path))
        if progress:
            print(
                f"  Highest clip {i}/{len(high_starts)} created: "
                f"{out_path.name} ({clip_duration:.1f}s)"
            )

    for i, start in enumerate(low_starts, start=1):
        clip_duration = _dynamic_duration_from_tone_change(
            combined,
            start_seconds=start,
            sample_period=sample_period,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration,
        )
        out_path = out_dir / f"{stem}_lowest_{i:02d}.mp4"
        _cut_clip(str(episode), start, clip_duration, str(out_path))
        outputs.append(str(out_path))
        if progress:
            print(
                f"  Lowest clip {i}/{len(low_starts)} created: "
                f"{out_path.name} ({clip_duration:.1f}s)"
            )

    return outputs


def _process_episode(
    episode_path: str,
    output_dir: str,
    max_high_clips: int,
    max_low_clips: int,
    min_clip_duration: float,
    max_clip_duration: float,
    sample_fps: float,
    prefer_gpu_decode: bool,
) -> List[str]:
    return list(
        extract_emotional_snippets(
            episode_path,
            output_dir,
            max_high_clips=max_high_clips,
            max_low_clips=max_low_clips,
            min_clip_duration=min_clip_duration,
            max_clip_duration=max_clip_duration,
            sample_fps=sample_fps,
            prefer_gpu_decode=prefer_gpu_decode,
            progress=False,
        )
    )


def extract_from_folder(
    input_dir: str = "input",
    output_dir: str = "output",
    *,
    max_high_clips: int = 20,
    max_low_clips: int = 10,
    min_clip_duration: float = 1.0,
    max_clip_duration: float = 11.0,
    sample_fps: float = 1.0,
    prefer_gpu_decode: bool = True,
    episode_workers: int = max(1, min(4, os.cpu_count() or 1)),
    progress: bool = True,
) -> List[str]:
    """
    Process all episode files in input_dir and save clips directly under output_dir.
    Episode processing is parallelized when episode_workers > 1.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    episodes = sorted(
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts
    )

    if not episodes:
        raise RuntimeError(
            f"No supported video files found in {in_dir}. "
            f"Supported extensions: {', '.join(sorted(allowed_exts))}"
        )
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")

    all_outputs: List[str] = []
    if progress:
        print(f"Found {len(episodes)} video(s) in {in_dir}")
    if episode_workers == 1 or len(episodes) == 1:
        for i, episode in enumerate(episodes, start=1):
            if progress:
                print(f"[{i}/{len(episodes)}] Processing: {episode.name}")
            clips = extract_emotional_snippets(
                str(episode),
                str(out_dir),
                max_high_clips=max_high_clips,
                max_low_clips=max_low_clips,
                min_clip_duration=min_clip_duration,
                max_clip_duration=max_clip_duration,
                sample_fps=sample_fps,
                prefer_gpu_decode=prefer_gpu_decode,
                progress=progress,
            )
            all_outputs.extend(clips)
            if progress:
                print(
                    f"[{i}/{len(episodes)}] Done: {episode.name} | "
                    f"clips this video: {len(clips)} | total clips: {len(all_outputs)}"
                )
    else:
        worker_count = min(episode_workers, len(episodes))
        if progress:
            print(f"Using {worker_count} episode worker(s)")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_episode = {
                executor.submit(
                    _process_episode,
                    str(episode),
                    str(out_dir),
                    max_high_clips,
                    max_low_clips,
                    min_clip_duration,
                    max_clip_duration,
                    sample_fps,
                    prefer_gpu_decode,
                ): episode
                for episode in episodes
            }
            completed = 0
            for future in as_completed(future_to_episode):
                episode = future_to_episode[future]
                clips = future.result()
                completed += 1
                all_outputs.extend(clips)
                if progress:
                    print(
                        f"[{completed}/{len(episodes)}] Done: {episode.name} | "
                        f"clips this video: {len(clips)} | total clips: {len(all_outputs)}"
                    )
    if progress:
        print(
            f"Completed processing {len(episodes)} video(s). "
            f"Total clips created: {len(all_outputs)}"
        )
    return all_outputs


if __name__ == "__main__":
    # Example:
    # Batch mode (default): python3 amv_snippets.py
    # Folder mode: python3 amv_snippets.py input output
    # Single file mode: python3 amv_snippets.py /path/to/episode.mp4 ./snippets
    import sys

    if len(sys.argv) == 1:
        # Auto-detect common folder names.
        if Path("input").is_dir():
            clips = extract_from_folder("input", "output")
        elif Path("input_episodes").is_dir():
            clips = extract_from_folder("input_episodes", "output_episodes")
        else:
            clips = extract_from_folder("input", "output")
    elif len(sys.argv) == 3 and Path(sys.argv[1]).is_dir():
        clips = extract_from_folder(sys.argv[1], sys.argv[2])
    elif len(sys.argv) >= 3:
        clips = extract_emotional_snippets(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage:\n"
            "  python3 amv_snippets.py\n"
            "  python3 amv_snippets.py <input_dir> <output_dir>\n"
            "  python3 amv_snippets.py <episode_path> <output_dir>"
        )
    print("Exported clips:")
    for c in clips:
        print(c)
