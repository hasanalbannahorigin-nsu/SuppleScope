from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    duration_ms: int
    width: int
    height: int
    fps: float


@dataclass
class SelectedFrame:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    scene_score: float
    selection_reason: str


def _bin(name: str, configured: str) -> str:
    path = configured or name
    found = shutil.which(path)
    if not found:
        raise RuntimeError(
            f"{name} was not found on PATH. Install FFmpeg and add it to PATH, "
            f"or set FFMPEG_PATH / FFPROBE_PATH."
        )
    return found


def probe_video(path: Path) -> VideoMeta:
    settings = get_settings()
    ffprobe = _bin("ffprobe", settings.ffprobe_path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration") or 0) * 1000
    video_stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    fps_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    num, _, den = fps_text.partition("/")
    fps = float(num) / float(den or 1) if den else float(num or 0)
    return VideoMeta(
        duration_ms=int(duration),
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=fps or 25.0,
    )


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    settings = get_settings()
    ffmpeg = _bin("ffmpeg", settings.ffmpeg_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1000:] if completed.stderr else "ffmpeg audio extract failed")
    return audio_path


def _phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_informative_frames(
    video_path: Path,
    *,
    sample_fps: float = 1.0,
    max_frames: int = 24,
    scene_threshold: float = 0.28,
    duplicate_hamming: int = 6,
) -> tuple[VideoMeta, list[SelectedFrame]]:
    meta = probe_video(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or meta.fps
    interval = max(int(round(native_fps / sample_fps)), 1)
    prev_hist = None
    candidates: list[SelectedFrame] = []
    index = 0
    kept_hashes: list[int] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
        take_time = index % interval == 0
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        scene_score = 1.0
        if prev_hist is not None:
            scene_score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_hist = hist
        take_scene = scene_score >= scene_threshold
        if not (take_time or take_scene):
            index += 1
            continue
        digest = _phash(frame)
        if any(_hamming(digest, existing) <= duplicate_hamming for existing in kept_hashes):
            index += 1
            continue
        reason = "scene_change" if take_scene and not take_time else "time_sample"
        if take_scene and take_time:
            reason = "time_and_scene"
        candidates.append(
            SelectedFrame(
                frame_index=len(candidates),
                timestamp_ms=timestamp_ms,
                image=frame.copy(),
                scene_score=scene_score,
                selection_reason=reason,
            )
        )
        kept_hashes.append(digest)
        index += 1

    capture.release()

    if not candidates:
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        if ok:
            candidates.append(
                SelectedFrame(
                    frame_index=0,
                    timestamp_ms=0,
                    image=frame,
                    scene_score=0.0,
                    selection_reason="fallback_first_frame",
                )
            )

    candidates.sort(key=lambda item: item.scene_score, reverse=True)
    selected = sorted(candidates[:max_frames], key=lambda item: item.timestamp_ms)
    for i, item in enumerate(selected):
        item.frame_index = i
    return meta, selected


def write_frame_jpeg(frame: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    dest.write_bytes(encoded.tobytes())
