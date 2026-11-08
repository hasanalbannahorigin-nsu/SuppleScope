from pathlib import Path

import numpy as np

from app.services import video


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == video.cv2.CAP_PROP_FPS:
            return 2.0
        if prop == video.cv2.CAP_PROP_POS_MSEC:
            return (self.position - 1) * 500
        return 0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self):
        pass


def test_extracts_time_sampled_frames_with_timestamps_and_ids(monkeypatch):
    split_scene = np.zeros((32, 32, 3), dtype=np.uint8)
    split_scene[:, 16:] = 255
    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8),
        split_scene,
        split_scene,
    ]
    monkeypatch.setattr(video, "probe_video", lambda _: video.VideoMeta(2000, 32, 32, 2.0))
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda _: FakeCapture(frames))

    metadata, selected = video.extract_informative_frames(
        Path("sample.mp4"),
        sample_fps=2.0,
        max_frames=4,
        scene_threshold=0.28,
    )

    assert metadata.duration_ms == 2000
    assert [frame.timestamp_ms for frame in selected] == [0, 1000]
    assert [frame.frame_index for frame in selected] == [0, 1]
    assert all(frame.image.size for frame in selected)


def test_extracts_frames_from_generated_mp4(tmp_path, monkeypatch):
    import pytest

    sample_path = tmp_path / "sample.mp4"
    writer = video.cv2.VideoWriter(
        str(sample_path),
        video.cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV has no MP4 video encoder available.")
    for frame_index in range(6):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if frame_index >= 3:
            frame[:, 32:] = 255
        writer.write(frame)
    writer.release()

    if not video.shutil.which(video.get_settings().ffprobe_path):
        monkeypatch.setattr(
            video,
            "probe_video",
            lambda _: video.VideoMeta(duration_ms=3000, width=64, height=64, fps=2.0),
        )

    metadata, frames = video.extract_informative_frames(sample_path, sample_fps=2.0)

    assert metadata.duration_ms > 0
    assert len(frames) >= 1
    assert all(frame.image.size for frame in frames)
