import numpy as np
import cv2
import pycine.raw as cine
from pathlib import Path

def load_from_cine(cine_path: str | Path, trim_range: tuple[int, int] | None = None, output_fps = 30) -> list[np.ndarray]:
    """
    Loads frames from a given `cine_path`, optionally trimming the frame range and resampling at `output_fps`.
    """
    print(f"Loading frames from {cine_path}...")
    header = cine.read_header(cine_path)
    capture_fps = header["setup"].FrameRate
    frame_count = header["cinefileheader"].ImageCount

    if trim_range is None:
        start_frame, end_frame = 0, frame_count
    else:
        start_frame, end_frame = trim_range
        if start_frame < 0 or end_frame < 0:
            raise ValueError("trim_range values must be non-negative")
        if end_frame < start_frame:
            raise ValueError("trim_range end must be greater than or equal to the start")
        end_frame = min(end_frame, frame_count)

    count = max(0, end_frame - start_frame)
    raw_frames, _, _ = cine.read_frames(cine_path, start_frame=start_frame, count=count) # type: ignore

    stride = max(1, round(capture_fps / output_fps))
    frame_indices = range(start_frame, end_frame, stride)

    selected_frames = []
    for actual_frame_number, frame in enumerate(raw_frames, start=start_frame):
        if actual_frame_number in frame_indices:
            selected_frames.append(frame)

    return selected_frames

def blur(frames: list[np.ndarray], amount: int = 5):
    return [cv2.GaussianBlur(frame, (amount, amount), 0) for frame in frames]

def mean(frames: list[np.ndarray]):
    stacked = np.stack(frames).astype(np.float64)
    averaged = stacked.mean(axis=0)
    return averaged.astype(frames[0].dtype)


def smooth(frames: list[np.ndarray], filter_size: int):
    if filter_size < 0:
        raise ValueError("filter_size must be >= 0")

    count = len(frames)
    stacked = np.stack(frames).astype(np.float64)

    smoothed = []
    for i in range(count):
        start = max(0, i - filter_size)
        end = min(count, i + filter_size + 1)
        window_avg = stacked[start:end].mean(axis=0)
        smoothed.append(window_avg.astype(frames[i].dtype))

    return smoothed