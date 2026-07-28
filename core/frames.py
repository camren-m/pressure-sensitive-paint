import numpy as np
import pycine.raw as cine
import pathlib

def load_from_cine(cine_path: str | pathlib.Path, trim_range: tuple[int, int] | None = None, output_fps = 30):
    """
    Loads all frames from a given `cine_path`, optionally trimming for `range`, and remapping frames at `output_fps`
    """
    header = cine.read_header(cine_path)
    capture_fps = header["setup"].FrameRate
    frame_count = header["cinefileheader"].ImageCount

    start_frame = trim_range[0] if trim_range else 0
    end_frame = trim_range[1] if trim_range else frame_count
    
    raw_frames, _, _ = cine.read_frames(cine_path, start_frame=start_frame, count=frame_count) # type: ignore
        
    stride = round(capture_fps / output_fps)
    if stride < 1:
        stride = 1
    
    frame_indices = range(start_frame, end_frame, stride)
    count = end_frame - start_frame

    selected_frames = []
    for i, frame in enumerate(raw_frames):
        actual_frame_number = start_frame + i
        if actual_frame_number in frame_indices:
            selected_frames.append(frame)

    return selected_frames

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