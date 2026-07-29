import numpy as np
import cv2
import pint
import core.frames as frames
import matplotlib.pyplot as plt
from pathlib import Path

ureg = pint.UnitRegistry()

def pressure_from_path(cine_path: Path):
    split = cine_path.stem.split(" ")

    if len(split) != 2:
        raise Exception(f"Failed to parse filename {cine_path.name} as pressure: malformed quantity/unit")

    quantity = split[0].replace("_", ".")
    unit = split[1]
    
    if not unit in ureg:
        raise Exception(f"Failed to parse filename {cine_path.name}: unit {unit} unknown")

    return float(quantity) * ureg.parse_units(unit, case_sensitive=False)

class ExperimentalCapture:
    @classmethod
    def from_cine(cls, cine_path: Path):
        return cls(frames.load_from_cine(cine_path), pressure_from_path(cine_path))

    @classmethod
    def from_folder(cls, folder_path: Path):
        return [cls.from_cine(f) for f in folder_path.iterdir() if f.is_file()]
    
    def __init__(self, frames: list[np.ndarray], pressure: pint.Quantity):
        self.frames = frames
        self.pressure = pressure
        self.__mean_frame = None

    def crop(self, rect: cv2.typing.Rect):
        cropped_frames = []

        x, y, w, h = rect

        for frame in self.frames:
            cropped_frames.append(frame[y : y + h, x: x + w])

        return ExperimentalCapture(cropped_frames, self.pressure)

    def mean_frame(self):
        """
        Returns the mean of all frames in this capture, lazily calculated
        """
        self.__mean_frame = self.__mean_frame or frames.mean(self.frames)
        return self.__mean_frame

def select_ROI(frame: np.ndarray):
    normalized_reference_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    
    return cv2.selectROI("Select region of interest", normalized_reference_frame, False)

def prompt_crop_captures(captures: list[ExperimentalCapture]):
    rect = select_ROI(captures[1].frames[1])
    return [f.crop(rect) for f in captures]

def create_pressure_intensity_map(captures: list[ExperimentalCapture]) -> dict[float, float]:
    pressure_intensity_map = {}
    
    for capture in captures:
        pressure_intensity_map[capture.pressure.to(ureg.pascal).magnitude] = capture.mean_frame().mean() / 255

    return pressure_intensity_map

def plot_pressure_v_intensity(pressure_intensity_map: dict[float, float]):
    sorted_data = sorted(pressure_intensity_map.items())
    x, y = zip(*sorted_data)
    plt.plot(x, y, marker="o", linestyle="-", color="blue")
    plt.xlabel("Surface Pressure (pascal)")
    plt.ylabel("Mean Intensity (unitless)")
    plt.grid(True)
    plt.show()
