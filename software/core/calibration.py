import numpy as np
import cv2
import pint
import math
import core.frames as frames
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
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

    def mean_frame(self, blur_amount = 7):
        """
        Returns the mean of all blurred frames in this capture, lazily calculated
        """
        assert blur_amount % 2 == 1, "blur_amount must be odd"
        self.__mean_frame = self.__mean_frame if self.__mean_frame is not None else frames.mean(frames.blur(self.frames, blur_amount))
        return self.__mean_frame

def select_ROI(frame: np.ndarray):
    normalized_reference_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    
    return cv2.selectROI("Select region of interest", normalized_reference_frame, False)

def prompt_crop_captures(captures: list[ExperimentalCapture]):
    rect = select_ROI(captures[1].mean_frame())
    return [f.crop(rect) for f in captures]

def get_I_ref(map: dict[float, float]) -> float:
    I_ref = next((I for P, I in map.items() if P == 0.0), None)
    if not I_ref:
        raise Exception("No wind-off capture provided. Cannot compute ratios")
    return I_ref

def create_pressure_intensity_map(captures: list[ExperimentalCapture]) -> dict[float, float]:
    pressure_intensity_map = {}
    
    for capture in captures:
        I = capture.mean_frame().mean() / 255
        pressure_intensity_map[capture.pressure.to(ureg.pascal).magnitude] = I

    return pressure_intensity_map

def fit_pressure_intensity_curve(map: dict[float, float], I_ref: float, P_ref: float) -> Polynomial:
    x = [P / P_ref for P in map.keys()]
    y = [I_ref / I for I in map.values()]
    return Polynomial.fit(x, y, 1)

def plot_pressure_v_intensity(pressure_intensity_map: dict[float, float], P_ref: float) -> Polynomial:
    I_ref = get_I_ref(pressure_intensity_map)
    poly = fit_pressure_intensity_curve(pressure_intensity_map, I_ref, P_ref)

    sorted_data = sorted([P / P_ref, I_ref / I] for P, I in pressure_intensity_map.items())
    x, y = zip(*sorted_data)
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, label=str(poly.convert()), color="black")
    plt.plot(x, y, marker="o", linestyle="-", color="blue")
    plt.title("Pressure ratio v. Intensity ratio")
    plt.xlabel("P / P_ref")
    plt.ylabel("I_ref / I")
    plt.grid(True)
    plt.legend()
    plt.show()
    return poly
