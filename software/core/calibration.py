import math

import numpy as np
import cv2
import pint
import gc
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from pathlib import Path
from core import frames, process
from core.capture import ExperimentalCapture

ureg = pint.UnitRegistry()

CALIBRATION_BLUR_AMOUNT = 7
DISPLAY_WIDTH = 1280

def select_ROI(frame: np.ndarray):
    normalized_reference_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    
    return cv2.selectROI("Select region of interest", normalized_reference_frame, False)

def respectfully_resize(image: cv2.typing.MatLike, width: float | None = None, height: float | None = None, inter: int = cv2.INTER_AREA):
    """
    Resizes `image` while maintaining aspect ratio.
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        assert width is not None, "Width must be provided if height is None"
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter) # type: ignore

def prompt_crop_captures(captures: list[ExperimentalCapture]) -> list[ExperimentalCapture]:
    reference_capture = next((c for c in captures if c.rel_pressure == 0.0), None)
    if not reference_capture:
        raise ValueError("No reference capture found")
    mean_frame = reference_capture.mean_frame()
    resized = respectfully_resize(mean_frame, width=DISPLAY_WIDTH)
    rect = select_ROI(resized)

    scale = mean_frame.shape[1] / float(resized.shape[1])
    rect = tuple(int(round(value * scale)) for value in rect)

    for capture in captures:
        cv2.imshow("cropped", cv2.normalize(capture.cropped(rect).mean_frame(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)) # type: ignore
        cv2.waitKey(0)
    return [f.cropped(rect) for f in captures]

def get_I_ref(map: dict[float, float], P_ref: float) -> float:
    I_ref = next((I for P, I in map.items() if P == P_ref), None)
    if not I_ref:
        raise Exception("No wind-off capture provided. Cannot compute ratios")
    return I_ref

def create_pressure_intensity_map(captures: list[ExperimentalCapture], P_ref: float) -> dict[float, float]:
    pressure_intensity_map = {}
    
    for capture in captures:
        I = cv2.GaussianBlur(capture.mean_frame(), (CALIBRATION_BLUR_AMOUNT, CALIBRATION_BLUR_AMOUNT), 0).mean() / 255
        assert capture.rel_pressure != None, "Capture has no pressure value"
        pressure_intensity_map[capture.rel_pressure.to(ureg.pascal).magnitude + P_ref] = I

    return pressure_intensity_map

def fit_pressure_intensity_curve(map: dict[float, float], I_ref: float, P_ref: float) -> Polynomial:
    x = [P / P_ref for P in map.keys()]
    y = [I_ref / I for I in map.values()]
    return Polynomial.fit(x, y, 1)

def plot_pressure_v_intensity(pressure_intensity_map: dict[float, float], P_ref: float) -> Polynomial:
    I_ref = get_I_ref(pressure_intensity_map, P_ref)
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
