import numpy as np
import cv2
import pint
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from pathlib import Path
from core import frames
from core.capture import ExperimentalCapture

ureg = pint.UnitRegistry()

PROCESSING_BLUR_AMOUNT = 7

def select_ROI(frame: np.ndarray):
    normalized_reference_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    
    return cv2.selectROI("Select region of interest", normalized_reference_frame, False)

def prompt_crop_captures(captures: list[ExperimentalCapture]):
    rect = select_ROI(captures[1].mean_frame())
    return [f.cropped(rect) for f in captures]

def get_I_ref(map: dict[float, float]) -> float:
    I_ref = next((I for P, I in map.items() if P == 0.0), None)
    if not I_ref:
        raise Exception("No wind-off capture provided. Cannot compute ratios")
    return I_ref

def create_pressure_intensity_map(captures: list[ExperimentalCapture]) -> dict[float, float]:
    pressure_intensity_map = {}
    
    for capture in captures:
        I = capture.mean_frame().mean() / 255
        assert capture.pressure
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
