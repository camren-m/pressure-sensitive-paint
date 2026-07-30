import numpy as np

PROCESSING_BLUR_AMOUNT = 7

def compute_ratio(wind_off_frame: np.ndarray, wind_on_frame: np.ndarray):
    a = wind_off_frame.astype(np.float64)
    b = wind_on_frame.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(a, b, where=(b != 0))
    return ratio

def ratio_to_pressure(ratio_map: np.ndarray, A: float, B: float, P_ref: float) -> np.ndarray:
    return P_ref * ((ratio_map - A)/B) # eq. 4

def normalize_map(pressure_map, p_min=None, p_max=None):
    if p_min is None:
        p_min = np.nanmin(pressure_map)
    if p_max is None:
        p_max = np.nanmax(pressure_map)
    normalized = (pressure_map - p_min) / (p_max - p_min) * 255
    normalized = np.clip(normalized, 0, 255)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized.astype(np.uint8)
