import cv2
import numpy as np
from pycine.raw import read_header, read_frames
import pandas as pd

color_map = cv2.COLORMAP_JET

def compute_ratio_sequence(frames, wind_off):
    if not frames:
        raise ValueError("No frames provided")
    return [compute_ratio(wind_off, frame) for frame in frames]

def compute_pressure_sequence(ratio_maps, A, B, Pref):
    if not ratio_maps:
        raise ValueError("No ratio maps provided")
    return [ratio_to_pressure(ratio_map, A, B, Pref) for ratio_map in ratio_maps]

def save_out(image, out_path):
    yes = cv2.imwrite(out_path, image)
    if not yes:
        raise IOError(f"Failed to save image to: {out_path}")

def compute_ratio(wind_off,wind_on):
    a = wind_off.astype(np.float64)
    b = wind_on.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(a, b, out=np.full_like(a, np.nan), where=(b != 0))
    return ratio

def ratio_to_pressure(ratio_map, A, B, Pref):
    pressure_map = Pref * ((ratio_map - A)/B)
    return pressure_map

def normalize_map(pressure_map, p_min=None, p_max=None):
    if p_min is None:
        p_min = np.nanmin(pressure_map)
    if p_max is None:
        p_max = np.nanmax(pressure_map)
    normalized = (pressure_map - p_min) / (p_max - p_min) * 255
    normalized = np.clip(normalized, 0, 255)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized.astype(np.uint8)

def load_calibration_table(path):
    return pd.read_csv(path)

def fit_calibration_from_table(table, I_ref, Pref):
    ratios = I_ref / table['intensity'].values
    pressures = table['pressure'].values
    
    x = pressures / Pref
    B, A = np.polyfit(x, ratios, 1)
    
    return A, B
