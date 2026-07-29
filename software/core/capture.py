import numpy as np
import cv2
import pint
import core.frames as frames
from pathlib import Path

ureg = pint.UnitRegistry()

PROCESSING_BLUR_AMOUNT = 17

def pressure_from_path(cine_path: Path) -> pint.Quantity | None:
    split = cine_path.stem.split(" ")

    if len(split) != 2:
        return None

    quantity = split[0].replace("_", ".")
    unit = split[1]
    
    if not unit in ureg:
        return None

    return float(quantity) * ureg.parse_units(unit, case_sensitive=False)

class ExperimentalCapture:
    @classmethod
    def from_cine(cls, cine_path: Path):
        return cls(frames.load_from_cine(cine_path), pressure_from_path(cine_path))

    @classmethod
    def from_folder(cls, folder_path: Path):
        return [cls.from_cine(f) for f in folder_path.iterdir() if f.is_file()]
    
    def __init__(self, frames: list[np.ndarray], pressure: pint.Quantity | None):
        self.frames = frames
        self.pressure = pressure
        self.__mean_frame = None

    def cropped(self, rect: cv2.typing.Rect):
        cropped_frames = []

        x, y, w, h = rect

        for frame in self.frames:
            cropped_frames.append(frame[y : y + h, x: x + w])

        return ExperimentalCapture(cropped_frames, self.pressure)

    def blurred(self, amount = 5):
        """
        Individually applies a gaussian blur to each frame of kernel `amount` x `amount`
        """
        assert amount % 2 == 1, "Blur amount must be odd"
        return ExperimentalCapture(frames.blur(self.frames, amount), self.pressure)


    def mean_frame(self):
        """
        Returns the mean of all blurred frames in this capture, lazily calculated
        """
        self.__mean_frame = self.__mean_frame if self.__mean_frame is not None else frames.mean(self.frames)
        return self.__mean_frame

    def post_process(self) -> np.ndarray:
        return self.blurred(PROCESSING_BLUR_AMOUNT).mean_frame()
