import numpy as np
import cv2
import pint
from core import frames
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
        self.rel_pressure = pressure
        self.__mean_frame = None

    def cropped(self, rect: cv2.typing.Rect):
        cropped_frames = []

        x, y, w, h = rect

        for frame in self.frames:
            cropped_frames.append(frame[y : y + h, x: x + w])

        return ExperimentalCapture(cropped_frames, self.rel_pressure)

    def mean_frame(self):
        """
        Returns the mean of all frames in this capture, lazily calculated
        """
        self.__mean_frame = self.__mean_frame if self.__mean_frame is not None else frames.mean(self.frames)
        return self.__mean_frame

    def align(self, reference_capture: "ExperimentalCapture") -> "ExperimentalCapture":
        """
        Aligns this capture to a reference capture using translational registration.
        """
        if reference_capture == self:
            return self

        reference_frame = reference_capture.mean_frame()
        reference_gray = cv2.normalize(reference_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
        reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)

        aligned_frames = []

        print("aligning capture...")
        for frame in self.frames:
            print("aligning frame...")
            frame_gray = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
            frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

            _, reference_mask = cv2.threshold(reference_gray, 1, 255, cv2.THRESH_BINARY)
            _, frame_mask = cv2.threshold(frame_gray, 1, 255, cv2.THRESH_BINARY)

            shift, _ = cv2.phaseCorrelate(reference_mask.astype(np.float32), frame_mask.astype(np.float32))
            if not np.isfinite(shift).all():
                aligned_frames.append(frame)
                continue

            tx = int(round(-shift[0]))
            ty = int(round(-shift[1]))
            matrix = np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)]], dtype=np.float32)
            aligned_frame = cv2.warpAffine(
                frame,
                matrix,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            aligned_frames.append(aligned_frame)
            print("frame aligned")

        print("capture aligned")
        return ExperimentalCapture(aligned_frames, self.rel_pressure)
