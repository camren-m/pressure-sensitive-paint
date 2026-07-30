import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "software"))

from core import frames


class LoadFromCineTests(unittest.TestCase):
    def test_load_from_cine_reads_only_trimmed_frame_range(self):
        seen_calls = []

        class Setup:
            FrameRate = 60

        class CineHeader:
            ImageCount = 10

        def fake_read_frames(path, start_frame=0, count=None):
            seen_calls.append((start_frame, count))
            frames_to_return = [f"frame-{i + start_frame}" for i in range(count or 0)]
            return frames_to_return, None, None

        with patch.object(frames, "cine") as cine_mock:
            cine_mock.read_header.return_value = {
                "setup": Setup(),
                "cinefileheader": CineHeader(),
            }
            cine_mock.read_frames.side_effect = fake_read_frames

            result = frames.load_from_cine(Path("sample.cine"), trim_range=(2, 6), output_fps=60)

        self.assertEqual(seen_calls, [(2, 4)])
        self.assertEqual(result, ["frame-2", "frame-3", "frame-4", "frame-5"])


if __name__ == "__main__":
    unittest.main()
