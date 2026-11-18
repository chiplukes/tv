import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tifffile import imwrite

from tv.camera import camera_metadata, camera_readout, neutral_gains, preview_rgb


class CameraTests(unittest.TestCase):
    def test_metadata_overrides_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "img1_blend3.tif"
            imwrite(
                path,
                np.zeros((2, 2), dtype=np.uint16),
                description=json.dumps({"black_level": 30, "white_level": 55000, "wb_applied": [2.73, 1, 1.41]}),
            )
            record = camera_metadata(path)
            self.assertEqual(record["black_level"], 30)
            self.assertEqual(record["white_level"], 55000)
            self.assertTrue(record["recognized"])

    def test_neutral_patch_and_readout(self):
        planes = [np.array([value, 65535, 0]) for value in (1000, 2000, 2000, 4000)]
        np.testing.assert_allclose(neutral_gains(planes, 0, 65535), [2, 1, 0.5])
        with self.assertRaises(ValueError):
            neutral_gains([np.zeros(4)] * 4, 24, 55250)
        counts, stops = camera_readout(104, 24, 184, 16)
        self.assertEqual(counts, 5)
        self.assertEqual(stops, 1)
        self.assertTrue(np.isinf(camera_readout(24, 24, 184, 16)[1]))

    def test_srgb_and_channel_order(self):
        planes = [np.full((2, 2), value, dtype=float) for value in (0.18, 0.18, 0.18, 0.18)]
        rgb = preview_rgb(planes, 0, 1, [1, 1, 1], 0, np.eye(3), 1, False)
        np.testing.assert_array_equal(rgb, np.full((2, 2, 3), 118))
        planes = [np.full((1, 1), value) for value in (1.0, 0.0, 0.0, 0.0)]
        np.testing.assert_array_equal(preview_rgb(planes, 0, 1, [1, 1, 1], 0, np.eye(3), 1, False), [[[255, 0, 0]]])

    def test_shoulder_preserves_headroom_and_neutrals(self):
        values = np.array([[0.0, 0.01, 0.05, 0.2, 1.0]])
        planes = [values] * 4
        rgb = preview_rgb(planes, 0, 1, [1, 1, 1], 4, np.eye(3), 1.4, True)
        self.assertTrue(np.all(np.diff(rgb[0, :, 0].astype(int)) > 0))
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
        self.assertEqual(rgb[0, -1, 0], 255)
