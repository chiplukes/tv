import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tifffile import imwrite

from tv.camera import camera_metadata, camera_readout, neutral_gains, preview_rgb
from tv.viewer import ZoomableCfaViewer


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

    def test_only_preset_white_balance_is_used_on_open(self):
        interpreter = tk.Tcl()
        viewer = ZoomableCfaViewer.__new__(ZoomableCfaViewer)
        viewer.filename = "img1_blend3.tif"
        viewer.camera_fields = {
            name: tk.StringVar(master=interpreter) for name in ("Black", "White", "WB R", "WB G", "WB B", "EV")
        }
        viewer.camera_enabled = tk.BooleanVar(master=interpreter)
        viewer.BAYER_PATTERNS = {}
        calls = []
        viewer._auto_camera = lambda **kwargs: calls.append(kwargs)
        metadata = {
            "black_level": 24,
            "white_level": 55250,
            "recognized": True,
            "wb_already_applied": [2.73, 1, 1.41],
            "channel_gains_already_applied": [4, 2, 3],
            "wb_applied": [7, 1, 9],
            "channel_gains_applied": [8, 1, 6],
        }
        for preset, expected in (
            ({}, [1, 1, 1]),
            ({"wb": [1, 1, 1], "fwd-matrix": np.eye(3).tolist()}, [1, 1, 1]),
            ({"wb": [4, 2, 3]}, [2, 1, 1.5]),
        ):
            with patch("tv.viewer.camera_metadata", return_value={**metadata, "color_matrix": preset}):
                viewer._load_camera_defaults()
            actual = [float(viewer.camera_fields[name].get()) for name in ("WB R", "WB G", "WB B")]
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_array_equal(viewer.camera_matrix, np.eye(3))
            self.assertEqual(calls[-1], {"balance": False, "exposure": True})

    def test_auto_buttons_change_only_their_own_fields(self):
        interpreter = tk.Tcl()
        viewer = ZoomableCfaViewer.__new__(ZoomableCfaViewer)
        values = {"Black": "0", "White": "10000", "WB R": "1.2", "WB G": "1", "WB B": "1.3", "EV": "5"}
        viewer.camera_fields = {name: tk.StringVar(master=interpreter, value=value) for name, value in values.items()}
        viewer.full_rows = viewer.full_cols = 32
        viewer.data = np.empty((32, 32))
        viewer.data[0::2, 0::2] = 500
        viewer.data[0::2, 1::2] = 1000
        viewer.data[1::2, 0::2] = 1000
        viewer.data[1::2, 1::2] = 2000
        viewer.get_bayer_offsets = lambda: ((0, 0), (0, 1), (1, 0), (1, 1))
        viewer.canvas = None
        viewer.redraw = lambda: None
        viewer._auto_exposure()
        self.assertAlmostEqual(float(viewer.camera_fields["EV"].get()), np.log2(1.8), places=4)
        for name in ("WB R", "WB G", "WB B"):
            self.assertEqual(viewer.camera_fields[name].get(), values[name])
        viewer.camera_fields["EV"].set("3.25")
        viewer._auto_white_balance()
        self.assertEqual(viewer.camera_fields["EV"].get(), "3.25")
        np.testing.assert_allclose(
            [float(viewer.camera_fields[n].get()) for n in ("WB R", "WB G", "WB B")], [2, 1, 0.5]
        )

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
