"""Headless regression tests for region and temporal analysis."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tifffile import imwrite

from tv.plotting import parse_regions, temporal_files
from tv.viewer import ZoomableCfaViewer


class PlottingTests(unittest.TestCase):
    def viewer(self):
        viewer = ZoomableCfaViewer.__new__(ZoomableCfaViewer)
        viewer.bit_msb, viewer.bit_lsb = 15, 0
        viewer.full_rows, viewer.full_cols = 6, 8
        viewer.dark_image_subtract_enabled = False
        viewer.dark_image_data = None
        viewer.row_correction_enabled = False
        viewer.cfa_mode = "Color"
        viewer.show_bayer_r = viewer.show_bayer_gr = viewer.show_bayer_gb = viewer.show_bayer_b = True
        viewer.BAYER_PATTERN = {(0, 0): "R", (0, 1): "Gr", (1, 0): "Gb", (1, 1): "B"}
        return viewer

    def test_regions_and_absolute_bayer_parity(self):
        regions = parse_regions('1,1,6,5,"b-"\n0,0,8,6,r.', (6, 8))
        viewer = self.viewer()
        data = np.arange(48).reshape(6, 8)
        np.testing.assert_array_equal(viewer._region_values(data, regions[0], "R"), data[2:5:2, 2:6:2].ravel())
        for text in ("0,0,9,6,b-", "2,0,1,6,b-", "0,0,8,6,invalid", ""):
            with self.assertRaises(ValueError):
                parse_regions(text, (6, 8))

    def test_signed_dark_and_per_frame_row_correction(self):
        viewer = self.viewer()
        viewer.dark_image_subtract_enabled = True
        viewer.dark_image_data = np.full((6, 8), 10)
        raw = np.full((6, 8), 5, dtype=np.uint16)
        np.testing.assert_array_equal(viewer._analysis_data(raw), np.full((6, 8), -5))
        viewer.row_correction_enabled = True
        viewer.row_corr_offset, viewer.row_corr_width = 0, 1
        raw[:, 1:7] = 3
        expected = np.zeros((6, 8))
        expected[:, 1:7] = -2
        np.testing.assert_array_equal(viewer._analysis_data(raw), expected)
        np.testing.assert_array_equal(viewer._analysis_data(raw + 20), expected)

    def test_temporal_units_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame_0.06ms.tif", "frame_100us.tif", "frame_0.001s.tiff", "other_2ms.tif", "frame_5.tif"):
                (root / name).touch()
            files, label = temporal_files(root / "frame_0.06ms.tif")
            np.testing.assert_allclose([t for t, _ in files], [0.06, 0.1, 1])
            self.assertEqual(label, "Time (ms)")

    def test_temporal_modes_fits_and_histogram_render(self):
        viewer = self.viewer()
        regions = [(1, 1, 7, 6, "b-"), (0, 0, 4, 4, "r.")]
        viewer._plot_regions = lambda: regions
        figures = []
        viewer._display_histogram = figures.append
        viewer.hist_sigma = 3
        viewer._on_hist_sigma_change = lambda: None
        with tempfile.TemporaryDirectory() as directory:
            for t in (1, 3, 8):
                imwrite(Path(directory) / f"frame_{t}ms.tif", np.full((6, 8), 2 * t + 7, dtype=np.uint16))
            viewer.filename = str(Path(directory) / "frame_1ms.tif")
            with patch("tv.viewer.messagebox.showerror", side_effect=AssertionError("Unexpected plot error")):
                for mode in ("Pixels", "Average", "Heatmap"):
                    viewer.temporal_mode_var = SimpleNamespace(get=lambda: mode)
                    viewer.on_temporal_scatterplot()
                    fig = figures[-1]
                    FigureCanvasAgg(fig).draw()
                    for ax in fig.axes[:4]:
                        fits = [line for line in ax.lines if "fit:" in line.get_label()]
                        self.assertEqual(len(fits), 2)
                        for line in fits:
                            np.testing.assert_allclose(line.get_ydata(), [9, 23])
                viewer._raw_full_bits = np.full((6, 8), 12, dtype=np.uint16)
                for merged in (False, True):
                    viewer.hist_merge_var = SimpleNamespace(get=lambda: merged)
                    viewer.on_histogram()
                    self.assertEqual(len(figures[-1].axes), 4 if merged else 8)
                    FigureCanvasAgg(figures[-1]).draw()


if __name__ == "__main__":
    unittest.main()
