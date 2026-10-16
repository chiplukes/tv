"""Headless regression tests for region and temporal analysis."""

import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tifffile import imwrite

from tv.plotting import TiffRegionReader, collect_temporal_samples, parse_regions, temporal_files
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
        viewer.temporal_start_var = SimpleNamespace(get=lambda: "")
        viewer.temporal_end_var = SimpleNamespace(get=lambda: "")
        viewer._run_plot_job = lambda load, done: done(load(None, threading.Event()))
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

    def test_temporal_range_uses_inclusive_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for time in (1, 2, 10, 20):
                (root / f"frame_{time}ms.tif").touch()
            current = root / "frame_1ms.tif"
            files, _ = temporal_files(current, "frame_2ms.tif", "frame_10ms.tif")
            self.assertEqual([time for time, _ in files], [2, 10])
            files, _ = temporal_files(current, "", "frame_10ms.tif")
            self.assertEqual([time for time, _ in files], [1, 2, 10])
            files, _ = temporal_files(current, "frame_10ms.tif", "")
            self.assertEqual([time for time, _ in files], [10, 20])
            for start, end in (
                ("missing.tif", ""),
                ("", "missing.tif"),
                ("frame_10ms.tif", "frame_2ms.tif"),
                ("frame_2ms.tif", "frame_2ms.tif"),
            ):
                with self.assertRaises(ValueError):
                    temporal_files(current, start, end)

    def test_temporal_range_autofill_preserves_edits_within_series(self):
        interpreter = tk.Tcl()

        viewer = self.viewer()
        viewer.temporal_start_var = tk.StringVar(master=interpreter)
        viewer.temporal_end_var = tk.StringVar(master=interpreter)
        viewer._temporal_range_key = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("frame_1ms.tif", "frame_2ms.tif", "frame_10ms.tif", "other_3ms.tif", "other_8ms.tif"):
                (root / name).touch()
            viewer.filename = str(root / "frame_1ms.tif")
            viewer._refresh_temporal_range()
            self.assertEqual(viewer.temporal_start_var.get(), "frame_1ms.tif")
            self.assertEqual(viewer.temporal_end_var.get(), "frame_10ms.tif")
            viewer.temporal_start_var.set("frame_2ms.tif")
            viewer.filename = str(root / "frame_2ms.tif")
            viewer._refresh_temporal_range()
            self.assertEqual(viewer.temporal_start_var.get(), "frame_2ms.tif")
            viewer._refresh_temporal_range(force=True)
            self.assertEqual(viewer.temporal_start_var.get(), "frame_1ms.tif")
            viewer.filename = str(root / "other_3ms.tif")
            viewer._refresh_temporal_range()
            self.assertEqual(viewer.temporal_start_var.get(), "other_3ms.tif")
            self.assertEqual(viewer.temporal_end_var.get(), "other_8ms.tif")
            viewer.filename = None
            viewer._refresh_temporal_range()
            self.assertEqual(viewer.temporal_start_var.get(), "")
            self.assertEqual(viewer.temporal_end_var.get(), "")

    def test_region_reader_mapped_stripped_and_tiled(self):
        data = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)
        rectangles = [(3, 5, 19, 25), (14, 17, 35, 36), (70, 60, 80, 64)]
        with tempfile.TemporaryDirectory() as directory:
            for kind, options in (
                ("mapped", {}),
                ("stripped", {"compression": "deflate", "rowsperstrip": 8}),
                ("tiled", {"compression": "deflate", "tile": (16, 16)}),
            ):
                path = Path(directory) / f"{kind}.tif"
                imwrite(path, data, **options)
                reader = TiffRegionReader(path, data.shape)
                try:
                    self.assertEqual(reader.mapping is not None, kind == "mapped")
                    if reader.mapping is None:
                        decode = reader.page.decode
                        calls = []

                        def counted(encoded, index, **kwargs):
                            calls.append(index)
                            return decode(encoded, index, **kwargs)

                        reader.page.__dict__["decode"] = counted
                        with patch.object(type(reader.page), "asarray", side_effect=AssertionError("full image read")):
                            actual = reader.read_many(rectangles)
                        self.assertGreater(len(calls), 0)
                        self.assertEqual(len(calls), len(set(calls)))
                        self.assertLess(len(calls), len(reader.page.dataoffsets))
                    else:
                        actual = reader.read_many(rectangles)
                    for rect, result in zip(rectangles, actual):
                        x0, y0, x1, y1 = rect
                        np.testing.assert_array_equal(result, data[y0:y1, x0:x1])
                finally:
                    reader.close()

    def test_large_sparse_tiff_reads_only_small_regions(self):
        from tifffile import memmap

        # A 600 MB logical image, sparsely allocated so this test stays cheap.
        shape = (15000, 20000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large_1ms.tif"
            raw = memmap(path, shape=shape, dtype="uint16", photometric="minisblack")
            raw[100:120, 200:230] = 1234
            raw.flush()
            raw._mmap.close()
            regions = [(200, 100, 230, 120, "b-")]
            samples = collect_temporal_samples(
                [(i, path) for i in range(50)], regions, ["Mono"], {}, shape, 15, 0, None, None, "Average"
            )
            self.assertEqual(len(samples[0][0]), 50)
            np.testing.assert_array_equal(np.concatenate(samples[0][0]), np.full(50, 1234))

    def test_roi_calibration_matches_full_frame_pipeline(self):
        viewer = self.viewer()
        viewer.bit_msb, viewer.bit_lsb = 11, 2
        viewer.dark_image_subtract_enabled = True
        viewer.dark_image_data = np.arange(48).reshape(6, 8) % 7
        viewer.row_correction_enabled = True
        viewer.row_corr_offset, viewer.row_corr_width = 0, 1
        regions = [(1, 1, 7, 6, "b-"), (2, 2, 5, 5, "r-")]
        channels = ["R", "Gr", "Gb", "B", "Mono"]
        raw = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 12) + 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_1ms.tif"
            for options in ({}, {"compression": "deflate", "rowsperstrip": 2}):
                imwrite(path, raw, **options)
                for mode in ("Pixels", "Average", "Heatmap"):
                    progress = []
                    samples = collect_temporal_samples(
                        [(1, path)],
                        regions,
                        channels,
                        viewer.BAYER_PATTERN,
                        raw.shape,
                        viewer.bit_msb,
                        viewer.bit_lsb,
                        viewer.dark_image_data,
                        (0, 1),
                        mode,
                        lambda n, total: progress.append((n, total)),
                    )
                    expected = viewer._analysis_data(raw)
                    for ri, region in enumerate(regions):
                        for ci, channel in enumerate(channels):
                            values = viewer._region_values(expected, region, channel)
                            if mode == "Average":
                                values = [values.mean()]
                            np.testing.assert_allclose(samples[ri][ci][0], values)
                    self.assertEqual(progress, [(1, 1)])
            cancelled = threading.Event()
            cancelled.set()
            self.assertIsNone(
                collect_temporal_samples(
                    [(1, path)],
                    regions,
                    channels,
                    viewer.BAYER_PATTERN,
                    raw.shape,
                    15,
                    0,
                    None,
                    None,
                    "Pixels",
                    cancelled=cancelled,
                )
            )

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
