"""CFA / Bayer-pattern TIFF viewer widgets."""

import io
import math
import os
import tkinter as tk
from tkinter import filedialog, ttk
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tifffile import imread


class ZoomableCfaViewer(tk.Tk):
    """
    A Tkinter GUI to visualize a large 2D NumPy array (CFA).

    - When zoomed out, displays a downsampled grayscale image (like 'imshow').
    - When zoomed in, displays the color-coded Bayer component grid.
    - Supports zooming (mouse wheel) and panning (click-drag).
    - Inspects pixel values on hover.
    """

    def __init__(self, cfa_data, filename=None):
        """
        Initializes the ZoomableCfaViewer.

        Args:
            cfa_data (np.ndarray): The FULL 2D NumPy array (e.g., 18k x 18k).
            filename (str): Optional filename to display in the title.
        """
        super().__init__()

        self.filename = filename
        if filename:
            basename = os.path.basename(filename)
            self.title(f"tv - {basename}")
        else:
            self.title("tv")

        self.geometry("800x600+50+50")

        # Bit-range rescaling: many 12-bit sensors are stored in TIFFs with
        # the value left-shifted into the upper bits of a 16-bit pixel so
        # the file looks correct in a typical image editor. `_raw_full_bits`
        # keeps the untouched loaded data; `data_raw` is the bit-extracted
        # view used by every downstream computation in this class.
        self._raw_full_bits = cfa_data
        self._dark_full_bits = None
        self.bit_msb = 15
        self.bit_lsb = 0

        self.data_raw = self._raw_full_bits
        self.data = self.data_raw  # active data — swapped when dark image subtraction is toggled

        if self.data.ndim != 2:
            raise ValueError(f"Input array must be 2D, but got {self.data.ndim} dimensions.")

        self.full_rows, self.full_cols = self.data.shape

        print("Calculating data range for intensity scaling...")
        self.data_min = float(self.data.min())
        self.data_max = float(self.data.max())
        print(f"Data range: {self.data_min} to {self.data_max}")

        # --- Viewport State ---
        self.view_x = 0.0
        self.view_y = 0.0
        self.zoom = 0.1
        self.gamma = 2.2
        self.cfa_mode = "Color"
        self.show_bayer_pixels = False
        self.bayer_pattern_type = "RGGB"

        # Bayer channel visibility (per Bayer component)
        self.show_bayer_r = True
        self.show_bayer_gr = True
        self.show_bayer_gb = True
        self.show_bayer_b = True

        # Color adjustment multipliers (per Bayer channel)
        self.color_adjust_r = 1.0
        self.color_adjust_gr = 1.0
        self.color_adjust_gb = 1.0
        self.color_adjust_b = 1.0

        # Row correction settings (dark columns on left and right edges)
        self.row_correction_enabled = False
        self.row_corr_offset = 64
        self.row_corr_width = 32
        self.row_corr_averages = None

        # Dark calibration image subtraction
        self.dark_image_data = None  # 2-D array loaded from a dark TIFF
        self.dark_image_subtract_enabled = False
        self.dark_image_path = None

        # Histogram settings
        self.hist_sigma = 3.0  # x-axis range in standard deviations

        # Flat-field analysis mode
        self.flat_mode_enabled = False
        self.flat_block_size = 6
        self.flat_deviation_range = 0.15  # +/- 15% of median shown as full-scale colour
        self.flat_display_mode = "Percentage"  # "Percentage" or "Absolute"
        self.flat_abs_deviation_range = 500.0  # +/- DN shown as full-scale colour in Absolute mode

        # Cached flat-field deviation data (populated by _render_flat)
        self._flat_deviation = None  # 2-D deviation array (block-level)
        self._flat_params = None  # dict with mapping info for hover lookup

        # Track the actual displayed image region (for letterboxing)
        self.display_x = 0
        self.display_y = 0
        self.display_w = 0
        self.display_h = 0

        # --- Panning State ---
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_view_x = 0
        self.pan_start_view_y = 0

        # --- Zoom Rectangle State ---
        self.zoom_rect_start = None
        self.zoom_rect_id = None
        self.zoom_rect_active = False

        # --- Histogram Rectangle State ---
        self.hist_rect_start = None
        self.hist_rect_id = None
        self.hist_rect_active = False

        # --- Resize State ---
        self.last_canvas_w = 0
        self.last_canvas_h = 0

        # --- Pixel Marking State ---
        self.marked_pixels = set()  # set of (x, y) raw array coords marked for annotation
        self.mark_min_pixel_size = 8  # minimum on-screen pixel size (px) before mark overlays are drawn
        self.GOTO_MIN_ZOOM = 16.0  # zoom level "Goto Pixel" zooms in to (unless already zoomed in further)

        # --- Bayer Pattern Definitions ---
        self.BAYER_PATTERNS = {
            "RGGB": {(0, 0): "R", (0, 1): "Gr", (1, 0): "Gb", (1, 1): "B"},
            "BGGR": {(0, 0): "B", (0, 1): "Gb", (1, 0): "Gr", (1, 1): "R"},
            "GRBG": {(0, 0): "Gr", (0, 1): "R", (1, 0): "B", (1, 1): "Gb"},
            "GBRG": {(0, 0): "Gb", (0, 1): "B", (1, 0): "R", (1, 1): "Gr"},
        }
        self.BAYER_PATTERN = self.BAYER_PATTERNS[self.bayer_pattern_type]

        # --- Create GUI Widgets ---
        self._build_info_panel()
        self._build_gamma_panel()
        self._build_bitdepth_panel()
        self._build_dark_panel()
        self._build_dark_image_panel()
        self._build_color_panel()
        self._build_flat_panel()
        self._build_goto_panel()
        self._build_debug_panel()
        self._build_canvas()

        self._update_bit_depth_label()

        # Set initial zoom to fit image to window
        self.zoom_to_fit()

    # ------------------------------------------------------------------
    # GUI construction helpers
    # ------------------------------------------------------------------

    def _build_info_panel(self):
        info_frame = ttk.Frame(self, padding="10 10 10 5", relief=tk.RIDGE)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)

        self.coord_label = ttk.Label(info_frame, text="Coords: (N/A)", width=25, font=("Courier", 12))
        self.coord_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.component_label = ttk.Label(info_frame, text="Component: N/A", width=18, font=("Courier", 12))
        self.component_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.value_label = ttk.Label(info_frame, text="Value: N/A", width=18, font=("Courier", 12))
        self.value_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.row_avg_label = ttk.Label(info_frame, text="Row Avg: N/A", width=18, font=("Courier", 12))
        self.row_avg_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.flat_dev_label = ttk.Label(info_frame, text="", width=18, font=("Courier", 12))
        self.flat_dev_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _build_gamma_panel(self):
        gamma_frame = ttk.Frame(self, padding="5 5 5 5")
        gamma_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(gamma_frame, text="Gamma:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 5))

        self.gamma_var = tk.DoubleVar(value=self.gamma)
        self.gamma_entry = ttk.Entry(gamma_frame, textvariable=self.gamma_var, width=6, font=("Courier", 10))
        self.gamma_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.gamma_entry.bind("<Return>", self.on_gamma_change)
        self.gamma_entry.bind("<FocusOut>", self.on_gamma_change)

        ttk.Label(gamma_frame, text="CFA Mode:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))

        self.cfa_mode_var = tk.StringVar(value=self.cfa_mode)
        self.cfa_mode_combo = ttk.Combobox(
            gamma_frame,
            textvariable=self.cfa_mode_var,
            values=["Color", "Mono"],
            width=8,
            font=("Courier", 10),
            state="readonly",
        )
        self.cfa_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.cfa_mode_combo.bind("<<ComboboxSelected>>", self.on_cfa_mode_change)

        ttk.Label(gamma_frame, text="Bayer:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))

        self.bayer_pattern_var = tk.StringVar(value=self.bayer_pattern_type)
        self.bayer_pattern_combo = ttk.Combobox(
            gamma_frame,
            textvariable=self.bayer_pattern_var,
            values=["RGGB", "BGGR", "GRBG", "GBRG"],
            width=6,
            font=("Courier", 10),
            state="readonly",
        )
        self.bayer_pattern_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.bayer_pattern_combo.bind("<<ComboboxSelected>>", self.on_bayer_pattern_change)

        self.show_bayer_pixels_var = tk.BooleanVar(value=self.show_bayer_pixels)
        self.show_bayer_pixels_check = ttk.Checkbutton(
            gamma_frame, text="Show CFA", variable=self.show_bayer_pixels_var, command=self.on_show_bayer_pixels_toggle
        )
        self.show_bayer_pixels_check.pack(side=tk.LEFT, padx=(20, 10))

        self.open_file_button = ttk.Button(gamma_frame, text="Open File…", command=self.on_open_file)
        self.open_file_button.pack(side=tk.LEFT, padx=(20, 5))

        self.reload_button = ttk.Button(gamma_frame, text="Reload", command=self.on_reload_file)
        self.reload_button.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_button = ttk.Button(gamma_frame, text="Reset View", command=self.on_reset_view)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            gamma_frame,
            text="Drag=pan | Shift+Drag=zoom | Ctrl+Drag=histogram | Scroll=zoom | Right-click=mark pixel",
            font=("Arial", 9),
            foreground="#666",
        ).pack(side=tk.LEFT, padx=10)

    def _build_bitdepth_panel(self):
        bitdepth_frame = ttk.Frame(self, padding="5 5 5 5")
        bitdepth_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(bitdepth_frame, text="Bit Range:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 10))

        ttk.Label(bitdepth_frame, text="MSB:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.bit_msb_var = tk.IntVar(value=self.bit_msb)
        self.bit_msb_entry = ttk.Entry(bitdepth_frame, textvariable=self.bit_msb_var, width=4, font=("Courier", 10))
        self.bit_msb_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.bit_msb_entry.bind("<Return>", self.on_bit_range_change)
        self.bit_msb_entry.bind("<FocusOut>", self.on_bit_range_change)

        ttk.Label(bitdepth_frame, text="LSB:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.bit_lsb_var = tk.IntVar(value=self.bit_lsb)
        self.bit_lsb_entry = ttk.Entry(bitdepth_frame, textvariable=self.bit_lsb_var, width=4, font=("Courier", 10))
        self.bit_lsb_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.bit_lsb_entry.bind("<Return>", self.on_bit_range_change)
        self.bit_lsb_entry.bind("<FocusOut>", self.on_bit_range_change)

        self.bit_autodetect_button = ttk.Button(
            bitdepth_frame, text="Auto Detect", command=self.on_auto_detect_bit_range
        )
        self.bit_autodetect_button.pack(side=tk.LEFT, padx=(0, 10))

        self.bit_depth_label = ttk.Label(bitdepth_frame, text="", font=("Courier", 9), foreground="#666")
        self.bit_depth_label.pack(side=tk.LEFT, padx=(10, 5))

    def _extract_bit_range(self, arr):
        """Rescale `arr` so bits [bit_lsb, bit_msb] occupy the full output range."""
        width = self.bit_msb - self.bit_lsb + 1
        mask = (1 << width) - 1
        return ((arr.astype(np.uint32) >> self.bit_lsb) & mask).astype(np.uint16)

    @staticmethod
    def _detect_bit_range(data):
        """Detect (msb, lsb) by finding the bits that are ever set across `data`.

        Bits that are always zero (e.g. the unused low bits when a 12-bit
        sensor value is left-shifted into a 16-bit pixel) are excluded.
        """
        any_bits = int(np.bitwise_or.reduce(data.ravel()))
        if any_bits == 0:
            return 15, 0
        msb = any_bits.bit_length() - 1
        lsb = (any_bits & -any_bits).bit_length() - 1
        return msb, lsb

    def _update_bit_depth_label(self):
        width = self.bit_msb - self.bit_lsb + 1
        max_val = (1 << width) - 1
        self.bit_depth_label.config(text=f"{width}-bit (0-{max_val})")

    def _apply_bit_range(self):
        """Recompute data_raw / dark_image_data from the unshifted source data."""
        self.data_raw = self._extract_bit_range(self._raw_full_bits)
        if self._dark_full_bits is not None:
            self.dark_image_data = self._extract_bit_range(self._dark_full_bits)

        self._recalculate_active_data()
        self._update_bit_depth_label()
        self.redraw()

    def on_bit_range_change(self, event=None):
        try:
            msb = int(self.bit_msb_var.get())
            lsb = int(self.bit_lsb_var.get())
            if 0 <= lsb <= msb <= 15:
                self.bit_msb = msb
                self.bit_lsb = lsb
                self._apply_bit_range()
            else:
                raise ValueError
        except (ValueError, tk.TclError):
            self.bit_msb_var.set(self.bit_msb)
            self.bit_lsb_var.set(self.bit_lsb)

    def on_auto_detect_bit_range(self):
        msb, lsb = self._detect_bit_range(self._raw_full_bits)
        self.bit_msb = msb
        self.bit_lsb = lsb
        self.bit_msb_var.set(msb)
        self.bit_lsb_var.set(lsb)
        self._apply_bit_range()

    def _build_dark_panel(self):
        dark_frame = ttk.Frame(self, padding="5 5 5 5")
        dark_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.row_correction_var = tk.BooleanVar(value=self.row_correction_enabled)
        self.row_correction_check = ttk.Checkbutton(
            dark_frame,
            text="Row Correction",
            variable=self.row_correction_var,
            command=self.on_row_correction_toggle,
        )
        self.row_correction_check.pack(side=tk.LEFT, padx=(10, 10))

        ttk.Label(dark_frame, text="Offset:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))

        self.row_corr_offset_var = tk.IntVar(value=self.row_corr_offset)
        self.row_corr_offset_entry = ttk.Entry(
            dark_frame, textvariable=self.row_corr_offset_var, width=6, font=("Courier", 10)
        )
        self.row_corr_offset_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.row_corr_offset_entry.bind("<Return>", self.on_row_corr_params_change)
        self.row_corr_offset_entry.bind("<FocusOut>", self.on_row_corr_params_change)

        ttk.Label(dark_frame, text="Width:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 5))

        self.row_corr_width_var = tk.IntVar(value=self.row_corr_width)
        self.row_corr_width_entry = ttk.Entry(
            dark_frame, textvariable=self.row_corr_width_var, width=6, font=("Courier", 10)
        )
        self.row_corr_width_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.row_corr_width_entry.bind("<Return>", self.on_row_corr_params_change)
        self.row_corr_width_entry.bind("<FocusOut>", self.on_row_corr_params_change)

        ttk.Label(dark_frame, text="Bayer Channels:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))

        self.bayer_r_var = tk.BooleanVar(value=self.show_bayer_r)
        ttk.Checkbutton(dark_frame, text="R", variable=self.bayer_r_var, command=self.on_bayer_channel_toggle).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        self.bayer_gr_var = tk.BooleanVar(value=self.show_bayer_gr)
        ttk.Checkbutton(dark_frame, text="Gr", variable=self.bayer_gr_var, command=self.on_bayer_channel_toggle).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        self.bayer_gb_var = tk.BooleanVar(value=self.show_bayer_gb)
        ttk.Checkbutton(dark_frame, text="Gb", variable=self.bayer_gb_var, command=self.on_bayer_channel_toggle).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        self.bayer_b_var = tk.BooleanVar(value=self.show_bayer_b)
        ttk.Checkbutton(dark_frame, text="B", variable=self.bayer_b_var, command=self.on_bayer_channel_toggle).pack(
            side=tk.LEFT, padx=(0, 10)
        )

    def _build_dark_image_panel(self):
        di_frame = ttk.Frame(self, padding="5 5 5 5")
        di_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.dark_image_subtract_var = tk.BooleanVar(value=self.dark_image_subtract_enabled)
        self.dark_image_subtract_check = ttk.Checkbutton(
            di_frame,
            text="Subtract Dark Image",
            variable=self.dark_image_subtract_var,
            command=self.on_dark_image_subtract_toggle,
            state=tk.DISABLED,
        )
        self.dark_image_subtract_check.pack(side=tk.LEFT, padx=(10, 10))

        self.load_dark_image_button = ttk.Button(di_frame, text="Load Dark Image…", command=self.on_load_dark_image)
        self.load_dark_image_button.pack(side=tk.LEFT, padx=(10, 10))

        self.dark_image_label = ttk.Label(
            di_frame,
            text="No dark image loaded",
            font=("Courier", 9),
            foreground="#666",
        )
        self.dark_image_label.pack(side=tk.LEFT, padx=(10, 5))

        ttk.Separator(di_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=(15, 15),
            pady=2,
        )
        ttk.Label(
            di_frame,
            text="Hist \u03c3:",
            font=("Courier", 10),
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.hist_sigma_var = tk.DoubleVar(value=self.hist_sigma)
        self.hist_sigma_entry = ttk.Entry(
            di_frame,
            textvariable=self.hist_sigma_var,
            width=5,
            font=("Courier", 10),
        )
        self.hist_sigma_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.hist_sigma_entry.bind(
            "<Return>",
            self._on_hist_sigma_change,
        )
        self.hist_sigma_entry.bind(
            "<FocusOut>",
            self._on_hist_sigma_change,
        )

    def _recalculate_active_data(self):
        """Recompute self.data from self.data_raw and the dark image, then refresh derived state."""
        if self.dark_image_subtract_enabled and self.dark_image_data is not None:
            self.data = np.clip(
                self.data_raw.astype(np.int32) - self.dark_image_data.astype(np.int32), 0, 65535
            ).astype(np.uint16)
        else:
            self.data = self.data_raw

        self.data_min = float(self.data.min())
        self.data_max = float(self.data.max())

        # Recompute row correction averages if active
        if self.row_correction_enabled:
            self.calculate_row_corr_averages()

    def _build_color_panel(self):
        color_frame = ttk.Frame(self, padding="5 5 5 5")
        color_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(color_frame, text="Color Adjust:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 10))

        ttk.Label(color_frame, text="R:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.color_adjust_r_var = tk.DoubleVar(value=self.color_adjust_r)
        self.color_adjust_r_entry = ttk.Entry(
            color_frame, textvariable=self.color_adjust_r_var, width=6, font=("Courier", 10)
        )
        self.color_adjust_r_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.color_adjust_r_entry.bind("<Return>", self.on_color_adjust_change)
        self.color_adjust_r_entry.bind("<FocusOut>", self.on_color_adjust_change)

        ttk.Label(color_frame, text="Gr:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.color_adjust_gr_var = tk.DoubleVar(value=self.color_adjust_gr)
        self.color_adjust_gr_entry = ttk.Entry(
            color_frame, textvariable=self.color_adjust_gr_var, width=6, font=("Courier", 10)
        )
        self.color_adjust_gr_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.color_adjust_gr_entry.bind("<Return>", self.on_color_adjust_change)
        self.color_adjust_gr_entry.bind("<FocusOut>", self.on_color_adjust_change)

        ttk.Label(color_frame, text="Gb:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.color_adjust_gb_var = tk.DoubleVar(value=self.color_adjust_gb)
        self.color_adjust_gb_entry = ttk.Entry(
            color_frame, textvariable=self.color_adjust_gb_var, width=6, font=("Courier", 10)
        )
        self.color_adjust_gb_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.color_adjust_gb_entry.bind("<Return>", self.on_color_adjust_change)
        self.color_adjust_gb_entry.bind("<FocusOut>", self.on_color_adjust_change)

        ttk.Label(color_frame, text="B:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.color_adjust_b_var = tk.DoubleVar(value=self.color_adjust_b)
        self.color_adjust_b_entry = ttk.Entry(
            color_frame, textvariable=self.color_adjust_b_var, width=6, font=("Courier", 10)
        )
        self.color_adjust_b_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.color_adjust_b_entry.bind("<Return>", self.on_color_adjust_change)
        self.color_adjust_b_entry.bind("<FocusOut>", self.on_color_adjust_change)

        self.export_button = ttk.Button(color_frame, text="Export View", command=self.on_export_view)
        self.export_button.pack(side=tk.LEFT, padx=(20, 10))

    def _build_flat_panel(self):
        flat_frame = ttk.Frame(self, padding="5 5 5 5")
        flat_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.flat_mode_var = tk.BooleanVar(value=self.flat_mode_enabled)
        ttk.Checkbutton(flat_frame, text="Flat", variable=self.flat_mode_var, command=self.on_flat_mode_toggle).pack(
            side=tk.LEFT, padx=(10, 10)
        )

        ttk.Label(flat_frame, text="Block Size:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.flat_block_size_var = tk.IntVar(value=self.flat_block_size)
        self.flat_block_size_entry = ttk.Entry(
            flat_frame, textvariable=self.flat_block_size_var, width=6, font=("Courier", 10)
        )
        self.flat_block_size_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.flat_block_size_entry.bind("<Return>", self.on_flat_block_size_change)
        self.flat_block_size_entry.bind("<FocusOut>", self.on_flat_block_size_change)

        ttk.Label(flat_frame, text="Display:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.flat_display_mode_var = tk.StringVar(value=self.flat_display_mode)
        self.flat_display_mode_combo = ttk.Combobox(
            flat_frame,
            textvariable=self.flat_display_mode_var,
            values=["Percentage", "Absolute"],
            width=10,
            font=("Courier", 10),
            state="readonly",
        )
        self.flat_display_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.flat_display_mode_combo.bind("<<ComboboxSelected>>", self.on_flat_display_mode_change)

        ttk.Label(flat_frame, text="Dev %:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.flat_deviation_var = tk.DoubleVar(value=self.flat_deviation_range * 100)
        self.flat_deviation_entry = ttk.Entry(
            flat_frame, textvariable=self.flat_deviation_var, width=6, font=("Courier", 10)
        )
        self.flat_deviation_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.flat_deviation_entry.bind("<Return>", self.on_flat_deviation_change)
        self.flat_deviation_entry.bind("<FocusOut>", self.on_flat_deviation_change)

        ttk.Label(flat_frame, text="Dev DN:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 5))
        self.flat_abs_deviation_var = tk.DoubleVar(value=self.flat_abs_deviation_range)
        self.flat_abs_deviation_entry = ttk.Entry(
            flat_frame, textvariable=self.flat_abs_deviation_var, width=6, font=("Courier", 10)
        )
        self.flat_abs_deviation_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.flat_abs_deviation_entry.bind("<Return>", self.on_flat_abs_deviation_change)
        self.flat_abs_deviation_entry.bind("<FocusOut>", self.on_flat_abs_deviation_change)

        self.flat_info_label = ttk.Label(flat_frame, text="", font=("Courier", 9), foreground="#666")
        self.flat_info_label.pack(side=tk.LEFT, padx=(10, 5))

    def _build_goto_panel(self):
        goto_frame = ttk.Frame(self, padding="5 5 5 5")
        goto_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(goto_frame, text="Goto Pixel:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 10))

        ttk.Label(goto_frame, text="X:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.goto_x_var = tk.IntVar(value=0)
        self.goto_x_entry = ttk.Entry(goto_frame, textvariable=self.goto_x_var, width=8, font=("Courier", 10))
        self.goto_x_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.goto_x_entry.bind("<Return>", self.on_goto_pixel)

        ttk.Label(goto_frame, text="Y:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.goto_y_var = tk.IntVar(value=0)
        self.goto_y_entry = ttk.Entry(goto_frame, textvariable=self.goto_y_var, width=8, font=("Courier", 10))
        self.goto_y_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.goto_y_entry.bind("<Return>", self.on_goto_pixel)

        self.goto_button = ttk.Button(goto_frame, text="Go", command=self.on_goto_pixel)
        self.goto_button.pack(side=tk.LEFT, padx=(0, 10))

        self.goto_status_label = ttk.Label(goto_frame, text="", font=("Courier", 9), foreground="#666")
        self.goto_status_label.pack(side=tk.LEFT, padx=(10, 5))

    def on_goto_pixel(self, event=None):
        """Center the view on the typed (x, y) pixel, zooming in if needed to see it clearly."""
        try:
            x = int(self.goto_x_var.get())
            y = int(self.goto_y_var.get())
        except (ValueError, tk.TclError):
            self.goto_status_label.config(text="Invalid coordinates", foreground="red")
            return

        if not (0 <= x < self.full_cols and 0 <= y < self.full_rows):
            self.goto_status_label.config(
                text=f"Out of range (0-{self.full_cols - 1}, 0-{self.full_rows - 1})", foreground="red"
            )
            return

        # Zoom in enough to see individual pixels (matches the mark overlay's
        # own visibility threshold), but don't zoom OUT if already closer in.
        self.zoom = max(self.zoom, self.GOTO_MIN_ZOOM)
        self.view_x = x - (self.display_w / self.zoom) / 2
        self.view_y = y - (self.display_h / self.zoom) / 2
        self._clamp_view()
        self.redraw()
        self._refresh_hover()
        self.goto_status_label.config(text=f"Jumped to ({x}, {y})", foreground="#060")

    def _build_debug_panel(self):
        debug_frame = ttk.Frame(self, padding="5 5 5 5")
        debug_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        ttk.Label(debug_frame, text="Debug Dump:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(10, 10))

        ttk.Label(debug_frame, text="Row:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.debug_row_var = tk.IntVar(value=0)
        self.debug_row_entry = ttk.Entry(debug_frame, textvariable=self.debug_row_var, width=8, font=("Courier", 10))
        self.debug_row_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(debug_frame, text="Start:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.debug_start_var = tk.IntVar(value=0)
        self.debug_start_entry = ttk.Entry(
            debug_frame, textvariable=self.debug_start_var, width=8, font=("Courier", 10)
        )
        self.debug_start_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(debug_frame, text="End:", font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        self.debug_end_var = tk.IntVar(value=100)
        self.debug_end_entry = ttk.Entry(debug_frame, textvariable=self.debug_end_var, width=8, font=("Courier", 10))
        self.debug_end_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.debug_dump_button = ttk.Button(debug_frame, text="Dump", command=self.on_debug_dump)
        self.debug_dump_button.pack(side=tk.LEFT, padx=(10, 10))

    def on_debug_dump(self):
        """Dump pixel values for the selected row and column range to stdout."""
        try:
            row = int(self.debug_row_var.get())
            col_start = int(self.debug_start_var.get())
            col_end = int(self.debug_end_var.get())
        except (ValueError, tk.TclError):
            print("Debug Dump: invalid row/start/end values.")
            return

        if row < 0 or row >= self.full_rows:
            print(f"Debug Dump: row {row} out of range [0, {self.full_rows - 1}]")
            return
        col_start = max(0, col_start)
        col_end = min(self.full_cols, col_end)
        if col_start >= col_end:
            print(f"Debug Dump: start ({col_start}) must be less than end ({col_end})")
            return

        row_data = self.data[row, col_start:col_end].copy()

        # Determine which Bayer channels are on this row
        # Even row -> pattern positions (0,0) and (0,1); Odd row -> (1,0) and (1,1)
        row_parity = row % 2
        even_comp = self.BAYER_PATTERN[(row_parity, 0)]  # component at even columns
        odd_comp = self.BAYER_PATTERN[(row_parity, 1)]  # component at odd columns

        comp_enabled = {
            "R": self.show_bayer_r,
            "Gr": self.show_bayer_gr,
            "Gb": self.show_bayer_gb,
            "B": self.show_bayer_b,
        }
        even_enabled = comp_enabled.get(even_comp, True)
        odd_enabled = comp_enabled.get(odd_comp, True)

        # If both channels enabled or mono mode, dump everything
        if self.cfa_mode == "Mono" or (even_enabled and odd_enabled):
            label = "all"
            values = row_data
            cols = np.arange(col_start, col_end)
        elif even_enabled:
            # Keep only even-column pixels (relative to the image, not the slice)
            mask = np.arange(col_start, col_end) % 2 == 0
            values = row_data[mask]
            cols = np.arange(col_start, col_end)[mask]
            label = even_comp
        elif odd_enabled:
            mask = np.arange(col_start, col_end) % 2 == 1
            values = row_data[mask]
            cols = np.arange(col_start, col_end)[mask]
            label = odd_comp
        else:
            print("Debug Dump: no Bayer channels enabled for this row.")
            return

        print(f"\n--- Debug Dump: row={row}  cols=[{col_start}, {col_end})  channel={label} ---")
        print(f"Pixels ({len(values)}):")
        # Print col:value pairs
        pairs = [f"  {c}: {v}" for c, v in zip(cols, values)]
        print("\n".join(pairs))
        print(f"Sum:    {np.sum(values.astype(np.int64))}")
        print(f"Mean:   {np.mean(values.astype(np.float64)):.2f}")
        print(f"StdDev: {np.std(values.astype(np.float64)):.2f}")
        print("--- End Dump ---\n")

    def _build_canvas(self):
        canvas_frame = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(5, 10))

        self.canvas = tk.Canvas(canvas_frame, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image_tk = None

        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.on_hover)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.bind("<MouseWheel>", self.on_scroll)  # Windows/Mac
        self.bind("<Button-4>", self.on_scroll)  # Linux (scroll up)
        self.bind("<Button-5>", self.on_scroll)  # Linux (scroll down)

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_photoimage(img):
        """Convert a PIL Image to a tk.PhotoImage without ImageTk.

        Pillow's ImageTk.PhotoImage relies on the _imagingtk C extension
        which fails on some Linux installations (standalone Python builds,
        missing Tk dev headers, etc.).  Writing PPM bytes and feeding them
        to tk.PhotoImage bypasses that extension entirely and works on
        every platform.
        """
        buf = io.BytesIO()
        img.save(buf, format="PPM")
        return tk.PhotoImage(data=buf.getvalue())

    def zoom_to_fit(self):
        """Calculates initial zoom to fit the whole array in the canvas."""
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w == 1 or canvas_h == 1:
            self.after(50, self.zoom_to_fit)
            return

        zoom_w = canvas_w / self.full_cols
        zoom_h = canvas_h / self.full_rows
        self.zoom = min(zoom_w, zoom_h)

        self.view_x = 0
        self.view_y = 0
        self.redraw()

    def get_bayer_component(self, x, y):
        """Gets the Bayer component for a given (x, y) coord."""
        if self.cfa_mode == "Mono":
            return "Mono"
        return self.BAYER_PATTERN[(y % 2, x % 2)]

    def get_bayer_offsets(self):
        """Returns the (row, col) offsets for R, Gr, Gb, B components."""
        r_pos = gr_pos = gb_pos = b_pos = None
        for pos, color in self.BAYER_PATTERN.items():
            if color == "R":
                r_pos = pos
            elif color == "Gr":
                gr_pos = pos
            elif color == "Gb":
                gb_pos = pos
            elif color == "B":
                b_pos = pos
        return r_pos, gr_pos, gb_pos, b_pos

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def redraw(self):  # noqa: C901 – complex but cohesive rendering pipeline
        """Draws the visible portion of the array to the canvas."""
        self.canvas.delete("all")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        full_aspect = self.full_cols / self.full_rows
        canvas_aspect = canvas_w / canvas_h

        if canvas_aspect > full_aspect:
            display_h = canvas_h
            display_w = int(canvas_h * full_aspect)
        else:
            display_w = canvas_w
            display_h = int(canvas_w / full_aspect)

        self.display_x = (canvas_w - display_w) // 2
        self.display_y = (canvas_h - display_h) // 2
        self.display_w = display_w
        self.display_h = display_h

        arr_w_visible = display_w / self.zoom
        arr_h_visible = display_h / self.zoom

        if self.cfa_mode == "Color":
            arr_cols = int(self.view_x + arr_w_visible) - int(self.view_x)
            arr_rows = int(self.view_y + arr_h_visible) - int(self.view_y)
            downsample = max(1, int(max(arr_cols / display_w, arr_rows / display_h)))

            if downsample > 1:
                downsample = 2 ** max(1, int(np.log2(downsample)))

            if downsample >= 2:
                align = downsample * 2
                x_start = max(0, (int(self.view_x) // align) * align)
                y_start = max(0, (int(self.view_y) // align) * align)
            else:
                x_start = max(0, (int(self.view_x) // 2) * 2)
                y_start = max(0, (int(self.view_y) // 2) * 2)

            x_end = min(self.full_cols, x_start + int(np.ceil(arr_w_visible / downsample) * downsample))
            y_end = min(self.full_rows, y_start + int(np.ceil(arr_h_visible / downsample) * downsample))
        else:
            x_start = max(0, int(self.view_x))
            y_start = max(0, int(self.view_y))
            x_end = min(self.full_cols, int(self.view_x + arr_w_visible) + 1)
            y_end = min(self.full_rows, int(self.view_y + arr_h_visible) + 1)

            arr_cols = x_end - x_start
            arr_rows = y_end - y_start
            downsample = max(1, int(max(arr_cols / display_w, arr_rows / display_h)))

            if downsample > 1:
                downsample = 2 ** int(np.log2(downsample))

        data_slice = self.data[y_start:y_end:downsample, x_start:x_end:downsample]

        if data_slice.size == 0:
            return

        self._last_render_params = {
            "x_start": x_start,
            "y_start": y_start,
            "downsample": downsample,
            "slice_h": data_slice.shape[0],
            "slice_w": data_slice.shape[1],
        }

        # Exact raw-array <-> screen mapping for the image about to be drawn.
        # This must match whatever _render_image is about to produce: the
        # Color debayer-cell path (2x2 Bayer quads collapsed into one output
        # pixel) has a different raw-pixel-per-output-pixel scale than the
        # Mono / "Show CFA" raw-pixel paths, so on_hover/right-click/mark
        # overlays need to know which one is in play to land on the right
        # pixel instead of drifting with zoom/pan.
        is_cell_mode = not self.flat_mode_enabled and self.cfa_mode == "Color" and not self.show_bayer_pixels
        if is_cell_mode:
            cell_step = downsample * 2
            grid_h = max(1, (y_end - y_start) // cell_step)
            grid_w = max(1, (x_end - x_start) // cell_step)
            raw_span_h = grid_h * cell_step
            raw_span_w = grid_w * cell_step
        else:
            raw_span_h = data_slice.shape[0] * downsample
            raw_span_w = data_slice.shape[1] * downsample

        self._render_geom = {
            "x_start": x_start,
            "y_start": y_start,
            "raw_span_w": raw_span_w,
            "raw_span_h": raw_span_h,
        }

        if self.row_correction_enabled and self.row_corr_averages is not None:
            corr_slice = self.row_corr_averages[y_start:y_end:downsample]
            data_slice = data_slice.astype(np.int32) - corr_slice[:, np.newaxis].astype(np.int32)
            data_slice = np.clip(data_slice, 0, 65535).astype(np.uint16)

        img = self._render_image(data_slice, x_start, y_start, x_end, y_end, downsample, display_w, display_h)

        img = img.resize((display_w, display_h), Image.NEAREST)
        self.image_tk = self._pil_to_photoimage(img)
        self.canvas.create_image(self.display_x, self.display_y, image=self.image_tk, anchor="nw")

        if self.flat_mode_enabled:
            self._draw_flat_scale_bar()

        if self.marked_pixels:
            self._draw_pixel_marks()

    def _render_image(self, data_slice, x_start, y_start, x_end, y_end, downsample, display_w, display_h):
        """Choose the right rendering path and return a PIL Image."""
        if self.flat_mode_enabled:
            return self._render_flat()
        if self.cfa_mode == "Color" and self.show_bayer_pixels:
            return self._render_bayer_pixels(data_slice, x_start, y_start, downsample)
        elif self.cfa_mode == "Color" and downsample >= 2:
            return self._render_color_downsampled(x_start, y_start, x_end, y_end, downsample)
        elif self.cfa_mode == "Color":
            return self._render_color_1to1(x_start, y_start, x_end, y_end)
        else:
            return self._render_mono(data_slice)

    def _gamma_scale_rgb(self, r_slice, gr_slice, gb_slice, b_slice):
        """Normalize, gamma-correct, colour-adjust, and return (r, g, b) uint8 arrays."""
        min_val, max_val = self.data_min, self.data_max
        if max_val == min_val:
            return (
                np.zeros_like(r_slice, dtype=np.uint8),
                np.zeros_like(gr_slice, dtype=np.uint8),
                np.zeros_like(b_slice, dtype=np.uint8),
            )

        inv_gamma = 1.0 / self.gamma
        r_norm = np.clip((r_slice.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)
        gr_norm = np.clip((gr_slice.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)
        gb_norm = np.clip((gb_slice.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)
        b_norm = np.clip((b_slice.astype(np.float32) - min_val) / (max_val - min_val), 0, 1)

        r_gamma = np.clip(np.power(r_norm, inv_gamma) * self.color_adjust_r, 0, 1)
        gr_gamma = np.clip(np.power(gr_norm, inv_gamma) * self.color_adjust_gr, 0, 1)
        gb_gamma = np.clip(np.power(gb_norm, inv_gamma) * self.color_adjust_gb, 0, 1)
        b_gamma = np.clip(np.power(b_norm, inv_gamma) * self.color_adjust_b, 0, 1)

        g_gamma = (gr_gamma + gb_gamma) * 0.5

        return (
            (255.0 * r_gamma).astype(np.uint8),
            (255.0 * g_gamma).astype(np.uint8),
            (255.0 * b_gamma).astype(np.uint8),
        )

    def _apply_row_corr_to_bayer_slices(
        self, r_slice, gr_slice, gb_slice, b_slice, y_start, h_cells, cell_step, offsets
    ):
        """Apply per-row correction to the four Bayer channel slices."""
        r_pos, gr_pos, gb_pos, b_pos = offsets

        r_rows = y_start + r_pos[0] + np.arange(h_cells) * cell_step
        gr_rows = y_start + gr_pos[0] + np.arange(h_cells) * cell_step
        gb_rows = y_start + gb_pos[0] + np.arange(h_cells) * cell_step
        b_rows = y_start + b_pos[0] + np.arange(h_cells) * cell_step

        def _correct(sl, rows):
            corrected = sl.astype(np.int32) - self.row_corr_averages[rows, np.newaxis].astype(np.int32)
            return np.clip(corrected, 0, 65535).astype(np.uint16)

        return (
            _correct(r_slice, r_rows),
            _correct(gr_slice, gr_rows),
            _correct(gb_slice, gb_rows),
            _correct(b_slice, b_rows),
        )

    def _render_color_downsampled(self, x_start, y_start, x_end, y_end, downsample):
        """Render colour image from downsampled 2×2 Bayer cells."""
        cell_step = downsample * 2
        h_cells = (y_end - y_start) // cell_step
        w_cells = (x_end - x_start) // cell_step

        if h_cells <= 0 or w_cells <= 0:
            return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), "RGB")

        r_pos, gr_pos, gb_pos, b_pos = self.get_bayer_offsets()

        r_slice = self.data[y_start + r_pos[0] :: cell_step, x_start + r_pos[1] :: cell_step][:h_cells, :w_cells]
        gr_slice = self.data[y_start + gr_pos[0] :: cell_step, x_start + gr_pos[1] :: cell_step][:h_cells, :w_cells]
        gb_slice = self.data[y_start + gb_pos[0] :: cell_step, x_start + gb_pos[1] :: cell_step][:h_cells, :w_cells]
        b_slice = self.data[y_start + b_pos[0] :: cell_step, x_start + b_pos[1] :: cell_step][:h_cells, :w_cells]

        if self.row_correction_enabled and self.row_corr_averages is not None:
            offsets = (r_pos, gr_pos, gb_pos, b_pos)
            r_slice, gr_slice, gb_slice, b_slice = self._apply_row_corr_to_bayer_slices(
                r_slice, gr_slice, gb_slice, b_slice, y_start, h_cells, cell_step, offsets
            )

        # Zero out disabled Bayer channels before debayer
        if not self.show_bayer_r:
            r_slice = np.zeros_like(r_slice)
        if not self.show_bayer_gr:
            gr_slice = np.zeros_like(gr_slice)
        if not self.show_bayer_gb:
            gb_slice = np.zeros_like(gb_slice)
        if not self.show_bayer_b:
            b_slice = np.zeros_like(b_slice)

        r_scaled, g_scaled, b_scaled = self._gamma_scale_rgb(r_slice, gr_slice, gb_slice, b_slice)

        rgb_img = np.zeros((h_cells, w_cells, 3), dtype=np.uint8)
        rgb_img[:, :, 0] = r_scaled
        rgb_img[:, :, 1] = g_scaled
        rgb_img[:, :, 2] = b_scaled
        return Image.fromarray(rgb_img, "RGB")

    def _render_bayer_pixels(self, data_slice, x_start, y_start, downsample):
        """Render individual Bayer-coloured pixels at high zoom."""
        min_val, max_val = self.data_min, self.data_max
        if max_val == min_val:
            scaled_slice = np.zeros_like(data_slice, dtype=np.uint8)
        else:
            normalized = (data_slice.astype(np.float32) - min_val) / (max_val - min_val)
            gamma_corrected = np.power(normalized, 1.0 / self.gamma)
            scaled_slice = (255.0 * gamma_corrected).astype(np.uint8)

        h, w = scaled_slice.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        y_indices = y_start + np.arange(h) * downsample
        x_indices = x_start + np.arange(w) * downsample
        y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing="ij")

        # Build per-component masks using the active Bayer pattern
        row_mod = y_grid % 2
        col_mod = x_grid % 2
        component_grid = np.empty((h, w), dtype="U2")
        for (rm, cm), comp in self.BAYER_PATTERN.items():
            component_grid[(row_mod == rm) & (col_mod == cm)] = comp

        is_r = component_grid == "R"
        is_gr = component_grid == "Gr"
        is_gb = component_grid == "Gb"
        is_b = component_grid == "B"

        base_intensity = (scaled_slice * 0.3).astype(np.uint8)

        if self.show_bayer_r:
            r_adj = np.clip(scaled_slice[is_r].astype(np.float32) * self.color_adjust_r, 0, 255).astype(np.uint8)
            r_base = np.clip(base_intensity[is_r].astype(np.float32) * self.color_adjust_r, 0, 255).astype(np.uint8)
            rgb_img[is_r, 0] = r_adj
            rgb_img[is_r, 1] = r_base
            rgb_img[is_r, 2] = r_base

        if self.show_bayer_gr:
            g_adj = np.clip(scaled_slice[is_gr].astype(np.float32) * self.color_adjust_gr, 0, 255).astype(np.uint8)
            g_base = np.clip(base_intensity[is_gr].astype(np.float32) * self.color_adjust_gr, 0, 255).astype(np.uint8)
            rgb_img[is_gr, 0] = g_base
            rgb_img[is_gr, 1] = g_adj
            rgb_img[is_gr, 2] = g_base

        if self.show_bayer_gb:
            g_adj = np.clip(scaled_slice[is_gb].astype(np.float32) * self.color_adjust_gb, 0, 255).astype(np.uint8)
            g_base = np.clip(base_intensity[is_gb].astype(np.float32) * self.color_adjust_gb, 0, 255).astype(np.uint8)
            rgb_img[is_gb, 0] = g_base
            rgb_img[is_gb, 1] = g_adj
            rgb_img[is_gb, 2] = g_base

        if self.show_bayer_b:
            b_adj = np.clip(scaled_slice[is_b].astype(np.float32) * self.color_adjust_b, 0, 255).astype(np.uint8)
            b_base = np.clip(base_intensity[is_b].astype(np.float32) * self.color_adjust_b, 0, 255).astype(np.uint8)
            rgb_img[is_b, 0] = b_base
            rgb_img[is_b, 1] = b_base
            rgb_img[is_b, 2] = b_adj

        return Image.fromarray(rgb_img, "RGB")

    def _render_color_1to1(self, x_start, y_start, x_end, y_end):
        """Render colour from 2×2 Bayer cells at roughly 1:1 zoom."""
        h_cells = (y_end - y_start) // 2
        w_cells = (x_end - x_start) // 2

        if h_cells <= 0 or w_cells <= 0:
            return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), "RGB")

        r_pos, gr_pos, gb_pos, b_pos = self.get_bayer_offsets()

        r_slice = self.data[y_start + r_pos[0] :: 2, x_start + r_pos[1] :: 2][:h_cells, :w_cells]
        gr_slice = self.data[y_start + gr_pos[0] :: 2, x_start + gr_pos[1] :: 2][:h_cells, :w_cells]
        gb_slice = self.data[y_start + gb_pos[0] :: 2, x_start + gb_pos[1] :: 2][:h_cells, :w_cells]
        b_slice = self.data[y_start + b_pos[0] :: 2, x_start + b_pos[1] :: 2][:h_cells, :w_cells]

        if self.row_correction_enabled and self.row_corr_averages is not None:
            offsets = (r_pos, gr_pos, gb_pos, b_pos)
            r_slice, gr_slice, gb_slice, b_slice = self._apply_row_corr_to_bayer_slices(
                r_slice, gr_slice, gb_slice, b_slice, y_start, h_cells, 2, offsets
            )

        # Zero out disabled Bayer channels before debayer
        if not self.show_bayer_r:
            r_slice = np.zeros_like(r_slice)
        if not self.show_bayer_gr:
            gr_slice = np.zeros_like(gr_slice)
        if not self.show_bayer_gb:
            gb_slice = np.zeros_like(gb_slice)
        if not self.show_bayer_b:
            b_slice = np.zeros_like(b_slice)

        r_scaled, g_scaled, b_scaled = self._gamma_scale_rgb(r_slice, gr_slice, gb_slice, b_slice)

        rgb_img = np.zeros((h_cells, w_cells, 3), dtype=np.uint8)
        rgb_img[:, :, 0] = r_scaled
        rgb_img[:, :, 1] = g_scaled
        rgb_img[:, :, 2] = b_scaled
        return Image.fromarray(rgb_img, "RGB")

    def _render_mono(self, data_slice):
        """Render a grayscale image from the data slice."""
        min_val, max_val = self.data_min, self.data_max
        if max_val == min_val:
            scaled_slice = np.zeros_like(data_slice, dtype=np.uint8)
        else:
            normalized = (data_slice.astype(np.float32) - min_val) / (max_val - min_val)
            gamma_corrected = np.power(normalized, 1.0 / self.gamma)
            scaled_slice = (255.0 * gamma_corrected).astype(np.uint8)
        return Image.fromarray(scaled_slice, "L")

    # ------------------------------------------------------------------
    # Flat-field analysis
    # ------------------------------------------------------------------

    def _get_active_flat_channel(self):
        """Return the name of the single enabled Bayer channel, or None if != 1 enabled."""
        enabled = []
        if self.show_bayer_r:
            enabled.append("R")
        if self.show_bayer_gr:
            enabled.append("Gr")
        if self.show_bayer_gb:
            enabled.append("Gb")
        if self.show_bayer_b:
            enabled.append("B")
        if len(enabled) == 1:
            return enabled[0]
        return None

    def _auto_select_flat_channel(self):
        """When flat mode needs exactly 1 channel, pick the first enabled one.

        If multiple are enabled, deselects the others and updates the UI.
        If none are enabled, defaults to R.
        Returns the selected channel name.
        """
        channel_map = [
            ("R", "show_bayer_r", "bayer_r_var"),
            ("Gr", "show_bayer_gr", "bayer_gr_var"),
            ("Gb", "show_bayer_gb", "bayer_gb_var"),
            ("B", "show_bayer_b", "bayer_b_var"),
        ]
        enabled = [(name, attr, var) for name, attr, var in channel_map if getattr(self, attr)]
        if not enabled:
            enabled = [("R", "show_bayer_r", "bayer_r_var")]

        chosen_name, chosen_attr, chosen_var = enabled[0]
        # Deselect everything except the chosen channel
        for name, attr, var in channel_map:
            setattr(self, attr, name == chosen_name)
            getattr(self, var).set(name == chosen_name)

        return chosen_name

    def _render_flat(self):
        """Render flat-field deviation map for the active Bayer channel.

        Respects viewport (zoom/pan) and dark correction settings.
        The top/bottom averages are always computed over the full image so
        the colour scale stays stable while panning.
        """
        channel = self._get_active_flat_channel()
        if channel is None:
            channel = self._auto_select_flat_channel()

        # Determine the Bayer sub-grid offset for the chosen channel
        for (r_off, c_off), name in self.BAYER_PATTERN.items():
            if name == channel:
                break

        # --- Full-image channel plane (for stable averages) ---
        full_plane = self.data[r_off::2, c_off::2].astype(np.float64)

        # Apply row correction if enabled (average the two correction rows per Bayer row)
        if self.row_correction_enabled and self.row_corr_averages is not None:
            full_rows = r_off + np.arange(full_plane.shape[0]) * 2
            full_plane = full_plane - self.row_corr_averages[full_rows, np.newaxis].astype(np.float64)
            full_plane = np.clip(full_plane, 0, 65535)

        # Full-image top/bottom medians (stable reference while panning)
        # Median is used instead of mean so dark rows/columns don't skew the reference.
        full_h = full_plane.shape[0]
        full_mid = full_h // 2
        top_avg = float(np.median(full_plane[:full_mid])) if full_mid > 0 else 1.0
        bot_avg = float(np.median(full_plane[full_mid:])) if full_mid < full_h else 1.0

        # --- Visible region in channel-plane coordinates ---
        # Channel plane is half the raw size in each dimension
        vx = max(0, int(self.view_x / 2))
        vy = max(0, int(self.view_y / 2))
        vis_w = max(1, int((self.display_w / self.zoom) / 2) + 1)
        vis_h = max(1, int((self.display_h / self.zoom) / 2) + 1)
        x_end = min(full_plane.shape[1], vx + vis_w)
        y_end = min(full_plane.shape[0], vy + vis_h)

        plane = full_plane[vy:y_end, vx:x_end]
        h, w = plane.shape
        if h == 0 or w == 0:
            return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), "RGB")

        # Block-average if requested
        bs = max(1, self.flat_block_size)
        if bs > 1:
            h_trim = (h // bs) * bs
            w_trim = (w // bs) * bs
            if h_trim == 0 or w_trim == 0:
                return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8), "RGB")
            trimmed = plane[:h_trim, :w_trim]
            plane = trimmed.reshape(h_trim // bs, bs, w_trim // bs, bs).mean(axis=(1, 3))
            h, w = plane.shape

        # Build per-row reference (top-half rows use top_avg, bottom-half rows use bot_avg)
        # Map visible rows back to full-image row index to decide top vs bottom
        row_avg = np.empty(h, dtype=np.float64)
        for i in range(h):
            # Approximate original channel-plane row for this output row
            orig_row = vy + (i * bs if bs > 1 else i)
            row_avg[i] = top_avg if orig_row < full_mid else bot_avg

        # Compute deviation depending on display mode
        if self.flat_display_mode == "Absolute":
            # Absolute deviation in DN
            deviation = plane - row_avg[:, np.newaxis]
            dev_range = self.flat_abs_deviation_range
        else:
            # Fractional deviation (percentage mode)
            with np.errstate(divide="ignore", invalid="ignore"):
                deviation = (plane - row_avg[:, np.newaxis]) / row_avg[:, np.newaxis]
            dev_range = self.flat_deviation_range

        # Cache deviation data for hover readout
        self._flat_deviation = deviation
        self._flat_params = {
            "r_off": r_off,
            "c_off": c_off,
            "vx": vx,
            "vy": vy,
            "bs": bs,
        }

        # Map deviation to colour: blue (below) → grey (at average) → red (above)
        norm = np.clip(deviation / dev_range, -1.0, 1.0) if dev_range != 0 else np.zeros_like(deviation)

        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        neg = norm < 0
        pos = norm >= 0
        abs_neg = np.abs(norm[neg])
        abs_pos = norm[pos]

        rgb_img[neg, 0] = (128 * (1.0 - abs_neg)).astype(np.uint8)
        rgb_img[neg, 1] = (128 * (1.0 - abs_neg)).astype(np.uint8)
        rgb_img[neg, 2] = (128 + 127 * abs_neg).astype(np.uint8)

        rgb_img[pos, 0] = (128 + 127 * abs_pos).astype(np.uint8)
        rgb_img[pos, 1] = (128 * (1.0 - abs_pos)).astype(np.uint8)
        rgb_img[pos, 2] = (128 * (1.0 - abs_pos)).astype(np.uint8)

        if self.flat_display_mode == "Absolute":
            self.flat_info_label.config(
                text=f"{channel}  top_med={top_avg:.1f}  bot_med={bot_avg:.1f}  dev=±{dev_range:.0f}DN  block={bs}"
            )
        else:
            self.flat_info_label.config(
                text=f"{channel}  top_med={top_avg:.1f}  bot_med={bot_avg:.1f}  dev=±{dev_range * 100:.1f}%  block={bs}"
            )

        return Image.fromarray(rgb_img, "RGB")

    def _draw_flat_scale_bar(self):
        """Draw a vertical colour-scale legend on the right side of the canvas."""
        bar_w = 20
        bar_h = min(200, self.display_h - 40)
        if bar_h < 20:
            return

        margin = 10
        x0 = self.display_x + self.display_w + margin
        # If not enough room on the right, overlay inside the image
        canvas_w = self.canvas.winfo_width()
        if x0 + bar_w + 50 > canvas_w:
            x0 = self.display_x + self.display_w - bar_w - 60

        y0 = self.display_y + (self.display_h - bar_h) // 2

        n_steps = bar_h

        # Build the gradient as a small image for efficiency
        grad = np.zeros((n_steps, bar_w, 3), dtype=np.uint8)
        for i in range(n_steps):
            # i=0 is top = +dev (red), i=n_steps-1 is bottom = -dev (blue)
            t = 1.0 - (i / max(1, n_steps - 1)) * 2.0  # +1 .. -1
            if t >= 0:
                grad[i, :, 0] = int(128 + 127 * t)
                grad[i, :, 1] = int(128 * (1.0 - t))
                grad[i, :, 2] = int(128 * (1.0 - t))
            else:
                a = abs(t)
                grad[i, :, 0] = int(128 * (1.0 - a))
                grad[i, :, 1] = int(128 * (1.0 - a))
                grad[i, :, 2] = int(128 + 127 * a)

        grad_img = Image.fromarray(grad, "RGB")
        self._flat_scale_tk = self._pil_to_photoimage(grad_img)
        self.canvas.create_image(x0, y0, image=self._flat_scale_tk, anchor="nw")

        # Border
        self.canvas.create_rectangle(x0, y0, x0 + bar_w, y0 + bar_h, outline="white", width=1)

        # Labels — show DN or % depending on display mode
        label_x = x0 + bar_w + 4
        if self.flat_display_mode == "Absolute":
            dev_dn = self.flat_abs_deviation_range
            top_lbl = f"+{dev_dn:.0f}"
            mid_lbl = "0"
            bot_lbl = f"-{dev_dn:.0f}"
        else:
            dev_pct = self.flat_deviation_range * 100
            top_lbl = f"+{dev_pct:.1f}%"
            mid_lbl = "0%"
            bot_lbl = f"-{dev_pct:.1f}%"
        self.canvas.create_text(label_x, y0, text=top_lbl, anchor="nw", fill="white", font=("Courier", 9))
        self.canvas.create_text(label_x, y0 + bar_h // 2, text=mid_lbl, anchor="w", fill="white", font=("Courier", 9))
        self.canvas.create_text(label_x, y0 + bar_h, text=bot_lbl, anchor="sw", fill="white", font=("Courier", 9))

    # ------------------------------------------------------------------
    # Pixel marking / annotation overlay
    # ------------------------------------------------------------------

    # Colours matched to the histogram's per-channel colour scheme.
    MARK_CHANNEL_COLORS = {"R": "red", "Gr": "green", "Gb": "darkgreen", "B": "blue"}

    def _raw_to_screen(self, x_arr, y_arr):
        """Map raw array coords to canvas screen coords (top-left of that pixel).

        Uses the exact geometry `redraw()` used for the image currently on
        screen (`self._render_geom`) rather than re-deriving it from
        view_x/zoom, since the latter doesn't account for the renderer's own
        start-coordinate rounding or (in Color debayer mode) its 2x2-cell
        grid — both of which would otherwise make the overlay drift out of
        registration with the actual pixel as you zoom/pan.
        """
        geom = getattr(self, "_render_geom", None)
        if geom is None:
            return (
                self.display_x + (x_arr - self.view_x) * self.zoom,
                self.display_y + (y_arr - self.view_y) * self.zoom,
            )
        return (
            self.display_x + (x_arr - geom["x_start"]) * self.display_w / geom["raw_span_w"],
            self.display_y + (y_arr - geom["y_start"]) * self.display_h / geom["raw_span_h"],
        )

    def _draw_pixel_marks(self):
        """Overlay highlight boxes + value readouts for marked pixels.

        Only drawn once individual pixels are large enough on screen to hold
        a border and legible text. In Mono mode each mark is a single raw
        pixel; in Color mode a mark expands to its full 2x2 Bayer quad so all
        four component values can be shown together. Skipped in Flat mode,
        whose display doesn't represent individual raw pixel positions.
        """
        if self.flat_mode_enabled:
            return
        geom = getattr(self, "_render_geom", None)
        if geom is None:
            return

        px_w = self.display_w / geom["raw_span_w"]
        px_h = self.display_h / geom["raw_span_h"]
        if min(px_w, px_h) < self.mark_min_pixel_size:
            return

        font_size = max(6, min(14, int(min(px_w, px_h) / 7)))
        font = ("Courier", font_size, "bold")

        # Visible raw-coordinate bounds (with a small margin for partially visible pixels)
        view_left = geom["x_start"] - 2
        view_top = geom["y_start"] - 2
        view_right = geom["x_start"] + geom["raw_span_w"] + 2
        view_bottom = geom["y_start"] + geom["raw_span_h"] + 2

        if self.cfa_mode == "Mono":
            for mx, my in self.marked_pixels:
                if not (view_left <= mx <= view_right and view_top <= my <= view_bottom):
                    continue
                x0, y0 = self._raw_to_screen(mx, my)
                x1, y1 = x0 + px_w, y0 + px_h
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=1)
                value = int(self.data[my, mx])
                self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(value), fill="red", font=font)
        else:
            # Expand each mark to its enclosing 2x2 Bayer quad, de-duplicating
            # quads shared by multiple marks.
            quads = {(mx - mx % 2, my - my % 2) for mx, my in self.marked_pixels}
            for qx, qy in quads:
                if not (view_left <= qx <= view_right and view_top <= qy <= view_bottom):
                    continue
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    rx, ry = qx + dx, qy + dy
                    if rx >= self.full_cols or ry >= self.full_rows:
                        continue
                    comp = self.get_bayer_component(rx, ry)
                    color = self.MARK_CHANNEL_COLORS.get(comp, "white")
                    x0, y0 = self._raw_to_screen(rx, ry)
                    x1, y1 = x0 + px_w, y0 + px_h
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=1)
                    value = int(self.data[ry, rx])
                    self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(value), fill="black", font=font)

    def on_flat_mode_toggle(self):
        self.flat_mode_enabled = self.flat_mode_var.get()
        self.redraw()

    def on_flat_block_size_change(self, event=None):
        try:
            val = int(self.flat_block_size_var.get())
            if val >= 1:
                self.flat_block_size = val
                if self.flat_mode_enabled:
                    self.redraw()
        except ValueError:
            self.flat_block_size_var.set(self.flat_block_size)

    def on_flat_display_mode_change(self, event=None):
        self.flat_display_mode = self.flat_display_mode_var.get()
        if self.flat_mode_enabled:
            self.redraw()

    def on_flat_deviation_change(self, event=None):
        try:
            val = float(self.flat_deviation_var.get())
            if val > 0:
                self.flat_deviation_range = val / 100.0
                if self.flat_mode_enabled:
                    self.redraw()
        except ValueError:
            self.flat_deviation_var.set(self.flat_deviation_range * 100)

    def on_flat_abs_deviation_change(self, event=None):
        try:
            val = float(self.flat_abs_deviation_var.get())
            if val > 0:
                self.flat_abs_deviation_range = val
                if self.flat_mode_enabled:
                    self.redraw()
        except ValueError:
            self.flat_abs_deviation_var.set(self.flat_abs_deviation_range)

    # ------------------------------------------------------------------
    # Info / hover
    # ------------------------------------------------------------------

    def _snap_to_enabled_channel(self, x, y):
        """Snap (x, y) to the nearest pixel whose Bayer channel is enabled.

        Returns (snapped_x, snapped_y).  If all channels are enabled or
        we're in Mono mode the coords are returned unchanged.
        """
        all_on = self.show_bayer_r and self.show_bayer_gr and self.show_bayer_gb and self.show_bayer_b
        if self.cfa_mode == "Mono" or all_on:
            return x, y

        enabled_map = {
            "R": self.show_bayer_r,
            "Gr": self.show_bayer_gr,
            "Gb": self.show_bayer_gb,
            "B": self.show_bayer_b,
        }

        # The four candidate 2×2 neighbours (same cell or adjacent)
        base_y = (y // 2) * 2
        base_x = (x // 2) * 2

        best = None
        best_dist = float("inf")
        for dy in range(2):
            for dx in range(2):
                cy, cx = base_y + dy, base_x + dx
                if 0 <= cx < self.full_cols and 0 <= cy < self.full_rows:
                    comp = self.BAYER_PATTERN[(cy % 2, cx % 2)]
                    if enabled_map.get(comp, False):
                        dist = abs(cx - x) + abs(cy - y)
                        if dist < best_dist:
                            best_dist = dist
                            best = (cx, cy)

        return best if best else (x, y)

    def _update_info(self, x_arr, y_arr):
        """Updates info panel based on GLOBAL array coordinates."""
        if 0 <= x_arr < self.full_cols and 0 <= y_arr < self.full_rows:
            # Snap to nearest enabled channel pixel
            x_arr, y_arr = self._snap_to_enabled_channel(x_arr, y_arr)

            active_value = self.data[y_arr, x_arr]
            raw_value = self.data_raw[y_arr, x_arr]
            component = self.get_bayer_component(x_arr, y_arr)

            self.coord_label.config(text=f"Coords: ({x_arr}, {y_arr})")
            self.component_label.config(text=f"Component: {component}")

            if self.row_correction_enabled and self.row_corr_averages is not None:
                corrected_value = self.get_corrected_value(y_arr, active_value)
                if self.dark_image_subtract_enabled and self.dark_image_data is not None:
                    dark_value = self.dark_image_data[y_arr, x_arr]
                    self.value_label.config(
                        text=f"Raw: {raw_value} | Dark: {dark_value} | Sub: {active_value} | Corr: {corrected_value}"
                    )
                else:
                    self.value_label.config(text=f"Raw: {raw_value} | Corr: {corrected_value}")
                row_avg = self.row_corr_averages[y_arr]
                self.row_avg_label.config(text=f"Row Avg: {row_avg:.1f}")
            elif self.dark_image_subtract_enabled and self.dark_image_data is not None:
                dark_value = self.dark_image_data[y_arr, x_arr]
                self.value_label.config(text=f"Raw: {raw_value} | Dark: {dark_value} | Sub: {active_value}")
                self.row_avg_label.config(text="Row Avg: N/A")
            else:
                self.value_label.config(text=f"Value: {raw_value}")
                self.row_avg_label.config(text="Row Avg: N/A")

            # Flat-field deviation readout
            if self.flat_mode_enabled and self._flat_deviation is not None and self._flat_params is not None:
                p = self._flat_params
                # Convert raw array coords to channel-plane coords
                cp_row = (y_arr - p["r_off"]) // 2
                cp_col = (x_arr - p["c_off"]) // 2
                # Convert to visible-region-relative, then to block index
                blk_row = (cp_row - p["vy"]) // p["bs"] if p["bs"] > 1 else (cp_row - p["vy"])
                blk_col = (cp_col - p["vx"]) // p["bs"] if p["bs"] > 1 else (cp_col - p["vx"])
                dev = self._flat_deviation
                if 0 <= blk_row < dev.shape[0] and 0 <= blk_col < dev.shape[1]:
                    val = dev[blk_row, blk_col]
                    if self.flat_display_mode == "Absolute":
                        self.flat_dev_label.config(text=f"Flat: {val:+.1f}DN")
                    else:
                        self.flat_dev_label.config(text=f"Flat: {val * 100.0:+.2f}%")
                else:
                    self.flat_dev_label.config(text="Flat: N/A")
            else:
                self.flat_dev_label.config(text="")
        else:
            self.coord_label.config(text="Coords: (N/A)")
            self.component_label.config(text="Component: N/A")
            self.value_label.config(text="Value: N/A")
            self.row_avg_label.config(text="Row Avg: N/A")
            self.flat_dev_label.config(text="")

    def _event_to_array_coords(self, event):
        """Map a canvas event position to (x, y) raw array coords, or None if outside the image."""
        if not (
            self.display_x <= event.x < self.display_x + self.display_w
            and self.display_y <= event.y < self.display_y + self.display_h
        ):
            return None

        adj_x = event.x - self.display_x
        adj_y = event.y - self.display_y

        # Use the renderer's own geometry (see redraw()) rather than
        # `_last_render_params`'s raw data-slice width: in Color debayer-cell
        # mode the on-screen grid is 2x2-Bayer-quad-per-cell, not
        # raw-pixel-per-cell, so scaling by the raw slice width there picked
        # the wrong pixel (roughly 2x off in each axis).
        geom = getattr(self, "_render_geom", None)
        if self.display_w > 0 and self.display_h > 0 and geom is not None:
            x_arr = geom["x_start"] + int(adj_x * geom["raw_span_w"] / self.display_w)
            y_arr = geom["y_start"] + int(adj_y * geom["raw_span_h"] / self.display_h)
        else:
            x_arr = int(self.view_x + adj_x / self.zoom)
            y_arr = int(self.view_y + adj_y / self.zoom)

        if 0 <= x_arr < self.full_cols and 0 <= y_arr < self.full_rows:
            return x_arr, y_arr
        return None

    def on_hover(self, event):
        """Handles mouse motion over the canvas to update info."""
        coords = self._event_to_array_coords(event)
        if coords is None:
            self._update_info(-1, -1)
        else:
            self._update_info(*coords)

    def on_right_click(self, event):
        """Shows a context menu to mark/unmark the pixel under the cursor."""
        coords = self._event_to_array_coords(event)
        if coords is None:
            return
        x_arr, y_arr = coords

        menu = tk.Menu(self.canvas, tearoff=0)
        is_marked = (x_arr, y_arr) in self.marked_pixels
        label = "Unmark Pixel" if is_marked else "Mark Pixel"
        menu.add_command(
            label=f"{label} ({x_arr}, {y_arr})",
            command=lambda: self._toggle_mark(x_arr, y_arr),
        )
        if self.marked_pixels:
            menu.add_separator()
            menu.add_command(label="Clear All Marks", command=self._clear_all_marks)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _toggle_mark(self, x_arr, y_arr):
        if (x_arr, y_arr) in self.marked_pixels:
            self.marked_pixels.discard((x_arr, y_arr))
        else:
            self.marked_pixels.add((x_arr, y_arr))
        self.redraw()

    def _clear_all_marks(self):
        self.marked_pixels.clear()
        self.redraw()

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def on_gamma_change(self, event=None):
        try:
            new_gamma = float(self.gamma_var.get())
            if new_gamma > 0:
                self.gamma = new_gamma
                self.redraw()
        except ValueError:
            self.gamma_var.set(self.gamma)

    def on_cfa_mode_change(self, event=None):
        self.cfa_mode = self.cfa_mode_var.get()
        self.redraw()

    def on_bayer_pattern_change(self, event=None):
        self.bayer_pattern_type = self.bayer_pattern_var.get()
        self.BAYER_PATTERN = self.BAYER_PATTERNS[self.bayer_pattern_type]
        print(f"Bayer pattern changed to: {self.bayer_pattern_type}")
        self.redraw()

    def on_show_bayer_pixels_toggle(self):
        self.show_bayer_pixels = self.show_bayer_pixels_var.get()
        self.redraw()

    def on_bayer_channel_toggle(self):
        # Snapshot the previous state (before reading the new checkbox values)
        prev = {
            "R": self.show_bayer_r,
            "Gr": self.show_bayer_gr,
            "Gb": self.show_bayer_gb,
            "B": self.show_bayer_b,
        }

        # Read what the user just clicked
        self.show_bayer_r = self.bayer_r_var.get()
        self.show_bayer_gr = self.bayer_gr_var.get()
        self.show_bayer_gb = self.bayer_gb_var.get()
        self.show_bayer_b = self.bayer_b_var.get()

        # In flat mode, behave like radio buttons: only the most recently
        # enabled channel stays on so the user can switch freely.
        if self.flat_mode_enabled:
            now = {
                "R": self.show_bayer_r,
                "Gr": self.show_bayer_gr,
                "Gb": self.show_bayer_gb,
                "B": self.show_bayer_b,
            }
            # Find the channel that was just turned ON (wasn't before, is now)
            newly_on = [ch for ch in now if now[ch] and not prev[ch]]
            channel_map = [
                ("R", "show_bayer_r", "bayer_r_var"),
                ("Gr", "show_bayer_gr", "bayer_gr_var"),
                ("Gb", "show_bayer_gb", "bayer_gb_var"),
                ("B", "show_bayer_b", "bayer_b_var"),
            ]
            if newly_on:
                # User clicked a new channel on — select only that one
                chosen = newly_on[0]
                for name, attr, var in channel_map:
                    setattr(self, attr, name == chosen)
                    getattr(self, var).set(name == chosen)
            else:
                # User unchecked the active one — pick the first still-enabled,
                # or fall back to auto-select if none remain
                enabled = [name for name, attr, _ in channel_map if getattr(self, attr)]
                if len(enabled) != 1:
                    self._auto_select_flat_channel()

        self.redraw()

    def on_color_adjust_change(self, event=None):
        try:
            r_val = float(self.color_adjust_r_var.get())
            gr_val = float(self.color_adjust_gr_var.get())
            gb_val = float(self.color_adjust_gb_var.get())
            b_val = float(self.color_adjust_b_var.get())
            if r_val > 0 and gr_val > 0 and gb_val > 0 and b_val > 0:
                self.color_adjust_r = r_val
                self.color_adjust_gr = gr_val
                self.color_adjust_gb = gb_val
                self.color_adjust_b = b_val
                self.redraw()
        except ValueError:
            self.color_adjust_r_var.set(self.color_adjust_r)
            self.color_adjust_gr_var.set(self.color_adjust_gr)
            self.color_adjust_gb_var.set(self.color_adjust_gb)
            self.color_adjust_b_var.set(self.color_adjust_b)

    def on_export_view(self):
        """Exports the current view as a TIFF file with raw Bayer values."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
            title="Export Current View",
        )
        if not filepath:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 0 or canvas_h <= 0:
            print("Canvas not ready for export")
            return

        arr_w_visible = canvas_w / self.zoom
        arr_h_visible = canvas_h / self.zoom

        if self.cfa_mode == "Color":
            arr_cols = int(self.view_x + arr_w_visible) - int(self.view_x)
            arr_rows = int(self.view_y + arr_h_visible) - int(self.view_y)
            downsample = max(1, int(max(arr_cols / canvas_w, arr_rows / canvas_h)))
            if downsample > 1:
                downsample = 2 ** max(1, int(np.log2(downsample)))
            downsample = max(1, downsample)
            if downsample >= 2:
                align = downsample * 2
                x_start = max(0, (int(self.view_x) // align) * align)
                y_start = max(0, (int(self.view_y) // align) * align)
            else:
                x_start = max(0, (int(self.view_x) // 2) * 2)
                y_start = max(0, (int(self.view_y) // 2) * 2)
            x_end = min(self.full_cols, x_start + int(np.ceil(arr_w_visible / downsample) * downsample))
            y_end = min(self.full_rows, y_start + int(np.ceil(arr_h_visible / downsample) * downsample))
        else:
            x_start = max(0, int(self.view_x))
            y_start = max(0, int(self.view_y))
            x_end = min(self.full_cols, int(self.view_x + arr_w_visible) + 1)
            y_end = min(self.full_rows, int(self.view_y + arr_h_visible) + 1)
            arr_cols = x_end - x_start
            arr_rows = y_end - y_start
            downsample = max(1, int(max(arr_cols / canvas_w, arr_rows / canvas_h)))
            if downsample > 1:
                downsample = 2 ** int(np.log2(downsample))

        export_data = self.data[y_start:y_end:downsample, x_start:x_end:downsample]
        if export_data.size == 0:
            print("No data to export")
            return

        rows, cols = export_data.shape
        if rows % 2 != 0:
            export_data = export_data[:-1, :]
            rows -= 1
        if cols % 2 != 0:
            export_data = export_data[:, :-1]
            cols -= 1
        if export_data.size == 0:
            print("No data to export after alignment")
            return

        try:
            from tifffile import imwrite

            imwrite(filepath, export_data)
            print(f"Exported {export_data.shape[0]}x{export_data.shape[1]} region to {filepath}")
            print(
                f"  Region: ({x_start}, {y_start}) to "
                f"({x_start + cols * downsample}, {y_start + rows * downsample}) with downsample={downsample}"
            )
        except Exception as e:
            print(f"Error exporting: {e}")

    def on_reset_view(self):
        self.view_x = 0.0
        self.view_y = 0.0
        self.zoom_to_fit()

    # ------------------------------------------------------------------
    # File loading (open / reload)
    # ------------------------------------------------------------------

    def _set_data(self, cfa_data, filename=None, preserve_view=False, preserve_bit_range=False, preserve_marks=False):
        """Replace the active image with new data and reset dependent state.

        When preserve_bit_range is True the current MSB/LSB selection is kept
        and applied to the new data.  When preserve_view is True the current
        zoom/pan are kept (clamped to the new image bounds) instead of
        re-fitting the viewport.  When preserve_marks is True, marked pixels
        are kept (dropping any that fall outside the new image bounds);
        otherwise all marks are cleared.
        """
        self._raw_full_bits = cfa_data
        self._dark_full_bits = None

        if not preserve_bit_range:
            self.bit_msb = 15
            self.bit_lsb = 0
            self.bit_msb_var.set(15)
            self.bit_lsb_var.set(0)
            self._update_bit_depth_label()

        self.data_raw = self._extract_bit_range(self._raw_full_bits)
        self.data = self.data_raw

        if self.data.ndim != 2:
            raise ValueError(f"Input array must be 2D, but got {self.data.ndim} dimensions.")

        self.full_rows, self.full_cols = self.data.shape

        print("Calculating data range for intensity scaling...")
        self.data_min = float(self.data.min())
        self.data_max = float(self.data.max())
        print(f"Data range: {self.data_min} to {self.data_max}")

        # Reset dark image state
        self.dark_image_data = None
        self.dark_image_subtract_enabled = False
        self.dark_image_path = None
        self.dark_image_subtract_var.set(False)
        self.dark_image_subtract_check.config(state=tk.DISABLED)
        self.dark_image_label.config(text="No dark image loaded", foreground="#666")

        # Reset row-correction cache (recompute if currently enabled)
        self.row_corr_averages = None
        if self.row_correction_enabled:
            self.calculate_row_corr_averages()

        # Reset flat-field cache
        self._flat_deviation = None
        self._flat_params = None

        # Reset (or filter) marked pixels
        if preserve_marks:
            self.marked_pixels = {
                (x, y) for x, y in self.marked_pixels if 0 <= x < self.full_cols and 0 <= y < self.full_rows
            }
        else:
            self.marked_pixels = set()

        self.filename = filename
        if filename:
            basename = os.path.basename(filename)
            self.title(f"tv - {basename}")
        else:
            self.title("tv")

        if preserve_view:
            self._clamp_view()
            self.redraw()
        else:
            self.zoom_to_fit()

    def _clamp_view(self):
        """Clamp view_x/view_y so the viewport stays within the image bounds."""
        if self.display_w <= 0 or self.display_h <= 0:
            self.zoom_to_fit()
            return
        arr_w = self.display_w / self.zoom
        arr_h = self.display_h / self.zoom
        self.view_x = max(0.0, min(self.view_x, max(0.0, self.full_cols - arr_w)))
        self.view_y = max(0.0, min(self.view_y, max(0.0, self.full_rows - arr_h)))

    def on_open_file(self):
        """Open a file dialog and load a new image."""
        file_path = filedialog.askopenfilename(
            title="Select TIFF/DNG file to view",
            filetypes=[("TIFF files", "*.tif;*.tiff;*.dng"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            data = read_tiff_2d(file_path)
            print(f"Loading file: {file_path}")
            print(f"Loaded image with shape: {data.shape}, dtype: {data.dtype}")
            self._set_data(data, file_path)
        except Exception as e:
            print(f"Error loading file: {e}")

    def on_reload_file(self):
        """Re-read the currently open file from disk."""
        if not self.filename:
            print("No file to reload (viewing mock data).")
            return
        try:
            data = read_tiff_2d(self.filename)
            print(f"Reloading file: {self.filename}")
            self._set_data(data, self.filename, preserve_view=True, preserve_bit_range=True, preserve_marks=True)
        except Exception as e:
            print(f"Error reloading file: {e}")

    # ------------------------------------------------------------------
    # Dark pixel correction
    # ------------------------------------------------------------------

    def calculate_row_corr_averages(self):
        """Calculate per-row averages from dark columns on both left and right edges."""
        left_start = self.row_corr_offset
        left_end = self.row_corr_offset + self.row_corr_width
        right_end = self.full_cols - self.row_corr_offset
        right_start = right_end - self.row_corr_width

        if left_end > self.full_cols or right_start < 0 or right_start < left_end:
            print(f"Invalid row correction params: offset={self.row_corr_offset}, width={self.row_corr_width}")
            return None

        print(f"Calculating row correction averages from cols {left_start}:{left_end} and {right_start}:{right_end}...")
        left_cols = self.data[:, left_start:left_end]
        right_cols = self.data[:, right_start:right_end]
        combined = np.hstack([left_cols, right_cols])
        self.row_corr_averages = np.mean(combined, axis=1)
        print(
            f"Row correction averages calculated. Range: {self.row_corr_averages.min():.1f}"
            f" to {self.row_corr_averages.max():.1f}"
        )
        return self.row_corr_averages

    def get_corrected_value(self, y_arr, raw_value):
        if not self.row_correction_enabled or self.row_corr_averages is None:
            return raw_value
        if 0 <= y_arr < self.full_rows:
            row_avg = self.row_corr_averages[y_arr]
            corrected = int(raw_value - row_avg)
            return max(0, min(corrected, 65535))
        return raw_value

    def on_row_correction_toggle(self):
        self.row_correction_enabled = self.row_correction_var.get()
        if self.row_correction_enabled and self.row_corr_averages is None:
            self.calculate_row_corr_averages()
        self.redraw()

    def on_row_corr_params_change(self, event=None):
        try:
            offset = int(self.row_corr_offset_var.get())
            width = int(self.row_corr_width_var.get())
            left_end = offset + width
            right_start = self.full_cols - offset - width
            if offset >= 0 and width > 0 and left_end <= self.full_cols and right_start >= left_end:
                self.row_corr_offset = offset
                self.row_corr_width = width
                if self.row_correction_enabled:
                    self.calculate_row_corr_averages()
                    self.redraw()
            else:
                self.row_corr_offset_var.set(self.row_corr_offset)
                self.row_corr_width_var.set(self.row_corr_width)
        except ValueError:
            self.row_corr_offset_var.set(self.row_corr_offset)
            self.row_corr_width_var.set(self.row_corr_width)

    def on_load_dark_image(self):
        """Open a file dialog to load a dark calibration TIFF."""
        file_path = filedialog.askopenfilename(
            title="Select dark calibration TIFF",
            filetypes=[("TIFF files", "*.tif;*.tiff;*.dng"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            dark_data = imread(file_path)
            if hasattr(dark_data, "base") and dark_data.base is not None:
                dark_data = np.array(dark_data, copy=True)
            if dark_data.shape != (self.full_rows, self.full_cols):
                print(
                    f"Dark image shape {dark_data.shape} does not match "
                    f"loaded image shape ({self.full_rows}, {self.full_cols})"
                )
                self.dark_image_label.config(text="Dimension mismatch!", foreground="red")
                return
            self._dark_full_bits = dark_data
            self.dark_image_data = self._extract_bit_range(dark_data)
            self.dark_image_path = file_path
            basename = os.path.basename(file_path)
            self.dark_image_label.config(text=f"Dark: {basename}", foreground="#060")
            self.dark_image_subtract_check.config(state=tk.NORMAL)
            print(f"Dark image loaded: {file_path} (shape={dark_data.shape}, dtype={dark_data.dtype})")
        except Exception as e:
            print(f"Error loading dark image: {e}")
            self.dark_image_label.config(text="Load error!", foreground="red")

    def on_dark_image_subtract_toggle(self):
        """Toggle dark image subtraction on/off."""
        self.dark_image_subtract_enabled = self.dark_image_subtract_var.get()
        self._recalculate_active_data()
        self.redraw()

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def _on_hist_sigma_change(self, event=None):
        """Update hist_sigma from the GUI entry."""
        try:
            val = float(self.hist_sigma_var.get())
            if val > 0:
                self.hist_sigma = val
        except ValueError:
            self.hist_sigma_var.set(self.hist_sigma)

    def _show_histogram(self, x0, y0, x1, y1):
        """Show per-Bayer-channel histograms for the selected region.

        Always shows all four Bayer channels.  If dark subtraction is
        active, a second row of histograms shows the signed (raw - dark)
        values which can be negative.

        The x-axis is centred on the mean and extends +/- hist_sigma
        standard deviations.
        """
        import matplotlib.pyplot as plt

        region_raw = self.data_raw[y0:y1, x0:x1]

        # Build coordinate grids (absolute array coords)
        rows_idx = np.arange(y0, y1)
        cols_idx = np.arange(x0, x1)
        col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)

        channels = ["R", "Gr", "Gb", "B"]
        channel_colors = {
            "R": "red",
            "Gr": "green",
            "Gb": "darkgreen",
            "B": "blue",
        }

        show_sub = self.dark_image_subtract_enabled and self.dark_image_data is not None
        dark_region = None
        if show_sub:
            dark_region = self.dark_image_data[y0:y1, x0:x1]

        n_hist_rows = 2 if show_sub else 1
        n_cols = len(channels)
        fig, axes = plt.subplots(
            n_hist_rows,
            n_cols,
            figsize=(4 * n_cols, 3.5 * n_hist_rows),
            squeeze=False,
        )
        sigma = self.hist_sigma

        for ci, ch_name in enumerate(channels):
            # Bayer position for this component
            ch_pos = None
            for pos, comp in self.BAYER_PATTERN.items():
                if comp == ch_name:
                    ch_pos = pos
                    break
            mask = (row_grid % 2 == ch_pos[0]) & (col_grid % 2 == ch_pos[1])
            raw_vals = region_raw[mask].astype(np.float64)
            color = channel_colors[ch_name]

            # --- Raw histogram ---
            ax = axes[0, ci]
            if raw_vals.size > 0:
                mean_r = np.mean(raw_vals)
                std_r = np.std(raw_vals)
                lo = mean_r - sigma * std_r
                hi = mean_r + sigma * std_r
                ax.hist(
                    raw_vals,
                    bins="auto",
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_xlim(lo, hi)
                ax.set_title(f"{ch_name} — Raw\n\u03bc={mean_r:.1f}  \u03c3={std_r:.1f}")
            else:
                ax.set_title(f"{ch_name} — Raw (no data)")
            ax.set_xlabel("DN")
            ax.set_ylabel("Count")

            # --- Subtracted histogram ---
            if show_sub:
                sub_vals = (raw_vals.astype(np.int64) - dark_region[mask].astype(np.int64)).astype(np.float64)
                ax2 = axes[1, ci]
                if sub_vals.size > 0:
                    mean_s = np.mean(sub_vals)
                    std_s = np.std(sub_vals)
                    lo_s = mean_s - sigma * std_s
                    hi_s = mean_s + sigma * std_s
                    ax2.hist(
                        sub_vals,
                        bins="auto",
                        color=color,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.3,
                    )
                    ax2.set_xlim(lo_s, hi_s)
                    ax2.set_title(f"{ch_name} — Sub\n\u03bc={mean_s:.1f}  \u03c3={std_s:.1f}")
                else:
                    ax2.set_title(f"{ch_name} — Sub (no data)")
                ax2.set_xlabel("DN")
                ax2.set_ylabel("Count")

        region_label = f"Region ({x0}, {y0})\u2013({x1}, {y1})  [{x1 - x0} \u00d7 {y1 - y0}]  (\u00b1{sigma:.1f}\u03c3)"
        fig.suptitle(region_label, fontsize=11)
        fig.tight_layout()
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Mouse / zoom / pan
    # ------------------------------------------------------------------

    def on_mouse_down(self, event):
        if event.state & 0x0004:  # Ctrl
            if (
                self.display_x <= event.x < self.display_x + self.display_w
                and self.display_y <= event.y < self.display_y + self.display_h
            ):
                self.hist_rect_active = True
                self.hist_rect_start = (event.x, event.y)
                self.canvas.config(cursor="cross")
        elif event.state & 0x0001:  # Shift
            if (
                self.display_x <= event.x < self.display_x + self.display_w
                and self.display_y <= event.y < self.display_y + self.display_h
            ):
                self.zoom_rect_active = True
                self.zoom_rect_start = (event.x, event.y)
                self.canvas.config(cursor="cross")
        else:
            self.canvas.config(cursor="fleur")
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_start_view_x = self.view_x
            self.pan_start_view_y = self.view_y

    def on_mouse_drag(self, event):
        if self.hist_rect_active:
            if self.hist_rect_id:
                self.canvas.delete(self.hist_rect_id)
            x0, y0 = self.hist_rect_start
            self.hist_rect_id = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="cyan", width=2, dash=(4, 4)
            )
        elif self.zoom_rect_active:
            if self.zoom_rect_id:
                self.canvas.delete(self.zoom_rect_id)
            x0, y0 = self.zoom_rect_start
            self.zoom_rect_id = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="yellow", width=2, dash=(4, 4)
            )
        elif self.zoom >= 1.0:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y

            arr_w_visible = self.display_w / self.zoom
            arr_h_visible = self.display_h / self.zoom

            dx_arr = dx / self.zoom
            dy_arr = dy / self.zoom

            self.view_x = self.pan_start_view_x - dx_arr
            self.view_y = self.pan_start_view_y - dy_arr

            max_view_x = max(0, self.full_cols - arr_w_visible)
            max_view_y = max(0, self.full_rows - arr_h_visible)
            self.view_x = max(0, min(self.view_x, max_view_x))
            self.view_y = max(0, min(self.view_y, max_view_y))

            self.redraw()

    def on_mouse_up(self, event):
        if self.hist_rect_active:
            if self.hist_rect_id:
                self.canvas.delete(self.hist_rect_id)
                self.hist_rect_id = None

            x0, y0 = self.hist_rect_start
            x1, y1 = event.x, event.y
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            if (x1 - x0) > 5 and (y1 - y0) > 5:
                adj_x0 = x0 - self.display_x
                adj_y0 = y0 - self.display_y
                adj_x1 = x1 - self.display_x
                adj_y1 = y1 - self.display_y

                arr_x0 = int(self.view_x + adj_x0 / self.zoom)
                arr_y0 = int(self.view_y + adj_y0 / self.zoom)
                arr_x1 = int(self.view_x + adj_x1 / self.zoom)
                arr_y1 = int(self.view_y + adj_y1 / self.zoom)

                arr_x0 = max(0, min(arr_x0, self.full_cols - 1))
                arr_y0 = max(0, min(arr_y0, self.full_rows - 1))
                arr_x1 = max(0, min(arr_x1, self.full_cols))
                arr_y1 = max(0, min(arr_y1, self.full_rows))

                if arr_x1 > arr_x0 and arr_y1 > arr_y0:
                    self._show_histogram(arr_x0, arr_y0, arr_x1, arr_y1)

            self.hist_rect_active = False
            self.hist_rect_start = None
            self.canvas.config(cursor="")
            return

        if self.zoom_rect_active:
            if self.zoom_rect_id:
                self.canvas.delete(self.zoom_rect_id)
                self.zoom_rect_id = None

            x0, y0 = self.zoom_rect_start
            x1, y1 = event.x, event.y
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            if (x1 - x0) > 10 and (y1 - y0) > 10:
                adj_x0 = x0 - self.display_x
                adj_y0 = y0 - self.display_y
                adj_x1 = x1 - self.display_x
                adj_y1 = y1 - self.display_y

                arr_x0 = self.view_x + adj_x0 / self.zoom
                arr_y0 = self.view_y + adj_y0 / self.zoom
                arr_x1 = self.view_x + adj_x1 / self.zoom
                arr_y1 = self.view_y + adj_y1 / self.zoom

                self.view_x = max(0.0, arr_x0)
                self.view_y = max(0.0, arr_y0)

                rect_width = arr_x1 - arr_x0
                rect_height = arr_y1 - arr_y0

                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()

                zoom_w = canvas_w / rect_width
                zoom_h = canvas_h / rect_height
                target_zoom = min(zoom_w, zoom_h)

                if target_zoom >= 1.0:
                    self.zoom = 2.0 ** round(math.log2(target_zoom))
                else:
                    self.zoom = 2.0 ** math.floor(math.log2(target_zoom))

                self.redraw()

            self.zoom_rect_active = False
            self.zoom_rect_start = None
            self.canvas.config(cursor="")
        else:
            self.canvas.config(cursor="")
            if (
                self.display_x <= event.x < self.display_x + self.display_w
                and self.display_y <= event.y < self.display_y + self.display_h
            ):
                adj_x = event.x - self.display_x
                adj_y = event.y - self.display_y
                x_arr = int(self.view_x + adj_x / self.zoom)
                y_arr = int(self.view_y + adj_y / self.zoom)
                self._update_info(x_arr, y_arr)
            else:
                self._update_info(-1, -1)

    def _refresh_hover(self):
        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        coords = self._event_to_array_coords(SimpleNamespace(x=x, y=y))
        if coords is None:
            self._update_info(-1, -1)
        else:
            self._update_info(*coords)

    def on_scroll(self, event):
        if event.num == 5 or event.delta < 0:
            zoom_in = False
        elif event.num == 4 or event.delta > 0:
            zoom_in = True
        else:
            return

        adj_x = event.x - self.display_x
        adj_y = event.y - self.display_y

        x_arr_before = self.view_x + adj_x / self.zoom
        y_arr_before = self.view_y + adj_y / self.zoom

        new_zoom = self.zoom * (2.0 if zoom_in else 0.5)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        min_zoom = min(canvas_w / self.full_cols, canvas_h / self.full_rows)

        if new_zoom < min_zoom:
            new_zoom = min_zoom
            self.view_x = 0.0
            self.view_y = 0.0
        else:
            self.view_x = max(0.0, x_arr_before - adj_x / new_zoom)
            self.view_y = max(0.0, y_arr_before - adj_y / new_zoom)

        self.zoom = new_zoom
        self.redraw()
        self._refresh_hover()

    def on_resize(self, event):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 2 or canvas_h < 2:
            return

        if abs(canvas_w - self.last_canvas_w) < 2 and abs(canvas_h - self.last_canvas_h) < 2:
            self.redraw()
            return

        self.last_canvas_w = canvas_w
        self.last_canvas_h = canvas_h

        max_zoom_w = canvas_w / self.full_cols
        max_zoom_h = canvas_h / self.full_rows
        max_zoom = min(max_zoom_w, max_zoom_h)

        target_zoom = 0.0625
        while target_zoom * 2.0 <= max_zoom:
            target_zoom *= 2.0

        if abs(self.zoom - 0.1) < 0.01 or self.zoom > max_zoom:
            self.zoom = target_zoom
            self.view_x = 0.0
            self.view_y = 0.0

        self.redraw()


# ------------------------------------------------------------------
# File loading
# ------------------------------------------------------------------


def read_tiff_2d(file_path):
    """Read a TIFF/DNG file and return it as a 2D NumPy array.

    Raises ValueError if the image is not 2D.
    """
    data = imread(file_path)
    if hasattr(data, "base") and data.base is not None:
        data = np.array(data, copy=True)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D array with shape {data.shape}")
    return data


def load_tiff_file():
    """Opens a file dialog to select and load a TIFF/DNG file.

    Returns:
        tuple: (data, file_path) where data is a 2D numpy array and
               file_path is the path to the loaded file. Both are None
               if no file was selected or loading failed.
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select TIFF/DNG file to view",
        filetypes=[("TIFF files", "*.tif;*.tiff;*.dng"), ("All files", "*.*")],
    )

    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        return None, None

    print(f"Loading file: {file_path}")
    try:
        print("Reading TIFF data into memory...")
        data = imread(file_path)

        if hasattr(data, "base") and data.base is not None:
            data = np.array(data, copy=True)

        print(f"Loaded image with shape: {data.shape}, dtype: {data.dtype}")

        if data.ndim != 2:
            print(f"Error: Expected 2D array, got {data.ndim}D array with shape {data.shape}")
            return None, None

        return data, file_path
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None
