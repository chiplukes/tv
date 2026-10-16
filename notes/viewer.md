# Viewer — Technical Details

`tv` is a single-window tkinter application built around the `ZoomableCfaViewer` class in `src/tv/viewer.py`. It displays raw 16-bit Bayer CFA TIFF images and provides tools for inspecting pixel values, adjusting rendering, and analysing flat-field uniformity.

## Architecture

| File | Purpose |
|---|---|
| `src/tv/__init__.py` | Package init, exports `__version__` |
| `src/tv/__main__.py` | CLI entry point — opens file dialog, falls back to mock data, launches `ZoomableCfaViewer` |
| `src/tv/viewer.py` | All GUI code: `ZoomableCfaViewer` class and `load_tiff_file()` helper |

The viewer holds the full raw 2D `numpy` array in memory as `data_raw`. The active working array `data` is either the same object or a dark-image-subtracted copy. All rendering reads from `data`, so dark image subtraction is transparent to every feature.

## Opening / Reloading Files

- **Open File…** loads a new TIFF/DNG file, resetting dark-image state, bit range, and viewport (re-fitted to the window).
- **Reload** re-reads the currently open file from disk (no-op when viewing mock data). It preserves the current bit range and view (zoom/pan), clamped to the new image bounds.

## GUI Layout (top to bottom)

1. **Info panel** — coordinates, Bayer component, raw/subtracted/corrected value, dark average
2. **Gamma / mode panel** — gamma entry, CFA Mode (Color/Mono), Bayer pattern selector, Show CFA checkbox, Open File / Reload / Reset View buttons
3. **Dark correction panel** — enable checkbox, dark column range entries, Bayer channel checkboxes (R, Gr, Gb, B)
4. **Dark image panel** — Subtract Dark Image checkbox, Load Dark Image button, filename label
5. **Colour adjust panel** — R/Gr/Gb/B gain multiplier entries, Export View button
6. **Flat-field panel** — Flat checkbox, block size, display mode (Percentage/Absolute), Dev %, Dev DN, info label
7. **Canvas** — the image display area

## Rendering Paths

`redraw()` computes the visible viewport, extracts a data slice, applies dark correction, then delegates to `_render_image()` which picks one of five paths:

| Path | When used |
|---|---|
| `_render_flat` | Flat mode enabled |
| `_render_bayer_pixels` | Color mode + Show CFA checked |
| `_render_color_downsampled` | Color mode, zoomed out (downsample ≥ 2) |
| `_render_color_1to1` | Color mode, ~1:1 zoom |
| `_render_mono` | Mono mode |

All colour paths use `_gamma_scale_rgb()` for normalisation, gamma correction, and colour-adjust multipliers.

### Bayer Debayering

The viewer does not perform full demosaicing. Instead it groups 2×2 Bayer cells and maps R, avg(Gr, Gb), B to the RGB channels. This is fast and sufficient for inspection purposes. Each Bayer channel (R, Gr, Gb, B) has its own independent gain multiplier applied before the green channels are averaged.

### Bayer Channel Toggles

Four independent checkboxes (R, Gr, Gb, B) control which Bayer components contribute to the rendered image. Disabled channels are zeroed before debayering. In flat mode, the checkboxes act as **radio buttons** — clicking a new channel auto-deselects the previous one.

## Dark Calibration Image

A separate dark TIFF of the same dimensions can be loaded and subtracted from the image data.

1. Click **Load Dark Image…** to select a TIFF file. It must have the same shape as the opened image.
2. Enable **Subtract Dark Image** to replace the active data with `clip(raw − dark, 0)`.
3. When enabled, `data_min` / `data_max` are recalculated from the subtracted result and dark row averages are recomputed if dark correction is also active.
4. The info panel shows `Raw: <original> | Sub: <subtracted>` on hover (plus `| Corr:` when dark column correction is also enabled).
5. Disabling the checkbox restores the original raw data.

## Dark Pixel Correction

Configurable via a column range (default 64–96). When enabled:

1. `calculate_dark_row_averages()` computes the mean of each row across the dark columns.
2. The per-row average is subtracted from every pixel in that row during rendering.
3. The info panel shows both raw and corrected values on hover.

Dark correction is applied in all render paths including flat mode.

## Flat-Field Analysis

Analyses spatial uniformity of a single Bayer channel. Steps:

1. Extract the selected channel's sub-grid (every other row/col based on the Bayer pattern offset).
2. Optionally apply dark correction to the channel plane.
3. Compute full-image top-half and bottom-half **medians** (stable reference while panning; median is used instead of mean so dark rows/columns don't skew the reference).
4. Extract the visible viewport region from the channel plane.
5. Optionally block-average using NxN blocks (default N=6).
6. Compute per-pixel fractional deviation from the appropriate half-average.
7. Map deviation to a diverging colour scale: blue (below average) → grey (at average) → red (above average).
8. Draw a vertical scale bar overlay showing ±deviation%.

### Display Modes

A combobox selects between two deviation display modes:

| Mode | Description |
|---|---|
| Percentage | Deviation as a fraction of the half-median (default). Scale bar and hover show `%`. |
| Absolute | Deviation as raw DN difference from the half-median. Scale bar and hover show `DN`. |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| Block Size | 6 | NxN block averaging on the channel plane (1 = no averaging) |
| Dev % | 15 | Full-scale range of the colour map in Percentage mode (±15% of median) |
| Dev DN | 500 | Full-scale range of the colour map in Absolute mode (±500 DN) |

## Navigation

| Action | Effect |
|---|---|
| Left-drag | Pan |
| Shift + left-drag | Zoom rectangle |
| Scroll wheel | Zoom in/out (powers of 2) |
| Reset View button | Fit entire image to window |

## Hover / Info Panel

The info panel updates on mouse movement. When only a subset of Bayer channels is enabled, the hover position **snaps to the nearest enabled-channel pixel** within the 2×2 Bayer cell so the readout always shows relevant values.

In **flat mode**, the info panel also displays the block-level deviation for the pixel under the cursor, matching the heat-map colour at the configured block size. The format depends on the display mode: `Flat: +1.23%` in Percentage mode or `Flat: +42.5DN` in Absolute mode.

## Export

The Export View button saves the current viewport's raw Bayer data (not the rendered RGB) as a 16-bit TIFF. The exported region is aligned to 2×2 Bayer cell boundaries and respects the current downsample level.

## Region plots

The plot controls sit to the right of the settings above the image. Drag the divider
to resize the two areas; wide settings rows remain accessible with the horizontal
scrollbar. Ctrl + left-drag appends a region to the editable plot region list. Each line is
`XSTART,YSTART,XEND,YEND,MATPLOTLIB_PLOTSTYLE`, for example `10,20,100,200,b-`.
End coordinates are exclusive. Type or delete lines to add, edit, or remove regions;
**Add full image** adds the entire frame. New regions cycle through default colors.

**Histogram** plots the current image, with a row per region and a column per enabled
Bayer component (a single column in Mono mode). **Merge histogram regions** combines
all listed samples into one histogram per channel. Overlapping regions contribute
pixels once for each region. The existing histogram sigma setting controls the visible
range around each distribution's mean.

**temporal_scatterplot** finds TIFF files in the current image's directory with the
same prefix and a numeric final underscore suffix, such as `capture_0.06ms.tif`,
`capture_0.1ms.tif`, and `capture_1ms.tif`. It supports scientific notation and
`s`, `ms`, `us` (also `µs`/`μs`), and `ns`; mixed time units are converted to the
current file's unit. Unitless numeric suffixes are also supported, separately from
unit-bearing names. At least two distinct times and matching image dimensions are
required. Files are sorted numerically, with actual time spacing on the X axis.

**Start file** and **End file** automatically show the first and last detected
filenames when a series is opened. Edit either filename to restrict plotting to that
inclusive range in numeric time order. Clear a field to leave that end unrestricted.
**Full range** restores the detected bounds. Opening another frame or reloading
within the same series preserves your edits; opening a different series fills in
its range. Invalid, reversed, or single-time ranges show an error before loading
image data.

Choose **Pixels** for every pixel, **Average** for one regional channel average per
frame, or **Heatmap** for pixel-count distributions at each time. Each region keeps
its style/color across channel panels. Heatmap color bars show counts per region;
overlapping heatmaps are overlaid. Every mode includes a dashed linear least-squares
fit of the per-frame averages for each region and channel, with the equation in the
legend. Average mode retains only averages between frames to reduce memory use.
Pixel and heatmap modes retain the selected samples and can use substantial memory
for large regions or long sequences.

Both plot types use the selected bit range, dark-image subtraction, and row correction.
Row references are recomputed for every temporal frame after dark subtraction.
Analysis preserves negative calibrated values. Gamma and color gains are display
adjustments and do not change plotted DN values. Color mode separates enabled Bayer
components using their absolute sensor coordinates; Mono mode uses all region pixels.

### Large temporal sequences

Temporal loading reads only the selected rectangles and, when row correction is
active, reference columns for the selected rows. Compatible uncompressed TIFFs are
memory-mapped read-only; other supported TIFFs decode only intersecting strips or
tiles, sharing decoded segments between overlapping regions. The TIFF storage layout
sets the minimum read size: a narrow region in a compressed full-width strip still
requires that entire strip to be decoded; a file stored as one compressed strip can
require decoding the whole frame.

Loading runs in a background thread. A progress window reports completed frames and
supports cancellation after the current frame. Calibration settings are captured when
the button is pressed. **Average** uses the least memory; **Pixels** still draws every
selected pixel and can take time to render when regions contain millions of samples.
Heatmap binning avoids building an extra concatenated copy of all pixel samples.

### Interactive axes

Every plot window includes a panel selector (or **All panels**), **X min**, **X max**,
**Y min**, **Y max**, and linear/log scale selectors. Edit the values and press
**Apply** to redraw without rereading images. **Reset axes** restores the original
limits and scales. Log scales require positive limits. When TkAgg is available, the
Matplotlib toolbar also supports pan, rectangle zoom, history, and saving. On Python
builds without TkAgg, the PNG fallback still supports the axis controls and **Save
plot…**.
