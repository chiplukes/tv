"""Region parsing and temporal-series discovery, independent of the GUI."""

import csv
import re
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D


def parse_regions(text, shape):
    """Parse exclusive-end rectangles and validate Matplotlib format strings."""
    regions = []
    rows, cols = shape
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = next(csv.reader([line], skipinitialspace=True))
            if len(fields) != 5:
                raise ValueError("expected XSTART,YSTART,XEND,YEND,PLOTSTYLE")
            x0, y0, x1, y1 = map(int, fields[:4])
            style = fields[4].strip().strip("'\"")
            if not (0 <= x0 < x1 <= cols and 0 <= y0 < y1 <= rows):
                raise ValueError("coordinates must be in bounds with start < end")
            # Let Matplotlib validate its own format-string syntax.
            from matplotlib.axes._base import _process_plot_format

            _process_plot_format(style)
            regions.append((x0, y0, x1, y1, style))
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"Region line {number}: {exc}") from exc
    if not regions:
        raise ValueError("Add at least one region (Ctrl+drag or type a line).")
    return regions


def style_properties(style):
    """Resolve a plot format to reusable line and color properties."""
    from matplotlib.axes._base import _process_plot_format

    linestyle, marker, color = _process_plot_format(style)
    line = Line2D([], [], linestyle=linestyle, marker=marker, color=color or "C0")
    return line.get_color(), line.get_linestyle(), line.get_marker()


def temporal_files(filename, start="", end=""):
    """Find TIFFs sharing the current filename prefix and a numeric final suffix."""
    path = Path(filename)
    pattern = re.compile(r"^(.*)_([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(s|ms|us|µs|μs|ns)?$")
    current = pattern.fullmatch(path.stem)
    if current is None:
        raise ValueError("Current filename must end in _NUMBER.tif or _NUMBERms.tif (also s/us/ns).")
    scales = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ns": 1e-9}
    unit = current[3]
    files = []
    for candidate in path.parent.iterdir():
        match = pattern.fullmatch(candidate.stem)
        if candidate.suffix.lower() not in (".tif", ".tiff") or match is None or match[1] != current[1]:
            continue
        if bool(match[3]) != bool(unit):
            continue
        value = float(match[2])
        if unit:
            value *= scales[match[3]] / scales[unit]
        if np.isfinite(value):
            files.append((value, candidate))
    files.sort(key=lambda item: (item[0], str(item[1])))
    # Slice by numeric temporal order, including both named endpoint files.
    names = [candidate.name for _, candidate in files]
    endpoints = []
    for label, value, default in (("Start", start, 0), ("End", end, len(files) - 1)):
        value = value.strip()
        if value and value not in names:
            raise ValueError(f"{label} file {value!r} is not in the detected temporal series.")
        endpoints.append(names.index(value) if value else default)
    first, last = endpoints
    if first > last:
        raise ValueError("Start file must come before or equal the end file in temporal order.")
    files = files[first : last + 1]
    if len({value for value, _ in files}) < 2:
        raise ValueError(
            "Temporal plotting needs at least two distinct times in the selected range with the same filename prefix."
        )
    return files, f"Time ({unit})" if unit else "Time (filename suffix)"


class TiffRegionReader:
    """Read only requested rectangles, using a read-only map or selected TIFF segments."""

    def __init__(self, filename, shape):
        from tifffile import TiffFile, memmap

        self.tiff = TiffFile(filename)
        self.mapping = None
        try:
            if tuple(self.tiff.series[0].shape) != tuple(shape):
                raise ValueError(f"{Path(filename).name}: image dimensions do not match the current image.")
            self.page = self.tiff.pages[0]
            if self.page.shape != tuple(shape) or self.page.samplesperpixel != 1:
                raise ValueError("Region reading requires a single 2D grayscale TIFF page.")
            try:
                self.mapping = memmap(filename, mode="r", series=0)
            except ValueError:
                pass
        except Exception:
            self.tiff.close()
            raise

    def close(self):
        if self.mapping is not None:
            self.mapping._mmap.close()
        self.tiff.close()

    def read_many(self, rectangles):
        """Copy rectangles, decoding each intersecting tile/strip at most once."""
        if self.mapping is not None:
            return [np.array(self.mapping[y0:y1, x0:x1]) for x0, y0, x1, y1 in rectangles]
        page = self.page
        outputs = [np.zeros((y1 - y0, x1 - x0), dtype=page.dtype) for x0, y0, x1, y1 in rectangles]
        height = page.tilelength if page.is_tiled else page.rowsperstrip
        width = page.tilewidth if page.is_tiled else page.imagewidth
        across = (page.imagewidth + width - 1) // width
        wanted = set()
        for x0, y0, x1, y1 in rectangles:
            for row in range(y0 // height, (y1 - 1) // height + 1):
                for col in range(x0 // width, (x1 - 1) // width + 1):
                    wanted.add(row * across + col)
        handle = self.tiff.filehandle
        for index in sorted(wanted):
            handle.seek(page.dataoffsets[index])
            encoded = handle.read(page.databytecounts[index])
            kwargs = {}
            if page.compression in (6, 7, 34892, 33007):
                kwargs = dict(jpegtables=page.jpegtables, jpegheader=page.jpegheader)
            segment, position, _ = page.decode(encoded, index, **kwargs)
            if segment is None:
                continue
            tile = segment[0, :, :, 0]
            top, left = position[2:4]
            for rect, output in zip(rectangles, outputs):
                x0, y0, x1, y1 = rect
                xa, ya = max(x0, left), max(y0, top)
                xb, yb = min(x1, left + tile.shape[1]), min(y1, top + tile.shape[0])
                if xa < xb and ya < yb:
                    output[ya - y0 : yb - y0, xa - x0 : xb - x0] = tile[ya - top : yb - top, xa - left : xb - left]
        return outputs


def collect_temporal_samples(
    files, regions, channels, pattern, shape, msb, lsb, dark, row_correction, mode, progress=None, cancelled=None
):
    """Calibrate only selected pixels and required dark reference columns."""
    samples = [[[] for _ in channels] for _ in regions]
    mask = (1 << (msb - lsb + 1)) - 1

    def extract(raw):
        return ((raw.astype(np.uint32) >> lsb) & mask).astype(np.float64)

    for frame, (_, path) in enumerate(files):
        if cancelled is not None and cancelled.is_set():
            return None
        rects = [r[:4] for r in regions]
        if row_correction:
            offset, width = row_correction
            cols = shape[1]
            if offset < 0 or width <= 0 or cols - offset - width < offset + width:
                raise ValueError("Invalid row correction columns.")
            for _, y0, _, y1, _ in regions:
                rects.extend([(offset, y0, offset + width, y1), (cols - offset - width, y0, cols - offset, y1)])
        reader = TiffRegionReader(path, shape)
        try:
            arrays = reader.read_many(rects)
        finally:
            reader.close()
        for ri, (x0, y0, x1, y1, _) in enumerate(regions):
            data = extract(arrays[ri])
            if dark is not None:
                data -= dark[y0:y1, x0:x1]
            if row_correction:
                references = []
                for j in (len(regions) + 2 * ri, len(regions) + 2 * ri + 1):
                    ref = extract(arrays[j])
                    rx0, ry0, rx1, ry1 = rects[j]
                    if dark is not None:
                        ref -= dark[ry0:ry1, rx0:rx1]
                    references.append(ref)
                row_mean = (references[0].sum(axis=1) + references[1].sum(axis=1)) / (2 * width)
                data -= row_mean[:, None]
            for ci, channel in enumerate(channels):
                if channel == "Mono":
                    values = data.ravel()
                else:
                    row, col = next(pos for pos, name in pattern.items() if name == channel)
                    values = data[(row - y0) % 2 :: 2, (col - x0) % 2 :: 2].ravel()
                if mode == "Average" and values.size:
                    values = np.array([values.mean()])
                samples[ri][ci].append(values.copy())
        if progress:
            progress(frame + 1, len(files))
    return samples


class PlotPanelLayout:
    """Switch between the original layout and an expanded single-channel layout."""

    def __init__(self, figure):
        self.figure = figure
        self.panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
        self.channels = list(dict.fromkeys(ax.get_label() for ax in self.panels))
        self.original = {
            ax: (ax.get_subplotspec(), ax.get_position(original=True).frozen(), ax.get_axes_locator(), ax.get_visible())
            for ax in figure.axes
        }
        self.colorbars = {}
        for ax in figure.axes:
            colorbar = getattr(ax, "_colorbar", None)
            if colorbar is not None:
                self.colorbars.setdefault(colorbar.mappable.axes, []).append(ax)

    def select(self, channel=None):
        """Keep plotted data and zoom limits while changing visibility and layout."""
        if channel is not None and channel not in self.channels:
            raise ValueError(f"Unknown plot channel: {channel}")
        if channel is None:
            for ax, (spec, position, locator, visible) in self.original.items():
                if spec is not None:
                    ax.set_subplotspec(spec)
                ax.set_axes_locator(locator)
                ax.set_position(position)
                ax.set_visible(visible)
            return

        selected = [ax for ax in self.panels if ax.get_label() == channel]
        for ax in self.figure.axes:
            ax.set_visible(False)
        grid = self.figure.add_gridspec(len(selected), 1, left=0.08, right=0.94, bottom=0.13, top=0.88, hspace=0.65)
        for row, ax in enumerate(selected):
            bars = self.colorbars.get(ax, [])
            if bars:
                nested = grid[row].subgridspec(1, len(bars) + 1, width_ratios=[1] + [0.025] * len(bars), wspace=0.25)
                spec = nested[0]
                for index, bar in enumerate(bars):
                    bar.set_subplotspec(nested[index + 1])
                    bar.set_visible(True)
            else:
                spec = grid[row]
            ax.set_subplotspec(spec)
            ax.set_visible(True)
