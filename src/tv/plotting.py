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


def temporal_files(filename):
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
    if len({value for value, _ in files}) < 2:
        raise ValueError("Temporal plotting needs at least two distinct times with the same filename prefix.")
    return files, f"Time ({unit})" if unit else "Time (filename suffix)"
