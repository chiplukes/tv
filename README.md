# tv — TIFF Viewer

A TIFF viewer for viewing and analyzing raw Bayer CFA images.

## Features

- **Zoomable CFA viewer** — pan (left-drag), scroll-wheel zoom, shift-drag zoom rectangle
- **Bayer pattern support** — RGGB, BGGR, GRBG, GBRG with per-component channel toggles (R / Gr / Gb / B)
- **Show CFA mode** — toggle to display individual Bayer-coloured pixels instead of debayered colour
- **Colour / Mono rendering** — debayered colour view or raw grayscale
- **Gamma correction** — adjustable gamma for intensity scaling
- **Colour adjustment** — per-channel R / G / B multipliers
- **Dark pixel correction** — per-row dark subtraction from configurable dark columns
- **Flat-field analysis** — deviation heat-map for a single Bayer channel with block averaging, top/bottom half references, and a colour scale bar
- **Hover inspection** — real-time coordinate, component, and value readout that snaps to enabled channels
- **Export** — save the current viewport as a raw TIFF

## Installation

```bash
uv sync
```

## Usage

```bash
uv run tv
```

Opens a file dialog to select a TIFF/DNG file. If no file is selected, mock gradient data is generated for demonstration.

See [notes/viewer.md](notes/viewer.md) for a detailed description of each feature.

## Development

```bash
uv run ruff check src/
uv run ruff format src/
```

## Command line tool

```bash
uv tool install . --force --reinstall
```

then you can just type tv to get this tool!