"""Display-only linear camera processing. Never modifies analytical pixel data."""

import json
from pathlib import Path

import numpy as np
from tifffile import TiffFile


def camera_metadata(filename):
    metadata = {}
    if filename:
        try:
            with TiffFile(filename) as tif:
                record = json.loads(tif.pages[0].description or "{}")
                if isinstance(record, dict):
                    metadata = record
        except (ValueError, OSError):
            pass
    readout = metadata.get("readout", Path(filename or "").stem.rsplit("_", 1)[-1]).lower()
    levels = {
        "dgh": (512, 47600),
        "sgh": (512, 47600),
        "dgl": (512, 55700),
        "sgl": (512, 55700),
        **{name: (24, 55250) for name in ("blend3", "blend6", "blend12", "hard")},
    }
    black, white = levels.get(readout, (0, 65535))
    metadata.setdefault("black_level", black)
    metadata.setdefault("white_level", white)
    metadata["recognized"] = readout in levels or "readout" in metadata
    return metadata


def preview_rgb(planes, black, white, wb, ev, matrix, saturation, shoulder):
    rgb = np.stack((planes[0], (planes[1] + planes[2]) * 0.5, planes[3]), axis=-1).astype(np.float32)
    rgb = (rgb - black) / (white - black)
    rgb *= np.asarray(wb, dtype=np.float32)
    # A common ceiling prevents one balanced channel clipping before the others.
    rgb = np.minimum(rgb, min(wb))
    gain = 2.0**ev
    rgb = (rgb * gain) @ np.asarray(matrix, dtype=np.float32).T
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rgb = luminance[..., None] + saturation * (rgb - luminance[..., None])
    rgb = np.maximum(rgb, 0)
    if shoulder:
        # Hue-preserving luminance compression with a white point at exposed clip.
        peak = rgb.max(axis=-1, keepdims=True)
        knee = 0.55
        headroom = max(gain * min(wb), 1.0)
        mapped = knee + (1 - knee) * np.log1p(np.maximum(peak - knee, 0)) / np.log1p(headroom - knee)
        rgb *= np.where(peak > knee, mapped / np.maximum(peak, 1e-12), 1)
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return np.round(srgb * 255).astype(np.uint8)


def neutral_gains(planes, black, white):
    """Estimate residual balance from unclipped, above-black channel samples."""
    means = []
    for plane in planes:
        values = np.asarray(plane, dtype=float)
        valid = values[(values > black + 0.005 * (white - black)) & (values < black + 0.95 * (white - black))] - black
        if not valid.size:
            raise ValueError("Neutral patch needs unsaturated, above-black pixels in every Bayer channel.")
        means.append(valid.mean())
    green = (means[1] + means[2]) / 2
    return np.array([green / means[0], 1.0, green / means[3]])


def camera_readout(value, black, white, units):
    signal = value - black
    counts = signal / units
    stops = np.log2((white - black) / signal) if signal > 0 and white > black else np.inf
    return counts, stops
