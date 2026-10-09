"""Phase 1: image normalization.

Loads the original floor-plan image unmodified, records its identity
(dimensions, image mode and container format), inspects its imaging
properties (background, contrast, noise and ink stroke width) and produces a
normalized working image for the later computer-vision phases.

This layer deliberately performs no thresholding, wall detection or room
detection. Nothing is cropped, so furniture, text and annotations stay in the
image. No denoising is applied and the image is never enlarged, so thin wall
strokes survive untouched.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from floorplanto3d.processing.image import open_image

# Working image: shrink-only cap on the longest side. Both axes share the
# same factor, so the aspect ratio is preserved exactly. The cap is generous
# so thin wall strokes are not averaged away on ordinary inputs.
NORMALIZE_MAX_SIDE_PX = 4000

# Background map: Gaussian sigma as a fraction of the shorter side, clamped
# from below. The kernel is far wider than any wall stroke, so individual
# strokes cannot pull the background estimate down.
BACKGROUND_SIGMA_SCALE = 20.0
BACKGROUND_SIGMA_MIN_PX = 16.0
# The map is estimated on a downscaled copy purely for speed; the map only
# carries illumination far larger than this factor anyway.
BACKGROUND_DOWNSCALE = 4

# Flat-fielding: background is scaled to this paper level. The divisor floor
# bounds the gain (max gain = target / floor) so ink-dense regions can never
# be blown out.
BACKGROUND_TARGET_LEVEL = 255.0
BACKGROUND_DIVISOR_FLOOR = 128.0

# Contrast normalization: linear stretch between these input percentiles.
# A monotone remap: faint thin strokes get darker (more visible), never lost.
CONTRAST_LOW_PERCENTILE = 1.0
CONTRAST_HIGH_PERCENTILE = 99.0
CONTRAST_MIN_RANGE = 1.0

# Inspection: paper/background gray level and polarity.
BACKGROUND_LEVEL_PERCENTILE = 98.0
BACKGROUND_POLARITY_LEVEL = 128.0

# Inspection: robust noise sigma (MAD scaled to a normal std) of the residual
# left after a median filter of this kernel size.
NOISE_MEDIAN_KERNEL = 3
NOISE_MAD_SCALE = 1.4826

# Inspection: ink stroke width from dark runs at or below this gray level,
# keeping only runs up to this length (longer runs are wall lengths, not
# stroke widths). Measurement only: no binary detection output is produced.
INK_DARK_LEVEL = 128
STROKE_RUN_MIN_PX = 1
STROKE_RUN_MAX_PX = 64

# Side-by-side comparison image.
COMPARISON_GAP_PX = 12
COMPARISON_LABEL_BAR_PX = 28
COMPARISON_CANVAS_COLOR = (255, 255, 255)
COMPARISON_LABEL_COLOR = (32, 32, 32)
COMPARISON_LABEL_LEFT = "original"
COMPARISON_LABEL_RIGHT = "normalized"

# Saved artifact filenames.
ORIGINAL_FILENAME = "original.png"
NORMALIZED_FILENAME = "normalized.png"
COMPARISON_FILENAME = "comparison.png"
REPORT_FILENAME = "report.json"

PNG_SAFE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"})

STROKE_WIDTH_METHOD = (
    "dark-run length statistics at gray <= "
    f"{INK_DARK_LEVEL} (inspection estimate only; no wall detection)"
)
NOISE_METHOD = "robust MAD of median-filter residual (noise + fine texture)"


@dataclass
class NormalizationResult:
    """Outputs of Phase 1 (image normalization)."""

    original: Image.Image
    normalized: Image.Image
    comparison: Image.Image
    report: dict[str, Any]


def normalize_image(source: str | Path | bytes | bytearray | Image.Image) -> NormalizationResult:
    """Run Phase 1 on a floor plan image.

    Args:
        source: File path, raw image bytes, or a PIL image.

    Returns:
        A :class:`NormalizationResult` with the untouched original, the
        normalized working image, the side-by-side comparison and the report.
    """
    original = open_image(source)
    width, height = original.size
    gray = _to_gray_array(original)

    background_sigma = max(
        BACKGROUND_SIGMA_MIN_PX, min(width, height) / BACKGROUND_SIGMA_SCALE
    )

    working = gray
    scale = 1.0
    if max(width, height) > NORMALIZE_MAX_SIDE_PX:
        scale = NORMALIZE_MAX_SIDE_PX / max(width, height)
        working = cv2.resize(
            working,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    normalized = _normalize_gray(working, background_sigma)
    normalized_image = Image.fromarray(normalized, mode="L")
    comparison = _build_comparison(original, normalized_image)

    report = {
        "original": {
            "width": width,
            "height": height,
            "mode": original.mode,
            "format": (original.format or "").upper() or None,
            "aspect_ratio": round(width / height, 6),
        },
        "normalized": {
            "width": normalized_image.width,
            "height": normalized_image.height,
            "mode": normalized_image.mode,
            "aspect_ratio": round(normalized_image.width / normalized_image.height, 6),
            "resized": scale != 1.0,
            "scale": round(scale, 6),
        },
        "inspection": _inspect(gray, normalized, background_sigma),
        "parameters": {
            "color_mode": "L",
            "resize_max_side_px": NORMALIZE_MAX_SIDE_PX,
            "resize_scale": round(scale, 6),
            "background_sigma_px": round(background_sigma, 3),
            "background_target_level": BACKGROUND_TARGET_LEVEL,
            "background_divisor_floor": BACKGROUND_DIVISOR_FLOOR,
            "contrast_low_percentile": CONTRAST_LOW_PERCENTILE,
            "contrast_high_percentile": CONTRAST_HIGH_PERCENTILE,
            "denoise": "none",
            "crop": "none",
        },
    }

    return NormalizationResult(
        original=original,
        normalized=normalized_image,
        comparison=comparison,
        report=report,
    )


def save_artifacts(result: NormalizationResult, directory: str | Path) -> dict[str, Path]:
    """Write ``original.png``, ``normalized.png``, ``comparison.png`` and
    ``report.json`` into ``directory`` and return the output paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "original": directory / ORIGINAL_FILENAME,
        "normalized": directory / NORMALIZED_FILENAME,
        "comparison": directory / COMPARISON_FILENAME,
        "report": directory / REPORT_FILENAME,
    }
    outputs["original"].write_bytes(png_bytes(result.original))
    outputs["normalized"].write_bytes(png_bytes(result.normalized))
    outputs["comparison"].write_bytes(png_bytes(result.comparison))
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def png_bytes(image: Image.Image) -> bytes:
    """Encode an image as PNG, converting only modes PNG cannot store."""
    if image.mode not in PNG_SAFE_MODES:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white and convert to RGB."""
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _to_gray_array(image: Image.Image) -> np.ndarray:
    """8-bit luma of the image, with transparency flattened onto white."""
    return np.asarray(_to_rgb(image).convert("L"))


def _background_map(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth illumination map of the image.

    Estimated on a downscaled copy (the map only carries illumination far
    wider than any stroke) and bilinearly restored to the input size.
    """
    height, width = gray.shape
    if min(height, width) >= 4 * BACKGROUND_DOWNSCALE:
        small = cv2.resize(
            gray,
            None,
            fx=1.0 / BACKGROUND_DOWNSCALE,
            fy=1.0 / BACKGROUND_DOWNSCALE,
            interpolation=cv2.INTER_AREA,
        )
        blurred = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma / BACKGROUND_DOWNSCALE))
        return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(gray, (0, 0), sigma)


def _normalize_gray(gray: np.ndarray, background_sigma: float) -> np.ndarray:
    """Flat-field the background and stretch the contrast.

    Flat-fielding scales each pixel by the local illumination ratio (bounded
    by ``BACKGROUND_DIVISOR_FLOOR``), then a percentile stretch maps the
    document's ink/paper range onto the full 8-bit range. Nothing here is a
    denoiser or a threshold: all ink keeps its relative ordering.
    """
    gray32 = gray.astype(np.float32)
    background = _background_map(gray32, background_sigma)
    gain = BACKGROUND_TARGET_LEVEL / np.maximum(background, BACKGROUND_DIVISOR_FLOOR)
    flattened = np.clip(gray32 * gain, 0.0, 255.0)

    low, high = (
        float(value)
        for value in np.percentile(
            flattened, [CONTRAST_LOW_PERCENTILE, CONTRAST_HIGH_PERCENTILE]
        )
    )
    span = max(high - low, CONTRAST_MIN_RANGE)
    stretched = (flattened - low) * (255.0 / span)
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, NOISE_MEDIAN_KERNEL)
    return float(NOISE_MAD_SCALE * np.median(np.abs(residual)))


def _iter_lines(mask: np.ndarray):
    for axis_mask in (mask, mask.T):
        yield from axis_mask


def _dark_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of every dark run along rows and columns.

    Runs touching the border of their line are censored (their true length is
    unknown), so only interior runs are reported.
    """
    lengths: list[np.ndarray] = []
    for line in _iter_lines(mask):
        padded = np.zeros(line.size + 2, dtype=np.int8)
        padded[1:-1] = line
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if starts.size and starts[0] == 0:
            starts, ends = starts[1:], ends[1:]
        if ends.size and ends[-1] == line.size:
            starts, ends = starts[:-1], ends[:-1]
        lengths.append(ends - starts)
    if not lengths:
        return np.array([], dtype=np.int64)
    return np.concatenate(lengths)


def _inspect(
    gray: np.ndarray, normalized: np.ndarray, background_sigma: float
) -> dict[str, Any]:
    """Measurement-only inspection of the source image."""
    gray32 = gray.astype(np.float32)
    background = _background_map(gray32, background_sigma)
    level = float(np.percentile(gray32, BACKGROUND_LEVEL_PERCENTILE))

    low, median, high = (
        float(value)
        for value in np.percentile(
            gray32, [CONTRAST_LOW_PERCENTILE, 50.0, CONTRAST_HIGH_PERCENTILE]
        )
    )

    runs = _dark_run_lengths(gray <= INK_DARK_LEVEL)
    widths = runs[(runs >= STROKE_RUN_MIN_PX) & (runs <= STROKE_RUN_MAX_PX)]
    if widths.size:
        p25, p50, p75, p90 = (
            round(float(value), 1) for value in np.percentile(widths, [25, 50, 75, 90])
        )
        stroke_width: dict[str, Any] = {
            "p25_px": p25,
            "median_px": p50,
            "p75_px": p75,
            "p90_px": p90,
            "sample_count": int(widths.size),
            "method": STROKE_WIDTH_METHOD,
        }
    else:
        stroke_width = {
            "p25_px": None,
            "median_px": None,
            "p75_px": None,
            "p90_px": None,
            "sample_count": 0,
            "method": STROKE_WIDTH_METHOD,
        }

    return {
        "background": {
            "level": round(level, 1),
            "polarity": "light" if level >= BACKGROUND_POLARITY_LEVEL else "dark",
            "gradient_std": round(float(background.std()), 2),
            "gradient_range": round(
                float(background.max() - background.min()), 2
            ),
        },
        "contrast": {
            "min": int(gray32.min()),
            "max": int(gray32.max()),
            "low": round(low, 1),
            "median": round(median, 1),
            "high": round(high, 1),
            "dynamic_range": round(high - low, 1),
            "rms_contrast": round(
                float(gray32.std() / max(float(gray32.mean()), 1e-6)), 4
            ),
        },
        "noise": {
            "sigma": round(_estimate_noise_sigma(gray), 2),
            "sigma_normalized": round(_estimate_noise_sigma(normalized), 2),
            "method": NOISE_METHOD,
        },
        "wall_thickness": stroke_width,
    }


def _build_comparison(original: Image.Image, normalized: Image.Image) -> Image.Image:
    """Side-by-side original vs normalized, labelled on a header bar."""
    left = _to_rgb(original)
    right = normalized.convert("RGB")
    gap = COMPARISON_GAP_PX

    canvas = Image.new(
        "RGB",
        (
            gap * 3 + left.width + right.width,
            COMPARISON_LABEL_BAR_PX + gap + max(left.height, right.height),
        ),
        COMPARISON_CANVAS_COLOR,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    left_x = gap
    right_x = gap * 2 + left.width
    label_y = COMPARISON_LABEL_BAR_PX // 2
    draw.text(
        (left_x, label_y),
        COMPARISON_LABEL_LEFT,
        fill=COMPARISON_LABEL_COLOR,
        font=font,
        anchor="lm",
    )
    draw.text(
        (right_x, label_y),
        COMPARISON_LABEL_RIGHT,
        fill=COMPARISON_LABEL_COLOR,
        font=font,
        anchor="lm",
    )

    top = COMPARISON_LABEL_BAR_PX + gap
    canvas.paste(left, (left_x, top))
    canvas.paste(right, (right_x, top))
    return canvas
