"""Phase 2: wall detection.

A wall in a floor plan is a *thick stroke with two edges*. Line detectors
such as the Hough transform lock onto each edge independently, so one
thick wall comes back as two duplicate or crossing lines. This phase
therefore looks for thick regions first and defers every line-level
decision.

Step A — foreground mask. The normalized working image is separated into
dark drawing pixels and light background twice: adaptive thresholding
(the primary mask, robust to uneven backgrounds) and global thresholding
(kept beside it for comparison). Furniture, text and annotations are
preserved: nothing is removed at this step.

Step B — axis structures. Morphological opening with long horizontal and
vertical kernels isolates long horizontal and vertical strokes from the
primary mask. The combined mask is a *diagnostic* only: it is never fed
into the room extractor, and no wall decision is made from it.

Step C — wall-candidate identification (continuity, consistent thickness,
parallel-edge pairing, plausible connectivity) is deliberately not
implemented here. It follows only after these diagnostics have been
inspected and the kernel starting values tuned.

Nothing is removed from any image and the inputs are only ever read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from floorplanto3d.processing.image import load_image, open_image
from floorplanto3d.processing.normalize import (
    COMPARISON_CANVAS_COLOR,
    COMPARISON_GAP_PX,
    COMPARISON_LABEL_BAR_PX,
    COMPARISON_LABEL_COLOR,
    png_bytes,
)

logger = logging.getLogger(__name__)

# Step A — foreground thresholding. The adaptive settings are starting
# values from the wall-detection spec. The global threshold (Otsu) exists
# for comparison only; the adaptive mask is the one Step B works from.
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 9
GLOBAL_THRESHOLD_METHOD = "otsu"

# Step B — axis structure extraction. Kernel lengths are starting values,
# not final settings: they may miss short walls, diagonal walls and broken
# strokes until they are tuned against the inspected outputs. Each kernel
# keeps only runs at least as long as its side.
HORIZONTAL_KERNEL_SIZE = (35, 1)
VERTICAL_KERNEL_SIZE = (1, 35)
AXIS_OPEN_OPERATION = "open"

# Side-by-side mask comparison labels (kept minimal so the masks stay
# readable underneath).
COMPARISON_LABEL_LEFT = "adaptive threshold"
COMPARISON_LABEL_RIGHT = "global threshold (otsu)"
COMPARISON_FONT_SCALE = 0.45
COMPARISON_FONT_THICKNESS = 1
COMPARISON_LABEL_INSET_PX = 6

# Saved artifact filenames.
FOREGROUND_MASK_FILENAME = "01_foreground_mask.png"
FOREGROUND_MASK_GLOBAL_FILENAME = "01_foreground_mask_global.png"
FOREGROUND_MASK_COMPARISON_FILENAME = "01_foreground_mask_comparison.png"
HORIZONTAL_STRUCTURES_FILENAME = "02_horizontal.png"
VERTICAL_STRUCTURES_FILENAME = "03_vertical.png"
COMBINED_STRUCTURES_FILENAME = "04_combined.png"
WALL_REPORT_FILENAME = "wall_report.json"


@dataclass
class WallDetectionResult:
    """Outputs of Phase 2 (wall detection), Steps A and B.

    All images are binary masks in the normalized working image's
    coordinate space (255 = foreground ink).
    """

    foreground_mask: Image.Image
    global_mask: Image.Image
    mask_comparison: Image.Image
    horizontal: Image.Image
    vertical: Image.Image
    combined: Image.Image
    report: dict[str, Any]


def detect_walls(
    normalized: str | Path | bytes | bytearray | Image.Image,
    original: str | Path | bytes | bytearray | Image.Image,
) -> WallDetectionResult:
    """Run Phase 2 (Steps A and B) on a floor plan.

    Args:
        normalized: The normalized working image produced by Phase 1. It is
            only read; detection never modifies it.
        original: The original floor plan image. Only its identity
            (dimensions) is recorded in the report; its pixels are untouched.

    Returns:
        A :class:`WallDetectionResult` with the Step A masks, the Step B
        structure masks and the ``wall_report.json`` document. No wall
        candidates are produced yet (Step C is pending inspection).
    """
    normalized_image = open_image(normalized)
    original_image = open_image(original)
    gray = _gray_array(normalized_image)

    adaptive = _adaptive_mask(gray)
    global_mask, otsu_value = _global_mask(gray)

    horizontal, vertical, combined = _axis_structures(adaptive)
    comparison = _comparison_view(adaptive, global_mask)

    report = _build_report(
        original_image,
        normalized_image,
        otsu_value,
        adaptive,
        global_mask,
        horizontal,
        vertical,
        combined,
    )

    logger.info(
        "wall detection (Steps A+B) complete: foreground %.2f%% adaptive / "
        "%.2f%% global, horizontal %d px, vertical %d px, combined %d px",
        report["summary"]["foreground_fraction_adaptive"] * 100.0,
        report["summary"]["foreground_fraction_global"] * 100.0,
        report["summary"]["horizontal_pixels"],
        report["summary"]["vertical_pixels"],
        report["summary"]["combined_pixels"],
    )
    return WallDetectionResult(
        foreground_mask=Image.fromarray(adaptive),
        global_mask=Image.fromarray(global_mask),
        mask_comparison=comparison,
        horizontal=Image.fromarray(horizontal),
        vertical=Image.fromarray(vertical),
        combined=Image.fromarray(combined),
        report=report,
    )


def save_artifacts(
    result: WallDetectionResult, directory: str | Path
) -> dict[str, Path]:
    """Write the Phase 2 artifacts into ``directory``.

    Writes ``01_foreground_mask.png``, ``01_foreground_mask_global.png``,
    ``01_foreground_mask_comparison.png``, ``02_horizontal.png``,
    ``03_vertical.png``, ``04_combined.png`` and ``wall_report.json``, and
    returns the output paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "foreground_mask": directory / FOREGROUND_MASK_FILENAME,
        "foreground_mask_global": directory / FOREGROUND_MASK_GLOBAL_FILENAME,
        "foreground_mask_comparison": directory / FOREGROUND_MASK_COMPARISON_FILENAME,
        "horizontal": directory / HORIZONTAL_STRUCTURES_FILENAME,
        "vertical": directory / VERTICAL_STRUCTURES_FILENAME,
        "combined": directory / COMBINED_STRUCTURES_FILENAME,
        "report": directory / WALL_REPORT_FILENAME,
    }
    outputs["foreground_mask"].write_bytes(png_bytes(result.foreground_mask))
    outputs["foreground_mask_global"].write_bytes(png_bytes(result.global_mask))
    outputs["foreground_mask_comparison"].write_bytes(png_bytes(result.mask_comparison))
    outputs["horizontal"].write_bytes(png_bytes(result.horizontal))
    outputs["vertical"].write_bytes(png_bytes(result.vertical))
    outputs["combined"].write_bytes(png_bytes(result.combined))
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def _gray_array(image: Image.Image) -> np.ndarray:
    """8-bit luma of an image via the project loader."""
    rgb, _, _, _ = load_image(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _adaptive_mask(gray: np.ndarray) -> np.ndarray:
    """Primary foreground mask (255 = ink), robust to uneven backgrounds."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def _global_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Comparison foreground mask (255 = ink) and the Otsu threshold used."""
    value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask, float(value)


def _axis_structures(
    binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long horizontal and vertical strokes of the foreground mask.

    Opening with a long axis-aligned kernel keeps only runs at least as
    long as the kernel, so these masks are structural diagnostics: short
    marks (much text and furniture) drop out, diagonal walls drop out
    entirely, and broken strokes drop out until the kernels are tuned.
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, HORIZONTAL_KERNEL_SIZE)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, VERTICAL_KERNEL_SIZE)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    combined = cv2.bitwise_or(horizontal, vertical)
    return horizontal, vertical, combined


def _comparison_view(adaptive: np.ndarray, global_mask: np.ndarray) -> Image.Image:
    """Side-by-side adaptive and global masks with minimal labels."""
    panels = [cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) for mask in (adaptive, global_mask)]
    gap = np.full((adaptive.shape[0], COMPARISON_GAP_PX, 3), COMPARISON_CANVAS_COLOR, np.uint8)
    row = np.hstack([panels[0], gap, panels[1]])

    canvas = np.full(
        (row.shape[0] + COMPARISON_LABEL_BAR_PX, row.shape[1], 3),
        COMPARISON_CANVAS_COLOR,
        np.uint8,
    )
    canvas[COMPARISON_LABEL_BAR_PX:, :] = row

    baseline = COMPARISON_LABEL_BAR_PX - COMPARISON_LABEL_INSET_PX
    cv2.putText(
        canvas,
        COMPARISON_LABEL_LEFT,
        (COMPARISON_LABEL_INSET_PX, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        COMPARISON_FONT_SCALE,
        COMPARISON_LABEL_COLOR,
        COMPARISON_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        COMPARISON_LABEL_RIGHT,
        (panels[0].shape[1] + COMPARISON_GAP_PX + COMPARISON_LABEL_INSET_PX, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        COMPARISON_FONT_SCALE,
        COMPARISON_LABEL_COLOR,
        COMPARISON_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return Image.fromarray(canvas)


def _ink_summary(mask: np.ndarray) -> tuple[int, float]:
    """Foreground pixel count and its fraction of the mask."""
    count = int(np.count_nonzero(mask))
    return count, round(count / max(1, mask.size), 6)


def _build_report(
    original: Image.Image,
    normalized: Image.Image,
    otsu_value: float,
    adaptive: np.ndarray,
    global_mask: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    combined: np.ndarray,
) -> dict[str, Any]:
    """The ``wall_report.json`` document."""
    adaptive_pixels, adaptive_fraction = _ink_summary(adaptive)
    global_pixels, global_fraction = _ink_summary(global_mask)
    return {
        "phase": {"number": 2, "name": "wall_detection"},
        "steps": {
            "a": "foreground mask (adaptive primary, global comparison); "
            "furniture and text preserved, nothing removed",
            "b": "long horizontal and vertical strokes by morphological "
            "opening (diagnostic only, not for the room extractor)",
            "c": "not implemented (wall-candidate identification follows "
            "after these outputs have been inspected)",
        },
        "input": {
            "original_width": original.width,
            "original_height": original.height,
            "normalized_width": normalized.width,
            "normalized_height": normalized.height,
            "coordinate_space": "normalized",
        },
        "parameters": {
            "adaptive_block_size": ADAPTIVE_BLOCK_SIZE,
            "adaptive_c": ADAPTIVE_C,
            "global_threshold_method": GLOBAL_THRESHOLD_METHOD,
            "global_threshold_value": round(otsu_value, 2),
            "horizontal_kernel": list(HORIZONTAL_KERNEL_SIZE),
            "vertical_kernel": list(VERTICAL_KERNEL_SIZE),
            "morph_operation": AXIS_OPEN_OPERATION,
        },
        "summary": {
            "foreground_pixels_adaptive": adaptive_pixels,
            "foreground_fraction_adaptive": adaptive_fraction,
            "foreground_pixels_global": global_pixels,
            "foreground_fraction_global": global_fraction,
            "horizontal_pixels": int(np.count_nonzero(horizontal)),
            "vertical_pixels": int(np.count_nonzero(vertical)),
            "combined_pixels": int(np.count_nonzero(combined)),
        },
    }
