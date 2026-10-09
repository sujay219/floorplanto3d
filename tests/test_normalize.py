"""Tests for Phase 1 (image normalization)."""

import json

import numpy as np
from PIL import Image, ImageDraw

from floorplanto3d.processing.normalize import (
    COMPARISON_GAP_PX,
    COMPARISON_LABEL_BAR_PX,
    NORMALIZE_MAX_SIDE_PX,
    normalize_image,
    save_artifacts,
)


def make_plan(width: int = 240, height: int = 120) -> Image.Image:
    """A tiny furnished-like plan: thin stroke, gray fill and a hairline."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [int(width * 0.1), int(height * 0.15), int(width * 0.9), int(height * 0.85)],
        outline=(0, 0, 0),
        width=3,
    )
    draw.rectangle(
        [int(width * 0.2), int(height * 0.3), int(width * 0.3), int(height * 0.5)],
        fill=(120, 120, 120),
    )
    draw.line(
        [int(width * 0.4), int(height * 0.2), int(width * 0.7), int(height * 0.7)],
        fill=(0, 0, 0),
        width=1,
    )
    return image


def test_records_dimensions_mode_and_format(tmp_path):
    path = tmp_path / "plan.png"
    make_plan().save(path)

    result = normalize_image(path)

    assert result.report["original"] == {
        "width": 240,
        "height": 120,
        "mode": "RGB",
        "format": "PNG",
        "aspect_ratio": 2.0,
    }


def test_original_pixels_are_not_modified():
    source = make_plan()

    result = normalize_image(source)

    assert result.original.mode == source.mode
    assert result.original.size == source.size
    assert result.original.tobytes() == source.tobytes()


def test_normalized_is_grayscale_at_original_size():
    result = normalize_image(make_plan())

    assert result.normalized.mode == "L"
    assert result.normalized.size == (240, 120)
    assert result.report["normalized"]["resized"] is False
    assert result.report["normalized"]["scale"] == 1.0


def test_resize_preserves_aspect_ratio():
    width = NORMALIZE_MAX_SIDE_PX + 8
    height = width // 2

    result = normalize_image(make_plan(width, height))

    report = result.report["normalized"]
    assert report["resized"] is True
    assert result.normalized.width == NORMALIZE_MAX_SIDE_PX
    assert abs(report["aspect_ratio"] - result.report["original"]["aspect_ratio"]) < 1e-3


def test_comparison_shows_original_and_normalized_side_by_side():
    source = make_plan()

    result = normalize_image(source)

    top = COMPARISON_LABEL_BAR_PX + COMPARISON_GAP_PX
    left_x = COMPARISON_GAP_PX
    right_x = COMPARISON_GAP_PX * 2 + source.width
    left = result.comparison.crop(
        (left_x, top, left_x + source.width, top + source.height)
    )
    right = result.comparison.crop(
        (right_x, top, right_x + source.width, top + source.height)
    )

    assert np.array_equal(np.asarray(left), np.asarray(source))
    assert np.array_equal(
        np.asarray(right),
        np.repeat(np.asarray(result.normalized)[:, :, None], 3, axis=2),
    )


def test_report_includes_inspection_and_parameters():
    result = normalize_image(make_plan())

    inspection = result.report["inspection"]
    assert set(inspection) == {"background", "contrast", "noise", "wall_thickness"}
    assert inspection["background"]["polarity"] == "light"
    assert inspection["wall_thickness"]["sample_count"] > 0

    parameters = result.report["parameters"]
    assert parameters["denoise"] == "none"
    assert parameters["crop"] == "none"


def test_save_artifacts_writes_images_and_report(tmp_path):
    result = normalize_image(make_plan())

    outputs = save_artifacts(result, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "comparison.png",
        "normalized.png",
        "original.png",
        "report.json",
    ]
    assert outputs["report"].name == "report.json"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["original"] == result.report["original"]
