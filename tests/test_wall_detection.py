"""Tests for Phase 2 (wall detection), Steps A and B."""

import json

import numpy as np
from PIL import Image, ImageDraw

from floorplanto3d.processing.wall_detection import (
    COMBINED_STRUCTURES_FILENAME,
    FOREGROUND_MASK_COMPARISON_FILENAME,
    FOREGROUND_MASK_FILENAME,
    FOREGROUND_MASK_GLOBAL_FILENAME,
    HORIZONTAL_STRUCTURES_FILENAME,
    VERTICAL_STRUCTURES_FILENAME,
    WALL_REPORT_FILENAME,
    detect_walls,
    save_artifacts,
)

REQUIRED_REPORT_KEYS = {"phase", "steps", "input", "parameters", "summary"}


def make_plan(width: int = 400, height: int = 300) -> Image.Image:
    """A plan with one thick horizontal and one thick vertical wall.

    The two walls cross near the left side. A short thin mark (text-like)
    sits away from both: it must survive into the foreground masks but drop
    out of the axis-structure masks.
    """
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 134, 360, 146], fill=(0, 0, 0))
    draw.rectangle([94, 30, 106, 270], fill=(0, 0, 0))
    draw.line([(298, 250), (310, 250)], fill=(0, 0, 0), width=2)
    return image


def masked(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask)


def has_ink(array: np.ndarray, x: int, y: int, radius: int = 1) -> bool:
    window = array[y - radius : y + radius + 1, x - radius : x + radius + 1]
    return bool(np.any(window == 255))


def test_foreground_masks_keep_all_ink():
    plan = make_plan()

    result = detect_walls(plan, plan)

    for mask in (result.foreground_mask, result.global_mask):
        array = masked(mask)
        assert has_ink(array, 200, 140)  # horizontal wall
        assert has_ink(array, 100, 60)  # vertical wall
        assert has_ink(array, 304, 250)  # thin text-like mark


def test_axis_structures_are_separated():
    plan = make_plan()

    result = detect_walls(plan, plan)
    horizontal = masked(result.horizontal)
    vertical = masked(result.vertical)

    # The horizontal wall survives only the horizontal opening, and the
    # vertical wall only the vertical one, each sampled away from the
    # crossing near x = 100.
    assert has_ink(horizontal, 200, 140)
    assert not has_ink(horizontal, 100, 60)
    assert has_ink(vertical, 100, 60)
    assert not has_ink(vertical, 200, 140)
    # The crossing itself belongs to both structures.
    assert has_ink(horizontal, 100, 140)
    assert has_ink(vertical, 100, 140)


def test_combined_is_union_of_axis_masks():
    plan = make_plan()

    result = detect_walls(plan, plan)

    expected = np.bitwise_or(
        masked(result.horizontal), masked(result.vertical)
    )
    assert np.array_equal(masked(result.combined), expected)


def test_short_marks_drop_out_of_axis_masks():
    plan = make_plan()

    result = detect_walls(plan, plan)

    for mask in (result.horizontal, result.vertical, result.combined):
        assert not has_ink(masked(mask), 304, 250, radius=2)


def test_inputs_are_not_modified():
    plan = make_plan()
    original = plan.resize((200, 150))
    plan_before = plan.tobytes()
    original_before = original.tobytes()

    detect_walls(plan, original)

    assert plan.tobytes() == plan_before
    assert original.tobytes() == original_before


def test_save_artifacts_writes_masks_and_report(tmp_path):
    plan = make_plan()

    result = detect_walls(plan, plan)
    outputs = save_artifacts(result, tmp_path)

    assert set(outputs) == {
        "foreground_mask",
        "foreground_mask_global",
        "foreground_mask_comparison",
        "horizontal",
        "vertical",
        "combined",
        "report",
    }
    for name, path in outputs.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    report = json.loads((tmp_path / WALL_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert set(report) == REQUIRED_REPORT_KEYS
    assert report["phase"] == {"number": 2, "name": "wall_detection"}
    assert report["input"]["coordinate_space"] == "normalized"
    assert report["input"]["normalized_width"] == 400
    assert report["input"]["normalized_height"] == 300
    for key in (
        "adaptive_block_size",
        "adaptive_c",
        "global_threshold_method",
        "horizontal_kernel",
        "vertical_kernel",
    ):
        assert key in report["parameters"]
    assert report["summary"]["combined_pixels"] > 0
