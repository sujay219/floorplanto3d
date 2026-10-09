"""Tests for Phase 2 (rectangle detection)."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from floorplanto3d.processing.rectangles import (
    LARGE_RECTANGLES_FILENAME,
    MEDIUM_RECTANGLES_FILENAME,
    RECTANGLE_REPORT_FILENAME,
    RECTANGLES_FILENAME,
    RECTANGLES_OVERLAY_FILENAME,
    SMALL_RECTANGLES_FILENAME,
    detect_rectangles,
    save_artifacts,
)

REQUIRED_RECORD_KEYS = {
    "id",
    "category",
    "bbox",
    "corners",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "angle_deg",
    "method",
    "quality_score",
}


def make_plan(width: int = 400, height: int = 300) -> Image.Image:
    """A plan with one large, one medium and one small rectangular structure.

    The large and medium structures are drawn as outlines (so they carry an
    inner and an outer face, i.e. obvious duplicates) and the small filled
    structure sits inside the large one (i.e. a nested rectangle).
    """
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 380, 280], outline=(0, 0, 0), width=4)
    draw.rectangle([60, 60, 145, 115], outline=(0, 0, 0), width=3)
    draw.rectangle([250, 200, 280, 230], fill=(0, 0, 0))
    return image


def test_detects_candidates_at_multiple_scales():
    plan = make_plan()

    result = detect_rectangles(plan, plan)

    summary = result.report["summary"]
    assert summary["total_rectangles"] >= 5
    assert summary["drawn_rectangles"] >= 3
    for category in ("small", "medium", "large"):
        assert summary["by_category"][category] >= 1


def test_every_candidate_has_a_unique_id_and_full_metadata():
    plan = make_plan()

    result = detect_rectangles(plan, plan)

    ids = [rect.id for rect in result.rectangles]
    assert len(ids) == len(set(ids))
    assert ids[0] == "rect_001"
    for record in result.metadata["rectangles"]:
        assert set(record) >= REQUIRED_RECORD_KEYS
        assert record["area"] > 0
        assert record["aspect_ratio"] >= 1.0
        assert 0.0 <= record["quality_score"] <= 1.0
        assert record["bbox"]["width"] > 0 and record["bbox"]["height"] > 0
        assert len(record["corners"]) == 4


def test_outline_faces_are_reported_as_duplicates():
    plan = make_plan()

    result = detect_rectangles(plan, plan)

    duplicates = result.report["duplicates"]
    assert result.report["summary"]["duplicate_count"] == len(duplicates)
    assert len(duplicates) >= 1
    for pair in duplicates:
        assert pair["representative"] != pair["duplicate"]
    duplicate_ids = {pair["duplicate"] for pair in duplicates}
    assert any(rect.duplicate_of is not None for rect in result.rectangles)
    assert duplicate_ids == {
        rect.id for rect in result.rectangles if rect.duplicate_of is not None
    }


def test_nested_rectangles_are_reported():
    plan = make_plan()

    result = detect_rectangles(plan, plan)

    nested = result.report["nested"]
    assert result.report["summary"]["nested_count"] == len(nested)
    by_category = {
        rect.category: rect for rect in result.rectangles if rect.duplicate_of is None
    }
    outer_id = by_category["large"].id
    inner_id = by_category["small"].id
    assert {"outer": outer_id, "inner": inner_id, "containment": 1.0} in nested
    assert by_category["small"].nested_in == outer_id


def test_inputs_are_not_modified():
    plan = make_plan()
    normalized = plan.convert("L")

    detect_rectangles(normalized, plan)

    assert plan.tobytes() == make_plan().tobytes()
    assert normalized.tobytes() == make_plan().convert("L").tobytes()


def test_rotated_rectangle_angle_is_reported():
    canvas = np.full((300, 400, 3), 255, dtype=np.uint8)
    box = cv2.boxPoints(((200.0, 150.0), (100.0, 50.0), 30.0))
    cv2.fillPoly(canvas, [box.astype(np.int32)], (0, 0, 0))
    plan = Image.fromarray(canvas)

    result = detect_rectangles(plan, plan)

    rotated = [
        rect
        for rect in result.rectangles
        if rect.category == "medium" and abs(rect.area - 5000) < 1000
    ]
    assert rotated, "the rotated rectangle should be detected"
    rect = rotated[0]
    assert rect.axis_aligned is False
    assert abs(abs(rect.angle_deg) - 30.0) < 3.0
    assert rect.method in ("contour_polygon", "contour_min_area_rect")


def test_coordinates_are_reported_in_original_image_space():
    normalized = Image.new("L", (200, 150), 255)
    ImageDraw.Draw(normalized).rectangle([20, 20, 60, 50], fill=0)
    original = Image.new("RGB", (400, 300), (255, 255, 255))

    result = detect_rectangles(normalized, original)

    record = result.metadata["rectangles"][0]
    assert result.metadata["image"]["coordinate_space"] == "original"
    assert abs(record["bbox"]["x"] - 40) <= 2
    assert abs(record["bbox"]["y"] - 40) <= 2
    assert abs(record["bbox"]["width"] - 80) <= 3
    assert abs(record["bbox"]["height"] - 60) <= 3
    assert result.report["parameters"]["scale_from_normalized"] == 2.0


def test_save_artifacts_writes_images_and_documents(tmp_path):
    plan = make_plan()
    result = detect_rectangles(plan, plan)

    outputs = save_artifacts(result, tmp_path)

    expected = {
        "overlay": RECTANGLES_OVERLAY_FILENAME,
        "small": SMALL_RECTANGLES_FILENAME,
        "medium": MEDIUM_RECTANGLES_FILENAME,
        "large": LARGE_RECTANGLES_FILENAME,
        "rectangles": RECTANGLES_FILENAME,
        "report": RECTANGLE_REPORT_FILENAME,
    }
    assert set(outputs) == set(expected)
    for key, filename in expected.items():
        assert outputs[key] == tmp_path / filename
        assert outputs[key].exists()
        assert outputs[key].stat().st_size > 0

    metadata = json.loads((tmp_path / RECTANGLES_FILENAME).read_text(encoding="utf-8"))
    report = json.loads(
        (tmp_path / RECTANGLE_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert len(metadata["rectangles"]) == len(result.rectangles)
    assert report["summary"]["total_rectangles"] == len(result.rectangles)
    assert "parameters" in report
    assert report["phase"] == {"number": 2, "name": "rectangle_detection"}


def test_views_share_the_original_canvas_size():
    plan = make_plan()

    result = detect_rectangles(plan, plan)

    for image in (result.overlay, result.small_view, result.medium_view, result.large_view):
        assert image.size == plan.size
        assert image.mode == "RGB"
