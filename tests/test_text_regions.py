"""Tests for Phase 2, Step C2 (proposed text/annotation regions).

Scenes are synthetic 600x450 plans with glyph-like marks (hollow small
boxes), thick wall strokes and large furniture outlines. Assertions check
that text-like groups are proposed and architectural shapes are not, that
the mask and overlay keep the source geometry and pixels, and that the
documents describe every region as a proposal — never as confirmed text.
"""

import json

import numpy as np
from PIL import Image, ImageDraw

from floorplanto3d.processing.text_regions import (
    TEXT_DETECTION_REPORT_FILENAME,
    TEXT_REGIONS_FILENAME,
    TEXT_REGIONS_MASK_FILENAME,
    TEXT_REGIONS_OVERLAY_FILENAME,
    detect_text_regions,
    main,
    save_artifacts,
)

WIDTH, HEIGHT = 600, 450
REQUIRED_REGION_KEYS = {
    "id",
    "role",
    "role_is_proposed",
    "detection_method",
    "polarity",
    "bbox",
    "dimensions",
    "quality",
    "quality_evidence",
    "component_count",
    "component_bboxes",
    "evidence",
    "uncertainty",
}
REQUIRED_DIMENSION_KEYS = {
    "width",
    "height",
    "aspect_ratio",
    "ink_pixels",
    "fill_ratio",
    "glyph_height_px",
    "stroke_width_px",
}
REQUIRED_REPORT_KEYS = {"phase", "steps", "summary", "input", "parameters", "constraints_respected", "limitations"}


def build(draw_fn) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw_fn(ImageDraw.Draw(image))
    return image


def run(plan: Image.Image):
    return detect_text_regions(plan, plan)


def glyph(draw: ImageDraw.ImageDraw, x: int, y: int, size=(14, 20), color=(0, 0, 0)) -> None:
    w, h = size
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)


def glyph_row(
    draw: ImageDraw.ImageDraw, x: int, y: int, count: int = 5, size=(14, 20), gap: int = 8,
    color=(0, 0, 0),
) -> None:
    for index in range(count):
        glyph(draw, x + index * (size[0] + gap), y, size=size, color=color)


def plan_like(draw: ImageDraw.ImageDraw) -> None:
    """A miniature plan: enough ink for Phase 1 plus a text row."""
    draw.rectangle([50, 300, 550, 316], fill=(0, 0, 0))
    draw.rectangle([280, 40, 296, 410], fill=(0, 0, 0))
    draw.rectangle([80, 340, 260, 420], fill=(40, 40, 40))
    text_row(draw)


def overlapping(regions, box):
    x0, y0, x1, y1 = box
    matches = []
    for region in regions:
        bx, by, bw, bh = region.bbox
        if not (bx + bw < x0 or bx > x1 or by + bh < y0 or by > y1):
            matches.append(region)
    return matches


def text_row(draw: ImageDraw.ImageDraw) -> None:
    glyph_row(draw, 60, 80, count=5)


def test_glyph_row_is_proposed_as_one_text_region():
    plan = build(text_row)

    regions = run(plan).regions

    matches = overlapping(regions, (55, 75, 175, 110))
    assert matches, "a row of glyph-like marks must produce a proposed region"
    region = matches[0]
    assert region.component_count >= 3, "the row must be grouped, not one box per mark"
    assert 0.0 <= region.quality <= 1.0
    assert region.role in {"text_line", "dimension_like", "room_label_like"}
    assert "spatial_grouping" in region.detection_method
    assert region.evidence or region.uncertainty


def test_walls_and_furniture_outlines_are_not_proposed_as_text():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([50, 300, 550, 316], fill=(0, 0, 0))
        draw.rectangle([280, 40, 296, 410], fill=(0, 0, 0))
        draw.rectangle([320, 60, 520, 200], outline=(0, 0, 0), width=3)
        draw.rectangle([60, 360, 240, 420], outline=(0, 0, 0), width=3)

    regions = run(build(draw_fn)).regions

    for box in ((50, 300, 550, 316), (280, 40, 296, 410), (320, 60, 520, 200), (60, 360, 240, 420)):
        assert not overlapping(regions, box), f"architectural/furniture shape at {box} must not be proposed"


def test_white_text_on_dark_fill_is_proposed():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([40, 60, 240, 160], fill=(60, 60, 60))
        glyph_row(draw, 60, 100, count=5, size=(12, 18), gap=10, color=(255, 255, 255))

    regions = run(build(draw_fn)).regions

    matches = overlapping(regions, (45, 90, 235, 130))
    assert matches, "light glyphs on a dark fill must be proposed"
    assert any(r.polarity == "light_on_dark" for r in matches)


def test_large_glyphs_are_proposed_as_room_labels():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        glyph_row(draw, 80, 150, count=4, size=(24, 30), gap=10)

    regions = run(build(draw_fn)).regions

    matches = overlapping(regions, (75, 145, 230, 185))
    assert matches
    assert any(r.role == "room_label_like" for r in matches)


def test_mask_is_binary_filled_bboxes_at_source_resolution():
    plan = build(text_row)

    result = run(plan)

    mask = np.asarray(result.mask)
    assert result.mask.size == plan.size
    assert set(np.unique(mask)) <= {0, 255}
    expected = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for region in result.regions:
        x, y, w, h = region.bbox
        expected[y : y + h, x : x + w] = 255
    assert np.array_equal(mask, expected), "the mask must be exactly the filled region boxes"


def test_overlay_is_at_source_resolution_and_inputs_untouched():
    plan = build(text_row)
    before = plan.tobytes()

    result = run(plan)

    assert result.overlay.size == plan.size
    assert plan.tobytes() == before, "the source image must not be modified"


def test_every_region_is_reported_with_geometry_and_quality():
    plan = build(text_row)

    document = run(plan).document

    assert document["status"].startswith("proposed")
    assert document["ocr"]["used"] is False
    assert document["image"]["coordinate_space"] == "original"
    assert document["mask"]["width"] == plan.width and document["mask"]["height"] == plan.height
    records = document["regions"]
    assert records
    ids = [record["id"] for record in records]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for record in records:
        assert set(record) == REQUIRED_REGION_KEYS
        assert set(record["dimensions"]) == REQUIRED_DIMENSION_KEYS
        assert record["role_is_proposed"] is True
        assert 0.0 <= record["quality"] <= 1.0
        assert record["detection_method"].startswith("connected_components")
        bbox = record["bbox"]
        assert 0 <= bbox["x"] < plan.width and 0 <= bbox["y"] < plan.height
        assert "classification" not in record, "regions must stay proposals"


def test_report_counts_regions_and_declares_limitations():
    plan = build(text_row)

    report = run(plan).report

    assert set(report) == REQUIRED_REPORT_KEYS
    assert report["phase"] == {"number": 3, "name": "text_regions", "step": "c2"}
    assert report["summary"]["total_regions"] >= 1
    assert sum(report["summary"]["by_role"].values()) == report["summary"]["total_regions"]
    assert report["summary"]["by_polarity"].keys() <= {"dark_on_light", "light_on_dark"}
    joined = " ".join(report["limitations"]).lower()
    assert "proposal" in joined or "proposed" in joined
    assert "ocr" in joined
    constraints = " ".join(report["constraints_respected"]).lower()
    assert "pixel" in constraints and "wall candidate" in constraints


def test_save_artifacts_writes_mask_overlay_and_documents(tmp_path):
    plan = build(text_row)

    result = run(plan)
    outputs = save_artifacts(result, tmp_path)

    assert set(outputs) == {"mask", "overlay", "regions", "report"}
    for name, path in outputs.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    mask = Image.open(tmp_path / TEXT_REGIONS_MASK_FILENAME)
    assert mask.size == plan.size
    overlay = Image.open(tmp_path / TEXT_REGIONS_OVERLAY_FILENAME)
    assert overlay.size == plan.size
    document = json.loads((tmp_path / TEXT_REGIONS_FILENAME).read_text(encoding="utf-8"))
    report = json.loads((tmp_path / TEXT_DETECTION_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["summary"]["total_regions"] == len(document["regions"])


def test_runner_generates_artifacts(tmp_path):
    plan = build(plan_like)
    image_path = tmp_path / "plan.png"
    plan.save(image_path)
    out_dir = tmp_path / "c2"

    main([str(image_path), str(out_dir)])

    saved = json.loads((out_dir / TEXT_REGIONS_FILENAME).read_text(encoding="utf-8"))
    assert saved["regions"], "the runner must propose regions for text-like marks"
    assert (out_dir / TEXT_REGIONS_MASK_FILENAME).exists()
    assert (out_dir / TEXT_REGIONS_OVERLAY_FILENAME).exists()
    assert (out_dir / TEXT_DETECTION_REPORT_FILENAME).exists()


def test_inputs_are_not_modified():
    plan = build(text_row)
    original = plan.resize((300, 225))
    plan_before = plan.tobytes()
    original_before = original.tobytes()

    detect_text_regions(plan, original)

    assert plan.tobytes() == plan_before
    assert original.tobytes() == original_before
