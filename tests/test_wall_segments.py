"""Tests for :mod:`floorplanto3d.processing.wall_segments`."""

from PIL import Image, ImageDraw

from floorplanto3d.processing.wall_segments import segment_to_dict, segment_walls


def make_masks() -> tuple[Image.Image, Image.Image]:
    horizontal = Image.new("L", (200, 150), 0)
    draw = ImageDraw.Draw(horizontal)
    draw.rectangle([10, 20, 189, 27], fill=255)
    vertical = Image.new("L", (200, 150), 0)
    draw = ImageDraw.Draw(vertical)
    draw.rectangle([60, 60, 65, 139], fill=255)
    return horizontal, vertical


def test_segments_one_per_component_with_orientation():
    horizontal, vertical = make_masks()
    result = segment_walls(horizontal, vertical)

    assert result.report["summary"]["total_segments"] == 2
    orientations = {segment.orientation for segment in result.segments}
    assert orientations == {"horizontal", "vertical"}
    ids = [segment.id for segment in result.segments]
    assert len(set(ids)) == len(ids)


def test_segments_measure_length_and_thickness():
    horizontal, vertical = make_masks()
    result = segment_walls(horizontal, vertical)
    by_orientation = {s.orientation: s for s in result.segments}

    horizontal_segment = by_orientation["horizontal"]
    assert horizontal_segment.length >= 175
    assert 5 <= horizontal_segment.thickness <= 10

    vertical_segment = by_orientation["vertical"]
    assert vertical_segment.length >= 75
    assert 4 <= vertical_segment.thickness <= 8


def test_segment_outline_matches_bbox():
    horizontal, vertical = make_masks()
    result = segment_walls(horizontal, vertical)

    for segment in result.segments:
        x, y, width, height = segment.bbox
        for px, py in segment.polygon:
            assert x - 1 <= px <= x + width + 1
            assert y - 1 <= py <= y + height + 1


def test_segment_to_dict_shape():
    horizontal, vertical = make_masks()
    result = segment_walls(horizontal, vertical)
    record = segment_to_dict(result.segments[0])

    assert record["id"].startswith("seg_")
    assert set(record) == {
        "id",
        "orientation",
        "length",
        "thickness",
        "area",
        "bbox",
        "polygon",
    }
    assert set(record["bbox"]) == {"x", "y", "width", "height"}
    assert all(set(point) == {"x", "y"} for point in record["polygon"])
