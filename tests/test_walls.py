"""Unit tests for the wall extraction stage."""

import cv2
import numpy as np
import pytest

from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.wall import Wall
from floorplanto3d.processing.preprocessing import preprocess
from floorplanto3d.processing.walls import (
    _estimate_stroke_width,
    _snap_to_axis,
    centre_on_stroke,
    detect_segments,
    extract_walls,
    measure_thickness,
    project_ink_profile,
)


def blank(width: int = 200, height: int = 200) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def stroke_at_thickness(thickness: int = 8, length: int = 120) -> np.ndarray:
    canvas = blank()
    cv2.line(canvas, (40, 100), (40 + length, 100), 0, thickness)
    return canvas


class TestStrokeWidthEstimation:
    def test_recovers_the_drawn_width(self):
        binary = preprocess(stroke_at_thickness(thickness=8)).binary
        assert _estimate_stroke_width(binary) == pytest.approx(8, abs=2)

    def test_thicker_stroke_yields_larger_estimate(self):
        thin = _estimate_stroke_width(preprocess(stroke_at_thickness(6)).binary)
        thick = _estimate_stroke_width(preprocess(stroke_at_thickness(8)).binary)
        assert thick > thin


class TestMeasureThickness:
    def test_measures_the_real_thickness(self):
        binary = preprocess(stroke_at_thickness(thickness=8)).binary
        thickness = measure_thickness(
            binary, Point2D(x=50, y=100), Point2D(x=150, y=100)
        )
        assert thickness == pytest.approx(8, abs=1.5)

    def test_returns_none_off_the_stroke(self):
        binary = preprocess(stroke_at_thickness()).binary
        # A line in empty space has no ink to measure.
        assert measure_thickness(binary, Point2D(x=50, y=20), Point2D(x=150, y=20)) is None

    def test_returns_none_for_a_degenerate_segment(self):
        binary = preprocess(stroke_at_thickness()).binary
        assert measure_thickness(binary, Point2D(x=50, y=100), Point2D(x=50, y=100)) is None


class TestSnapToAxis:
    def _segment(self, start, end):
        from floorplanto3d.processing.walls import StrokeCandidate

        return StrokeCandidate(start=start, end=end, stroke_id=-1)

    def test_removes_horizontal_drift(self):
        segment = self._segment(Point2D(x=0, y=50), Point2D(x=100, y=53))
        snapped = _snap_to_axis(segment)
        assert snapped.start.y == snapped.end.y == 52.0

    def test_removes_vertical_drift(self):
        segment = self._segment(Point2D(x=50, y=0), Point2D(x=53, y=100))
        snapped = _snap_to_axis(segment)
        assert snapped.start.x == snapped.end.x == 52.0

    def test_leaves_a_real_diagonal_alone(self):
        segment = self._segment(Point2D(x=0, y=0), Point2D(x=100, y=100))
        snapped = _snap_to_axis(segment)
        assert snapped.start.y == 0.0
        assert snapped.end.y == 100.0


class TestCentreOnStroke:
    def test_moves_a_face_line_to_the_centre(self):
        binary = preprocess(stroke_at_thickness(thickness=10)).binary
        segments = detect_segments(binary, min_length=40)
        assert segments
        for segment in segments:
            centred = centre_on_stroke(binary, segment)
            thickness = measure_thickness(binary, centred.start, centred.end)
            # Sitting on the centre, the probe finds roughly half the stroke
            # on each side; the reported thickness stays close to the truth.
            assert thickness == pytest.approx(10, abs=3)


class TestDetectSegments:
    def test_finds_a_single_line(self):
        binary = preprocess(stroke_at_thickness()).binary
        assert len(detect_segments(binary, min_length=30)) >= 1

    def test_returns_nothing_for_a_blank_page(self):
        binary = preprocess(blank()).binary
        assert detect_segments(binary, min_length=30) == []


class TestExtractWalls:
    def test_no_walls_in_a_blank_page(self):
        assert extract_walls(preprocess(blank()).binary) == []

    def test_wall_ids_are_sequential_and_stable(self):
        canvas = blank()
        cv2.rectangle(canvas, (40, 40), (160, 160), 0, 8)
        walls = extract_walls(preprocess(canvas).binary)
        assert [w.id for w in walls] == [f"wall_{i + 1:03d}" for i in range(len(walls))]


class TestProjectInkProfile:
    def test_profile_marks_the_stroke_as_inked(self):
        binary = preprocess(stroke_at_thickness(thickness=8, length=150)).binary
        wall = Wall(
            id="w",
            start=Point2D(x=45, y=100),
            end=Point2D(x=165, y=100),
            thickness=8,
        )
        positions, has_ink = project_ink_profile(binary, wall)
        assert len(positions) == len(has_ink)
        assert has_ink.mean() > 0.9

    def test_profile_detects_a_gap(self):
        canvas = blank()
        cv2.line(canvas, (40, 100), (190, 100), 0, 8)
        cv2.line(canvas, (100, 100), (130, 100), 255, 8)
        wall = Wall(
            id="w", start=Point2D(x=44, y=100), end=Point2D(x=186, y=100), thickness=8
        )
        _, has_ink = project_ink_profile(preprocess(canvas).binary, wall)
        assert (~has_ink).any(), "the erased span should read as missing ink"

    def test_degenerate_wall_gives_an_empty_profile(self):
        wall = Wall(id="w", start=Point2D(x=5, y=5), end=Point2D(x=5, y=5))
        positions, has_ink = project_ink_profile(preprocess(blank()).binary, wall)
        assert len(positions) == 0
        assert len(has_ink) == 0
