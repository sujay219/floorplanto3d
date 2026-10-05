"""Domain model tests."""

import pytest

from floorplanto3d.models.floor_plan import FloorPlan, ImageInfo, Scale, Units
from floorplanto3d.models.geometry import (
    BoundingBox,
    Line2D,
    Point2D,
    classify_orientation,
)
from floorplanto3d.models.opening import Door, OpeningType, Window
from floorplanto3d.models.room import Room
from floorplanto3d.models.wall import Wall


class TestPoint2D:
    def test_distance_is_euclidean(self):
        assert Point2D(x=0, y=0).distance_to(Point2D(x=3, y=4)) == 5.0

    def test_midpoint(self):
        assert Point2D(x=0, y=0).midpoint(Point2D(x=4, y=6)) == Point2D(x=2, y=3)

    def test_arithmetic(self):
        a, b = Point2D(x=1, y=2), Point2D(x=3, y=5)
        assert a + b == Point2D(x=4, y=7)
        assert b - a == Point2D(x=2, y=3)
        assert a * 3 == Point2D(x=3, y=6)


class TestLine2D:
    def test_length_and_midpoint(self):
        line = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        assert line.length == 10.0
        assert line.midpoint == Point2D(x=5, y=0)

    def test_angle_is_normalised(self):
        # Angles are reported in [0, 180) so a wall has one representation
        # regardless of which endpoint is listed first.
        up = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=0, y=10))
        down = Line2D(start=Point2D(x=0, y=10), end=Point2D(x=0, y=0))
        assert up.angle_deg == down.angle_deg == 90.0

        right = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        left = Line2D(start=Point2D(x=10, y=0), end=Point2D(x=0, y=0))
        assert right.angle_deg == left.angle_deg == 0.0

    def test_offset_is_parallel_and_signed(self):
        line = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        moved = line.offset(3)
        assert moved.start == Point2D(x=0, y=3)
        assert moved.end == Point2D(x=10, y=3)

    def test_degenerate_line_has_zero_unit_vector(self):
        line = Line2D(start=Point2D(x=1, y=1), end=Point2D(x=1, y=1))
        assert line.is_degenerate
        assert line.unit_vector == Point2D(x=0, y=0)

    def test_distance_to_point(self):
        line = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        assert line.distance_to_point(Point2D(x=5, y=4)) == pytest.approx(4.0)

    def test_closest_point_is_clamped_to_segment(self):
        line = Line2D(start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        assert line.closest_point(Point2D(x=-5, y=3)) == Point2D(x=0, y=0)
        assert line.closest_point(Point2D(x=99, y=3)) == Point2D(x=10, y=0)


class TestBoundingBox:
    def test_dimensions(self):
        box = BoundingBox(x1=10, y1=20, x2=40, y2=60)
        assert box.width == 30
        assert box.height == 40
        assert box.area == 1200
        assert box.center == Point2D(x=25, y=40)

    def test_iou_of_identical_boxes_is_one(self):
        box = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        b = BoundingBox(x1=20, y1=20, x2=30, y2=30)
        assert a.iou(b) == 0.0


class TestClassifyOrientation:
    def test_axis_aligned(self):
        assert classify_orientation(100, 0) == "horizontal"
        assert classify_orientation(0, 100) == "vertical"

    def test_diagonal(self):
        assert classify_orientation(100, 100) == "diagonal"

    def test_small_noise_counts_as_axis_aligned(self):
        assert classify_orientation(100, 2) == "horizontal"


class TestWall:
    def test_thickness_defaults_to_unknown_not_invented(self):
        wall = Wall(id="w1", start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))
        assert wall.thickness is None

    def test_orientation(self):
        assert Wall(id="w", start=Point2D(x=0, y=0), end=Point2D(x=10, y=0)).is_horizontal
        assert Wall(id="w", start=Point2D(x=0, y=0), end=Point2D(x=0, y=10)).is_vertical

    def test_face_lines_straddle_the_centreline(self):
        wall = Wall(
            id="w",
            start=Point2D(x=0, y=0),
            end=Point2D(x=10, y=0),
            thickness=4,
        )
        left, right = wall.face_lines(2)
        assert {left.start.y, right.start.y} == {-2.0, 2.0}


class TestRoom:
    def test_shoelace_area(self):
        room = Room(
            id="r",
            polygon=[
                Point2D(x=0, y=0),
                Point2D(x=10, y=0),
                Point2D(x=10, y=4),
                Point2D(x=0, y=4),
            ],
        )
        assert room.computed_area == pytest.approx(40.0)
        assert room.perimeter == pytest.approx(28.0)

    def test_open_polygon_is_not_closed(self):
        room = Room(
            id="r",
            polygon=[Point2D(x=0, y=0), Point2D(x=10, y=0), Point2D(x=5, y=5), Point2D(x=1, y=1)],
        )
        assert room.is_closed is False


class TestOpenings:
    def test_door_defaults_to_door_type(self):
        door = Door(id="d1", wall_id="w1", start=Point2D(x=0, y=0), end=Point2D(x=3, y=4))
        assert door.type is OpeningType.DOOR

    def test_window_defaults_to_window_type(self):
        window = Window(id="wn1", wall_id="w1", start=Point2D(x=0, y=0), end=Point2D(x=3, y=4))
        assert window.type is OpeningType.WINDOW

    def test_width_is_measured_along_the_opening(self):
        window = Window(id="w", wall_id="w1", start=Point2D(x=0, y=0), end=Point2D(x=0, y=6))
        assert window.width == pytest.approx(6.0)
        assert window.center == Point2D(x=0, y=3)


class TestFloorPlan:
    def test_defaults_report_pixels_not_millimetres(self):
        plan = FloorPlan(image=ImageInfo(width=10, height=10))
        assert plan.units is Units.PX
        assert plan.scale.is_known is False

    def test_wall_lookup(self):
        wall = Wall(id="wall_007", start=Point2D(x=0, y=0), end=Point2D(x=1, y=1))
        plan = FloorPlan(image=ImageInfo(width=1, height=1), walls=[wall])
        assert plan.wall_by_id("wall_007") is wall
        assert plan.wall_by_id("nope") is None

    def test_scale_becomes_known_when_supplied(self):
        assert Scale(pixels_per_unit=4.0, unit=Units.MM).is_known is True
        assert Scale(pixels_per_unit=0.0).is_known is False
