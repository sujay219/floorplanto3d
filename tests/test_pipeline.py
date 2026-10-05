"""End-to-end pipeline tests against synthetic plans with known ground truth.

Each fixture has an exact expected geometry, so these assert real accuracy
rather than merely that the pipeline returned *something*.
"""

import pytest
from PIL import Image

from fixtures.plans import (
    blank_plan,
    four_room_plan,
    to_rgb,
    two_room_plan,
    window_plan,
)
from floorplanto3d import FloorPlanProcessor
from floorplanto3d.errors import InsufficientGeometryError
from floorplanto3d.models.floor_plan import Units


def run(plan, **kwargs):
    """Process a grayscale fixture as an RGB PIL image."""
    return FloorPlanProcessor(**kwargs).process(Image.fromarray(to_rgb(plan)))


def vertical_walls(floor_plan):
    return [w for w in floor_plan.walls if w.is_vertical]


def horizontal_walls(floor_plan):
    return [w for w in floor_plan.walls if w.is_horizontal]


class TestTwoRoomPlan:
    """Outer walls at x=50/300/550, y=50/350; door gap in the divider."""

    def test_finds_expected_wall_count(self):
        result = run(two_room_plan())
        assert len(result.walls) == 5

    def test_finds_both_rooms(self):
        result = run(two_room_plan())
        assert len(result.rooms) == 2

    def test_room_areas_match_the_drawing(self):
        # Each room spans 250x300px of interior space.
        for room in run(two_room_plan()).rooms:
            assert room.area == pytest.approx(250 * 300, rel=0.05)

    def test_divider_wall_is_at_the_right_x(self):
        divider = vertical_walls(run(two_room_plan()))[1]
        assert divider.start.x == pytest.approx(300, abs=3)
        assert divider.is_vertical

    def test_measures_wall_thickness(self):
        # The fixture draws 8px strokes.
        thicknesses = [w.thickness for w in run(two_room_plan()).walls if w.thickness]
        assert thicknesses
        assert all(abs(t - 8) <= 1 for t in thicknesses)

    def test_finds_the_door_in_the_divider(self):
        result = run(two_room_plan())
        assert len(result.doors) == 1
        door = result.doors[0]
        # The gap is 40px tall and centred near y=200.
        assert door.width == pytest.approx(40, abs=4)
        assert door.center.y == pytest.approx(200, abs=12)
        assert result.wall_by_id(door.wall_id) is not None


class TestFourRoomPlan:
    """2x2 grid: outer walls plus two internal dividers, four door gaps."""

    def test_finds_four_rooms(self):
        assert len(run(four_room_plan()).rooms) == 4

    def test_rooms_are_comparable_in_size(self):
        areas = [room.area for room in run(four_room_plan()).rooms]
        assert max(areas) / min(areas) < 1.3

    def test_finds_all_four_openings(self):
        result = run(four_room_plan())
        assert len(result.doors) + len(result.windows) == 4

    def test_no_bogus_giant_opening(self):
        # A regression produced one 329px "door" from a merged pair of gaps.
        for opening in run(four_room_plan()).openings:
            assert opening.width < 100

    def test_internal_walls_sit_on_their_true_axes(self):
        result = run(four_room_plan())
        # The internal horizontal wall is at y=300 and must not drift.
        internals = [
            w for w in horizontal_walls(result) if abs(w.start.y - 300) < 20
        ]
        assert internals
        assert all(w.start.y == pytest.approx(w.end.y, abs=0.01) for w in internals)


class TestWindowPlan:
    """One room, two short gaps (windows) and one long gap (door)."""

    def test_separates_the_long_gap_as_a_door(self):
        result = run(window_plan())
        assert len(result.doors) == 1
        assert result.doors[0].width > 50

    def test_classifies_the_two_short_gaps_as_windows(self):
        result = run(window_plan())
        assert len(result.windows) == 2
        for window in result.windows:
            assert window.width == pytest.approx(30, abs=6)

    def test_single_room(self):
        assert len(run(window_plan()).rooms) == 1


class TestUnitsContract:
    def test_reports_pixels_when_no_scale_is_given(self):
        result = run(two_room_plan())
        assert result.units is Units.PX
        assert result.scale.pixels_per_unit is None
        assert any("pixels" in w for w in result.diagnostics.warnings)

    def test_accepts_string_unit(self):
        processor = FloorPlanProcessor()
        result = processor.process(
            Image.fromarray(to_rgb(two_room_plan())), pixels_per_unit=0.1, unit="mm"
        )
        assert result.units is Units.MM
        assert result.scale.pixels_per_unit == 0.1

    def test_rejects_unknown_unit(self):
        processor = FloorPlanProcessor()
        with pytest.raises(ValueError, match="Unknown unit"):
            processor.process(Image.fromarray(to_rgb(two_room_plan())), unit="parsecs")


class TestEmptyAndInvalidInput:
    def test_blank_image_raises_by_default(self):
        with pytest.raises(InsufficientGeometryError):
            run(blank_plan())

    def test_blank_image_can_return_an_empty_plan(self):
        result = run(blank_plan(), fail_on_empty=False)
        assert result.walls == []
        assert any("blank" in w or "no wall" in w for w in result.diagnostics.warnings)


class TestDiagnostics:
    def test_counts_match_the_collections(self):
        result = run(four_room_plan())
        d = result.diagnostics
        assert d.wall_count == len(result.walls)
        assert d.room_count == len(result.rooms)
        assert d.door_count == len(result.doors)
        assert d.window_count == len(result.windows)
        assert d.opening_count == len(result.doors) + len(result.windows)

    def test_processing_time_is_recorded(self):
        assert run(two_room_plan()).diagnostics.processing_ms > 0


class TestDeterminism:
    def test_same_input_gives_identical_output(self):
        plan = two_room_plan()
        first = run(plan)
        second = run(plan)
        assert len(first.walls) == len(second.walls)
        assert [w.id for w in first.walls] == [w.id for w in second.walls]
        assert [r.id for r in first.rooms] == [r.id for r in second.rooms]
