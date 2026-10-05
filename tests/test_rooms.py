"""Unit tests for room detection: snapping, extension and polygonisation."""

import pytest

from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.wall import Wall
from floorplanto3d.processing.rooms import (
    _node_walls,
    extend_walls_to_intersections,
    extract_rooms,
    room_adjacency,
    snap_wall_endpoints,
)


def rectangle_walls(gap: float = 0.0, inset: float = 0.0) -> list[Wall]:
    """Four walls forming a closed rectangle, optionally with endpoint gaps."""
    x0, y0, x1, y1 = 10 + inset, 10 + inset, 190 - inset, 190 - inset
    return [
        Wall(id="w1", start=Point2D(x=x0, y=y1 + gap), end=Point2D(x=x0, y=y0 - gap)),
        Wall(id="w2", start=Point2D(x=x0 - gap, y=y0), end=Point2D(x=x1 + gap, y=y0)),
        Wall(id="w3", start=Point2D(x=x1, y=y0 - gap), end=Point2D(x=x1, y=y1 + gap)),
        Wall(id="w4", start=Point2D(x=x1 + gap, y=y1), end=Point2D(x=x0 - gap, y=y1)),
    ]


def divider_wall(x: float = 100.0, gap: float = 0.0) -> Wall:
    return Wall(
        id="w5",
        start=Point2D(x=x, y=190 + gap),
        end=Point2D(x=x, y=10 - gap),
    )


class TestSnapWallEndpoints:
    def test_touching_endpoints_are_merged(self):
        walls = [
            Wall(id="a", start=Point2D(x=0, y=0), end=Point2D(x=10, y=0)),
            Wall(id="b", start=Point2D(x=11, y=0), end=Point2D(x=20, y=0)),
        ]
        snapped = snap_wall_endpoints(walls)
        assert snapped[0].end.x == pytest.approx(snapped[1].start.x)

    def test_distant_walls_are_untouched(self):
        walls = rectangle_walls()
        snapped = snap_wall_endpoints(walls, tolerance=1.0)
        assert [(w.start, w.end) for w in snapped] == [(w.start, w.end) for w in walls]

    def test_terminates_on_nearby_endpoints(self):
        # A cluster of many mutually close endpoints must not loop forever.
        walls = [
            Wall(id=f"w{i}", start=Point2D(x=i, y=0), end=Point2D(x=i, y=50))
            for i in range(12)
        ]
        snapped = snap_wall_endpoints(walls, tolerance=6.0)
        assert len(snapped) == 12


class TestExtendWallsToIntersections:
    def test_short_divider_reaches_the_horizontal_walls(self):
        walls = rectangle_walls() + [divider_wall(gap=6.0)]
        extended = extend_walls_to_intersections(walls)
        divider = next(w for w in extended if w.id == "w5")
        # Both ends should now reach the outer walls at y=10 and y=190.
        assert divider.start.y == pytest.approx(190, abs=1)
        assert divider.end.y == pytest.approx(10, abs=1)

    def test_untouched_when_nothing_is_near(self):
        walls = rectangle_walls()
        extended = extend_walls_to_intersections(walls)
        assert len(extended) == len(walls)

    def test_does_not_extend_far_beyond_the_limit(self):
        walls = rectangle_walls() + [divider_wall(gap=500.0)]
        extended = extend_walls_to_intersections(walls, max_extension=15.0)
        divider = next(w for w in extended if w.id == "w5")
        assert abs(divider.start.y - 190) > 15


class TestNodeWalls:
    def test_a_single_line_is_returned_as_is(self):
        walls = [Wall(id="w", start=Point2D(x=0, y=0), end=Point2D(x=10, y=0))]
        assert len(_node_walls(walls)) == 1

    def test_a_rectangle_nodes_into_its_four_edges(self):
        segments = _node_walls(rectangle_walls())
        assert len(segments) == 4


class TestExtractRooms:
    def test_closed_rectangle_is_one_room(self):
        rooms = extract_rooms(rectangle_walls())
        assert len(rooms) == 1
        assert rooms[0].area == pytest.approx(180 * 180, rel=0.02)

    def test_rectangle_with_divider_gives_two_rooms(self):
        walls = rectangle_walls() + [divider_wall()]
        rooms = extract_rooms(walls)
        assert len(rooms) == 2
        for room in rooms:
            assert room.area == pytest.approx(90 * 180, rel=0.05)

    def test_open_geometry_yields_no_rooms(self):
        walls = [
            Wall(id="a", start=Point2D(x=0, y=0), end=Point2D(x=50, y=0)),
            Wall(id="b", start=Point2D(x=0, y=0), end=Point2D(x=0, y=50)),
        ]
        assert extract_rooms(walls) == []

    def test_too_few_walls_short_circuits(self):
        assert extract_rooms(rectangle_walls()[:2]) == []

    def test_rooms_are_ordered_largest_first(self):
        walls = rectangle_walls() + [divider_wall(x=60.0), divider_wall(x=140.0)]
        rooms = extract_rooms(walls)
        areas = [room.area for room in rooms]
        assert areas == sorted(areas, reverse=True)

    def test_every_room_lists_its_bounding_walls(self):
        rooms = extract_rooms(rectangle_walls())
        assert rooms[0].wall_ids
        assert set(rooms[0].wall_ids) >= {"w1", "w2", "w3", "w4"}

    def test_polygons_have_at_least_three_vertices(self):
        for room in extract_rooms(rectangle_walls() + [divider_wall()]):
            assert len(room.polygon) >= 3


class TestRoomAdjacency:
    def test_adjacent_rooms_reference_each_other(self):
        rooms = extract_rooms(rectangle_walls() + [divider_wall()])
        adjacency = room_adjacency(rooms)
        left, right = rooms[0].id, rooms[1].id
        assert right in adjacency[left]
        assert left in adjacency[right]
