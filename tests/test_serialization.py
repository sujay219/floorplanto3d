"""Serialization tests: the JSON contract in docs/api.md."""

import json

import pytest

from fixtures.plans import to_rgb, two_room_plan
from floorplanto3d.models.floor_plan import FloorPlan, ImageInfo, Units
from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.opening import Door, Window
from floorplanto3d.models.room import Room
from floorplanto3d.models.wall import Wall
from floorplanto3d.serialization.json import from_json, to_json


def sample_plan() -> FloorPlan:
    wall = Wall(
        id="wall_001",
        start=Point2D(x=0, y=0),
        end=Point2D(x=500, y=0),
        thickness=150,
    )
    room = Room(
        id="room_001",
        polygon=[
            Point2D(x=0, y=0),
            Point2D(x=500, y=0),
            Point2D(x=500, y=400),
            Point2D(x=0, y=400),
        ],
        area=200000.0,
        wall_ids=["wall_001"],
    )
    door = Door(
        id="door_001",
        wall_id="wall_001",
        start=Point2D(x=100, y=0),
        end=Point2D(x=190, y=0),
    )
    return FloorPlan(
        units=Units.MM,
        image=ImageInfo(width=1000, height=800, format="PNG"),
        walls=[wall],
        rooms=[room],
        doors=[door],
        windows=[Window(id="window_001", wall_id="wall_001",
                        start=Point2D(x=300, y=0), end=Point2D(x=400, y=0))],
        openings=[door],
    )


class TestWireFormat:
    def test_top_level_keys(self):
        payload = json.loads(to_json(sample_plan()))
        assert set(payload) == {
            "version", "units", "scale", "image", "walls", "rooms",
            "doors", "windows", "diagnostics",
        }

    def test_points_are_objects_not_arrays(self):
        payload = json.loads(to_json(sample_plan()))
        wall = payload["walls"][0]
        assert wall["centerline"]["start"] == {"x": 0.0, "y": 0.0}
        assert wall["centerline"]["end"] == {"x": 500.0, "y": 0.0}

    def test_wall_carries_measured_thickness(self):
        wall = json.loads(to_json(sample_plan()))["walls"][0]
        assert wall["thickness"] == 150.0
        assert wall["orientation"] == "horizontal"
        assert wall["length"] == 500.0

    def test_unknown_thickness_is_null_not_invented(self):
        wall = Wall(id="w", start=Point2D(x=0, y=0), end=Point2D(x=1, y=0))
        plan = FloorPlan(image=ImageInfo(width=1, height=1), walls=[wall])
        assert json.loads(to_json(plan))["walls"][0]["thickness"] is None

    def test_opening_references_its_wall(self):
        payload = json.loads(to_json(sample_plan()))
        assert payload["doors"][0]["wall_id"] == "wall_001"
        assert payload["doors"][0]["type"] == "door"

    def test_room_polygon_and_area(self):
        room = json.loads(to_json(sample_plan()))["rooms"][0]
        assert len(room["polygon"]) == 4
        assert room["area"] == 200000.0

    def test_units_are_reported_explicitly(self):
        assert json.loads(to_json(sample_plan()))["units"] == "mm"


class TestRoundTrip:
    def test_round_trip_preserves_geometry(self):
        original = sample_plan()
        restored = from_json(to_json(original))

        assert len(restored.walls) == len(original.walls)
        assert len(restored.rooms) == len(original.rooms)
        assert len(restored.doors) == len(original.doors)
        assert len(restored.windows) == len(original.windows)
        assert restored.units is Units.MM

    def test_round_trip_preserves_wall_endpoints(self):
        original = sample_plan()
        restored = from_json(to_json(original))
        assert restored.walls[0].start == original.walls[0].start
        assert restored.walls[0].end == original.walls[0].end
        assert restored.walls[0].thickness == original.walls[0].thickness

    def test_round_trip_preserves_room_polygon(self):
        original = sample_plan()
        restored = from_json(to_json(original))
        assert restored.rooms[0].polygon == original.rooms[0].polygon

    def test_round_trip_preserves_opening_links(self):
        restored = from_json(to_json(sample_plan()))
        assert restored.doors[0].wall_id == "wall_001"
        assert restored.windows[0].wall_id == "wall_001"

    def test_round_trip_of_a_real_detection(self):
        from PIL import Image

        from floorplanto3d import FloorPlanProcessor

        plan = FloorPlanProcessor().process(
            Image.fromarray(to_rgb(two_room_plan()))
        )
        restored = from_json(to_json(plan))
        assert len(restored.walls) == len(plan.walls)
        assert len(restored.rooms) == len(plan.rooms)
        assert restored.units is Units.PX


def test_invalid_json_is_rejected():
    with pytest.raises(json.JSONDecodeError):
        from_json("{not json")
