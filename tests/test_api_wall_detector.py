"""Smoke tests for the ``/detect2d/wall-detector`` API wiring.

The endpoint function is invoked directly (no HTTP client dependency):
this exercises the full path example plan -> normalize -> Step A/B masks ->
wall segmentation -> serialization. The bundled example is replaced with a
small generated plan so the tests stay fast.
"""

import asyncio
import importlib
import json

from PIL import Image, ImageDraw
from starlette.requests import Request

from floorplanto3d.api.app import EXAMPLE_PLAN_PATH, create_app
from floorplanto3d.processing.normalize import png_bytes

ENDPOINT = "/detect2d/wall-detector"


def make_plan() -> Image.Image:
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 140, 350, 154], fill=(0, 0, 0))
    return image


def call_endpoint(monkeypatch) -> dict:
    # Patch on the module object: ``floorplanto3d.api.app`` resolves to the
    # FastAPI instance through the package namespace (``api/__init__.py``).
    app_module = importlib.import_module("floorplanto3d.api.app")
    monkeypatch.setattr(
        app_module,
        "_example_plan_bytes",
        lambda: png_bytes(make_plan()),
    )
    application = create_app()
    route = next(
        route
        for route in application.routes
        if getattr(route, "path", None) == ENDPOINT
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": ENDPOINT,
            "headers": [],
            "query_string": b"",
        }
    )
    response = asyncio.run(route.endpoint(request))
    return json.loads(response.body)


def test_route_is_registered():
    paths = {
        getattr(route, "path", None) for route in create_app().routes
    }
    assert ENDPOINT in paths


def test_bundled_example_plan_exists():
    assert EXAMPLE_PLAN_PATH.is_file()


def test_endpoint_returns_images_and_segments(monkeypatch):
    payload = call_endpoint(monkeypatch)

    assert payload["phase"] == {"number": 3, "name": "wall_detector"}
    assert payload["images"]["normalized"]["filename"] == "normalized.png"
    assert payload["images"]["combined"]["filename"] == "04_combined.png"
    for name in ("normalized", "combined"):
        image = payload["images"][name]
        assert image["media_type"] == "image/png"
        assert image["data"].startswith("data:image/png;base64,")
    assert payload["report"]["coordinate_space"] == "normalized"
    assert payload["segments"], "the endpoint must return wall segments"


def test_segments_carry_threshold_measurements(monkeypatch):
    payload = call_endpoint(monkeypatch)

    for segment in payload["segments"]:
        assert segment["length"] > 0
        assert segment["thickness"] > 0
        assert segment["orientation"] in ("horizontal", "vertical")
        bbox = segment["bbox"]
        assert bbox["width"] > 0 and bbox["height"] > 0
        assert len(segment["polygon"]) >= 3


def test_endpoint_applies_no_thresholds(monkeypatch):
    payload = call_endpoint(monkeypatch)

    summary = payload["report"]["summary"]
    assert summary["total_segments"] == len(payload["segments"])
