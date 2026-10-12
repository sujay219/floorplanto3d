"""Smoke tests for the ``/detect2d/wall-candidates`` API wiring.

The endpoint function is invoked directly (no HTTP client dependency):
this exercises the full path upload → normalize → Step A/B masks →
Step C candidates → serialization.
"""

import asyncio
import io
import json

from fastapi import UploadFile
from PIL import Image, ImageDraw
from starlette.datastructures import Headers
from starlette.requests import Request

from floorplanto3d.api.app import create_app
from floorplanto3d.processing.normalize import png_bytes

ENDPOINT = "/detect2d/wall-candidates"


def make_plan() -> Image.Image:
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 140, 350, 154], fill=(0, 0, 0))
    return image


def call_endpoint(plan: Image.Image) -> dict:
    application = create_app()
    route = next(
        route
        for route in application.routes
        if getattr(route, "path", None) == ENDPOINT
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": ENDPOINT,
            "headers": [],
            "query_string": b"",
        }
    )
    upload = UploadFile(
        file=io.BytesIO(png_bytes(plan)),
        filename="plan.png",
        headers=Headers({"content-type": "image/png"}),
    )
    response = asyncio.run(route.endpoint(request, upload))
    return json.loads(response.body)


def test_route_is_registered():
    paths = {
        getattr(route, "path", None) for route in create_app().routes
    }
    assert ENDPOINT in paths


def test_endpoint_returns_candidates_report_and_overlay():
    payload = call_endpoint(make_plan())

    assert payload["phase"] == {"number": 3, "name": "wall_candidates", "step": "c"}
    assert payload["candidates"], "the endpoint must return stroke candidates"
    assert payload["report"]["summary"]["total_candidates"] == len(payload["candidates"])
    overlay = payload["images"]["overlay"]
    assert overlay["filename"] == "wall_candidates_overlay.png"
    assert overlay["data"].startswith("data:image/png;base64,")
    ids = [candidate["id"] for candidate in payload["candidates"]]
    assert len(set(ids)) == len(ids)


def test_endpoint_preserves_and_never_classifies_candidates():
    payload = call_endpoint(make_plan())

    for candidate in payload["candidates"]:
        assert 0.0 <= candidate["confidence"] <= 1.0
        assert candidate["evidence"] or candidate["uncertainty"]
        assert "classification" not in candidate
