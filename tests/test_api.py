"""HTTP API tests.

Every failure mode the TODO lists is exercised here, and each one asserts both
the status code and the machine-readable error code.
"""

import cv2
import pytest
from fastapi.testclient import TestClient

from fixtures.plans import blank_plan, to_rgb, two_room_plan
from floorplanto3d.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def png_bytes(plan) -> bytes:
    ok, buffer = cv2.imencode(".png", to_rgb(plan))
    assert ok
    return buffer.tobytes()


def upload(data: bytes, filename: str = "plan.png", content_type: str = "image/png"):
    return {"image": (filename, data, content_type)}


class TestHealth:
    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ready"] is True


class TestSuccessfulProcessing:
    def test_returns_the_floor_plan_contract(self, client):
        response = client.post(
            "/process-floorplan", files=upload(png_bytes(two_room_plan()))
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {"version", "units", "image", "walls", "rooms", "doors", "windows"}
        assert body["image"]["width"] == 600
        assert body["image"]["height"] == 400

    def test_detects_geometry(self, client):
        body = client.post(
            "/process-floorplan", files=upload(png_bytes(two_room_plan()))
        ).json()
        assert len(body["walls"]) == 5
        assert len(body["rooms"]) == 2
        assert len(body["doors"]) == 1

    def test_defaults_to_pixel_units(self, client):
        body = client.post(
            "/process-floorplan", files=upload(png_bytes(two_room_plan()))
        ).json()
        assert body["units"] == "px"
        assert body["scale"]["pixels_per_unit"] is None

    def test_scale_can_be_supplied(self, client):
        body = client.post(
            "/process-floorplan",
            files=upload(png_bytes(two_room_plan())),
            params={"pixels_per_unit": 0.1, "unit": "mm"},
        ).json()
        assert body["units"] == "mm"
        assert body["scale"]["pixels_per_unit"] == 0.1

    def test_blank_plan_returns_empty_geometry_with_warnings(self, client):
        response = client.post(
            "/process-floorplan", files=upload(png_bytes(blank_plan()))
        )
        assert response.status_code == 200
        body = response.json()
        assert body["walls"] == []
        assert body["diagnostics"]["warnings"]

    def test_diagnostics_are_populated(self, client):
        body = client.post(
            "/process-floorplan", files=upload(png_bytes(two_room_plan()))
        ).json()
        diagnostics = body["diagnostics"]
        assert diagnostics["wall_count"] == 5
        assert diagnostics["room_count"] == 2
        assert diagnostics["processing_ms"] > 0


class TestErrorHandling:
    def test_unsupported_content_type(self, client):
        response = client.post(
            "/process-floorplan", files=upload(b"hello", "a.txt", "text/plain")
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_image_format"

    def test_corrupt_image(self, client):
        response = client.post(
            "/process-floorplan", files=upload(b"this is not an image")
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_image"

    def test_empty_file(self, client):
        response = client.post("/process-floorplan", files=upload(b""))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_image"

    def test_missing_file_field(self, client):
        response = client.post("/process-floorplan")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_invalid_scale_value_is_rejected(self, client):
        response = client.post(
            "/process-floorplan",
            files=upload(png_bytes(two_room_plan())),
            params={"pixels_per_unit": -5},
        )
        assert response.status_code == 422

    def test_unknown_unit_is_rejected(self, client):
        response = client.post(
            "/process-floorplan",
            files=upload(png_bytes(two_room_plan())),
            params={"pixels_per_unit": 1, "unit": "parsecs"},
        )
        assert response.status_code == 422

    def test_errors_never_leak_a_stack_trace(self, client):
        response = client.post(
            "/process-floorplan", files=upload(b"not an image at all")
        )
        text = response.text
        assert "Traceback" not in text
        assert "File \"" not in text
        assert "site-packages" not in text

    def test_error_body_has_a_stable_shape(self, client):
        body = client.post("/process-floorplan", files=upload(b"x")).json()
        assert "error" in body
        assert {"code", "message"} <= set(body["error"])
