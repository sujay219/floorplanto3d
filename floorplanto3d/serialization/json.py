"""JSON serialization for :class:`FloorPlan`.

The wire format is documented in ``docs/api.md``. Coordinates are always
emitted as ``{"x": ..., "y": ...}`` objects and always carry the unit of the
document so a consumer never has to guess.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from PIL import Image

from floorplanto3d.models.floor_plan import FloorPlan
from floorplanto3d.models.opening import Door, Window
from floorplanto3d.models.room import Room
from floorplanto3d.models.wall import Wall
from floorplanto3d.processing.normalize import (
    COMPARISON_FILENAME,
    NORMALIZED_FILENAME,
    ORIGINAL_FILENAME,
    NormalizationResult,
    png_bytes,
)
from floorplanto3d.processing.rectangles import (
    LARGE_RECTANGLES_FILENAME,
    MEDIUM_RECTANGLES_FILENAME,
    RECTANGLES_OVERLAY_FILENAME,
    SMALL_RECTANGLES_FILENAME,
    RectangleDetectionResult,
)

JsonDict = dict[str, Any]


def point_to_dict(point: Any) -> JsonDict:
    return {"x": round(point.x, 3), "y": round(point.y, 3)}


def wall_to_dict(wall: Wall) -> JsonDict:
    payload: JsonDict = {
        "id": wall.id,
        "centerline": {
            "start": point_to_dict(wall.start),
            "end": point_to_dict(wall.end),
        },
        "length": round(wall.length, 3),
        "angle_deg": round(wall.angle_deg, 3),
        "orientation": wall.orientation,
        "thickness": round(wall.thickness, 3) if wall.thickness is not None else None,
    }
    if wall.confidence is not None:
        payload["confidence"] = wall.confidence
    return payload


def opening_to_dict(opening: Door | Window) -> JsonDict:
    payload: JsonDict = {
        "id": opening.id,
        "type": opening.type.value,
        "wall_id": opening.wall_id,
        "start": point_to_dict(opening.start),
        "end": point_to_dict(opening.end),
        "width": round(opening.width, 3),
        "angle_deg": round(opening.angle_deg, 3),
        "center": point_to_dict(opening.center),
    }
    extra = {
        "swing_direction": getattr(opening, "swing_direction", None),
        "sill_height": getattr(opening, "sill_height", None),
        "head_height": getattr(opening, "head_height", None),
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    if opening.confidence is not None:
        payload["confidence"] = opening.confidence
    return payload


def room_to_dict(room: Room) -> JsonDict:
    return {
        "id": room.id,
        "name": room.name,
        "polygon": [point_to_dict(point) for point in room.polygon],
        "area": room.area if room.area is not None else round(room.computed_area, 3),
        "perimeter": round(room.perimeter, 3),
        "wall_ids": room.wall_ids,
    }


def floor_plan_to_dict(floor_plan: FloorPlan) -> JsonDict:
    """Serialize a floor plan to its wire representation."""
    return {
        "version": floor_plan.version,
        "units": floor_plan.units.value,
        "scale": {
            "pixels_per_unit": floor_plan.scale.pixels_per_unit,
            "unit": floor_plan.scale.unit.value,
            "source": floor_plan.scale.source,
        },
        "image": {
            "width": floor_plan.image.width,
            "height": floor_plan.image.height,
            "format": floor_plan.image.format,
            "color_space": floor_plan.image.color_space,
        },
        "walls": [wall_to_dict(wall) for wall in floor_plan.walls],
        "rooms": [room_to_dict(room) for room in floor_plan.rooms],
        "doors": [opening_to_dict(door) for door in floor_plan.doors],
        "windows": [opening_to_dict(window) for window in floor_plan.windows],
        "diagnostics": floor_plan.diagnostics.model_dump(),
    }


def parse2d_to_dict(floor_plan: FloorPlan) -> JsonDict:
    """Serialize only the detected dark lines (walls) for the 2D parser.

    A focused subset of :func:`floor_plan_to_dict` used by the ``/parse2d``
    endpoint: it carries the darker ink lines (the wall centrelines) and the
    image canvas, leaving out rooms, openings and scale so a 2D consumer can
    draw the detected lines directly in image-pixel space.
    """
    return {
        "version": floor_plan.version,
        "units": floor_plan.units.value,
        "image": {
            "width": floor_plan.image.width,
            "height": floor_plan.image.height,
        },
        "walls": [wall_to_dict(wall) for wall in floor_plan.walls],
        "diagnostics": floor_plan.diagnostics.model_dump(),
    }


def detect2d_to_dict(result: NormalizationResult) -> JsonDict:
    """Serialize the Phase 1 (normalization) result for ``/detect2d``.

    Carries the normalization report (dimensions, inspection and
    preprocessing parameters) together with the three generated artifacts
    (``original.png``, ``normalized.png``, ``comparison.png``) as PNG data
    URIs. No thresholding, wall or room detection output exists at this
    phase.
    """
    return {
        "phase": {"number": 1, "name": "normalization"},
        "report": result.report,
        "images": {
            "original": _image_payload(result.original, ORIGINAL_FILENAME),
            "normalized": _image_payload(result.normalized, NORMALIZED_FILENAME),
            "comparison": _image_payload(result.comparison, COMPARISON_FILENAME),
        },
    }


def rectangles_to_dict(result: RectangleDetectionResult) -> JsonDict:
    """Serialize the Phase 2 (rectangle detection) result.

    Carries every rectangular candidate with its geometry and detection
    metadata, the phase report (summary, duplicate/nesting relationships and
    detection parameters) and the four generated overlays as PNG data URIs.
    Candidates are never removed: duplicates stay in the list, flagged with
    ``duplicate_of``.
    """
    return {
        "phase": {"number": 2, "name": "rectangle_detection"},
        "rectangles": result.metadata["rectangles"],
        "report": result.report,
        "images": {
            "overlay": _image_payload(result.overlay, RECTANGLES_OVERLAY_FILENAME),
            "small": _image_payload(result.small_view, SMALL_RECTANGLES_FILENAME),
            "medium": _image_payload(result.medium_view, MEDIUM_RECTANGLES_FILENAME),
            "large": _image_payload(result.large_view, LARGE_RECTANGLES_FILENAME),
        },
    }


def _image_payload(image: Image.Image, filename: str) -> JsonDict:
    data = base64.b64encode(png_bytes(image)).decode("ascii")
    return {
        "filename": filename,
        "media_type": "image/png",
        "data": f"data:image/png;base64,{data}",
    }


def to_json(floor_plan: FloorPlan, *, indent: int | None = None, **kwargs: Any) -> str:
    """Serialize a floor plan to a JSON string."""
    return json.dumps(floor_plan_to_dict(floor_plan), indent=indent, **kwargs)


def from_json(data: str | bytes) -> FloorPlan:
    """Reconstruct a :class:`FloorPlan` from its JSON form."""
    payload: JsonDict = json.loads(data)
    return FloorPlan.model_validate(_floor_plan_from_wire(payload))


def _floor_plan_from_wire(payload: JsonDict) -> JsonDict:
    """Map the wire format back onto the model's field names."""
    mapped = {
        "version": payload["version"],
        "units": payload.get("units", "px"),
        "scale": {
            "pixels_per_unit": (payload.get("scale") or {}).get("pixels_per_unit"),
            "unit": (payload.get("scale") or {}).get("unit", "mm"),
            "source": (payload.get("scale") or {}).get("source"),
        },
        "image": payload["image"],
        "walls": [
            {
                "id": wall["id"],
                "start": wall["centerline"]["start"],
                "end": wall["centerline"]["end"],
                "thickness": wall.get("thickness"),
                "confidence": wall.get("confidence"),
            }
            for wall in payload.get("walls", [])
        ],
        "rooms": [
            {
                "id": room["id"],
                "name": room.get("name"),
                "polygon": room["polygon"],
                "wall_ids": room.get("wall_ids", []),
                "area": room.get("area"),
            }
            for room in payload.get("rooms", [])
        ],
        "doors": [
            {
                "id": opening["id"],
                "wall_id": opening["wall_id"],
                "start": opening["start"],
                "end": opening["end"],
                "confidence": opening.get("confidence"),
            }
            for opening in payload.get("doors", [])
        ],
        "windows": [
            {
                "id": opening["id"],
                "wall_id": opening["wall_id"],
                "start": opening["start"],
                "end": opening["end"],
                "confidence": opening.get("confidence"),
            }
            for opening in payload.get("windows", [])
        ],
    }
    mapped["diagnostics"] = payload.get("diagnostics", {})
    return mapped


def to_file(floor_plan: FloorPlan, path: str | Path, *, indent: int = 2) -> None:
    """Write a floor plan to a JSON file."""
    Path(path).write_text(to_json(floor_plan, indent=indent), encoding="utf-8")


def from_file(path: str | Path) -> FloorPlan:
    """Read a floor plan from a JSON file."""
    return from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "floor_plan_to_dict",
    "from_file",
    "from_json",
    "parse2d_to_dict",
    "rectangles_to_dict",
    "to_file",
    "to_json",
    "wall_to_dict",
]