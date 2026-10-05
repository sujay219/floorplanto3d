"""Root FloorPlan model and its serialization contract."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from floorplanto3d.models.opening import Door, Window
from floorplanto3d.models.room import Room
from floorplanto3d.models.wall import Wall

SCHEMA_VERSION = "1.0"


class Units(str, Enum):
    """Unit of measure for all coordinates in the document.

    ``PX`` means the source image carried no usable scale and all coordinates
    are image pixels. Real-world units are only emitted once a scale has been
    supplied via :class:`Scale`.
    """

    PX = "px"
    MM = "mm"
    CM = "cm"
    M = "m"
    IN = "in"
    FT = "ft"


class ImageInfo(BaseModel):
    """Source image metadata."""

    width: int
    height: int
    format: str | None = None
    color_space: str = "sRGB"


class Scale(BaseModel):
    """Scale factor relating image pixels to real-world units.

    ``pixels_per_unit`` is the number of image pixels that correspond to one
    unit of ``unit``. It is ``None`` when the image provides no scale, in which
    case all coordinates remain in ``px``.
    """

    pixels_per_unit: float | None = None
    unit: Units = Units.MM
    source: str | None = None
    confidence: float | None = None

    @property
    def is_known(self) -> bool:
        return self.pixels_per_unit is not None and self.pixels_per_unit > 0


class Diagnostics(BaseModel):
    """Non-fatal information about a processing run."""

    wall_count: int = 0
    room_count: int = 0
    door_count: int = 0
    window_count: int = 0
    opening_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    processing_ms: float | None = None


class FloorPlan(BaseModel):
    """Renderer-independent floor plan description.

    This model is the library's public contract. It carries wall centrelines,
    measured wall thicknesses where recoverable, openings expressed relative
    to their host wall, and room polygons. It has no dependency on any
    renderer.
    """

    version: str = SCHEMA_VERSION
    units: Units = Units.PX
    scale: Scale = Field(default_factory=Scale)
    image: ImageInfo
    walls: list[Wall] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    openings: list[Door | Window] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to the wire format described in ``docs/api.md``."""
        from floorplanto3d.serialization.json import floor_plan_to_dict

        return floor_plan_to_dict(self)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to a JSON string."""
        import json

        return json.dumps(self.model_dump(), **kwargs)

    @property
    def all_openings(self) -> list[Door | Window]:
        """Doors and windows combined."""
        return [*self.doors, *self.windows]

    def wall_by_id(self, wall_id: str) -> Wall | None:
        for wall in self.walls:
            if wall.id == wall_id:
                return wall
        return None