"""Scale handling: pixel coordinates to real-world units."""

from __future__ import annotations

import logging

from floorplanto3d.models.floor_plan import FloorPlan, Scale, Units
from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.opening import Door, Window

logger = logging.getLogger(__name__)

# Rough real-world ranges used only to sanity-check a user-supplied scale.
PLAUSIBLE_RANGES: dict[Units, tuple[float, float]] = {
    # (min, max) plausible pixels-per-unit for architectural drawings.
    Units.MM: (0.02, 2.0),
    Units.CM: (0.2, 20.0),
    Units.M: (2.0, 200.0),
    Units.IN: (5.0, 500.0),
    Units.FT: (60.0, 6000.0),
}


def validate_scale(pixels_per_unit: float | None, unit: Units) -> tuple[bool, str | None]:
    """Check a scale against plausible architectural ranges."""
    if pixels_per_unit is None:
        return True, None
    if pixels_per_unit <= 0:
        return False, "pixels_per_unit must be positive"
    low, high = PLAUSIBLE_RANGES.get(unit, (0.0, float("inf")))
    if not low <= pixels_per_unit <= high:
        return False, (
            f"{pixels_per_unit:g} pixels per {unit.value} is outside the plausible "
            f"range {low:g}-{high:g}; check the scale reference"
        )
    return True, None


def build_scale(
    pixels_per_unit: float | None,
    unit: Units = Units.MM,
    source: str | None = None,
) -> tuple[Scale, list[str]]:
    """Build a :class:`Scale`, warning when the value looks implausible."""
    warnings: list[str] = []

    ok, message = validate_scale(pixels_per_unit, unit)
    if not ok:
        warnings.append(message or "implausible scale rejected")
        pixels_per_unit = None

    if pixels_per_unit is None:
        return Scale(unit=unit, source=source), warnings

    logger.info("Using scale %g pixels per %s", pixels_per_unit, unit.value)
    return Scale(pixels_per_unit=pixels_per_unit, unit=unit, source=source), warnings


def convert_point(x: float, y: float, scale: Scale) -> tuple[float, float]:
    """Convert pixel coordinates into the scale's unit."""
    if not scale.is_known:
        return x, y
    factor = 1.0 / scale.pixels_per_unit
    return x * factor, y * factor


def target_units(scale: Scale) -> Units:
    """The unit that geometry should be reported in."""
    return scale.unit if scale.is_known else Units.PX


def scale_floor_plan(floor_plan: FloorPlan, scale: Scale) -> FloorPlan:
    """Return a copy of ``floor_plan`` with geometry expressed in ``scale``'s unit.

    Wall centrelines, thicknesses, opening endpoints and room polygons are all
    measured from the image, so each is multiplied by ``1 / pixels_per_unit``.
    Stored areas are multiplied by the square of that factor, since area scales
    with the square of length.

    Vertical real-world measures (``sill_height``, ``head_height``) are left
    untouched: they are not derived from the image plane and are already
    expressed in ``scale.unit``.

    When the scale is unknown the plan is returned unchanged.
    """
    if not scale.is_known:
        return floor_plan

    factor = 1.0 / scale.pixels_per_unit

    def point(p: Point2D) -> Point2D:
        return Point2D(x=p.x * factor, y=p.y * factor)

    walls = [
        w.model_copy(
            update={
                "start": point(w.start),
                "end": point(w.end),
                "thickness": None if w.thickness is None else w.thickness * factor,
            }
        )
        for w in floor_plan.walls
    ]

    def opening_copy(o: Door | Window) -> Door | Window:
        return o.model_copy(update={"start": point(o.start), "end": point(o.end)})

    doors = [opening_copy(d) for d in floor_plan.doors]
    windows = [opening_copy(w) for w in floor_plan.windows]

    rooms = [
        r.model_copy(
            update={
                "polygon": [point(p) for p in r.polygon],
                "area": None if r.area is None else r.area * factor**2,
            }
        )
        for r in floor_plan.rooms
    ]

    return floor_plan.model_copy(
        update={
            "units": target_units(scale),
            "scale": scale,
            "walls": walls,
            "rooms": rooms,
            "doors": doors,
            "windows": windows,
            "openings": [opening_copy(o) for o in floor_plan.openings],
        }
    )