"""Room detection.

Walls are turned into a planar graph (noding them at every intersection) and
the faces of that graph become rooms. Two faces of the graph are always
produced: the bounded regions inside the building, and the unbounded outside.
The unbounded face is identified by its sign and dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.room import Room
from floorplanto3d.models.wall import Wall

logger = logging.getLogger(__name__)

_EXTENSION_OVERSHOOT = 0.25


@dataclass
class RoomCandidate:
    """A polygonised face and the walls that bound it."""

    polygon: Polygon
    wall_ids: list[str]


def _wall_line(wall: Wall) -> LineString:
    return LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)])


def snap_wall_endpoints(walls: list[Wall], tolerance: float = 8.0) -> list[Wall]:
    """Pull wall endpoints together so walls that meet actually touch.

    Detected centrelines stop a few pixels short of each other where walls
    meet, because a Hough run is clipped to the ink it can see. Left as is,
    those few pixels keep the graph open and ``polygonize`` finds no face at
    all. Each endpoint within ``tolerance`` of another endpoint is moved to
    their midpoint. ``unary_union`` then trims any overshoot back to the
    crossing point so a divider never spills through an outer wall.
    """
    points: list[Point2D] = []
    for wall in walls:
        points.extend([wall.start, wall.end])

    count = len(points)
    visited = [False] * count
    moved: dict[int, Point2D] = {}

    for seed in range(count):
        if visited[seed]:
            continue

        cluster = [seed]
        visited[seed] = True
        head = 0
        while head < len(cluster):
            current = points[cluster[head]]
            head += 1
            for other in range(count):
                if visited[other]:
                    continue
                if current.distance_to(points[other]) <= tolerance:
                    visited[other] = True
                    cluster.append(other)

        if len(cluster) > 1:
            cx = sum(points[index].x for index in cluster) / len(cluster)
            cy = sum(points[index].y for index in cluster) / len(cluster)
            centroid = Point2D(x=cx, y=cy)
            for index in cluster:
                moved[index] = centroid

    if not moved:
        return walls

    snapped: list[Wall] = []
    for wall_index, wall in enumerate(walls):
        start = moved.get(wall_index * 2, wall.start)
        end = moved.get(wall_index * 2 + 1, wall.end)
        snapped.append(wall.model_copy(update={"start": start, "end": end}))

    logger.info("Snapped %d endpoints to close the wall graph", len(moved))
    return snapped


def extend_walls_to_intersections(
    walls: list[Wall], max_extension: float = 15.0
) -> list[Wall]:
    """Extend each wall to meet any perpendicular wall it is close to.

    Hough runs stop at the edge of the ink they trace, so a vertical divider
    stops half a wall-thickness short of the horizontal wall it joins. Extending
    each endpoint to the crossing point (capped at ``max_extension``) lets
    ``polygonize`` actually close the room. Extensions overshoot the crossing
    by a hair so the pair forms a true crossing that noding cannot miss; the
    overshoot is trimmed back by ``unary_union``.
    """
    extended: list[Wall] = []

    for wall in walls:
        start = wall.start
        end = wall.end
        length = wall.length
        if length < 1e-6:
            extended.append(wall)
            continue

        ux = (end.x - start.x) / length
        uy = (end.y - start.y) / length

        for sign in (-1.0, 1.0):
            base = start if sign < 0 else end
            # Probe outward from the endpoint, i.e. away from the wall body.
            dx, dy = ux * sign, uy * sign
            best: Point2D | None = None
            for other in walls:
                if other.id == wall.id:
                    continue
                other_length = other.length
                if other_length < 1e-6:
                    continue
                # Only perpendicular partners can extend a wall's endpoint.
                oux = (other.end.x - other.start.x) / other_length
                ouy = (other.end.y - other.start.y) / other_length
                cross = abs(dx * ouy - dy * oux)
                if cross < 0.5:  # roughly parallel
                    continue
                point = _intersect_point(base, dx, dy, other, max_extension)
                if point is None:
                    continue
                if best is None or base.distance_to(point) < base.distance_to(best):
                    best = point

            if best is not None:
                if base.distance_to(best) > 1e-9:
                    # A T-junction that only touches the partner within float
                    # noise is invisible to exact noding; a slight overshoot
                    # turns it into a real crossing that is then trimmed.
                    best = Point2D(
                        x=best.x + dx * _EXTENSION_OVERSHOOT,
                        y=best.y + dy * _EXTENSION_OVERSHOOT,
                    )
                if sign < 0:
                    start = best
                else:
                    end = best

        extended.append(wall.model_copy(update={"start": start, "end": end}))

    return extended


def _intersect_point(
    base: Point2D, ux: float, uy: float, other: Wall, max_extension: float
) -> Point2D | None:
    """Where the ray ``base + t*(ux, uy)`` crosses ``other``'s centreline.

    ``(ux, uy)`` is the outward direction from the endpoint being extended.
    Returns ``None`` unless the crossing lies ahead of ``base`` within
    ``max_extension`` and within the other wall's own span.
    """
    other_length = other.length
    if other_length < 1e-6:
        return None

    ovx = (other.end.x - other.start.x) / other_length
    ovy = (other.end.y - other.start.y) / other_length

    denominator = ux * ovy - uy * ovx
    if abs(denominator) < 1e-9:
        return None

    # Vector from the probe origin to the other wall's start.
    rx = other.start.x - base.x
    ry = other.start.y - base.y

    t = (rx * ovy - ry * ovx) / denominator
    if t < 0 or t > max_extension:
        return None

    crossing = Point2D(x=base.x + t * ux, y=base.y + t * uy)

    # Confirm the crossing lands on the other wall's own span, measured from
    # the crossing itself rather than from the wall's start corner.
    offset_x = crossing.x - other.start.x
    offset_y = crossing.y - other.start.y
    along = offset_x * ovx + offset_y * ovy
    if along < -max_extension or along > other_length + max_extension:
        return None

    return crossing


def _node_walls(walls: list[Wall]) -> list[LineString]:
    """Split walls at mutual intersections and return the noded segments.

    Without noding, ``shapely.polygonize`` cannot close a T-junction: the
    dividing wall of a two-room plan stops short of the outer wall and leaves a
    gap that no face can form. ``unary_union`` also *trims* overshooting walls
    back to the crossing point, which is what keeps a divider from spilling
    through the outer wall and merging both rooms into one face.
    """
    lines = [_wall_line(wall) for wall in walls]
    if len(lines) < 2:
        return lines

    noded = unary_union(lines)
    if noded.geom_type == "LineString":
        return [noded]

    segments: list[LineString] = []
    for geom in getattr(noded, "geoms", []):
        if geom.geom_type == "LineString":
            segments.append(geom)
        elif geom.geom_type == "MultiLineString":
            segments.extend(
                part for part in geom.geoms if part.geom_type == "LineString"
            )
    return segments


def _assign_walls_to_face(
    face: Polygon, walls: list[Wall], tolerance: float = 2.0
) -> list[str]:
    """Ids of walls whose centreline touches the face boundary."""
    ids: list[str] = []
    boundary = face.boundary
    for wall in walls:
        line = _wall_line(wall)
        if boundary.distance(line) <= tolerance:
            ids.append(wall.id)
    return ids


def extract_rooms(
    walls: list[Wall],
    *,
    min_area: float = 100.0,
    simplify_tolerance: float = 1.0,
) -> list[Room]:
    """Recover enclosed rooms from a set of walls.

    Args:
        walls: Detected walls.
        min_area: Discard faces smaller than this, in square pixels.
        simplify_tolerance: Douglas-Peucker tolerance for the output polygon.

    Returns:
        A list of :class:`Room` objects, largest first.
    """
    if len(walls) < 3:
        logger.info("Fewer than 3 walls; skipping room detection")
        return []

    # Snap first so extensions are computed against the final wall positions:
    # extending before snapping lets a later snap pull a joined wall a fraction
    # of a pixel away from a T-junction and reopen the graph.
    walls = snap_wall_endpoints(walls)
    walls = extend_walls_to_intersections(walls)
    segments = _node_walls(walls)
    faces = list(polygonize(segments))
    if not faces:
        logger.info("Polygonisation produced no faces")
        return []

    min_area = max(0.0, float(min_area))
    rooms: list[Room] = []

    for face in faces:
        if not face.is_valid:
            face = face.buffer(0)
            if face.is_empty or face.geom_type != "Polygon":
                continue

        # The exterior face is unbounded; its area is astronomically large.
        area = face.area
        if area <= min_area or area > float(np.inf):
            continue

        if simplify_tolerance > 0:
            face = face.simplify(simplify_tolerance, preserve_topology=True)
            if face.is_empty or face.geom_type != "Polygon":
                continue
            area = face.area
            if area <= min_area:
                continue

        ring = _outer_ring(face)
        if len(ring) < 3:
            continue

        polygon = [Point2D(x=float(x), y=float(y)) for x, y in ring]
        rooms.append(
            Room(
                id=f"room_{len(rooms) + 1:03d}",
                polygon=polygon,
                wall_ids=_assign_walls_to_face(face, walls),
                area=round(float(area), 2),
            )
        )

    # Largest room first gives a stable, useful ordering to API consumers.
    rooms.sort(key=lambda room: room.area or 0.0, reverse=True)
    for index, room in enumerate(rooms, start=1):
        room.id = f"room_{index:03d}"

    logger.info("Extracted %d rooms", len(rooms))
    return rooms


def _outer_ring(polygon: Polygon) -> list[tuple[float, float]]:
    """Exterior ring without the duplicated closing vertex."""
    coords = list(polygon.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def room_adjacency(rooms: list[Room]) -> dict[str, list[str]]:
    """Map each room id to the ids of rooms sharing a wall."""
    adjacency: dict[str, list[str]] = {room.id: [] for room in rooms}
    for i, left in enumerate(rooms):
        left_ring = LineString([(p.x, p.y) for p in left.polygon])
        for right in rooms[i + 1 :]:
            right_ring = LineString([(p.x, p.y) for p in right.polygon])
            if left_ring.distance(right_ring) <= 1.0:
                adjacency[left.id].append(right.id)
                adjacency[right.id].append(left.id)
    return adjacency


__all__ = ["RoomCandidate", "extract_rooms", "room_adjacency"]