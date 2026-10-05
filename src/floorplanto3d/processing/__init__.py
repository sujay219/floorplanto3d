"""Processing stages for floor plan analysis."""

from floorplanto3d.processing.image import load_image
from floorplanto3d.processing.preprocessing import PreprocessResult, preprocess
from floorplanto3d.processing.scale import build_scale, convert_point, target_units
from floorplanto3d.processing.walls import (
    extract_walls,
    measure_thickness,
    project_ink_profile,
)
from floorplanto3d.processing.openings import classify_openings, extract_openings, find_gaps
from floorplanto3d.processing.rooms import extract_rooms, room_adjacency

__all__ = [
    "PreprocessResult",
    "build_scale",
    "classify_openings",
    "convert_point",
    "extract_openings",
    "extract_rooms",
    "extract_walls",
    "find_gaps",
    "load_image",
    "measure_thickness",
    "preprocess",
    "project_ink_profile",
    "room_adjacency",
    "target_units",
]