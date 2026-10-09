"""FloorPlanTo3D: floor plan images in, renderer-independent geometry out."""

from floorplanto3d.errors import (
    FloorPlanError,
    ImageTooLargeError,
    InsufficientGeometryError,
    InvalidImageError,
    MissingImageError,
    ProcessingError,
    UnsupportedImageFormatError,
)
from floorplanto3d.models import (
    Door,
    FloorPlan,
    Point2D,
    Room,
    Scale,
    Units,
    Wall,
    Window,
)
from floorplanto3d.pipeline.processor import FloorPlanProcessor

__version__ = "0.2.0"

__all__ = [
    "Door",
    "FloorPlan",
    "FloorPlanError",
    "FloorPlanProcessor",
    "ImageTooLargeError",
    "InsufficientGeometryError",
    "InvalidImageError",
    "MissingImageError",
    "Point2D",
    "ProcessingError",
    "Room",
    "Scale",
    "Units",
    "UnsupportedImageFormatError",
    "Wall",
    "Window",
    "__version__",
]