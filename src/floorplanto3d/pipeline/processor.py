"""Main processing pipeline.

    FloorPlanProcessor
          -> preprocess   (binarise, deskew)
          -> walls        (strokes -> centrelines + thickness)
          -> openings     (gaps along walls -> doors and windows)
          -> rooms        (node walls -> polygonise -> faces)
          -> FloorPlan

FastAPI is not involved anywhere in this pipeline; it only calls into it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from floorplanto3d.errors import InsufficientGeometryError, ProcessingError
from floorplanto3d.models.floor_plan import (
    Diagnostics,
    FloorPlan,
    ImageInfo,
    Units,
)
from floorplanto3d.processing.image import (
    DEFAULT_MAX_DIMENSION,
    DEFAULT_MAX_PIXELS,
    load_image,
)
from floorplanto3d.processing.openings import extract_openings
from floorplanto3d.processing.preprocessing import preprocess
from floorplanto3d.processing.rooms import extract_rooms
from floorplanto3d.processing.scale import build_scale, scale_floor_plan
from floorplanto3d.processing.walls import extract_walls

logger = logging.getLogger(__name__)

ImageSource = str | Path | bytes | bytearray | Image.Image


class FloorPlanProcessor:
    """Converts a floor plan image into a :class:`FloorPlan`.

    The processor is stateless and safe to reuse across requests.

    Example:
        >>> processor = FloorPlanProcessor()
        >>> floor_plan = processor.process("plan.png")
        >>> floor_plan.units
        <Units.PX: 'px'>
    """

    def __init__(
        self,
        *,
        min_wall_length: float = 20.0,
        min_gap_length: float = 12.0,
        min_room_area: float = 400.0,
        deskew: bool = True,
        rectify: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        fail_on_empty: bool = True,
    ) -> None:
        """Configure the processor.

        Args:
            min_wall_length: Shortest centreline accepted as a wall, in pixels.
            min_gap_length: Shortest wall gap accepted as an opening, in pixels.
            min_room_area: Smallest polygonised face accepted as a room.
            deskew: Correct small global rotation before analysis.
            rectify: Warp a rotated or keystoned rectangular outline onto an
                exact rectangle before analysis.
            max_pixels: Reject images above this pixel count.
            max_dimension: Reject images with a side above this length.
            fail_on_empty: Raise when no wall is found instead of returning an
                empty plan. Useful for tests and strict deployments.
        """
        self.min_wall_length = min_wall_length
        self.min_gap_length = min_gap_length
        self.min_room_area = min_room_area
        self.deskew = deskew
        self.rectify = rectify
        self.max_pixels = max_pixels
        self.max_dimension = max_dimension
        self.fail_on_empty = fail_on_empty

    # -- public entry points ---------------------------------------------

    def process(
        self,
        image: ImageSource,
        *,
        pixels_per_unit: float | None = None,
        unit: Units = Units.MM,
        scale_source: str | None = None,
    ) -> FloorPlan:
        """Process a floor plan from a path, bytes or a PIL image.

        Args:
            image: File path, raw image bytes, or a PIL image.
            pixels_per_unit: Scale reference. When omitted, coordinates are
                reported in pixels and ``units`` is ``"px"``.
            unit: Real-world unit that ``pixels_per_unit`` refers to.
            scale_source: Provenance note for the scale.

        Returns:
            The detected :class:`FloorPlan`.

        Raises:
            InvalidImageError: The image could not be decoded.
            ImageTooLargeError: The image exceeds the configured limits.
            InsufficientGeometryError: No walls were found and
                ``fail_on_empty`` is set.
        """
        started = time.perf_counter()
        array, width, height, image_format = load_image(
            image,
            max_pixels=self.max_pixels,
            max_dimension=self.max_dimension,
        )
        # Accept "mm" as well as Units.MM from programmatic callers.
        try:
            unit = Units(unit)
        except ValueError as exc:
            raise ValueError(
                f"Unknown unit {unit!r}; expected one of "
                f"{', '.join(u.value for u in Units)}"
            ) from exc

        return self._run(
            array,
            width,
            height,
            image_format,
            started,
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            scale_source=scale_source,
        )

    def process_bytes(
        self,
        image_bytes: bytes,
        *,
        pixels_per_unit: float | None = None,
        unit: Units = Units.MM,
        scale_source: str | None = None,
    ) -> FloorPlan:
        """Process a floor plan supplied as raw encoded image bytes."""
        return self.process(
            image_bytes,
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            scale_source=scale_source,
        )

    def process_pil(
        self,
        image: Image.Image,
        *,
        pixels_per_unit: float | None = None,
        unit: Units = Units.MM,
        scale_source: str | None = None,
    ) -> FloorPlan:
        """Process a floor plan supplied as a PIL image."""
        return self.process(
            image,
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            scale_source=scale_source,
        )

    # -- pipeline ---------------------------------------------------------

    def _run(
        self,
        array: np.ndarray,
        width: int,
        height: int,
        image_format: str | None,
        started: float,
        *,
        pixels_per_unit: float | None,
        unit: Units,
        scale_source: str | None,
    ) -> FloorPlan:
        warnings: list[str] = []

        pre = preprocess(array, deskew=self.deskew, rectify=self.rectify)
        warnings.extend(pre.warnings)

        # Rectification and rotation change the canvas, so all downstream
        # geometry is expressed against the processed image dimensions.
        width, height = pre.binary.shape[1], pre.binary.shape[0]

        walls = extract_walls(
            pre.binary,
            min_edge_length=self.min_wall_length,
        )
        if not walls and self.fail_on_empty:
            raise InsufficientGeometryError(
                "No wall geometry could be detected in the image",
                image_width=width,
                image_height=height,
                warnings=warnings,
            )
        if not walls:
            warnings.append("no wall geometry detected")

        doors, windows = extract_openings(
            pre.binary,
            walls,
            width,
            height,
            min_gap_length=self.min_gap_length,
        )

        rooms = extract_rooms(walls, min_area=self.min_room_area)

        scale, scale_warnings = build_scale(pixels_per_unit, unit, scale_source)
        warnings.extend(scale_warnings)
        if not scale.is_known:
            warnings.append(
                "no scale supplied; coordinates are reported in image pixels"
            )

        if not rooms and walls:
            warnings.append("no enclosed rooms could be recovered from the walls")

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        diagnostics = Diagnostics(
            wall_count=len(walls),
            room_count=len(rooms),
            door_count=len(doors),
            window_count=len(windows),
            opening_count=len(doors) + len(windows),
            warnings=warnings,
            processing_ms=round(elapsed_ms, 2),
        )

        floor_plan = FloorPlan(
            image=ImageInfo(
                width=width,
                height=height,
                format=image_format,
            ),
            walls=walls,
            rooms=rooms,
            doors=doors,
            windows=windows,
            openings=[*doors, *windows],
            diagnostics=diagnostics,
        )

        # Detection runs in image pixels; convert once here so the document is
        # emitted in real-world units whenever a scale is known.
        floor_plan = scale_floor_plan(floor_plan, scale)

        logger.info(
            "Pipeline complete in %.1fms: %d walls, %d rooms, %d doors, %d windows",
            elapsed_ms,
            len(walls),
            len(rooms),
            len(doors),
            len(windows),
        )
        return floor_plan


__all__ = ["FloorPlanProcessor", "ProcessingError"]