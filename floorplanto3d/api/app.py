"""FastAPI service.

The HTTP layer is deliberately thin: it decodes the upload, calls
:class:`FloorPlanProcessor`, and maps library errors onto structured responses.
No image-processing logic lives here.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from floorplanto3d.errors import (
    FloorPlanError,
    ImageTooLargeError,
    InsufficientGeometryError,
    InvalidImageError,
    MissingImageError,
    UnsupportedImageFormatError,
)
from floorplanto3d.logging_config import configure_logging, get_logger
from floorplanto3d.models.floor_plan import SCHEMA_VERSION, Units
from floorplanto3d.pipeline.processor import FloorPlanProcessor
from floorplanto3d.processing.normalize import normalize_image
from floorplanto3d.processing.rectangles import detect_rectangles
from floorplanto3d.processing.wall_detection import detect_walls
from floorplanto3d.serialization.json import (
    detect2d_to_dict,
    floor_plan_to_dict,
    parse2d_to_dict,
    rectangles_to_dict,
    wall_detection_to_dict,
)

logger = get_logger("api")
configure_logging()

# Accepted upload types. Some clients send a generic type, so bytes are checked
# against the decoder as well and only the obvious non-images are rejected.
ALLOWED_CONTENT_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/tiff",
        "image/bmp",
        "image/gif",
        "image/webp",
        "application/octet-stream",
    }
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


class _State:
    processor: FloorPlanProcessor | None = None


state = _State()


def get_processor() -> FloorPlanProcessor:
    """FastAPI dependency returning the shared processor."""
    if state.processor is None:
        raise HTTPException(
            status_code=503, detail={"code": "not_ready", "message": "Processor not initialised"}
        )
    return state.processor


async def _read_upload(request: Request, image: UploadFile) -> tuple[bytes, str]:
    """Validate an uploaded image and return ``(contents, content_type)``.

    Shared by the processing endpoints so both enforce the same upload
    contract: content-type allow-list, empty-file and size guards, and a
    ``request.received`` log record.
    """
    if image.filename is None or not image.content_type:
        raise MissingImageError("No image was supplied")

    content_type = image.content_type.split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedImageFormatError(
            f"Content type {content_type!r} is not a supported image type",
            content_type=content_type,
            supported=sorted(ALLOWED_CONTENT_TYPES),
        )

    contents = await image.read()
    logger.info(
        "request.received",
        extra={
            "image_filename": image.filename,
            "image_content_type": content_type,
            "image_bytes": len(contents),
            "client_host": request.client.host if request.client else None,
        },
    )

    if len(contents) == 0:
        raise InvalidImageError("Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ImageTooLargeError(
            f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit",
            bytes=len(contents),
            limit=MAX_UPLOAD_BYTES,
        )
    return contents, content_type


def create_app(*, fail_on_empty: bool = False) -> FastAPI:
    """Build the ASGI application.

    Args:
        fail_on_empty: Return a plan with no geometry instead of a 422 when no
            wall is detected. Useful in batch pipelines; ``False`` is safer
            because it surfaces bad inputs immediately.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state.processor = FloorPlanProcessor(fail_on_empty=fail_on_empty)
        logger.info(
            "service.start",
            extra={"schema_version": SCHEMA_VERSION, "fail_on_empty": fail_on_empty},
        )
        yield
        state.processor = None
        logger.info("service.stop")

    application = FastAPI(
        title="FloorPlanTo3D",
        version="0.2.0",
        description=(
            "Converts a floor plan image into renderer-independent geometry: "
            "wall centrelines, measured thicknesses, openings and room polygons."
        ),
        lifespan=lifespan,
    )

    # The service is consumed by a browser frontend on a separate origin, so it
    # must allow cross-origin requests. It is a stateless, unauthenticated
    # processing endpoint, hence the permissive CORS policy.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(FloorPlanError)
    async def _library_error(_: Request, exc: FloorPlanError) -> JSONResponse:
        # Log the detail server-side; return only the structured error.
        logger.error(
            "request.failed",
            extra={"code": exc.code, "details": exc.details},
        )
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    @application.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "The request could not be parsed",
                    "details": {"errors": _safe_errors(exc)},
                }
            },
        )

    @application.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Never leak a stack trace to the client.
        logger.exception("request.crashed", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred while processing the image",
                }
            },
        )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "ok", "version": application.version, "ready": state.processor is not None}

    @application.post("/process-floorplan")
    async def process_floorplan(
        request: Request,
        image: Annotated[UploadFile, File(description="Floor plan image")],
        pixels_per_unit: Annotated[
            float | None,
            Query(
                description=(
                    "Number of image pixels per unit of real-world length. "
                    "Omit when the image carries no scale: geometry is then "
                    "returned in pixels with units=\"px\"."
                ),
                gt=0,
            ),
        ] = None,
        unit: Annotated[
            Units, Query(description="Real-world unit that pixels_per_unit refers to")
        ] = Units.MM,
        processor: Annotated[FloorPlanProcessor, Depends(get_processor)] = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Process a floor plan image and return its geometry.

        Returns the floor plan document described in ``docs/api.md``.
        """
        started = time.perf_counter()
        contents, _ = await _read_upload(request, image)

        floor_plan = processor.process_bytes(
            contents,
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            scale_source="query:pixels_per_unit" if pixels_per_unit else None,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        diagnostics = floor_plan.diagnostics
        logger.info(
            "request.completed",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "image_width": floor_plan.image.width,
                "image_height": floor_plan.image.height,
                "walls": len(floor_plan.walls),
                "rooms": len(floor_plan.rooms),
                "doors": len(floor_plan.doors),
                "windows": len(floor_plan.windows),
                "units": floor_plan.units.value,
                "warnings": diagnostics.warnings,
            },
        )

        return JSONResponse(content=floor_plan_to_dict(floor_plan))

    @application.post("/parse2d")
    async def parse2d(
        request: Request,
        image: Annotated[UploadFile, File(description="Floor plan image")],
        processor: Annotated[FloorPlanProcessor, Depends(get_processor)] = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Parse a floor plan and return the detected dark lines as JSON.

        Returns a focused document with the image canvas and the wall
        centrelines (the darker ink lines), tailored for 2D rendering. See
        ``docs/api.md`` for the exact shape.
        """
        started = time.perf_counter()
        contents, _ = await _read_upload(request, image)

        floor_plan = processor.process_bytes(contents)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "request.completed",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "image_width": floor_plan.image.width,
                "image_height": floor_plan.image.height,
                "walls": len(floor_plan.walls),
                "units": floor_plan.units.value,
            },
        )

        return JSONResponse(content=parse2d_to_dict(floor_plan))

    @application.post("/detect2d")
    async def detect2d(
        request: Request,
        image: Annotated[UploadFile, File(description="Floor plan image")],
    ) -> JSONResponse:
        """Run Phase 1 (image normalization) on a floor plan.

        Returns the untouched original, the normalized working image, their
        side-by-side comparison and the normalization report, as described in
        ``docs/api.md``. No thresholding, wall detection or room detection is
        performed at this phase.
        """
        started = time.perf_counter()
        contents, _ = await _read_upload(request, image)

        result = normalize_image(contents)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "request.completed",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "image_width": result.report["original"]["width"],
                "image_height": result.report["original"]["height"],
                "phase": "normalization",
            },
        )

        return JSONResponse(content=detect2d_to_dict(result))

    @application.post("/detect2d/rectangles")
    async def detect2d_rectangles(
        request: Request,
        image: Annotated[UploadFile, File(description="Floor plan image")],
    ) -> JSONResponse:
        """Run Phase 2 (rectangle detection) on a floor plan.

        Normalizes the upload (Phase 1 input stage), detects rectangular
        candidates on the normalized image and returns them with overlays
        drawn on the original image, as described in ``docs/api.md``.
        Candidates are inspection data only: nothing is removed from the
        image and no semantic labels are assigned.
        """
        started = time.perf_counter()
        contents, _ = await _read_upload(request, image)

        normalization = normalize_image(contents)
        result = detect_rectangles(normalization.normalized, normalization.original)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "request.completed",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "image_width": result.report["input"]["original_width"],
                "image_height": result.report["input"]["original_height"],
                "phase": "rectangle_detection",
                "rectangles": result.report["summary"]["total_rectangles"],
            },
        )

        return JSONResponse(content=rectangles_to_dict(result))

    @application.post("/detect2d/walls")
    async def detect2d_walls(
        request: Request,
        image: Annotated[UploadFile, File(description="Floor plan image")],
    ) -> JSONResponse:
        """Run Phase 2 (wall detection) on a floor plan.

        Normalizes the upload (Phase 1 input stage), then separates dark
        drawing pixels from the background (Step A) and extracts long
        horizontal and vertical structures (Step B), as described in
        ``docs/api.md``. These are diagnostic masks only: furniture and
        text are preserved, nothing is removed, and no wall candidates are
        decided yet — that is Step C, pending inspection of these outputs.
        """
        started = time.perf_counter()
        contents, _ = await _read_upload(request, image)

        normalization = normalize_image(contents)
        result = detect_walls(normalization.normalized, normalization.original)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "request.completed",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "image_width": result.report["input"]["original_width"],
                "image_height": result.report["input"]["original_height"],
                "phase": "wall_detection",
            },
        )

        return JSONResponse(content=wall_detection_to_dict(result))

    return application


def _safe_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Validation errors with any raw input stripped out."""
    cleaned = []
    for error in exc.errors():
        cleaned.append(
            {
                "type": error.get("type"),
                "location": [str(part) for part in error.get("loc", [])],
                "message": error.get("msg"),
            }
        )
    return cleaned


app = create_app()

# Re-exported so callers can catch the library's error types directly.
__all__ = ["InsufficientGeometryError", "app", "create_app"]
