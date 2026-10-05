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
from floorplanto3d.serialization.json import floor_plan_to_dict

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
