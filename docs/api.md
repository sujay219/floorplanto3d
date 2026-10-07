# API Reference

## Library

### Processing an image

```python
from floorplanto3d import FloorPlanProcessor

processor = FloorPlanProcessor()

floor_plan = processor.process("plan.png")            # file path
floor_plan = processor.process_bytes(image_bytes)      # encoded bytes
floor_plan = processor.process_pil(pil_image)          # PIL Image
```

`FloorPlanProcessor` is stateless and safe to reuse across requests.

### Constructor options

| Argument | Default | Purpose |
|---|---|---|
| `min_wall_length` | `20.0` | Shortest centreline accepted as a wall, in pixels. |
| `min_gap_length` | `12.0` | Shortest wall gap accepted as an opening, in pixels. |
| `min_room_area` | `400.0` | Smallest polygonised face accepted as a room. |
| `deskew` | `True` | Correct small global rotation before analysis. |
| `max_pixels` | `50_000_000` | Reject images above this pixel count. |
| `max_dimension` | `12_000` | Reject images with a side above this length. |
| `fail_on_empty` | `True` | Raise when no wall is found instead of returning an empty plan. |

### Scale

`process`, `process_bytes`, and `process_pil` all accept:

| Argument | Default | Purpose |
|---|---|---|
| `pixels_per_unit` | `None` | Image pixels per unit of real-world length. |
| `unit` | `Units.MM` | Unit that `pixels_per_unit` refers to. |
| `scale_source` | `None` | Provenance note recorded in the output. |

Omit the scale when the image carries none: `units` stays `"px"` and
`scale.pixels_per_unit` is `null`.

```python
floor_plan = processor.process("plan.png", pixels_per_unit=0.1, unit="mm")
```

### Serialization

```python
from floorplanto3d.serialization.json import to_json, from_json, to_file, from_file

payload = to_json(floor_plan, indent=2)   # str
restored = from_json(payload)             # FloorPlan
to_file(floor_plan, "plan.json")
floor_plan = from_file("plan.json")
```

Round-tripping preserves walls, rooms, openings, units, and geometry.

### Errors

All errors inherit from `FloorPlanError` and carry `.code` and `.details`.

| Exception | Code |
|---|---|
| `MissingImageError` | `missing_image` |
| `UnsupportedImageFormatError` | `unsupported_image_format` |
| `ImageTooLargeError` | `image_too_large` |
| `InvalidImageError` | `invalid_image` |
| `InsufficientGeometryError` | `insufficient_geometry` |
| `ProcessingError` | `processing_error` |

## HTTP service

```bash
uvicorn floorplanto3d.api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs are served at `/docs`.

### `GET /health`

```json
{ "status": "ok", "version": "0.2.0", "ready": true }
```

### `POST /process-floorplan`

`multipart/form-data` with a required `image` file.

| Query parameter | Type | Default | Purpose |
|---|---|---|---|
| `pixels_per_unit` | positive float | — | Scale reference. Omit for pixel output. |
| `unit` | `px`, `mm`, `cm`, `m`, `in`, `ft` | `mm` | Unit `pixels_per_unit` refers to. |

```bash
curl -F "image=@plan.png" http://localhost:8000/process-floorplan

curl -F "image=@plan.png" \
     "http://localhost:8000/process-floorplan?pixels_per_unit=0.1&unit=mm"
```

Accepted content types: `image/png`, `image/jpeg`, `image/tiff`, `image/bmp`,
`image/gif`, `image/webp`, `application/octet-stream`. Uploads are capped at
50 MB.

### Response format

Coordinates are always `{"x": ..., "y": ...}` objects, never bare arrays.

```json
{
  "version": "1.0",
  "units": "px",
  "scale": {
    "pixels_per_unit": null,
    "unit": "mm",
    "source": null
  },
  "image": {
    "width": 600,
    "height": 400,
    "format": "PNG",
    "color_space": "sRGB"
  },
  "walls": [
    {
      "id": "wall_001",
      "centerline": { "start": {"x": 49, "y": 51}, "end": {"x": 49, "y": 351} },
      "length": 300.0,
      "angle_deg": 90.0,
      "orientation": "vertical",
      "thickness": 8.0,
      "confidence": 1.0
    }
  ],
  "rooms": [
    {
      "id": "room_001",
      "name": null,
      "polygon": [
        {"x": 49, "y": 51}, {"x": 49, "y": 351},
        {"x": 301, "y": 351}, {"x": 301, "y": 51}
      ],
      "area": 75600.0,
      "perimeter": 1104.0,
      "wall_ids": ["wall_001", "wall_002", "wall_004", "wall_005"]
    }
  ],
  "doors": [
    {
      "id": "door_001",
      "type": "door",
      "wall_id": "wall_002",
      "start": {"x": 301, "y": 181.433},
      "end": {"x": 301, "y": 220.567},
      "width": 39.135,
      "angle_deg": 90.0,
      "center": {"x": 301, "y": 201}
    }
  ],
  "windows": [],
  "diagnostics": {
    "wall_count": 5,
    "room_count": 2,
    "door_count": 1,
    "window_count": 0,
    "opening_count": 1,
    "warnings": ["no scale supplied; coordinates are reported in image pixels"],
    "processing_ms": 24.97
  }
}
```

Above is real output for `examples/plans/two-room.png`: a 600x400 plan with
outer walls at x=50 and x=550, a divider at x=300, and a 40px door gap.

### Field notes

**`units`** is `"px"` unless a scale is supplied. Pixel coordinates are never
labelled as millimetres, and `thickness` is `null` when it could not be
measured. No dimension is invented from an image that does not contain one.

**`thickness`** is measured perpendicular to the centreline and is `null` when
no ink was found, which distinguishes "not measurable" from zero.

**`orientation`** is `horizontal`, `vertical`, or `diagonal`.

**`angle_deg`** is normalised to `[0, 180)` so a wall has one representation
regardless of endpoint order.

**Openings** are gaps expressed relative to their host wall via `wall_id`, and
carry `start`/`end` on the wall centreline so a consumer can cut the wall
without re-deriving the position.

**`rooms`** are ordered largest first. `wall_ids` lists the bounding walls.

**`diagnostics.warnings`** is non-fatal. It records absent scale, blank input,
and unrecoverable rooms. An empty `warnings` array means a clean run.

### Errors

Every failure returns a structured body and never a stack trace.

```json
{ "error": { "code": "unsupported_image_format", "message": "...", "details": {} } }
```

| Status | Code | Cause |
|---|---|---|
| 400 | `missing_image` | No image supplied. |
| 413 | `image_too_large` | Beyond size limits. |
| 415 | `unsupported_image_format` | Content type or format not decodable. |
| 422 | `invalid_image` | Corrupt, empty, or unrecognisable. |
| 422 | `insufficient_geometry` | No wall geometry recovered. |
| 422 | `invalid_request` | Request could not be parsed. |
| 500 | `internal_error` | Unexpected failure. |

## Logging

| Variable | Default | Purpose |
|---|---|---|
| `FLOORPLANTO3D_LOG_LEVEL` | `INFO` | Log level. |
| `FLOORPLANTO3D_LOG_FORMAT` | text | Set to `json` for one JSON object per line. |

Records cover request receipt, image dimensions, per-stage progress, duration,
and detection counts. Image contents are never logged.

## Known limitations

- Rooms are recovered from axis-aligned wall geometry; plans relying on
  diagonal or curved walls will under-detect.
- Doors and windows are distinguished by measured gap size within a single
  plan. A plan with only one kind of opening, or with all gaps the same size,
  cannot be resolved further and all gaps are reported as doors.
- Dimension annotations are not parsed, so real-world units require the caller
  to supply `pixels_per_unit`.
- Furniture and fixtures are ignored; only walls, openings, and rooms are
  extracted.
