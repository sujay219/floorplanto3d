# FloorPlanTo3D

Converts floor plan images into renderer-independent 3D geometry.

```
floor-plan image  →  analysis  →  wall centrelines, openings, rooms  →  JSON
```

The output is plain data with no dependency on Unity, Babylon.js, React, or any
other renderer.

## Install

Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[api]"
```

## Library use

```python
from floorplanto3d import FloorPlanProcessor

processor = FloorPlanProcessor()
floor_plan = processor.process("plan.png")

print(floor_plan.units)                      # Units.PX
print(len(floor_plan.walls), "walls")
print(len(floor_plan.rooms), "rooms")
print(floor_plan.to_json(indent=2))
```

Also accepts raw bytes and PIL images:

```python
floor_plan = processor.process_bytes(image_bytes)
floor_plan = processor.process_pil(pil_image)
```

### Scale

Floor plan images usually carry no usable scale, so coordinates are reported in
pixels and `units` is `"px"`. Nothing is invented. Supply a scale to get
real-world units:

```python
floor_plan = processor.process("plan.png", pixels_per_unit=0.1, unit="mm")
assert floor_plan.units.value == "mm"
```

## Service

```bash
.venv/bin/uvicorn floorplanto3d.api.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -F "image=@plan.png" http://localhost:8000/process-floorplan
curl http://localhost:8000/health
```

Interactive docs at `http://localhost:8000/docs`.

## Output

Coordinates are always `{"x": ..., "y": ...}` objects.

```json
{
  "version": "1.0",
  "units": "px",
  "scale": { "pixels_per_unit": null, "unit": "mm", "source": null },
  "image": { "width": 600, "height": 400, "format": "PNG", "color_space": "sRGB" },
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
      "polygon": [{"x": 49, "y": 51}, {"x": 49, "y": 351}, {"x": 301, "y": 351}, {"x": 301, "y": 51}],
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
      "center": {"x": 301, "y": 201}
    }
  ],
  "windows": [],
  "diagnostics": {
    "wall_count": 5, "room_count": 2, "door_count": 1,
    "window_count": 0, "opening_count": 1,
    "warnings": [], "processing_ms": 24.97
  }
}
```

`thickness` is `null` when it could not be measured, and `scale.pixels_per_unit`
is `null` when no scale was supplied. Neither is ever guessed.

## Examples

```bash
.venv/bin/python examples/make_examples.py
```

Writes synthetic plans to `examples/plans/`. These are test drawings with no
dimension annotations, so they process in pixel units.

## Logging

```bash
export FLOORPLANTO3D_LOG_LEVEL=DEBUG
export FLOORPLANTO3D_LOG_FORMAT=json
```

Logs cover request receipt, image dimensions, stage progress, duration, and
detection counts. Image contents are never logged.

## Errors

All inherit from `FloorPlanError` and carry a stable `code`. Over HTTP, failures
return `{"error": {"code": ..., "message": ...}}` with no stack trace.

| Code | Status |
|---|---|
| `missing_image` | 400 |
| `image_too_large` | 413 |
| `unsupported_image_format` | 415 |
| `invalid_image` | 422 |
| `insufficient_geometry` | 422 |
| `invalid_request` | 422 |
| `internal_error` | 500 |

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline design and the reasoning behind each stage |
| [docs/api.md](docs/api.md) | Full API and JSON reference |
| [docs/original-architecture.md](docs/original-architecture.md) | Analysis of the original repositories |
| [docs/extraction-map.md](docs/extraction-map.md) | What was kept, replaced, or discarded, and why |

## Limitations

- Room detection assumes axis-aligned walls; plans with diagonal or curved
  walls will under-detect.
- Doors and windows are separated by measured gap size within a plan. When all
  gaps are the same size they are all reported as doors, since the plan offers
  no evidence to distinguish them.
- Dimension annotations are not parsed; supply `pixels_per_unit` yourself.
- Furniture and fixtures are ignored.
