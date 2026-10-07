# FloorPlanTo3D

**Turn a floor plan image into clean, renderer-independent 3D geometry.**

![Python](https://img.shields.io/badge/python-3.14%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Schema](https://img.shields.io/badge/schema-1.0-informational)

FloorPlanTo3D reads a 2D floor plan — a scan, a rendered drawing, or a screenshot —
and emits the data a 3D engine actually needs: wall centrelines with measured
thickness, door and window openings hosed on their wall, and room polygons with
area and perimeter.

```
floor-plan image  →  analysis  →  wall centrelines, openings, rooms  →  JSON
```

The output is plain data. There is no dependency on Unity, Babylon.js, Three.js,
React, or any other renderer — you consume the JSON and build the scene however
you like. The same document drives a WebGL viewer, a Blender import, or a
headless geometry check.

## Why

Floor plan images are everywhere, but extracting usable geometry from them is
usually a bespoke, throwaway script. FloorPlanTo3D packages the hard parts —
rectification, binarisation, wall centring, opening detection, and room
polygonisation — behind one small API and a stable JSON contract.

It is also deliberate about **not inventing data**. When a value cannot be
measured, it is reported as `null` rather than guessed. A 3D consumer that wants
to assume a default wall thickness can; one that needs the truth gets the truth.

## Features

- **Wall centrelines with measured thickness** — not bounding boxes, not masks.
  Thickness is probed perpendicular to each wall and is `null` when unmeasurable.
- **Openings as wall-relative gaps** — each door/window carries its host
  `wall_id`, endpoints, width and centre, so a renderer can cut the hole exactly.
- **Plan-relative door/window classification** — thresholds are derived from the
  drawing's own gaps, not from an assumed page size.
- **Room polygons** — recovered by noding the walls into a planar graph and
  polygonising its faces; each room lists the walls that bound it, plus area and
  perimeter (shoelace).
- **Optional real-world scale** — supply `pixels_per_unit` and the whole
  document is converted once, at the end, into mm/cm/m/in/ft.
- **Rectification** — a rotated or keystoned plan is warped so its outline
  becomes an exact rectangle, absorbing arbitrary rotation and perspective
  skew; plans without a rectangular outline get a small-angle deskew.
- **A thin FastAPI service** — the HTTP layer only decodes uploads and maps
  errors; all vision logic stays in the library.
- **Structured logging** — one JSON line per event, with stage timings and
  detection counts. Image contents are never logged.
- **Stable, typed output** — Pydantic models and a documented wire format
  (`schema` version `1.0`).

## Install

Requires **Python 3.14+**.

```bash
git clone https://github.com/sujay219/floorplanto3d.git
cd FloorPlanTo3D
python3 -m venv .venv
.venv/bin/pip install -e ".[api]"
```

Swap the extra to taste:

| Extra | Adds |
|---|---|
| `.[api]` | FastAPI, Uvicorn, `python-multipart` — the HTTP service |
| `.[dev]` | Ruff — linting and formatting |
| `.[api,dev]` | Both |

Core runtime dependencies: OpenCV (headless), NumPy, Pillow, Pydantic, Shapely.

## Quick start

```python
from floorplanto3d import FloorPlanProcessor

processor = FloorPlanProcessor()
floor_plan = processor.process("plan.png")

print(floor_plan.units)              # Units.PX
print(len(floor_plan.walls), "walls")
print(len(floor_plan.rooms), "rooms")
print(floor_plan.to_json(indent=2))
```

The processor is stateless and safe to reuse across requests. It also accepts
raw bytes and PIL images:

```python
floor_plan = processor.process_bytes(image_bytes)
floor_plan = processor.process_pil(pil_image)
```

### Adding a real-world scale

Images usually carry no usable scale, so coordinates are reported in pixels and
`units` is `"px"`. Supply a scale to get real units — nothing is inferred:

```python
floor_plan = processor.process("plan.png", pixels_per_unit=0.1, unit="mm")

assert floor_plan.units.value == "mm"
print(floor_plan.walls[0].length)    # now in millimetres
```

`pixels_per_unit` is the number of image pixels that correspond to **one** unit.
A scale outside the plausible architectural range for the chosen unit is
rejected with a warning rather than silently applied.

## Output

Coordinates are always `{"x": ..., "y": ...}` objects, and the document always
declares its unit so a consumer never has to guess. This is the result of
processing the bundled two-room example, abridged for readability:

```json
{
  "version": "1.0",
  "units": "px",
  "scale": { "pixels_per_unit": null, "unit": "mm", "source": null },
  "image": { "width": 600, "height": 400, "format": "PNG", "color_space": "sRGB" },
  "walls": [
    {
      "id": "wall_001",
      "centerline": { "start": { "x": 51.0, "y": 346.0 }, "end": { "x": 51.0, "y": 56.0 } },
      "length": 290.0,
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
        { "x": 49.0, "y": 51.0 },
        { "x": 49.0, "y": 351.0 },
        { "x": 301.0, "y": 351.0 },
        { "x": 301.0, "y": 51.0 }
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
      "start": { "x": 301.0, "y": 181.433 },
      "end": { "x": 301.0, "y": 220.567 },
      "width": 39.135,
      "angle_deg": 90.0,
      "center": { "x": 301.0, "y": 201.0 }
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
    "processing_ms": 23.05
  }
}
```

Field notes:

- `thickness` is `null` when it could not be measured from the image.
- `scale.pixels_per_unit` is `null` when no scale was supplied.
- `walls[].orientation` is one of `horizontal`, `vertical`, `diagonal`.
- `rooms[].area` uses the document's unit — square px, square mm, and so on.
- `doors` and `windows` are also merged into `openings` on the model
  (`floor_plan.all_openings`).

## HTTP service

```bash
.venv/bin/uvicorn floorplanto3d.api.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -F "image=@plan.png" "http://localhost:8000/process-floorplan"
curl "http://localhost:8000/process-floorplan?pixels_per_unit=0.1&unit=mm" -F "image=@plan.png"
curl http://localhost:8000/health
```

Interactive OpenAPI docs are served at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and readiness probe |
| `POST` | `/process-floorplan` | Multipart image upload → floor plan JSON |

Uploads are capped at **50 MB**. Accepted formats: PNG, JPEG, TIFF, BMP, GIF,
WEBP (plus `application/octet-stream`, since some clients send a generic type —
the bytes are still validated by the decoder).

## How it works

```
FloorPlanProcessor
      │
      ├─ preprocess   rectify + binarise     → ink mask
      ├─ walls        strokes → centrelines  → Wall[]  (+ measured thickness)
      ├─ openings     gaps along walls       → Door[], Window[]
      ├─ rooms        noded graph → faces    → Room[]
      └─ scale        pixels → real units    → FloorPlan
```

**1. Preprocess.** Convert to grayscale and blur away compression ringing.
Detect the plan's outer quadrilateral and warp it onto an exact rectangle with
a homography, which straightens arbitrary rotation and perspective skew in one
step (rectification is skipped for non-rectangular outlines such as L-shaped
buildings). When no outline is found, fall back to estimating a small global
rotation from the ink's minimum-area rectangle and deskewing that.
Binarise with an *adaptive* Gaussian threshold, because scans routinely mix
shaded and unshaded regions. A light morphological close reconnects pinholes
without bridging real door gaps.

**2. Walls.** Run a probabilistic Hough transform to recover axis-aligned runs.
Merge collinear segments *before* centring — a door splits one wall into several
fragments, and centring each fragment separately would leave a thick band that
hides the opening. Snap near-axis segments to their exact axis, centre each line
on the middle of its ink, drop overlapping duplicates, then measure thickness by
probing outward perpendicular to the centreline.

**3. Openings.** Project the ink mask onto each wall centreline. The runs with
no ink are the gaps — these are the doors and windows. Speckle runs are filled,
and a gap that consumes almost an entire wall is discarded as a detection
artefact rather than an opening. Classification is **plan-relative**: the widest
gap is the door reference (a doorway is the largest break a plan contains); a
gap at most `1 / 1.2` of that is a window. If every gap is essentially the same
size the plan gives no evidence to separate the two, so they are all reported as
doors — a mislabelled opening is far less harmful to a 3D consumer than a
mislabelled room.

**4. Rooms.** Extend walls to meet the perpendicular walls they nearly touch,
snap clustered endpoints together, then node the walls at every intersection
with a union operation. `shapely.polygonize` turns the noded graph into faces;
the unbounded exterior face is discarded, small faces are filtered, and each
remaining polygon is simplified. Every room records the ids of the walls that
bound it.

**5. Scale.** Detection always runs in image pixels. If a scale is known, the
finished document is converted exactly once: lengths by `1 / pixels_per_unit`,
areas by the square of that factor. Vertical measures such as `sill_height` are
left alone — they are not derived from the image plane.

## Configuration

`FloorPlanProcessor` takes keyword-only tuning parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `min_wall_length` | `20.0` | Shortest centreline accepted as a wall (px) |
| `min_gap_length` | `12.0` | Shortest wall gap accepted as an opening (px) |
| `min_room_area` | `400.0` | Smallest polygonised face accepted as a room |
| `rectify` | `True` | Warp a rotated or keystoned rectangular outline onto an exact rectangle |
| `deskew` | `True` | Correct small global rotation when rectification does not apply |
| `max_pixels` | `50_000_000` | Reject images above this pixel count |
| `max_dimension` | `12_000` | Reject images with a side above this length |
| `fail_on_empty` | `True` | Raise instead of returning an empty plan |

```python
processor = FloorPlanProcessor(min_room_area=200.0, deskew=False)
```

For batch work where an empty result is legitimate, construct the service with
`create_app(fail_on_empty=False)` or set the flag on the processor.

## Errors

Every error inherits from `FloorPlanError` and carries a stable `code`. Over
HTTP, failures return `{"error": {"code": ..., "message": ...}}` with no stack
trace; the detail is logged server-side.

| Code | HTTP | Raised when |
|---|---|---|
| `missing_image` | 400 | No image was supplied |
| `image_too_large` | 413 | Upload or dimensions exceed the limits |
| `unsupported_image_format` | 415 | Format is not decodable |
| `invalid_image` | 422 | Bytes are empty or not a recognisable image |
| `insufficient_geometry` | 422 | No walls were detected |
| `invalid_request` | 422 | Request could not be parsed |
| `internal_error` | 500 | Unexpected failure (never leaks a trace) |

```python
from floorplanto3d import InsufficientGeometryError

try:
    processor.process("blank.png")
except InsufficientGeometryError as exc:
    print(exc.code, exc.details)
```

## Logging

Structured logging is on by default and reports request/image metadata, stage
progress, durations and detection counts — **never** image contents.

```bash
export FLOORPLANTO3D_LOG_LEVEL=DEBUG     # default INFO
export FLOORPLANTO3D_LOG_FORMAT=json     # default human-readable text
```

## Examples

Regenerate the bundled synthetic plans and see them processed:

```bash
.venv/bin/python examples/make_examples.py
```

Results on those plans (pixel units — they carry no dimension annotations):

| Plan | Walls | Rooms | Doors | Windows | Notes |
|---|---:|---:|---:|---:|---|
| `two-room` | 5 | 2 | 1 | 0 | Divider with a single doorway |
| `four-room` | 6 | 4 | 4 | 0 | 2×2 grid, one door per internal wall |
| `windows` | 4 | 1 | 1 | 2 | Two short gaps plus one long gap |
| `skewed` | 5 | 2 | 1 | 0 | Rotated 10° and keystoned, rectified before analysis |
| `blank` | — | — | — | — | Raises `InsufficientGeometryError` |

## Project layout

```
src/floorplanto3d/
├── api/            FastAPI app (thin HTTP wrapper)
├── models/         Pydantic contract: FloorPlan, Wall, Room, Door, Window
├── pipeline/       FloorPlanProcessor — orchestrates the stages
├── processing/     image, preprocessing, rectify, walls, openings, rooms, scale
└── serialization/  wire-format JSON
examples/           synthetic plans and their generator
```

## Limitations

- Room detection assumes **axis-aligned** walls. Plans with diagonal or curved
  walls will under-detect.
- Doors and windows are separated by **gap size** within a plan. When every gap
  is the same size they are all reported as doors, because the drawing offers no
  evidence to distinguish them.
- Dimension annotations are **not parsed**. Supply `pixels_per_unit` yourself.
- Furniture, fixtures and text are ignored.
- A connected outer outline becomes one ink blob; detection relies on the Hough
  pass rather than contour tracing for exactly this reason.

## Development

```bash
.venv/bin/pip install -e ".[api,dev]"
.venv/bin/ruff check .
```

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).
