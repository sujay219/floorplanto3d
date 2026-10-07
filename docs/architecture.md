# Architecture

## Scope

```
floor-plan image
      ↓
floor-plan analysis
      ↓
architectural geometry
      ↓
normalized FloorPlan representation
      ↓
JSON
```

The library contains no Babylon.js, Unity, React, or renderer-specific code.
The output is plain data.

## Package layout

```
FloorPlanTo3D/
├── pyproject.toml
├── README.md
├── examples/                 synthetic plans + generator
├── src/floorplanto3d/
│   ├── models/               domain types, no I/O
│   ├── processing/           image analysis stages
│   ├── pipeline/             stage orchestration
│   ├── serialization/        JSON wire format
│   ├── api/                  FastAPI service
│   ├── errors.py             error hierarchy
│   └── logging_config.py     log formatting
└── docs/
```

Layering is strict. `models` depends on nothing. `processing` depends on
`models`. `pipeline` depends on both. `api` depends on `pipeline` and never on
`processing` internals. A stage cannot call another stage through the
processor.

## Pipeline

`FloorPlanProcessor.process()` runs five stages and returns a `FloorPlan`.

```
image (path | bytes | PIL)
   │
   ├── 1. image.load_image          decode, validate, flatten alpha, RGB
   │
   ├── 2. preprocessing.preprocess  grayscale → blur → deskew → adaptive
   │                                threshold → binary ink mask
   │
   ├── 3. walls.extract_walls       ink mask → wall centrelines + thickness
   │
   ├── 4. openings.extract_openings ink mask + walls → doors and windows
   │
   ├── 5. rooms.extract_rooms       walls → closed polygons
   │
   └── FloorPlan                    + scale + diagnostics
```

Openings and rooms both depend on the wall set but are independent of each
other, so either can run without the other.

## Wall extraction

The hard part of floor-plan analysis is turning ink into wall centrelines.

**Why not contours.** Walls in a plan meet at corners, so every wall belongs to
one connected ink component. `cv2.findContours` returns the building outline as
a single contour — measured directly, a two-room plan yields *one* contour of
area 156438 spanning every wall. Worse, a thick stroke's contour edges are its
two *faces*, not its centre, so a naive reading places the wall on an edge and
makes the opposite wall look like a duplicate.

**The pipeline used instead:**

1. **Hough transform** (`detect_segments`) — floor plans are dominated by
   horizontal and vertical strokes, and `HoughLinesP` recovers them straight
   from the ink mask. On a two-room plan this yields 50 segments whose
   orientations split cleanly into 18 horizontal and 32 vertical.
2. **Collinear merge** (`_collinear_merge`) — groups segments sharing a
   direction and line. This runs *before* centring: a door splits one wall into
   several fragments, each sitting on its own edge, so centring them
   individually leaves the wall as a multi-pixel band and hides the opening.
3. **Axis snapping** (`_snap_to_axis`) — removes residual drift. A wall that
   drifts by 4px over its length never closes against its neighbours, so the
   rooms behind it silently fail to polygonise.
4. **Centring** (`centre_on_stroke`) — steps perpendicular to the line and
   picks the offset where the ink run is most symmetric, snapping the result to
   whole pixels.
5. **Deduplication** (`_dedupe_overlapping`) — drops segments almost wholly
   contained in another, after checking they are genuinely collinear.
6. **Thickness** (`measure_thickness`) — probes perpendicular to the centreline
   from many sample points along the wall and takes the median run, doubled.
   Returns `None` when no ink is found rather than guessing.

## Opening detection

A door or window is a *gap* in a wall, not a separate mark. Recognising the
symbol drawn on a plan would have to account for per-architect conventions and
per-scan styles, so gaps are measured instead:

1. `project_ink_profile` samples the ink mask along each wall centreline,
   returning a boolean per position.
2. `_drop_speckle` removes runs shorter than the noise tolerance. Filtering
   whole runs rather than individual samples is essential — bridging
   sample-by-sample leaks across a real opening and destroys it. An earlier
   implementation did exactly that and silently erased every 40px door.
3. `classify_openings` splits gaps using thresholds derived from the gaps in
   *this* plan, never an absolute constant: the widest gap is the door
   reference, and anything materially shorter is a window. When every gap is
   the same size the plan offers no evidence, so all are reported as doors
   rather than guessing arbitrarily.

Every measured gap is reported. Discarding one would leave a wall that a 3D
engine would render without the correct hole.

## Room detection

Rooms are faces of the planar graph formed by the walls, recovered with Shapely:

1. `extend_walls_to_intersections` — a Hough run stops at the edge of the ink
   it traces, so a divider stops short of the wall it joins. Each endpoint is
   extended to the crossing point, capped at `max_extension`.
2. `snap_wall_endpoints` — endpoints within tolerance are merged to their
   centroid, closing the remaining few-pixel gaps. Implemented as a flood fill
   so it always terminates.
3. `_node_walls` — `unary_union` splits walls at mutual intersections and trims
   overshoot back to the crossing point.
4. `polygonize` produces the faces; the unbounded exterior face is discarded by
   its area, and rooms are ordered largest first.

Noding matters: without it a two-room plan yields one merged face instead of
two, or no face at all.

## Coordinate systems and scale

Two spaces, and the contract never blurs them.

| Space | Meaning |
|---|---|
| Image pixels | Where geometry was measured in the source image. |
| Real-world units | Available only once a scale is supplied. |

**`units` is `"px"` unless a scale is supplied.** A floor plan image normally
carries no usable scale — it would have to come from a dimension annotation,
which the library does not parse. Rather than assume one, the document states
`"px"` and sets `scale.pixels_per_unit` to `null`. The original's habit of
labelling pixel coordinates `"mm"` and hardcoding a `150mm` thickness
fabricates dimensions the source does not contain, which the TODO explicitly
forbids.

Supply a scale to get real-world units:

```python
processor.process("plan.png", pixels_per_unit=0.1, unit="mm")
```

`build_scale` rejects implausible values against per-unit plausible ranges and
falls back to pixels with a warning rather than emitting nonsense.

## Errors

`errors.py` defines a hierarchy where every error carries a stable `code` and
an HTTP `status_code`. The API maps library errors to responses directly, so no
message string is ever parsed.

| Code | Status | Meaning |
|---|---|---|
| `missing_image` | 400 | No image supplied. |
| `unsupported_image_format` | 415 | Content type or format not decodable. |
| `image_too_large` | 413 | Beyond pixel or dimension limits. |
| `invalid_image` | 422 | Corrupt, empty, or unrecognisable. |
| `insufficient_geometry` | 422 | No wall geometry recovered. |
| `invalid_request` | 422 | Request could not be parsed. |
| `processing_error` | 500 | Unexpected internal failure. |

## Logging

Structured records for request receipt, image dimensions, per-stage progress,
duration, and detection counts. Image contents are never logged.

Text by default, JSON with `FLOORPLANTO3D_LOG_FORMAT=json`. Level via
`FLOORPLANTO3D_LOG_LEVEL`. Handlers carry a `_floorplanto3d` marker so
re-configuration is idempotent and a host application's handlers are left
alone; `propagate` is disabled to avoid duplicate output.

## Extension points

- **Different detector.** Replace `processing/walls.extract_walls`. The domain
  model, JSON contract, and API are unaffected.
- **ML detection.** A modern segmentation model can supply centrelines and
  thicknesses in place of the Hough stage. `processing/scale.py` and
  `models/floor_plan.py` already carry per-entity confidence.
- **Dimension recognition.** Parse dimension annotations and supply
  `pixels_per_unit`; the unit contract already supports it.
- **Other output formats.** Add a module beside `serialization/json.py`.
