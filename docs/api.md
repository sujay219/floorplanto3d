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

### `POST /parse2d`

`multipart/form-data` with a required `image` file. Same accepted content types
and 50 MB cap as `/process-floorplan`, but no scale parameters: coordinates are
always returned in image pixels.

Returns only the detected dark lines (the wall centrelines) and the image
canvas, for consumers that just want to draw the plan in 2D.

```bash
curl -F "image=@plan.png" http://localhost:8000/parse2d
```

```json
{
  "version": "1.0",
  "units": "px",
  "image": { "width": 600, "height": 400 },
  "walls": [
    {
      "id": "wall_001",
      "centerline": { "start": {"x": 49, "y": 51}, "end": {"x": 49, "y": 351} },
      "length": 300.0,
      "angle_deg": 90.0,
      "orientation": "vertical",
      "thickness": 8.0
    }
  ],
  "diagnostics": {
    "wall_count": 5,
    "room_count": 2,
    "door_count": 1,
    "window_count": 0,
    "opening_count": 1,
    "warnings": ["no scale supplied; coordinates are reported in image pixels"],
    "processing_ms": 21.4
  }
}
```

`walls` is the list of detected darker lines, each a wall centreline in image
pixels with the same per-wall fields documented above. `image` gives the pixel
canvas so a renderer can scale the lines to fit. Errors are identical to
`/process-floorplan`.

### `POST /detect2d`

`multipart/form-data` with a required `image` file. Same accepted content types
and 50 MB cap as `/process-floorplan`, no scale parameters.

Runs **Phase 1 (image normalization)** only: the original image is loaded
unmodified, its dimensions, image mode and format are recorded, the
background, contrast, noise and ink stroke width are inspected, and a
normalized working image is produced. Nothing is cropped (furniture, text and
annotations stay), the aspect ratio is preserved, and no denoising or
thresholding is applied. Wall, room and opening detection are later phases and
are not performed here.

```bash
curl -F "image=@plan.png" http://localhost:8000/detect2d
```

```json
{
  "phase": { "number": 1, "name": "normalization" },
  "report": {
    "original": {
      "width": 1356,
      "height": 1708,
      "mode": "RGB",
      "format": "PNG",
      "aspect_ratio": 0.793911
    },
    "normalized": {
      "width": 1356,
      "height": 1708,
      "mode": "L",
      "aspect_ratio": 0.793911,
      "resized": false,
      "scale": 1.0
    },
    "inspection": {
      "background": {
        "level": 245.0,
        "polarity": "light",
        "gradient_std": 6.31,
        "gradient_range": 44.8
      },
      "contrast": {
        "min": 0,
        "max": 255,
        "low": 24.0,
        "median": 231.0,
        "high": 252.0,
        "dynamic_range": 228.0,
        "rms_contrast": 0.3172
      },
      "noise": {
        "sigma": 2.94,
        "sigma_normalized": 3.5,
        "method": "robust MAD of median-filter residual (noise + fine texture)"
      },
      "wall_thickness": {
        "p25_px": 2.0,
        "median_px": 4.0,
        "p75_px": 9.0,
        "p90_px": 15.0,
        "sample_count": 4821,
        "method": "dark-run length statistics at gray <= 128 (inspection estimate only; no wall detection)"
      }
    },
    "parameters": {
      "color_mode": "L",
      "resize_max_side_px": 4000,
      "resize_scale": 1.0,
      "background_sigma_px": 67.8,
      "background_target_level": 255.0,
      "background_divisor_floor": 128.0,
      "contrast_low_percentile": 1.0,
      "contrast_high_percentile": 99.0,
      "denoise": "none",
      "crop": "none"
    }
  },
  "images": {
    "original": { "filename": "original.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "normalized": { "filename": "normalized.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "comparison": { "filename": "comparison.png", "media_type": "image/png", "data": "data:image/png;base64,..." }
  }
}
```

`report` is the Phase 1 JSON report (also saved as `report.json` when
artifacts are written): image dimensions and the preprocessing parameters,
plus the inspection measurements. `images` carries the three generated
artifacts — `original.png` (unmodified), `normalized.png` (flat-fielded and
contrast-stretched grayscale) and `comparison.png` (the two side by side) —
as PNG data URIs. `inspection.wall_thickness` is a stroke-width estimate for
inspection only; no wall detection has run. Errors are identical to
`/process-floorplan`.

### `POST /detect2d/walls`

`multipart/form-data` with a required `image` file. Same accepted content types
and 50 MB cap as `/detect2d`, no scale parameters.

Runs **Phase 2 (wall detection)**: the upload is normalized (Phase 1 input
stage), then Step A separates dark drawing pixels from the light background
(adaptive thresholding as the primary mask, global Otsu thresholding kept
beside it for comparison) and Step B extracts long horizontal and vertical
strokes by morphological opening. These are diagnostic masks only: furniture
and text are preserved, nothing is removed from any image, and no wall
decisions are made here — see `POST /detect2d/wall-candidates` for Step C
(wall-candidate identification). The combined mask is never fed
to the room extractor.

```bash
curl -F "image=@plan.png" http://localhost:8000/detect2d/walls
```

```json
{
  "phase": { "number": 2, "name": "wall_detection" },
  "report": {
    "steps": { "a": "...", "b": "...", "c": "not implemented (wall-candidate identification follows after these outputs have been inspected)" },
    "input": {
      "original_width": 1108,
      "original_height": 1340,
      "normalized_width": 1108,
      "normalized_height": 1340,
      "coordinate_space": "normalized"
    },
    "parameters": {
      "adaptive_block_size": 31,
      "adaptive_c": 9,
      "global_threshold_method": "otsu",
      "global_threshold_value": 134.0,
      "horizontal_kernel": [35, 1],
      "vertical_kernel": [1, 35],
      "morph_operation": "open"
    },
    "summary": {
      "foreground_pixels_adaptive": 245573,
      "foreground_fraction_adaptive": 0.1654,
      "foreground_pixels_global": 273814,
      "foreground_fraction_global": 0.1844,
      "horizontal_pixels": 97769,
      "vertical_pixels": 65277,
      "combined_pixels": 158113
    }
  },
  "images": {
    "foreground_mask": { "filename": "01_foreground_mask.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "foreground_mask_global": { "filename": "01_foreground_mask_global.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "foreground_mask_comparison": { "filename": "01_foreground_mask_comparison.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "horizontal": { "filename": "02_horizontal.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "vertical": { "filename": "03_vertical.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "combined": { "filename": "04_combined.png", "media_type": "image/png", "data": "data:image/png;base64,..." }
  }
}
```

`report` is the `wall_report.json` document: input dimensions, the
thresholding and kernel parameters (starting values, pending tuning) and
pixel summaries of each mask. `images` carries the six diagnostic masks as
PNG data URIs — Step A's adaptive and global foreground masks plus their
side-by-side comparison, then Step B's horizontal, vertical and combined
structure masks. All masks use the normalized image's coordinate space
(255 = ink). Errors are identical to `/process-floorplan`.

### `POST /detect2d/wall-candidates`

`multipart/form-data` with a required `image` file. Same accepted content types
and 50 MB cap as `/detect2d`, no scale parameters.

Runs **Phase 2 (wall detection), Step C (wall-candidate identification)**: the
upload is normalized (Phase 1 input stage), the Step A/B masks are produced,
and straight stroke candidates are extracted from the foreground mask.
Collinear edge fragments are consolidated (a gap is bridged only when the gap
itself carries ink), approximately parallel edges a plausible stroke width
apart are paired into one centreline, duplicate candidates are merged, and
every candidate is measured. Candidates are **proposals, not confirmed
walls**: nothing is classified, nothing is discarded (uncertain single-edge
candidates are preserved with their uncertainty), and the production wall,
opening and room extraction is untouched.

```bash
curl -F "image=@plan.png" http://localhost:8000/detect2d/wall-candidates
```

```json
{
  "phase": { "number": 3, "name": "wall_candidates", "step": "c" },
  "candidates": [
    {
      "id": "cand_001",
      "geometry": {
        "start": { "x": 832.0, "y": 84.5 },
        "end": { "x": 1236.0, "y": 84.5 },
        "polygon": [ { "x": 832.0, "y": 80.0 }, { "x": 1236.0, "y": 80.0 }, { "x": 1236.0, "y": 89.0 }, { "x": 832.0, "y": 89.0 } ],
        "bbox": { "x": 832, "y": 80, "width": 404, "height": 9 }
      },
      "length": 404.0,
      "angle_deg": 0.0,
      "orientation": "horizontal",
      "estimated_thickness": 9.0,
      "thickness_consistency": 0.0,
      "straightness": 0.9326,
      "continuity": 0.995,
      "edge_pairing": {
        "paired": true,
        "edge_ids": ["edge_001", "edge_002"],
        "angle_delta_deg": 0.0,
        "distance_px": 9.0,
        "overlap_fraction": 1.0,
        "score": 1.0,
        "probed_thickness_px": 4.0
      },
      "axis_support": { "horizontal": 0.995, "vertical": 0.015, "combined": 0.995 },
      "confidence": 0.7853,
      "confidence_evidence": { "pairing": 1.0, "continuity": 0.995, "straightness": 0.9326, "thickness_consistency": 0.0 },
      "evidence": ["two approximately parallel edges paired: Δangle 0.0°, distance 9.0 px, overlap 1.0"],
      "uncertainty": ["thickness varies along the stroke"],
      "merged_from": []
    }
  ],
  "report": { "phase": {}, "steps": {}, "summary": {}, "input": {}, "parameters": {}, "limitations": [] },
  "images": {
    "overlay": { "filename": "wall_candidates_overlay.png", "media_type": "image/png", "data": "data:image/png;base64,..." }
  }
}
```

`candidates` is the `wall_candidates.json` list, in original-image pixel
space. Each candidate carries its geometry (centreline, stroke-band polygon,
bounding box), length, angle and orientation (`horizontal`, `vertical` or
`diagonal`), estimated thickness with consistency, straightness, continuity,
edge-pairing evidence (paired edge IDs, angle delta, distance, overlap and
score), Step B mask support (recorded as evidence only), a confidence score
with its per-signal breakdown, and written `evidence` and `uncertainty`
lists. Confidence is geometric coherence only; no candidate is a confirmed
wall. `report` is the `candidate_report.json` document (summary, the
scale-aware detection parameters and the known limitations).
`images.overlay` is the original image with candidate centrelines, stroke
bands and IDs drawn over it (blue = paired edges, orange = single-edge /
uncertain). Errors are identical to `/process-floorplan`.

### `GET /detect2d/wall-detector`

No request body and no parameters: the input is the bundled
`examples/plans/real-time.png` example plan.

Normalizes the fixed example and returns one document for the
wall-detector UI: the Phase 1 `normalized.png`, the Step B
`04_combined.png` structure mask and every wall stroke of the masks as a
measured segment. No threshold is applied server-side — clients filter
segments (for example by `thickness` and `length`) and redraw the
survivors themselves.

```bash
curl http://localhost:8000/detect2d/wall-detector
```

```json
{
  "phase": { "number": 3, "name": "wall_detector" },
  "images": {
    "normalized": { "filename": "normalized.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "combined": { "filename": "04_combined.png", "media_type": "image/png", "data": "data:image/png;base64,..." }
  },
  "segments": [
    {
      "id": "seg_001",
      "orientation": "horizontal",
      "length": 179.0,
      "thickness": 8.4,
      "area": 1504,
      "bbox": { "x": 10, "y": 20, "width": 179, "height": 8 },
      "polygon": [ { "x": 10.0, "y": 20.0 }, { "x": 188.0, "y": 20.0 }, { "x": 188.0, "y": 27.0 }, { "x": 10.0, "y": 27.0 } ]
    }
  ],
  "report": {
    "phase": { "number": 3, "name": "wall_segments" },
    "coordinate_space": "normalized",
    "summary": { "total_segments": 214, "horizontal_segments": 118, "vertical_segments": 96 }
  }
}
```

`segments` lists one entry per connected component of the Step B
horizontal and vertical masks: its `orientation`, `length` (bounding-box
extent along the stroke axis), `thickness` (mean stroke width,
`area / length`), `area`, `bbox` and outer-contour `polygon`. Everything
is in the **normalized image's pixel space** — the same space as the two
images — so a client can erase failing segments from `04_combined.png`
and reproduce that mask exactly when no threshold is applied. The masks
are axis-aligned by construction, so no diagonal strokes appear.
`report` carries the coordinate space and per-orientation counts. Errors
are identical to `/process-floorplan`.

### `POST /detect2d/rectangles`

`multipart/form-data` with a required `image` file. Same accepted content types
and 50 MB cap as `/detect2d`, no scale parameters.

Runs **Phase 2 (rectangle detection)**: the upload is normalized (Phase 1 input
stage), rectangular candidates are detected on the normalized image at
multiple scales (small structures such as tables and furniture, medium such as
beds and fixtures, large such as rooms and balconies) and drawn as labelled
outlines over the unmodified original image. Detection combines ink-contour
polygon approximation, rotated minimum-area rectangles and sealed background
components; rotated rectangles are supported. Candidates are inspection data
only: nothing is removed from any image, and no semantic label ("bedroom",
"furniture", ...) is assigned.

```bash
curl -F "image=@plan.png" http://localhost:8000/detect2d/rectangles
```

```json
{
  "phase": { "number": 2, "name": "rectangle_detection" },
  "rectangles": [
    {
      "id": "rect_001",
      "category": "large",
      "bbox": { "x": 118, "y": 96, "width": 1120, "height": 1420 },
      "corners": [
        { "x": 118.0, "y": 96.0 },
        { "x": 1238.0, "y": 96.0 },
        { "x": 1238.0, "y": 1516.0 },
        { "x": 118.0, "y": 1516.0 }
      ],
      "width": 1120.0,
      "height": 1420.0,
      "area": 1590400.0,
      "aspect_ratio": 1.2679,
      "angle_deg": 0.0,
      "axis_aligned": true,
      "method": "contour_polygon",
      "quality_score": 0.9987,
      "duplicate_of": null,
      "nested_in": null
    }
  ],
  "report": {
    "summary": {
      "total_rectangles": 87,
      "drawn_rectangles": 61,
      "by_category": { "small": 42, "medium": 31, "large": 14 },
      "by_method": { "contour_polygon": 55, "contour_min_area_rect": 12, "region_component": 20 },
      "duplicate_count": 26,
      "nested_count": 18
    },
    "duplicates": [
      { "representative": "rect_001", "duplicate": "rect_014", "method": "region_component", "iou": 0.9412, "area_ratio": 0.9412 }
    ],
    "nested": [
      { "outer": "rect_001", "inner": "rect_007", "containment": 1.0 }
    ],
    "input": {
      "original_width": 1356,
      "original_height": 1708,
      "normalized_width": 1356,
      "normalized_height": 1708
    },
    "parameters": { "ink_threshold": "otsu", "min_rectangularity": 0.7, "...": "..." }
  },
  "images": {
    "overlay": { "filename": "rectangles_overlay.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "small": { "filename": "small_rectangles.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "medium": { "filename": "medium_rectangles.png", "media_type": "image/png", "data": "data:image/png;base64,..." },
    "large": { "filename": "large_rectangles.png", "media_type": "image/png", "data": "data:image/png;base64,..." }
  }
}
```

`rectangles` is the `rectangles.json` document: every candidate with a unique
ID, axis-aligned `bbox` and rotated `corners`, `width` (longer side), `height`,
`area`, `aspect_ratio`, `angle_deg` (long-side direction in original-image
coordinates, folded to (-90, 90], 0 = axis-aligned), the detection `method`
(`contour_polygon`, `contour_min_area_rect` or `region_component`) and a
geometric `quality_score` (shape area over rectangle area, not a semantic
confidence). No candidate is ever dropped: duplicates stay in the list flagged
with `duplicate_of`, nested rectangles carry `nested_in`. `report` is the
`rectangle_report.json` document: counts by size category and method, the
obvious duplicate and nested pairs, and every detection parameter. `images`
carries the four overlays (outlines and IDs drawn on a copy of the original,
colour-coded by size, never filled) as PNG data URIs. Errors are identical to
`/process-floorplan`.

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
