# Extraction Map

Every file in the two reference repositories, what it did, and whether the
rewrite keeps, replaces, or discards it.

## FloorPlanTo3D-API/

| Original file | Purpose | Decision | Reason |
|---|---|---|---|
| `application.py` | Flask server, single `POST /` route | **Replace** | HTTP glue. Rebuilt as a thin FastAPI layer; the image-processing logic it contained is gone. |
| `requirements.txt` | 123 pinned packages | **Discard** | TensorFlow 1.15 / Keras 2.0.8 / numpy 1.19 all have no Python 3.14 wheels. Replaced by 5 current packages. |
| `runtime.txt` | Python version pin (3.6.x) | **Discard** | Target is 3.14. |
| `readme.md` | Setup notes | **Discard** | Describes a workflow that no longer applies. |
| `mrcnn/__init__.py` | Package init | **Discard** | Vendored Matterport library. |
| `mrcnn/config.py` | `Config` with `NUM_CLASSES = 4`, resize mode | **Discard** | TF 1.x graph-mode configuration. The class vocabulary (wall/window/door) is preserved as `models/opening.py`. |
| `mrcnn/model.py` | Mask R-CNN ResNet graph, `mold_image`, `detect` | **Replace** | Imports `keras.engine` and `distutils`, both gone. Detection is rebuilt on classical CV (see below). |
| `mrcnn/parallel_model.py` | Multi-GPU `MaskRCNN` | **Discard** | TF 1.x only; adds nothing to a single-image service. |
| `mrcnn/utils.py` | `extract_bboxes`, `compute_iou`, `apply_box_mask` | **Discard** | Generic NumPy bounding-box utilities with no domain logic; the equivalents here are purpose-built. |
| `mrcnn/visualize.py` | Matplotlib overlay + `plot_images` | **Discard** | Rendering. Explicitly out of scope. |
| `weights/maskrcnn_15_epochs.h5` | Trained weights | **Unavailable** | **Not in the repository.** No trained model could be ported even in principle. |

## FloorPlanTo3D-unityClient/

| Original file | Purpose | Decision | Reason |
|---|---|---|---|
| `Builder.cs` | Parses JSON, creates GameObjects per detection | **Discard** | Unity-side. The schema it consumed is superseded by `docs/api.md`. |
| `WallMesh.cs` | Builds a 3D box from a bounding box | **Discard** | Renderer-specific, and axis-aligned-only. |
| `CubeMeshData.cs` | Shared mesh vertex data | **Discard** | Unity-specific. |

## Why detection was rewritten rather than ported

Three independent blockers, any one of which is decisive:

1. **The weights do not exist.** `weights/maskrcnn_15_epochs.h5` is absent, so
   there is no trained model to port.
2. **TensorFlow 1.15 cannot install on Python 3.14.** No wheels exist, and
   TF 1.x links against OpenSSL 1.1, which is what the TODO set out to avoid.
3. **The vendored model targets a removed Keras API.** `mrcnn/model.py` uses
   `keras.engine` and `distutils.version`, none of which exist in current
   Keras or Python.

A modern ML detector (TensorFlow 2.x or a PyTorch segmentation model) remains a
reasonable future addition. The current design isolates detection behind
`processing/walls.py`, so swapping the implementation does not change the
domain model, the JSON contract, or the HTTP layer.

## What the rewrite actually preserves

| Aspect | How |
|---|---|
| Pipeline shape | image → elements → renderer-independent JSON |
| Class vocabulary | `wall`, `door`, `window` (`models/opening.py`) |
| Wall → centreline representation | `Wall.centerline`, but derived from geometry rather than a bounding box |
| Bounding-box output for consumers | `models/geometry.BoundingBox` remains available |
| Both entry points | `FloorPlanProcessor` (library) and `POST /process-floorplan` (HTTP) |

## Dependency mapping

| Original | Replacement | Why |
|---|---|---|
| tensorflow 1.15 + keras 2.0.8 | `opencv-contrib-python-headless` | Classical CV has no deep-learning runtime constraint and installs cleanly on 3.14. |
| `mrcnn.model.mold_image` | `processing/preprocessing.preprocess` | Deskew + adaptive threshold replaces letterboxing. |
| `extract_bboxes` / Mask R-CNN ROIs | `processing/walls.detect_segments` | Hough transform finds wall runs directly from ink. |
| `utils.compute_iou` | `BoundingBox.iou` | Kept as a model utility for downstream consumers. |
| flask 2.0.1 | `fastapi` + `uvicorn` | Async, typed, and gives structured validation errors. |
| `numpy.lib.function_base.average` | `statistics` / direct reductions | `numpy.lib.function_base` is private and deprecated. |
| matplotlib / jupyter stack | — | Tooling, not runtime. |

## New modules

| Module | Responsibility |
|---|---|
| `models/geometry.py` | `Point2D`, `Line2D`, `BoundingBox`, orientation classification |
| `models/wall.py` | Wall as a centreline with measured thickness |
| `models/opening.py` | `Door` / `Window` as a gap in a host wall |
| `models/room.py` | Room polygon, area, perimeter, bounding walls |
| `models/floor_plan.py` | Root document, `Units`, `Scale`, `Diagnostics` |
| `processing/image.py` | Decoding and validation (format, size, dimension limits) |
| `processing/preprocessing.py` | Grayscale, blur, deskew, adaptive threshold |
| `processing/walls.py` | Hough segments → centrelines, thickness measurement, ink projection |
| `processing/openings.py` | Gap detection and door/window classification |
| `processing/rooms.py` | Endpoint snapping, intersection extension, polygonisation |
| `processing/scale.py` | Scale validation and unit selection |
| `serialization/json.py` | Wire format and round-trip |
| `pipeline/processor.py` | Stage orchestration, no HTTP |
| `api/app.py` | Upload decoding, error mapping, structured logging |
| `errors.py` | Error hierarchy with stable `code` values |
| `logging_config.py` | Text and JSON log formatters |
