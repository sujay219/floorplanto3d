# Original Architecture

Findings from inspecting `FloorPlanTo3D-API/` and `FloorPlanTo3D-unityClient/`.
Neither repository was modified.

## Repository contents

```
floorplan/
├── FloorPlanTo3D-API/
│   ├── application.py          Flask app, 176 lines
│   ├── requirements.txt        123 pinned packages (UTF-16 encoded)
│   ├── runtime.txt
│   └── mrcnn/                  Matterport Mask R-CNN, vendored
│       ├── config.py
│       ├── model.py
│       ├── parallel_model.py
│       ├── utils.py
│       └── visualize.py
└── FloorPlanTo3D-unityClient/
    ├── Builder.cs              JSON -> Unity GameObjects
    ├── WallMesh.cs             2D wall box -> 3D mesh
    └── CubeMeshData.cs
```

## The actual core algorithm

There is no geometric reconstruction in the original project. The pipeline is:

```
image
  ↓  PIL → numpy, RGB
  ↓  mold_image()          resize/letterbox to the network's input shape
  ↓  MaskRCNN.detect()     instance segmentation, 3 classes
  ↓  r['rois'], r['class_ids']
bounding boxes + class labels
  ↓
JSON
```

The only post-processing is in `normalizePoints()`, which sums door bounding-box
dimensions and divides by the door count to produce `averageDoor`. Nothing
converts a bounding box into a wall centreline, thickness, or orientation.

### Detection classes

`getClassNames()` maps Mask R-CNN class ids:

| id | name    |
|----|---------|
| 1  | `wall`  |
| 2  | `window`|
| 3  | `door`  |

### Weights

`application.py` loads `weights/maskrcnn_15_epochs.h5` by name. **This file is
not present in the repository**, so the original service cannot run as shipped
and no trained weights are available to port.

## API contract of the original

`POST /` with `multipart/form-data`, field `image`.

```json
{
  "points":  [{"x1": 12.0, "y1": 30.0, "x2": 400.0, "y2": 44.0}],
  "classes": [{"name": "wall"}],
  "Width":   2048,
  "Height":  1536,
  "averageDoor": 842.5
}
```

Coordinate handling is inconsistent: `normalizePoints()` multiplies
`bb[0]` by `normalizingY` and `bb[1]` by `normalizingX` (both hardcoded to `1`,
so it is a no-op), and `turnSubArraysToJson()` then swaps to `x1 = obj[1]`,
`y1 = obj[0]`. The net effect happens to be a correct x/y mapping, but only
because the normalisation constants are `1`.

`averageDoor` divides by `doorCount`, which is `0` when a plan has no doors, so
any doorless image raises `ZeroDivisionError` and returns HTTP 500.

## Unity client contract

`Builder.cs` reads a local file (`D:/Mask_RCNN/unity/unityData.json`) and maps
each detection to a GameObject:

- `wall` → `WallMesh` component, given the bounding box corners
- `window`, `door` → a bare GameObject with no mesh

`WallMesh.cs` builds the 3D box. Two consequences:

1. A wall is rendered as a box spanning its **bounding box**, so a wall is
   always treated as axis-aligned. A bounding box has no orientation, so a
   diagonal wall cannot be represented.
2. Windows and doors create empty GameObjects. Openings are not subtracted
   from walls, so nothing is cut out of the geometry.

## Dependency analysis

`requirements.txt` pins 123 packages. The chain that matters:

| Package | Pinned | Problem on Python 3.14 |
|---------|--------|------------------------|
| tensorflow | 1.15.3 | Last TF 1.x; no wheels for 3.14, needs OpenSSL 1.1 |
| keras | 2.0.8 | Pre-3.0 Keras, incompatible with modern TF |
| h5py | 2.10.0 | Built against OpenSSL 1.1 |
| numpy | 1.19.5 | No 3.14 wheels |
| flask | 2.0.1 | Superseded |
| matplotlib, jupyter, qtconsole, widgetsnbextension, alabaster, astroid | various | Notebook/tooling, unused at runtime |

The runtime dependencies that actually matter are: TensorFlow 1.15, Keras 2.0.8,
Flask, Pillow, numpy, and the vendored `mrcnn`. Everything from `jupyter-client`
to `xonsh` is development or notebook tooling.

The vendored `mrcnn/model.py` imports `keras.engine` and `distutils.version`,
both removed in modern Keras and Python 3.12+ respectively. Porting it means
porting a 2017-era Mask R-CNN implementation onto a Keras API that no longer
exists in the same shape.

## Data flow summary

```
Client
  └─ POST / (multipart image)
       └─ application.py :: prediction()
            ├─ PIL.Image.open(stream)
            ├─ myImageLoader()          numpy array
            ├─ mrcnn.model.mold_image() letterbox
            ├─ MaskRCNN.detect()        TF graph session
            ├─ normalizePoints()        average door size
            └─ jsonify()                boxes + labels + dimensions

Unity
  └─ Builder.cs
       ├─ JsonUtility.FromJson()
       ├─ GameObject per detection
       └─ WallMesh.setPoints(x1,y1,x2,y2)  → axis-aligned box
```

## What the original does not do

- No wall centrelines, thickness, or orientation
- No room or polygon detection of any kind
- No opening subtraction from walls
- No scale handling; all coordinates are pixels
- No wall merging or junction resolution
- No 3D conversion in Python; that lives in Unity

## Consequences for the rewrite

Because the original produced bounding boxes and its weights are absent, there
is no trained-model behaviour to preserve. The reusable idea is the *shape* of
the pipeline (image → elements → renderer-independent JSON) and the class
vocabulary (wall / door / window). Everything else is a fresh implementation,
which is what `extraction-map.md` records in detail.
