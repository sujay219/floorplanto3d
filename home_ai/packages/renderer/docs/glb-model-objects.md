# GLB model objects

How `BabylonRenderer` renders `SceneObject`s whose type maps to a glTF/GLB asset
(`MODEL_URLS` in `src/index.ts`, currently `bed` → `bed.glb` at the repo root).

## Pipeline

1. `buildObject` dispatches to `buildModelObject` when `MODEL_URLS[object.type]` is set;
   every other type still renders as a box.
2. The GLB is loaded once per URL via `SceneLoader.LoadAssetContainerAsync` (glTF plugin
   comes from the `babylonjs-loaders` side-effect import). The container promise is cached
   per renderer so re-renders only instantiate, never refetch.
3. `placeModelInstance` instantiates the container into the scene with
   `instantiateModelsToScene`, renaming every node to `object_<id>__<sourceName>` so the
   standard `render()` cleanup catches it on the next render.

## Determinism guard

`render()` increments `renderGeneration`. A model load started by an older render is
discarded when it resolves (`buildModelObject` compares generations before instantiating).
This keeps output deterministic under React StrictMode double-render or rapid re-renders:
exactly one instance per scene object.

## Coordinate and size mapping (the non-obvious part)

Scene space is Z-up, millimeters. glTF is Y-up, meters. The node graph is:

- `object_<id>` (pivot) — carries `object.position` / `object.rotation` from scene data.
- `object_<id>__oriented` (child) — carries the model→scene conversion.
- model root nodes (children of `oriented`) — untouched glTF local transforms.

The measured model-space bounding box (`min`/`max` over `oriented.getChildMeshes()`,
measured while every node is identity) maps to the object's declared `dimensions` exactly,
like the box renderer does:

| model axis | scene axis | size source |
|------------|------------|-------------|
| X          | X (width)  | `dimensions.width` |
| Y (up)     | Z (height) | `dimensions.height` |
| Z          | −Y (depth) | `dimensions.depth` |

So `oriented.scaling = (width/extX, height/extY, depth/extZ)` and
`oriented.rotation.x = MODEL_UP_TO_SCENE_UP` (π/2 maps +Y → +Z, +Z → −Y in Babylon).
`oriented.position` then recenters the bbox on the pivot's plan origin and puts its
bottom at z = 0, so `pivot.position` means the same as the box mesh's `position`:
plan center at (x, y), base at z.

If the declared dimensions do not match the model's aspect ratio the model stretches to
fill them — the dimensions field is the source of truth, same as for box objects.

## Naming contract with `render()` cleanup

- `object_<id>` — pivot transform node, disposed (recursively, with materials) by the
  `transformNodes` filter in `render()`.
- `object_<id>__oriented` and `object_<id>__<sourceName>` — caught by the same filter /
  the `object_` mesh filter.

`dispose()` disposes cached asset containers when the renderer goes away.
