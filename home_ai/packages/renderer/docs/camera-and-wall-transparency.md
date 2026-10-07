# Camera and wall transparency

This doc explains how the Babylon renderer positions the camera and how it
decides when a wall should become translucent so the interior of a room stays
visible.

## Camera

The renderer uses a `BABYLON.ArcRotateCamera`:

- Initial angles: `alpha = -PI / 4`, `beta = PI / 3`
- Default radius: `2400` mm (see `defaultCameraRadius`)
- Radius limits: `400` to `20000` mm
- Up vector: `(0, 0, 1)` (height runs along Z)
- Target: the center of the primary room in the plan plane
  (`dimensions.width / 2`, `dimensions.depth / 2`, `0`)

The plan is the XY plane and height is Z. An `ArcRotateCamera` orbits a fixed
target using spherical coordinates:

- `alpha` rotates around the target in the plan (XY) plane
- `beta` controls elevation above the target
- `radius` is the distance from the target to the camera

The camera is re-centered on every `render()` call via `ensureCamera`, which
sets the target and resets `alpha`, `beta`, and `radius`.

## Why walls fade

When the camera orbits a room, the walls nearest to the camera can block the
view of objects inside the room. To keep the interior visible, those walls are
made translucent. The camera-change observable calls `syncWallTransparency`
after every camera move, so the fading follows the orbit in real time.

## How obstruction is determined

The logic lives in `syncWallTransparency` (`packages/renderer/src/index.ts`).

Wall segment meshes carry `metadata.wallId`, which points back to the parent
`Wall` in the renderer's `wallById` map. Segment meshes are grouped by
`wallId`, and obstruction is computed once per semantic wall:

1. Compute the wall midpoint from the parent wall's `start`/`end`.
2. Compute the wall's 2D normal in the plan (XY) plane:
   `normal = (-direction.y, direction.x)` (normalized by wall length).
3. Compute the 2D direction from the wall midpoint to the camera.
4. Compute the 2D direction from the wall midpoint to the room center.
5. Project both directions onto the wall normal (dot products).
6. A wall obstructs the room when the camera and room center are on opposite
   sides of the wall: `roomSide * cameraSide < 0`.

A wall is treated as exterior when the room's interior lies entirely on one
side of it. With a single reference room center, this is decided purely by the
sign of `roomSide`: the outward face is the side opposite that center. The
camera is "outside" when it is on that opposite side, in which case the wall
fades. No explicit `type` field is required, so the decision stays independent
of how the scene data was produced.

Because the fade is keyed on `wallId` instead of mesh ordering or segment
count, splitting a wall into more segments does not change which walls fade.

The room center is the primary shell center (`target.dimensions.width / 2`,
`target.dimensions.depth / 2`, `0`).

## Alpha values

- Opaque: `alpha = 1`
- Translucent (obstructing): `alpha = 0.35`

Wall materials are opaque by default. Only the outward-facing exterior face is
switched to `transparencyMode = MATERIAL_ALPHABLEND` (alpha 0.35) when it
obstructs the camera; otherwise it stays opaque, so the depth buffer handles it.

## Shared walls and materials

Before meshes are built, `groupContiguousWallSegments`
(`packages/renderer/src/wallRuns.ts`) merges adjacent segments that share a
wall, share the same `top`/`bottom`, and touch end-to-end with no opening
between them into a single "run". Each run becomes one `BABYLON.Mesh` with one
`MultiMaterial` (interior + exterior sub-materials), so a wall that was split
into several coplanar segments renders as one continuous surface.

This matters for transparency: when the camera is outside, adjacent coplanar
transparent meshes would otherwise alpha-blend independently (and their
coincident end caps would show through), producing visible seams at segment
boundaries. Merging removes the internal caps and applies the fade to a single
material, so the exterior face fades uniformly.

Each run's material is named `wallInteriorMaterial_${run.id}` and
`wallExteriorMaterial_${run.id}`, so changing one run's alpha does not affect
other walls. The interior-facing face always stays opaque, while only the
exterior-facing face can fade. Run meshes carry `metadata.wallId`, so every
derived run remains traceable to its parent wall and fades together with it.

## Key methods

- `ensureCamera` — creates/re-centers the orbit camera.
- `syncWallTransparency` — updates each wall's exterior material alpha from the
  current camera position.
