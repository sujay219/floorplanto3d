import { describe, expect, it } from 'vitest';
import type { FloorPlanJson } from './floorplan';
import { floorplanToScene } from './floorplan';

const MILLIMETERS_PER_PIXEL = 10;

const sampleFloorplan = (): FloorPlanJson => ({
  version: '1.0',
  units: 'px',
  scale: { pixels_per_unit: null, unit: 'mm', source: null },
  image: { width: 600, height: 400, format: 'PNG', color_space: 'sRGB' },
  walls: [
    { id: 'wall_001', centerline: { start: { x: 51, y: 346 }, end: { x: 51, y: 56 } }, thickness: 8 },
    { id: 'wall_002', centerline: { start: { x: 51, y: 56 }, end: { x: 301, y: 56 } }, thickness: 8 },
    { id: 'wall_004', centerline: { start: { x: 301, y: 346 }, end: { x: 301, y: 56 } }, thickness: 8 },
    { id: 'wall_005', centerline: { start: { x: 51, y: 346 }, end: { x: 301, y: 346 } }, thickness: 8 },
  ],
  rooms: [
    {
      id: 'room_001',
      name: null,
      polygon: [
        { x: 49, y: 51 },
        { x: 49, y: 351 },
        { x: 301, y: 351 },
        { x: 301, y: 51 },
      ],
      wall_ids: ['wall_001', 'wall_002', 'wall_004', 'wall_005'],
    },
  ],
  doors: [
    {
      id: 'door_001',
      type: 'door',
      wall_id: 'wall_002',
      start: { x: 301, y: 181.433 },
      end: { x: 301, y: 220.567 },
      width: 39.135,
    },
  ],
  windows: [],
  diagnostics: { wall_count: 4, room_count: 1, door_count: 1, window_count: 0, opening_count: 1 },
});

describe('floorplanToScene', () => {
  it('scales a pixel floor plan into millimeters', () => {
    const scene = floorplanToScene(sampleFloorplan());

    expect(scene.units).toBe('mm');
    expect(scene.rooms).toHaveLength(1);

    const room = scene.rooms[0];
    expect(room.dimensions.width).toBeCloseTo((301 - 49) * MILLIMETERS_PER_PIXEL);
    expect(room.dimensions.depth).toBeCloseTo((351 - 51) * MILLIMETERS_PER_PIXEL);
    expect(room.dimensions.height).toBe(560);
    expect(room.walls).toHaveLength(4);
    expect(room.doors).toHaveLength(1);
    expect(room.windows).toHaveLength(0);
    expect(room.objects).toHaveLength(0);
  });

  it('normalizes geometry to the origin', () => {
    const room = floorplanToScene(sampleFloorplan()).rooms[0];

    const minX = Math.min(...room.walls.flatMap((wall) => [wall.start.x, wall.end.x]));
    const minY = Math.min(...room.walls.flatMap((wall) => [wall.start.y, wall.end.y]));

    expect(minX).toBeCloseTo((51 - 49) * MILLIMETERS_PER_PIXEL);
    expect(minY).toBeCloseTo((56 - 51) * MILLIMETERS_PER_PIXEL);
  });

  it('scales wall thickness and opening widths', () => {
    const room = floorplanToScene(sampleFloorplan()).rooms[0];

    expect(room.walls[0].thickness).toBeCloseTo(8 * MILLIMETERS_PER_PIXEL);
    expect(room.doors[0].width).toBeCloseTo(39.135 * MILLIMETERS_PER_PIXEL);
  });

  it('treats a millimeter floor plan as identity-scaled', () => {
    const floorplan = sampleFloorplan();
    floorplan.units = 'mm';

    const room = floorplanToScene(floorplan).rooms[0];

    expect(room.dimensions.width).toBeCloseTo(301 - 49);
    expect(room.dimensions.depth).toBeCloseTo(351 - 51);
    expect(room.walls[0].thickness).toBeCloseTo(8);
  });

  it('throws when the floor plan has no geometry', () => {
    expect(() => floorplanToScene({})).toThrow();
  });
});
