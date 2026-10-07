import { describe, expect, it } from 'vitest';
import type { Opening, Wall, WallSegment } from '@home-ai/scene-schema';
import { generateWallSegments } from './index';

const EPS = 1e-9;

const createWall = (overrides: Partial<Wall> = {}): Wall => ({
  id: 'wall_1',
  start: { x: 0, y: 0, z: 0 },
  end: { x: 5000, y: 0, z: 0 },
  thickness: 100,
  height: 2800,
  ...overrides,
});

const createOpening = (overrides: Partial<Opening> = {}): Opening => ({
  id: 'opening_1',
  wallId: 'wall_1',
  position: 0.45,
  width: 900,
  bottom: 0,
  height: 2100,
  type: 'door',
  ...overrides,
});

const longitudinalRange = (segment: WallSegment): { min: number; max: number } => {
  const dx = Math.abs(segment.end.x - segment.start.x);
  const dy = Math.abs(segment.end.y - segment.start.y);

  if (dx >= dy) {
    return { min: Math.min(segment.start.x, segment.end.x), max: Math.max(segment.start.x, segment.end.x) };
  }

  return { min: Math.min(segment.start.y, segment.end.y), max: Math.max(segment.start.y, segment.end.y) };
};

const overlaps = (a: WallSegment, b: WallSegment): boolean => {
  const aRange = longitudinalRange(a);
  const bRange = longitudinalRange(b);
  const overlapsLongitudinally = aRange.min < bRange.max - EPS && bRange.min < aRange.max - EPS;
  const overlapsVertically = a.bottom < b.top - EPS && b.bottom < a.top - EPS;
  return overlapsLongitudinally && overlapsVertically;
};

describe('generateWallSegments', () => {
  it('returns one full-height segment for a wall without openings', () => {
    const segments = generateWallSegments([createWall()], []);

    expect(segments).toHaveLength(1);
    expect(segments[0].bottom).toBe(0);
    expect(segments[0].top).toBe(2800);
    expect(segments[0].start).toEqual({ x: 0, y: 0, z: 0 });
    expect(segments[0].end).toEqual({ x: 5000, y: 0, z: 0 });
  });

  it('splits a door wall into two full-height runs plus a header', () => {
    // Door from x=1800 to x=2700, bottom=0, height=2100.
    const segments = generateWallSegments(
      [createWall()],
      [createOpening({ position: 0.45, width: 900, bottom: 0, height: 2100 })],
    );

    expect(segments).toHaveLength(3);

    const fullHeight = segments.filter((segment) => segment.bottom === 0 && segment.top === 2800);
    expect(fullHeight).toHaveLength(2);
    expect(fullHeight.map(longitudinalRange)).toEqual([
      { min: 0, max: 1800 },
      { min: 2700, max: 5000 },
    ]);

    const header = segments.find((segment) => segment.bottom > 0);
    expect(header).toBeDefined();
    expect(header?.bottom).toBe(2100);
    expect(header?.top).toBe(2800);
    expect(longitudinalRange(header as WallSegment)).toEqual({ min: 1800, max: 2700 });

    expect(segments.some((segment) => longitudinalRange(segment).min === 0 && longitudinalRange(segment).max === 5000 && segment.bottom > 0)).toBe(false);

    for (let i = 0; i < segments.length; i++) {
      for (let j = i + 1; j < segments.length; j++) {
        expect(overlaps(segments[i], segments[j])).toBe(false);
      }
    }
  });

  it('splits a window into full-height runs plus sill and header', () => {
    // Window from x=1000 to x=2200, sill=900, height=1200.
    const segments = generateWallSegments(
      [createWall()],
      [createOpening({ position: 0.32, width: 1200, bottom: 900, height: 1200, type: 'window' })],
    );

    expect(segments).toHaveLength(4);

    const fullHeight = segments.filter((segment) => segment.bottom === 0 && segment.top === 2800);
    expect(fullHeight).toHaveLength(2);
    expect(fullHeight.map(longitudinalRange)).toEqual([
      { min: 0, max: 1000 },
      { min: 2200, max: 5000 },
    ]);

    const sill = segments.find((segment) => segment.bottom === 0 && segment.top === 900);
    const header = segments.find((segment) => segment.bottom === 2100 && segment.top === 2800);
    expect(sill).toBeDefined();
    expect(header).toBeDefined();
    expect(longitudinalRange(sill as WallSegment)).toEqual({ min: 1000, max: 2200 });
    expect(longitudinalRange(header as WallSegment)).toEqual({ min: 1000, max: 2200 });
  });

  it('supports multiple sorted openings without overlapping geometry', () => {
    const door = createOpening({ id: 'door', position: 0.45, width: 900, bottom: 0, height: 2100 });
    const window = createOpening({
      id: 'window',
      position: 0.18,
      width: 800,
      bottom: 900,
      height: 1200,
      type: 'window',
    });

    const segments = generateWallSegments([createWall()], [window, door]);

    expect(segments).toHaveLength(6);
    expect(segments.filter((segment) => segment.bottom === 0 && segment.top === 2800)).toHaveLength(3);

    for (let i = 0; i < segments.length; i++) {
      for (let j = i + 1; j < segments.length; j++) {
        expect(overlaps(segments[i], segments[j])).toBe(false);
      }
    }
  });

  it('does not overlap geometry for adjacent openings', () => {
    const door = createOpening({ id: 'door', position: 0.45, width: 900, bottom: 0, height: 2100 });
    const window = createOpening({
      id: 'window',
      position: 0.61,
      width: 700,
      bottom: 900,
      height: 1200,
      type: 'window',
    });

    // door = [1800, 2700], window = [2700, 3400] (touching).
    const segments = generateWallSegments([createWall()], [door, window]);

    expect(segments).toHaveLength(5);

    for (let i = 0; i < segments.length; i++) {
      for (let j = i + 1; j < segments.length; j++) {
        expect(overlaps(segments[i], segments[j])).toBe(false);
      }
    }
  });

  it('skips overlapping openings instead of emitting overlapping geometry', () => {
    const door = createOpening({ id: 'door', position: 0.45, width: 900, bottom: 0, height: 2100 });
    const overlapping = createOpening({ id: 'overlap', position: 0.49, width: 900, bottom: 0, height: 2100 });

    const segments = generateWallSegments([createWall()], [door, overlapping]);

    expect(segments).toHaveLength(3);
    const header = segments.find((segment) => segment.bottom === 2100);
    expect(longitudinalRange(header as WallSegment)).toEqual({ min: 1800, max: 2700 });
  });

  it('skips openings that extend outside the wall', () => {
    const segments = generateWallSegments(
      [createWall()],
      [createOpening({ position: 0.5, width: 6000, bottom: 0, height: 2100 })],
    );

    expect(segments).toHaveLength(1);
    expect(segments[0].bottom).toBe(0);
    expect(segments[0].top).toBe(2800);
  });

  it('works for walls oriented along the Y axis', () => {
    const yWall = createWall({
      start: { x: 0, y: 0, z: 0 },
      end: { x: 0, y: 5000, z: 0 },
    });

    const segments = generateWallSegments(
      [yWall],
      [createOpening({ position: 0.45, width: 900, bottom: 0, height: 2100 })],
    );

    expect(segments).toHaveLength(3);

    const fullHeight = segments.filter((segment) => segment.bottom === 0 && segment.top === 2800);
    expect(fullHeight).toHaveLength(2);
    expect(fullHeight.map(longitudinalRange)).toEqual([
      { min: 0, max: 1800 },
      { min: 2700, max: 5000 },
    ]);

    const header = segments.find((segment) => segment.bottom === 2100);
    expect(longitudinalRange(header as WallSegment)).toEqual({ min: 1800, max: 2700 });
  });
});
