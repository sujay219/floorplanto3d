import type { Opening, Vector3, Wall, WallSegment } from '@home-ai/scene-schema';

export interface BoundingBox {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
}

export interface CollisionResult {
  collides: boolean;
  overlap?: number;
  axis?: 'x' | 'y' | 'z';
}

export interface Position2D {
  x: number;
  z: number;
}

export interface Rotation3D {
  x: number;
  y: number;
  z: number;
}

export interface DistanceResult {
  value: number;
  unit: 'mm';
}

export const createBoundingBox = (
  x: number,
  y: number,
  z: number,
  width: number,
  depth: number,
  height: number,
): BoundingBox => ({
  minX: x,
  minY: y,
  minZ: z,
  maxX: x + width,
  maxY: y + height,
  maxZ: z + depth,
});

export const calculateDistance = (from: Position2D, to: Position2D): DistanceResult => ({
  value: Math.hypot(to.x - from.x, to.z - from.z),
  unit: 'mm',
});

export const detectCollision = (
  aMin: number,
  aMax: number,
  bMin: number,
  bMax: number,
): CollisionResult => {
  const overlap = Math.min(aMax, bMax) - Math.max(aMin, bMin);

  return {
    collides: overlap > 0,
    overlap: overlap > 0 ? overlap : 0,
    axis: 'x',
  };
};

const SEGMENT_EPSILON = 1e-9;

interface OpeningInterval {
  start: number;
  end: number;
  bottom: number;
  top: number;
  opening: Opening;
}

const subtract = (a: Vector3, b: Vector3): Vector3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const add = (a: Vector3, b: Vector3): Vector3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const scale = (a: Vector3, scalar: number): Vector3 => ({ x: a.x * scalar, y: a.y * scalar, z: a.z * scalar });
const vectorLength = (a: Vector3): number => Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

const buildOpeningIntervals = (wall: Wall, wallLength: number, openings: Opening[]): OpeningInterval[] => {
  const intervals: OpeningInterval[] = [];

  for (const opening of openings) {
    if (opening.width <= 0 || opening.height <= 0) {
      continue;
    }

    const center = opening.position * wallLength;
    const start = center - opening.width / 2;
    const end = center + opening.width / 2;

    if (start < -SEGMENT_EPSILON || end > wallLength + SEGMENT_EPSILON) {
      continue;
    }

    const bottom = Math.max(0, Math.min(opening.bottom, wall.height));
    const top = Math.max(0, Math.min(opening.bottom + opening.height, wall.height));

    if (top <= bottom + SEGMENT_EPSILON) {
      continue;
    }

    intervals.push({
      start: Math.max(0, start),
      end: Math.min(wallLength, end),
      bottom,
      top,
      opening,
    });
  }

  return intervals;
};

export const generateWallSegments = (walls: Wall[], openings: Opening[]): WallSegment[] => {
  const segments: WallSegment[] = [];
  const openingsByWall = new Map<string, Opening[]>();

  for (const opening of openings) {
    const wallOpenings = openingsByWall.get(opening.wallId) ?? [];
    wallOpenings.push(opening);
    openingsByWall.set(opening.wallId, wallOpenings);
  }

  for (const wall of walls) {
    const wallVector = subtract(wall.end, wall.start);
    const wallLength = vectorLength(wallVector);

    if (wallLength <= SEGMENT_EPSILON) {
      continue;
    }

    const direction = scale(wallVector, 1 / wallLength);
    const intervals = buildOpeningIntervals(wall, wallLength, openingsByWall.get(wall.id) ?? []);
    intervals.sort((a, b) => a.start - b.start);

    const nonOverlapping: OpeningInterval[] = [];
    for (const interval of intervals) {
      const previous = nonOverlapping[nonOverlapping.length - 1];
      if (previous && interval.start < previous.end - SEGMENT_EPSILON) {
        continue;
      }
      nonOverlapping.push(interval);
    }

    const addSegment = (start: number, end: number, bottom: number, top: number): void => {
      if (end - start <= SEGMENT_EPSILON || top - bottom <= SEGMENT_EPSILON) {
        return;
      }

      segments.push({
        id: `${wall.id}_seg_${segments.length}`,
        wallId: wall.id,
        start: add(wall.start, scale(direction, start)),
        end: add(wall.start, scale(direction, end)),
        bottom,
        top,
        thickness: wall.thickness,
        type: wall.type,
        material: wall.material,
      });
    };

    let cursor = 0;
    for (const interval of nonOverlapping) {
      if (interval.start > cursor + SEGMENT_EPSILON) {
        addSegment(cursor, interval.start, 0, wall.height);
      }
      cursor = Math.max(cursor, interval.end);
    }

    if (cursor < wallLength - SEGMENT_EPSILON) {
      addSegment(cursor, wallLength, 0, wall.height);
    }

    for (const interval of nonOverlapping) {
      if (interval.bottom > SEGMENT_EPSILON) {
        addSegment(interval.start, interval.end, 0, interval.bottom);
      }

      if (interval.top < wall.height - SEGMENT_EPSILON) {
        addSegment(interval.start, interval.end, interval.top, wall.height);
      }
    }
  }

  return segments;
};
