import type { Vector3, Wall, WallSegment } from '@home-ai/scene-schema';

export interface WallRun {
  wallId: string;
  id: string;
  start: Vector3;
  end: Vector3;
  bottom: number;
  top: number;
  type?: 'exterior' | 'interior';
}

const EPS = 1e-6;

const sameHeight = (a: WallSegment, b: WallSegment): boolean =>
  Math.abs(a.bottom - b.bottom) < EPS && Math.abs(a.top - b.top) < EPS;

const isContiguous = (a: WallSegment, b: WallSegment): boolean =>
  Math.abs(a.end.x - b.start.x) < EPS &&
  Math.abs(a.end.y - b.start.y) < EPS &&
  Math.abs(a.end.z - b.start.z) < EPS;

const distanceAlongWall = (wall: Wall, point: Vector3): number => {
  const dx = wall.end.x - wall.start.x;
  const dy = wall.end.y - wall.start.y;
  const dz = wall.end.z - wall.start.z;
  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (length < EPS) {
    return 0;
  }

  return (
    ((point.x - wall.start.x) * dx +
      (point.y - wall.start.y) * dy +
      (point.z - wall.start.z) * dz) /
    length
  );
};

/**
 * Groups wall segments into contiguous runs. Adjacent segments are merged only
 * when they share a wall, share the same top/bottom (same cross-section), and
 * touch end-to-end with no opening between them. This lets coplanar segments of
 * one wall be rendered as a single mesh (and a single transparent surface)
 * while preserving real openings and height changes as separate runs.
 */
export const groupContiguousWallSegments = (
  segments: WallSegment[],
  wallsById: Map<string, Wall>,
): WallRun[] => {
  const runs: WallRun[] = [];
  const segmentsByWall = new Map<string, WallSegment[]>();

  for (const segment of segments) {
    const list = segmentsByWall.get(segment.wallId) ?? [];
    list.push(segment);
    segmentsByWall.set(segment.wallId, list);
  }

  segmentsByWall.forEach((wallSegments, wallId) => {
    const wall = wallsById.get(wallId);
    if (!wall) {
      return;
    }

    const sorted = [...wallSegments].sort(
      (a, b) => distanceAlongWall(wall, a.start) - distanceAlongWall(wall, b.start),
    );

    let run: WallSegment[] = [];
    const flush = (): void => {
      if (run.length === 0) {
        return;
      }

      const first = run[0];
      const last = run[run.length - 1];
      runs.push({
        wallId,
        id: first.id,
        start: first.start,
        end: last.end,
        bottom: first.bottom,
        top: first.top,
        type: first.type,
      });
      run = [];
    };

    for (const segment of sorted) {
      const previous = run[run.length - 1];
      if (previous && sameHeight(previous, segment) && isContiguous(previous, segment)) {
        run.push(segment);
      } else {
        flush();
        run = [segment];
      }
    }
    flush();
  });

  return runs;
};
