import { describe, expect, it } from 'vitest';
import type { Wall, WallSegment } from '@home-ai/scene-schema';
import { groupContiguousWallSegments } from './wallRuns';

const wall: Wall = {
  id: 'wall_1',
  start: { x: 0, y: 0, z: 0 },
  end: { x: 300, y: 0, z: 0 },
  thickness: 100,
  height: 2800,
  type: 'exterior',
};

const segment = (
  id: string,
  startX: number,
  endX: number,
  bottom = 0,
  top = 2800,
): WallSegment => ({
  id,
  wallId: 'wall_1',
  start: { x: startX, y: 0, z: 0 },
  end: { x: endX, y: 0, z: 0 },
  bottom,
  top,
  thickness: 100,
  type: 'exterior',
});

describe('groupContiguousWallSegments', () => {
  it('merges three contiguous full-height segments into one run', () => {
    const segments = [
      segment('seg_a', 0, 100),
      segment('seg_b', 100, 200),
      segment('seg_c', 200, 300),
    ];

    const runs = groupContiguousWallSegments(segments, new Map([['wall_1', wall]]));

    expect(runs).toHaveLength(1);
    expect(runs[0].start).toEqual({ x: 0, y: 0, z: 0 });
    expect(runs[0].end).toEqual({ x: 300, y: 0, z: 0 });
    expect(runs[0].bottom).toBe(0);
    expect(runs[0].top).toBe(2800);
  });

  it('keeps segments separated by an opening as distinct runs', () => {
    const segments = [segment('seg_a', 0, 100), segment('seg_c', 200, 300)];

    const runs = groupContiguousWallSegments(segments, new Map([['wall_1', wall]]));

    expect(runs).toHaveLength(2);
    expect(runs.map((run) => run.start.x)).toEqual([0, 200]);
  });

  it('keeps segments with different heights as distinct runs', () => {
    const segments = [
      segment('seg_full', 0, 150),
      segment('seg_header', 150, 300, 2100, 2800),
    ];

    const runs = groupContiguousWallSegments(segments, new Map([['wall_1', wall]]));

    expect(runs).toHaveLength(2);
  });

  it('merges contiguous header segments that share a height', () => {
    const segments = [
      segment('seg_header_a', 0, 150, 2100, 2800),
      segment('seg_header_b', 150, 300, 2100, 2800),
    ];

    const runs = groupContiguousWallSegments(segments, new Map([['wall_1', wall]]));

    expect(runs).toHaveLength(1);
    expect(runs[0].start).toEqual({ x: 0, y: 0, z: 0 });
    expect(runs[0].end).toEqual({ x: 300, y: 0, z: 0 });
    expect(runs[0].bottom).toBe(2100);
    expect(runs[0].top).toBe(2800);
  });

  it('does not merge segments from different walls', () => {
    const otherWall: Wall = {
      id: 'wall_2',
      start: { x: 300, y: 0, z: 0 },
      end: { x: 600, y: 0, z: 0 },
      thickness: 100,
      height: 2800,
      type: 'exterior',
    };

    const segments = [
      segment('seg_a', 0, 300),
      {
        ...segment('seg_b', 300, 600),
        wallId: 'wall_2',
      },
    ];

    const runs = groupContiguousWallSegments(
      segments,
      new Map([
        ['wall_1', wall],
        ['wall_2', otherWall],
      ]),
    );

    expect(runs).toHaveLength(2);
  });
});
