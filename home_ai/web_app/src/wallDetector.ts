import { Detect2DImagePayload, FLOORPLAN_URL } from './detect2d';

export interface WallSegmentPoint {
  x: number;
  y: number;
}

export interface WallSegment {
  id: string;
  orientation: 'horizontal' | 'vertical';
  length: number;
  thickness: number;
  area: number;
  bbox: { x: number; y: number; width: number; height: number };
  polygon: WallSegmentPoint[];
}

export interface WallDetectorReport {
  phase: { number: number; name: string };
  coordinate_space: string;
  summary: {
    total_segments: number;
    horizontal_segments: number;
    vertical_segments: number;
  };
}

export interface WallDetectorResult {
  phase: { number: number; name: string };
  images: {
    normalized: Detect2DImagePayload;
    combined: Detect2DImagePayload;
  };
  segments: WallSegment[];
  report: WallDetectorReport;
}

export async function fetchWallDetector(): Promise<WallDetectorResult> {
  const response = await fetch(`${FLOORPLAN_URL}/detect2d/wall-detector`);

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message =
      payload?.error?.message ?? `FloorPlanTo3D /detect2d/wall-detector failed (${response.status})`;
    throw new Error(message);
  }

  return payload as WallDetectorResult;
}
