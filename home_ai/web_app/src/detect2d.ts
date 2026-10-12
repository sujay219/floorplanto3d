const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env ?? {};
export const FLOORPLAN_URL = env.VITE_FLOORPLAN_URL ?? 'http://localhost:8000';

export interface Detect2DImagePayload {
  filename: string;
  media_type: string;
  data: string;
}

export interface Detect2DReport {
  original: {
    width: number;
    height: number;
    mode: string;
    format: string | null;
    aspect_ratio: number;
  };
  normalized: {
    width: number;
    height: number;
    mode: string;
    aspect_ratio: number;
    resized: boolean;
    scale: number;
  };
  inspection: {
    background: {
      level: number;
      polarity: 'light' | 'dark';
      gradient_std: number;
      gradient_range: number;
    };
    contrast: {
      min: number;
      max: number;
      low: number;
      median: number;
      high: number;
      dynamic_range: number;
      rms_contrast: number;
    };
    noise: {
      sigma: number;
      sigma_normalized: number;
      method: string;
    };
    wall_thickness: {
      p25_px: number | null;
      median_px: number | null;
      p75_px: number | null;
      p90_px: number | null;
      sample_count: number;
      method: string;
    };
  };
  parameters: Record<string, number | string>;
}

export interface Detect2DResult {
  phase: { number: number; name: string };
  report: Detect2DReport;
  images: {
    original: Detect2DImagePayload;
    normalized: Detect2DImagePayload;
    comparison: Detect2DImagePayload;
  };
}

export interface WallDetectionReport {
  phase: { number: number; name: string };
  steps: { a: string; b: string; c: string };
  input: {
    original_width: number;
    original_height: number;
    normalized_width: number;
    normalized_height: number;
    coordinate_space: string;
  };
  parameters: Record<string, number | string | number[] | string[]>;
  summary: Record<string, number>;
}

export interface WallDetectionResult {
  phase: { number: number; name: string };
  report: WallDetectionReport;
  images: {
    foreground_mask: Detect2DImagePayload;
    foreground_mask_global: Detect2DImagePayload;
    foreground_mask_comparison: Detect2DImagePayload;
    horizontal: Detect2DImagePayload;
    vertical: Detect2DImagePayload;
    combined: Detect2DImagePayload;
  };
}

export interface WallCandidatePoint {
  x: number;
  y: number;
}

export interface WallCandidate {
  id: string;
  geometry: {
    start: WallCandidatePoint;
    end: WallCandidatePoint;
    polygon: WallCandidatePoint[];
    bbox: { x: number; y: number; width: number; height: number };
  };
  length: number;
  angle_deg: number;
  orientation: string;
  estimated_thickness: number;
  thickness_consistency: number;
  straightness: number;
  continuity: number;
  edge_pairing: {
    paired: boolean;
    edge_ids: string[];
    angle_delta_deg: number | null;
    distance_px: number | null;
    overlap_fraction: number | null;
    score: number;
    probed_thickness_px?: number;
  };
  axis_support: Record<string, number>;
  confidence: number;
  confidence_evidence: Record<string, number>;
  evidence: string[];
  uncertainty: string[];
  merged_from: string[];
}

export interface WallCandidatesReport {
  phase: { number: number; name: string; step: string };
  steps: Record<string, string>;
  summary: {
    total_candidates: number;
    paired_edge_candidates: number;
    single_edge_candidates: number;
    merged_source_count: number;
    by_orientation: Record<string, number>;
    mean_confidence: number;
    low_confidence_candidates: number;
  };
  input: Record<string, number | string>;
  parameters: Record<string, unknown>;
  limitations: string[];
}

export interface WallCandidatesResult {
  phase: { number: number; name: string; step: string };
  candidates: WallCandidate[];
  report: WallCandidatesReport;
  images: {
    overlay: Detect2DImagePayload;
  };
}

export async function detectFloorPlan2D(file: File): Promise<Detect2DResult> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${FLOORPLAN_URL}/detect2d`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message = payload?.error?.message ?? `FloorPlanTo3D /detect2d failed (${response.status})`;
    throw new Error(message);
  }

  return payload as Detect2DResult;
}

export async function detectWalls(file: File): Promise<WallDetectionResult> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${FLOORPLAN_URL}/detect2d/walls`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message = payload?.error?.message ?? `FloorPlanTo3D /detect2d/walls failed (${response.status})`;
    throw new Error(message);
  }

  return payload as WallDetectionResult;
}

export async function detectWallCandidates(file: File): Promise<WallCandidatesResult> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${FLOORPLAN_URL}/detect2d/wall-candidates`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message =
      payload?.error?.message ?? `FloorPlanTo3D /detect2d/wall-candidates failed (${response.status})`;
    throw new Error(message);
  }

  return payload as WallCandidatesResult;
}
