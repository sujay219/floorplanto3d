const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env ?? {};
const FLOORPLAN_URL = env.VITE_FLOORPLAN_URL ?? 'http://localhost:8000';

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
