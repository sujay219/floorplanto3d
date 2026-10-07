import { floorplanToScene } from '@home-ai/scene-schema';
import type { Scene } from '@home-ai/scene-schema';

const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env ?? {};
const FLOORPLAN_URL = env.VITE_FLOORPLAN_URL ?? 'http://localhost:8000';

export async function processFloorPlanImage(file: File): Promise<Scene> {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${FLOORPLAN_URL}/process-floorplan`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message = payload?.error?.message ?? `FloorPlanTo3D failed (${response.status})`;
    throw new Error(message);
  }

  return floorplanToScene(payload);
}
