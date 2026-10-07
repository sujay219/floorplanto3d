import type { Dimensions, Door, Room, Scene, Vector3, Wall, Window } from './index';

export interface FloorPlanPoint {
  x: number;
  y: number;
}

export interface FloorPlanWall {
  id: string;
  centerline?: { start?: FloorPlanPoint; end?: FloorPlanPoint };
  thickness?: number | null;
}

export interface FloorPlanRoom {
  id: string;
  name?: string | null;
  polygon?: FloorPlanPoint[];
  wall_ids?: string[];
}

export interface FloorPlanOpening {
  id: string;
  type?: 'door' | 'window';
  wall_id?: string;
  start?: FloorPlanPoint;
  end?: FloorPlanPoint;
  width?: number;
  swing_direction?: string | null;
  sill_height?: number | null;
}

export interface FloorPlanDiagnostics {
  wall_count?: number;
  room_count?: number;
  door_count?: number;
  window_count?: number;
  opening_count?: number;
  warnings?: string[];
  processing_ms?: number;
}

export interface FloorPlanJson {
  version?: string;
  units?: string;
  scale?: { pixels_per_unit?: number | null; unit?: string; source?: string | null };
  image?: { width?: number; height?: number; format?: string; color_space?: string };
  walls?: FloorPlanWall[];
  rooms?: FloorPlanRoom[];
  doors?: FloorPlanOpening[];
  windows?: FloorPlanOpening[];
  diagnostics?: FloorPlanDiagnostics;
}

export interface FloorPlanToSceneOptions {
  millimetersPerPixel?: number;
  roomHeight?: number;
  wallThickness?: number;
  doorWidth?: number;
  doorHeight?: number;
  windowWidth?: number;
  windowHeight?: number;
  windowSillHeight?: number;
}

export const UNIT_TO_MM: Record<string, number> = {
  mm: 1,
  cm: 10,
  m: 1000,
  in: 25.4,
  ft: 304.8,
};

const DEFAULTS = {
  millimetersPerPixel: 10,
  roomHeight: 800,
  wallThickness: 100,
  doorWidth: 600,
  doorHeight: 630,
  windowWidth: 1200,
  windowHeight: 360,
  windowSillHeight: 180,
} as const;

const SWING_MAP: Record<string, 'left' | 'right'> = {
  left: 'left',
  inward: 'left',
  right: 'right',
  outward: 'right',
};

interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function unitScale(units: string | undefined, millimetersPerPixel: number): number {
  const resolved = units ?? 'px';
  if (resolved in UNIT_TO_MM) {
    return UNIT_TO_MM[resolved];
  }
  if (resolved === 'px') {
    return millimetersPerPixel;
  }
  return 1;
}

function planBounds(floorplan: FloorPlanJson): Bounds | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const include = (point: FloorPlanPoint | undefined): void => {
    if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') {
      return;
    }
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  };

  for (const wall of floorplan.walls ?? []) {
    include(wall.centerline?.start);
    include(wall.centerline?.end);
  }

  for (const room of floorplan.rooms ?? []) {
    for (const point of room.polygon ?? []) {
      include(point);
    }
  }

  for (const opening of [...(floorplan.doors ?? []), ...(floorplan.windows ?? [])]) {
    include(opening.start);
    include(opening.end);
  }

  if (minX > maxX || minY > maxY) {
    return null;
  }

  return { minX, minY, maxX, maxY };
}

function toVector(point: FloorPlanPoint, scale: number, minX: number, minY: number): Vector3 {
  return { x: (point.x - minX) * scale, y: (point.y - minY) * scale, z: 0 };
}

function scaled(value: number | null | undefined, scale: number, fallback: number): number {
  return value == null ? fallback : value * scale;
}

function mapSwing(raw: string | null | undefined): 'left' | 'right' | undefined {
  return raw ? SWING_MAP[raw] : undefined;
}

function buildWalls(
  walls: FloorPlanWall[],
  scale: number,
  bounds: Bounds,
  roomHeight: number,
  options: FloorPlanToSceneOptions,
): Wall[] {
  const thickness = options.wallThickness ?? DEFAULTS.wallThickness;
  const result: Wall[] = [];

  for (const wall of walls) {
    const start = wall.centerline?.start;
    const end = wall.centerline?.end;
    if (!start || !end) {
      continue;
    }

    result.push({
      id: wall.id,
      start: toVector(start, scale, bounds.minX, bounds.minY),
      end: toVector(end, scale, bounds.minX, bounds.minY),
      thickness: scaled(wall.thickness, scale, thickness),
      height: roomHeight,
    });
  }

  return result;
}

function buildDoors(
  openings: FloorPlanOpening[],
  scale: number,
  bounds: Bounds,
  options: FloorPlanToSceneOptions,
): Door[] {
  const result: Door[] = [];

  for (const opening of openings) {
    const start = opening.start;
    const end = opening.end;
    if (!start || !end) {
      continue;
    }

    const swing = mapSwing(opening.swing_direction);
    result.push({
      id: opening.id,
      start: toVector(start, scale, bounds.minX, bounds.minY),
      end: toVector(end, scale, bounds.minX, bounds.minY),
      width: scaled(opening.width, scale, options.doorWidth ?? DEFAULTS.doorWidth),
      height: options.doorHeight ?? DEFAULTS.doorHeight,
      ...(swing ? { swing } : {}),
    });
  }

  return result;
}

function buildWindows(
  openings: FloorPlanOpening[],
  scale: number,
  bounds: Bounds,
  options: FloorPlanToSceneOptions,
): Window[] {
  const result: Window[] = [];

  for (const opening of openings) {
    const start = opening.start;
    const end = opening.end;
    if (!start || !end) {
      continue;
    }

    result.push({
      id: opening.id,
      start: toVector(start, scale, bounds.minX, bounds.minY),
      end: toVector(end, scale, bounds.minX, bounds.minY),
      width: scaled(opening.width, scale, options.windowWidth ?? DEFAULTS.windowWidth),
      height: options.windowHeight ?? DEFAULTS.windowHeight,
      sillHeight: opening.sill_height ?? options.windowSillHeight ?? DEFAULTS.windowSillHeight,
    });
  }

  return result;
}

export function floorplanToScene(floorplan: FloorPlanJson, options: FloorPlanToSceneOptions = {}): Scene {
  if (!floorplan || typeof floorplan !== 'object') {
    throw new Error('Floor plan must be a JSON object.');
  }

  const scale = unitScale(floorplan.units, options.millimetersPerPixel ?? DEFAULTS.millimetersPerPixel);
  const bounds = planBounds(floorplan);

  if (!bounds) {
    throw new Error('Floor plan contains no geometry.');
  }

  const width = (bounds.maxX - bounds.minX) * scale;
  const depth = (bounds.maxY - bounds.minY) * scale;

  if (width <= 0 || depth <= 0) {
    throw new Error('Floor plan has zero extent.');
  }

  const roomHeight = options.roomHeight ?? DEFAULTS.roomHeight;
  const walls = buildWalls(floorplan.walls ?? [], scale, bounds, roomHeight, options);
  const doors = buildDoors(floorplan.doors ?? [], scale, bounds, options);
  const windows = buildWindows(floorplan.windows ?? [], scale, bounds, options);

  const dimensions: Dimensions = { width, depth, height: roomHeight };

  const room: Room = {
    id: 'floorplan',
    name: 'Floor Plan',
    dimensions,
    walls,
    doors,
    windows,
    objects: [],
    metadata: {
      source: 'floorplanto3d',
      sourceVersion: floorplan.version,
      sourceUnits: floorplan.units,
      wallCount: floorplan.diagnostics?.wall_count ?? walls.length,
      roomCount: floorplan.diagnostics?.room_count,
      doorCount: floorplan.diagnostics?.door_count ?? doors.length,
      windowCount: floorplan.diagnostics?.window_count ?? windows.length,
    },
  };

  return {
    version: 1,
    units: 'mm',
    rooms: [room],
    metadata: {
      name: 'Floor Plan',
      description: 'Scene adapted from FloorPlanTo3D output.',
    },
  };
}
