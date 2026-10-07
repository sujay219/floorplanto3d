export type SceneObjectType =
  | 'base_cabinet'
  | 'drawer_cabinet'
  | 'wall_cabinet'
  | 'loft'
  | 'countertop'
  | 'sink'
  | 'hob'
  | 'tall_unit'
  | 'wardrobe'
  | 'bed';

export type AnchorType = 'absolute' | 'wall' | 'object' | 'room';
export type AnchorPosition = 'left' | 'right' | 'above' | 'below' | 'center' | 'next_available';

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export type Rotation = Vector3;

export interface Dimensions {
  width: number;
  depth: number;
  height: number;
}

export interface Material {
  name: string;
  color?: string;
  finish?: string;
}

export interface Transform {
  position: Vector3;
  rotation: Rotation;
  scale?: Vector3;
}

export interface Anchor {
  type: AnchorType;
  targetId?: string;
  position?: AnchorPosition;
  offset?: Vector3;
}

export interface SceneMetadata {
  name?: string;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface Wall {
  id: string;
  start: Vector3;
  end: Vector3;
  thickness: number;
  height: number;
  type?: 'exterior' | 'interior';
  material?: Material;
}

export interface Door {
  id: string;
  start: Vector3;
  end: Vector3;
  width: number;
  height: number;
  swing?: 'left' | 'right';
}

export interface Window {
  id: string;
  start: Vector3;
  end: Vector3;
  width: number;
  height: number;
  sillHeight?: number;
}

export interface Opening {
  id: string;
  wallId: string;
  position: number; // normalized [0,1]
  width: number;
  bottom: number;
  height: number;
  type: 'door' | 'window';
}

export interface WallSegment {
  id: string;
  wallId: string;
  start: Vector3;
  end: Vector3;
  bottom: number;
  top: number;
  thickness: number;
  type?: 'exterior' | 'interior';
  material?: Material;
}

export interface Room {
  id: string;
  name?: string;
  dimensions: Dimensions;
  walls: Wall[];
  doors: Door[];
  windows: Window[];
  objects: SceneObject[];
  metadata?: Record<string, string | number | boolean | undefined>;
}

export interface SceneObject {
  id: string;
  type: SceneObjectType;
  name?: string;
  position: Vector3;
  rotation: Rotation;
  dimensions: Dimensions;
  material?: Material;
  anchor?: Anchor;
  metadata?: Record<string, string | number | boolean | undefined>;
}

export interface Scene {
  version: number;
  units: 'mm';
  rooms: Room[];
  metadata: SceneMetadata;
}

export interface SceneError {
  code: string;
  message: string;
  path?: string;
}

export interface OperationResult {
  success: boolean;
  scene?: Scene;
  errors: SceneError[];
}

export interface AddObjectOperation {
  op: 'ADD_OBJECT';
  roomId: string;
  object: SceneObject;
}

export interface RemoveObjectOperation {
  op: 'REMOVE_OBJECT';
  roomId: string;
  objectId: string;
}

export interface MoveObjectOperation {
  op: 'MOVE_OBJECT';
  roomId: string;
  objectId: string;
  position: Vector3;
}

export interface ResizeObjectOperation {
  op: 'RESIZE_OBJECT';
  roomId: string;
  objectId: string;
  dimensions: Dimensions;
}

export interface RotateObjectOperation {
  op: 'ROTATE_OBJECT';
  roomId: string;
  objectId: string;
  rotation: Rotation;
}

export interface UpdateObjectOperation {
  op: 'UPDATE_OBJECT';
  roomId: string;
  objectId: string;
  patch: Partial<SceneObject>;
}

export type SceneOperation =
  | AddObjectOperation
  | RemoveObjectOperation
  | MoveObjectOperation
  | ResizeObjectOperation
  | RotateObjectOperation
  | UpdateObjectOperation;

export const isFiniteNumber = (value: number): boolean => typeof value === 'number' && Number.isFinite(value);

export const isFiniteVector3 = (value: Vector3): boolean =>
  isFiniteNumber(value.x) && isFiniteNumber(value.y) && isFiniteNumber(value.z);

export const isValidDimensions = (dimensions: Dimensions): boolean =>
  isFiniteNumber(dimensions.width) &&
  isFiniteNumber(dimensions.depth) &&
  isFiniteNumber(dimensions.height) &&
  dimensions.width > 0 &&
  dimensions.depth > 0 &&
  dimensions.height > 0;

export const validSceneObjectTypes: SceneObjectType[] = [
  'base_cabinet',
  'drawer_cabinet',
  'wall_cabinet',
  'loft',
  'countertop',
  'sink',
  'hob',
  'tall_unit',
  'wardrobe',
  'bed',
];

export const isValidSceneObject = (object: SceneObject): boolean => {
  if (!object || !object.id || !object.type) {
    return false;
  }

  return validSceneObjectTypes.includes(object.type) && isFiniteVector3(object.position) && isFiniteVector3(object.rotation) && isValidDimensions(object.dimensions);
};

export const createEmptyScene = (): Scene => ({
  version: 1,
  units: 'mm',
  rooms: [],
  metadata: {
    name: 'New Scene',
  },
});

export const createRoom = (room: Partial<Room> & Pick<Room, 'id' | 'dimensions'>): Room => ({
  id: room.id,
  name: room.name ?? room.id,
  dimensions: room.dimensions,
  walls: room.walls ?? [],
  doors: room.doors ?? [],
  windows: room.windows ?? [],
  objects: room.objects ?? [],
  metadata: room.metadata ?? {},
});

export const createWall = (wall: Partial<Wall> & Pick<Wall, 'id' | 'start' | 'end'>): Wall => ({
  id: wall.id,
  start: wall.start,
  end: wall.end,
  thickness: wall.thickness ?? 100,
  height: wall.height ?? 2800,
  type: wall.type,
  material: wall.material,
});

export const createDoor = (door: Partial<Door> & Pick<Door, 'id' | 'start' | 'end'>): Door => ({
  id: door.id,
  start: door.start,
  end: door.end,
  width: door.width ?? 900,
  height: door.height ?? 2100,
  swing: door.swing,
});

export const createWindow = (window: Partial<Window> & Pick<Window, 'id' | 'start' | 'end'>): Window => ({
  id: window.id,
  start: window.start,
  end: window.end,
  width: window.width ?? 1200,
  height: window.height ?? 1200,
  sillHeight: window.sillHeight,
});

export const createSceneObject = (
  object: Partial<SceneObject> & Pick<SceneObject, 'id' | 'type' | 'position' | 'rotation' | 'dimensions'>,
): SceneObject => ({
  id: object.id,
  type: object.type,
  name: object.name ?? object.id,
  position: object.position,
  rotation: object.rotation,
  dimensions: object.dimensions,
  material: object.material,
  anchor: object.anchor,
  metadata: object.metadata ?? {},
});

export const findRoomById = (scene: Scene, roomId: string): Room | undefined => scene.rooms.find((room) => room.id === roomId);

export const getObjectById = (scene: Scene, objectId: string): SceneObject | undefined =>
  scene.rooms.flatMap((room) => room.objects).find((object) => object.id === objectId);

export const cloneScene = (scene: Scene): Scene => deserializeScene(serializeScene(scene));

export const validateScene = (scene: Scene): SceneError[] => {
  const errors: SceneError[] = [];

  if (!scene || typeof scene !== 'object') {
    return [{ code: 'INVALID_SCENE', message: 'Scene is required.' }];
  }

  if (scene.units !== 'mm') {
    errors.push({ code: 'INVALID_UNITS', message: 'Scene units must be mm.' });
  }

  if (!Array.isArray(scene.rooms)) {
    errors.push({ code: 'INVALID_ROOMS', message: 'Scene rooms must be an array.' });
    return errors;
  }

  for (const room of scene.rooms) {
    if (!room || !room.id) {
      errors.push({ code: 'INVALID_ROOM', message: 'Each room requires an id.' });
      continue;
    }

    if (!isValidDimensions(room.dimensions)) {
      errors.push({ code: 'INVALID_ROOM_DIMENSIONS', message: `Room ${room.id} has invalid dimensions.`, path: `rooms.${room.id}.dimensions` });
    }

    const objectIds = new Set<string>();
    for (const object of room.objects) {
      if (!object || !object.id) {
        errors.push({ code: 'INVALID_OBJECT', message: `Room ${room.id} contains an object without an id.`, path: `rooms.${room.id}.objects` });
        continue;
      }

      if (objectIds.has(object.id)) {
        errors.push({ code: 'DUPLICATE_OBJECT_ID', message: `Duplicate object id: ${object.id}.`, path: `rooms.${room.id}.objects.${object.id}` });
      }
      objectIds.add(object.id);

      if (!isValidSceneObject(object)) {
        errors.push({ code: 'INVALID_OBJECT', message: `Object ${object.id} is invalid.`, path: `rooms.${room.id}.objects.${object.id}` });
      }
    }
  }

  return errors;
};

const updateRoomObject = (room: Room, objectId: string, updater: (object: SceneObject) => SceneObject): Room => ({
  ...room,
  objects: room.objects.map((object) => (object.id === objectId ? updater(object) : object)),
});

export interface SceneEngine {
  apply(scene: Scene, operation: SceneOperation): Scene | OperationResult;
  applyMany(scene: Scene, operations: SceneOperation[]): Scene | OperationResult;
}

export class SceneEngine {
  validateOperation(scene: Scene, operation: SceneOperation): SceneError[] {
    if (!operation || typeof operation !== 'object') {
      return [{ code: 'INVALID_OPERATION', message: 'Operation is required.' }];
    }

    const errors: SceneError[] = [];
    const room = findRoomById(scene, operation.roomId);

    if (!room && 'roomId' in operation) {
      errors.push({ code: 'INVALID_ROOM_REFERENCE', message: `Room ${operation.roomId} does not exist.`, path: 'roomId' });
    }

    switch (operation.op) {
      case 'ADD_OBJECT': {
        if (!isValidSceneObject(operation.object)) {
          errors.push({ code: 'INVALID_OBJECT', message: 'Object payload is invalid.', path: 'object' });
        }
        if (getObjectById(scene, operation.object.id)) {
          errors.push({ code: 'DUPLICATE_OBJECT_ID', message: `Object ${operation.object.id} already exists.`, path: 'object.id' });
        }
        break;
      }
      case 'REMOVE_OBJECT': {
        if (!room || !room.objects.some((object) => object.id === operation.objectId)) {
          errors.push({ code: 'OBJECT_NOT_FOUND', message: `Object ${operation.objectId} does not exist in room ${operation.roomId}.`, path: 'objectId' });
        }
        break;
      }
      case 'MOVE_OBJECT': {
        if (!getObjectById(scene, operation.objectId)) {
          errors.push({ code: 'OBJECT_NOT_FOUND', message: `Object ${operation.objectId} does not exist.`, path: 'objectId' });
        }
        if (!isFiniteVector3(operation.position)) {
          errors.push({ code: 'INVALID_POSITION', message: 'Move operation position is invalid.', path: 'position' });
        }
        break;
      }
      case 'RESIZE_OBJECT': {
        if (!getObjectById(scene, operation.objectId)) {
          errors.push({ code: 'OBJECT_NOT_FOUND', message: `Object ${operation.objectId} does not exist.`, path: 'objectId' });
        }
        if (!isValidDimensions(operation.dimensions)) {
          errors.push({ code: 'INVALID_DIMENSIONS', message: 'Resize operation dimensions are invalid.', path: 'dimensions' });
        }
        break;
      }
      case 'ROTATE_OBJECT': {
        if (!getObjectById(scene, operation.objectId)) {
          errors.push({ code: 'OBJECT_NOT_FOUND', message: `Object ${operation.objectId} does not exist.`, path: 'objectId' });
        }
        if (!isFiniteVector3(operation.rotation)) {
          errors.push({ code: 'INVALID_ROTATION', message: 'Rotate operation rotation is invalid.', path: 'rotation' });
        }
        break;
      }
      case 'UPDATE_OBJECT': {
        if (!getObjectById(scene, operation.objectId)) {
          errors.push({ code: 'OBJECT_NOT_FOUND', message: `Object ${operation.objectId} does not exist.`, path: 'objectId' });
        }
        if (operation.patch && 'dimensions' in operation.patch && operation.patch.dimensions && !isValidDimensions(operation.patch.dimensions)) {
          errors.push({ code: 'INVALID_DIMENSIONS', message: 'Updated dimensions are invalid.', path: 'patch.dimensions' });
        }
        if (operation.patch && 'position' in operation.patch && operation.patch.position && !isFiniteVector3(operation.patch.position)) {
          errors.push({ code: 'INVALID_POSITION', message: 'Updated position is invalid.', path: 'patch.position' });
        }
        if (operation.patch && 'rotation' in operation.patch && operation.patch.rotation && !isFiniteVector3(operation.patch.rotation)) {
          errors.push({ code: 'INVALID_ROTATION', message: 'Updated rotation is invalid.', path: 'patch.rotation' });
        }
        break;
      }
      default: {
        errors.push({ code: 'UNSUPPORTED_OPERATION', message: 'Unsupported operation type.', path: 'op' });
      }
    }

    return errors;
  }

  apply(scene: Scene, operation: SceneOperation): Scene | OperationResult {
    const validationErrors = this.validateOperation(scene, operation);

    if (validationErrors.length > 0) {
      return {
        success: false,
        errors: validationErrors,
      };
    }

    const nextScene = this.applyInternal(scene, operation);
    const sceneErrors = validateScene(nextScene);

    if (sceneErrors.length > 0) {
      return {
        success: false,
        errors: sceneErrors,
      };
    }

    return nextScene;
  }

  applyInternal(scene: Scene, operation: SceneOperation): Scene {
    switch (operation.op) {
      case 'ADD_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? { ...room, objects: [...room.objects, { ...operation.object }] }
              : room,
          ),
        };
      }
      case 'REMOVE_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? { ...room, objects: room.objects.filter((object) => object.id !== operation.objectId) }
              : room,
          ),
        };
      }
      case 'MOVE_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? updateRoomObject(room, operation.objectId, (object) => ({ ...object, position: { ...operation.position } }))
              : room,
          ),
        };
      }
      case 'RESIZE_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? updateRoomObject(room, operation.objectId, (object) => ({ ...object, dimensions: { ...operation.dimensions } }))
              : room,
          ),
        };
      }
      case 'ROTATE_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? updateRoomObject(room, operation.objectId, (object) => ({ ...object, rotation: { ...operation.rotation } }))
              : room,
          ),
        };
      }
      case 'UPDATE_OBJECT': {
        return {
          ...scene,
          rooms: scene.rooms.map((room) =>
            room.id === operation.roomId
              ? updateRoomObject(room, operation.objectId, (object) => ({
                  ...object,
                  ...operation.patch,
                  material: operation.patch.material ?? object.material,
                  anchor: operation.patch.anchor ?? object.anchor,
                  metadata: operation.patch.metadata ?? object.metadata,
                  position: operation.patch.position ?? object.position,
                  rotation: operation.patch.rotation ?? object.rotation,
                  dimensions: operation.patch.dimensions ?? object.dimensions,
                }))
              : room,
          ),
        };
      }
      default:
        return scene;
    }
  }

  applyMany(scene: Scene, operations: SceneOperation[]): Scene | OperationResult {
    let current = cloneScene(scene);

    for (const operation of operations) {
      const next = this.apply(current, operation);
      if (next && typeof next === 'object' && 'success' in next && !next.success) {
        return next;
      }
      current = next as Scene;
    }

    return current;
  }
}

export const applyOperation = (scene: Scene, operation: SceneOperation): Scene => {
  const engine = new SceneEngine();
  const result = engine.apply(scene, operation);

  if (result && typeof result === 'object' && 'success' in result && !result.success) {
    const details = result.errors.map((error) => error.message).join('; ');
    throw new Error(`Invalid operation: ${details}`);
  }

  return result as Scene;
};

export const serializeScene = (scene: Scene): string => JSON.stringify(sortObjectKeys(scene), null, 2);

export const deserializeScene = (data: string): Scene => {
  const parsed = JSON.parse(data) as Scene;
  const errors = validateScene(parsed);

  if (errors.length > 0) {
    throw new Error(errors.map((error) => error.message).join('; '));
  }

  return parsed;
};

const sortObjectKeys = <T>(value: T): T => {
  if (Array.isArray(value)) {
    return value.map((item) => sortObjectKeys(item)) as T;
  }

  if (value && typeof value === 'object') {
    const sortedEntries = Object.keys(value as Record<string, unknown>)
      .sort()
      .reduce<Record<string, unknown>>((accumulator, key) => {
        accumulator[key] = sortObjectKeys((value as Record<string, unknown>)[key]);
        return accumulator;
      }, {});

    return sortedEntries as T;
  }

  return value;
};

export class SceneHistory {
  private history: Scene[];
  private index: number;

  constructor(initialScene: Scene) {
    this.history = [cloneScene(initialScene)];
    this.index = 0;
  }

  current(): Scene {
    return cloneScene(this.history[this.index]);
  }

  push(scene: Scene): Scene {
    this.history = this.history.slice(0, this.index + 1);
    this.history.push(cloneScene(scene));
    this.index = this.history.length - 1;
    return this.current();
  }

  undo(): Scene {
    if (this.index === 0) {
      return this.current();
    }

    this.index -= 1;
    return this.current();
  }

  redo(): Scene {
    if (this.index >= this.history.length - 1) {
      return this.current();
    }

    this.index += 1;
    return this.current();
  }
}

export const kitchenBasic: Scene = {
  version: 1,
  units: 'mm',
  metadata: {
    name: 'Kitchen Basic',
    description: 'Small reference kitchen scene.',
  },
  rooms: [
    {
      id: 'room_1',
      name: 'Kitchen',
      dimensions: { width: 4000, depth: 3000, height: 2800 },
      walls: [
        createWall({ id: 'wall_north', start: { x: 0, y: 0, z: 0 }, end: { x: 4000, y: 0, z: 0 }, thickness: 120, height: 2800 }),
        createWall({ id: 'wall_east', start: { x: 4000, y: 0, z: 0 }, end: { x: 4000, y: 0, z: 3000 }, thickness: 120, height: 2800 }),
        createWall({ id: 'wall_south', start: { x: 4000, y: 0, z: 3000 }, end: { x: 0, y: 0, z: 3000 }, thickness: 120, height: 2800 }),
        createWall({ id: 'wall_west', start: { x: 0, y: 0, z: 3000 }, end: { x: 0, y: 0, z: 0 }, thickness: 120, height: 2800 }),
      ],
      doors: [
        createDoor({ id: 'door_01', start: { x: 3000, y: 0, z: 0 }, end: { x: 3600, y: 0, z: 0 }, width: 600, height: 2100 }),
      ],
      windows: [
        createWindow({ id: 'window_01', start: { x: 500, y: 0, z: 3000 }, end: { x: 1800, y: 0, z: 3000 }, width: 1300, height: 1200, sillHeight: 900 }),
      ],
      objects: [
        createSceneObject({
          id: 'sink_01',
          type: 'sink',
          position: { x: 2500, y: 0, z: 500 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 500, depth: 500, height: 200 },
          material: { name: 'stainless_steel', color: '#d1d5db' },
          anchor: { type: 'wall', targetId: 'wall_north', position: 'left' },
        }),
        createSceneObject({
          id: 'cabinet_01',
          type: 'drawer_cabinet',
          name: 'Drawer cabinet',
          position: { x: 2500, y: 0, z: 1000 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 600, depth: 600, height: 900 },
          material: { name: 'oak', color: '#b8925c' },
          anchor: { type: 'object', targetId: 'sink_01', position: 'right' },
        }),
        createSceneObject({
          id: 'cabinet_02',
          type: 'base_cabinet',
          name: 'Base cabinet',
          position: { x: 3200, y: 0, z: 1000 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 600, depth: 600, height: 900 },
          material: { name: 'oak', color: '#b8925c' },
          anchor: { type: 'object', targetId: 'cabinet_01', position: 'right' },
        }),
        createSceneObject({
          id: 'countertop_01',
          type: 'countertop',
          name: 'Countertop',
          position: { x: 2500, y: 900, z: 1000 },
          rotation: { x: 0, y: 0, z: 0 },
          dimensions: { width: 1500, depth: 600, height: 60 },
          material: { name: 'stone', color: '#d9d9d9' },
          anchor: { type: 'object', targetId: 'cabinet_01', position: 'above' },
        }),
      ],
      metadata: { style: 'urban' },
    },
  ],
};

export { kitchenBasicFixture } from './fixtures/kitchen-basic';
export * from './floorplan';
