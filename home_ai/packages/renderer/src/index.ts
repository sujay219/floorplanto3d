/// <reference path="./assets.d.ts" />
import * as BABYLON from 'babylonjs';
import 'babylonjs-loaders';
import type { Scene, SceneObject, SceneObjectType, Wall, Opening } from '@home-ai/scene-schema';
import { generateWallSegments } from '@home-ai/geometry';
import { groupContiguousWallSegments } from './wallRuns';
import type { WallRun } from './wallRuns';
import exteriorWallTexture from '../../textures/wall3.jpg';
import interiorWallTexture from '../../textures/wall_white.png';
import bedModelUrl from '../../../bed.glb';

export const SHOW_AXIS = true;

const CORNER_CAMERA_HEIGHT_FACTOR = 0.55;
const CORNER_CAMERA_INSET = 180;
const CORNER_LOOK_SENSITIVITY = 0.005;
const CORNER_PITCH_LIMIT = (Math.PI / 2) * 0.95;

const MODEL_UP_TO_SCENE_UP = Math.PI / 2;
const MODEL_URLS: Partial<Record<SceneObjectType, string>> = {
  bed: bedModelUrl,
};

export interface SceneRenderer {
  initialize(canvas: HTMLCanvasElement): Promise<void>;
  render(scene: Scene): void;
  dispose(): void;
}

export class BabylonRenderer implements SceneRenderer {
  private engine: BABYLON.Engine | null = null;
  private scene: BABYLON.Scene | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private camera: BABYLON.ArcRotateCamera | null = null;
  private freeCamera: BABYLON.FreeCamera | null = null;
  private activeCameraKind: 'arc' | 'free' = 'arc';
  private cornerYaw = 0;
  private cornerPitch = 0;
  private cornerLookActive = false;
  private cornerLastPointerX = 0;
  private cornerLastPointerY = 0;
  private currentRoomCenter: BABYLON.Vector3 | null = null;
  private cameraInitialized = false;
  private zoomChangeListeners = new Set<(radius: number) => void>();
  private lastReportedRadius: number | null = null;
  private readonly defaultCameraRadius = 2400;
  private readonly gridAlpha = 0.5;
  private readonly gridSpacing = 150;
  private wallThickness = 80.0;
  private wallById = new Map<string, Wall>();
  private renderGeneration = 0;
  private modelContainers = new Map<string, Promise<BABYLON.AssetContainer>>();
  private readonly eagleAlpha = -Math.PI / 4;
  private readonly eagleBeta = Math.PI / 3;
  private roomHeight = 2800;
  private activeCameraMode: 'eagle' | 'corner' = 'eagle';
  private cameraViewListeners = new Set<(mode: 'eagle' | 'corner') => void>();

  getRadius(): number {
    return this.camera?.radius ?? this.defaultCameraRadius;
  }

  setZoom(radius: number): number {
    if (!this.camera) {
      return this.defaultCameraRadius;
    }

    const clampedRadius = Math.min(8000, Math.max(400, radius));
    this.camera.radius = clampedRadius;
    this.reportZoomChange(clampedRadius);
    return clampedRadius;
  }

  adjustZoom(deltaY: number): number {
    if (!this.camera) {
      return this.defaultCameraRadius;
    }

    return this.setZoom(this.camera.radius - deltaY * 0.2);
  }

  setWallThickness(thickness: number): number {
    const nextThickness = Math.min(400, Math.max(20, thickness));
    this.wallThickness = nextThickness;
    return nextThickness;
  }

  onZoomChange(listener: (radius: number) => void): () => void {
    this.zoomChangeListeners.add(listener);
    if (this.camera) {
      listener(this.getRadius());
    }

    return () => {
      this.zoomChangeListeners.delete(listener);
    };
  }

  private reportZoomChange(radius: number): void {
    if (this.lastReportedRadius === radius) {
      return;
    }

    this.lastReportedRadius = radius;
    this.zoomChangeListeners.forEach((listener) => listener(radius));
  }

  getActiveCameraMode(): 'eagle' | 'corner' {
    return this.activeCameraMode;
  }

  onCameraViewChange(listener: (mode: 'eagle' | 'corner') => void): () => void {
    this.cameraViewListeners.add(listener);
    return () => {
      this.cameraViewListeners.delete(listener);
    };
  }

  resetToEagleView(): void {
    if (!this.camera) {
      return;
    }

    const center = this.currentRoomCenter ?? BABYLON.Vector3.Zero();
    this.applyEagleView(new BABYLON.Vector3(center.x, center.y, 0));
  }

  focusCornerCamera(corner: { x: number; y: number }): void {
    const freeCamera = this.freeCamera;
    const center = this.currentRoomCenter;

    if (!freeCamera || !center) {
      return;
    }

    const eyeHeight = this.roomHeight * CORNER_CAMERA_HEIGHT_FACTOR;
    freeCamera.position.copyFrom(new BABYLON.Vector3(corner.x, corner.y, eyeHeight));

    const dx = center.x - corner.x;
    const dy = center.y - corner.y;
    this.cornerYaw = Math.atan2(dy, dx);
    this.cornerPitch = 0;
    this.cornerLookActive = false;

    this.applyCornerLook();

    this.setActiveCamera('free');
    this.setActiveCameraMode('corner');
  }

  private applyCornerLook(): void {
    const freeCamera = this.freeCamera;
    if (!freeCamera) {
      return;
    }

    const cosPitch = Math.cos(this.cornerPitch);
    const forward = new BABYLON.Vector3(
      cosPitch * Math.cos(this.cornerYaw),
      cosPitch * Math.sin(this.cornerYaw),
      Math.sin(this.cornerPitch),
    );

    freeCamera.setTarget(freeCamera.position.add(forward));
  }

  private handleCornerPointer(pointerInfo: BABYLON.PointerInfo): void {
    if (this.activeCameraKind !== 'free') {
      return;
    }

    const event = pointerInfo.event as PointerEvent;

    if (pointerInfo.type === BABYLON.PointerEventTypes.POINTERDOWN) {
      this.cornerLookActive = true;
      this.cornerLastPointerX = event.clientX;
      this.cornerLastPointerY = event.clientY;
      return;
    }

    if (pointerInfo.type === BABYLON.PointerEventTypes.POINTERUP) {
      this.cornerLookActive = false;
      return;
    }

    if (pointerInfo.type === BABYLON.PointerEventTypes.POINTERMOVE && this.cornerLookActive) {
      const deltaX = event.clientX - this.cornerLastPointerX;
      const deltaY = event.clientY - this.cornerLastPointerY;
      this.cornerLastPointerX = event.clientX;
      this.cornerLastPointerY = event.clientY;

      this.cornerYaw += deltaX * CORNER_LOOK_SENSITIVITY;
      this.cornerPitch -= deltaY * CORNER_LOOK_SENSITIVITY;
      this.cornerPitch = Math.max(-CORNER_PITCH_LIMIT, Math.min(CORNER_PITCH_LIMIT, this.cornerPitch));

      this.applyCornerLook();
    }
  }

  private applyEagleView(target: BABYLON.Vector3): void {
    if (!this.camera) {
      return;
    }

    this.camera.setTarget(target);
    this.camera.alpha = this.eagleAlpha;
    this.camera.beta = this.eagleBeta;
    this.camera.radius = this.defaultCameraRadius;
    this.setActiveCamera('arc');
    this.setActiveCameraMode('eagle');
  }

  private setActiveCamera(mode: 'arc' | 'free'): void {
    const sceneInstance = this.scene;
    const arc = this.camera;
    const free = this.freeCamera;

    if (!sceneInstance || !arc || !free) {
      return;
    }

    sceneInstance.activeCamera = mode === 'free' ? free : arc;

    if (this.activeCameraKind === mode) {
      return;
    }

    if (mode === 'free') {
      arc.detachControl();
    } else {
      arc.attachControl(this.canvas ?? undefined, true);
    }

    this.activeCameraKind = mode;
  }

  private handleCameraChanged(): void {
    if (this.scene && this.currentRoomCenter) {
      this.syncWallTransparency(this.scene, this.currentRoomCenter);
    }

    if (this.camera) {
      this.reportZoomChange(this.camera.radius);
    }
  }

  private setActiveCameraMode(mode: 'eagle' | 'corner'): void {
    if (this.activeCameraMode === mode) {
      return;
    }

    this.activeCameraMode = mode;
    this.cameraViewListeners.forEach((listener) => listener(mode));
  }

  private ensureCamera(target: BABYLON.Vector3, sceneInstance: BABYLON.Scene): void {
    if (!this.camera) {
      this.camera = new BABYLON.ArcRotateCamera('camera', this.eagleAlpha, this.eagleBeta, 6000, target, sceneInstance);
      this.camera.upVector = new BABYLON.Vector3(0, 0, 1);
      this.camera.attachControl(this.canvas ?? undefined, true);
      this.camera.lowerRadiusLimit = 400;
      this.camera.upperRadiusLimit = 20000;
      sceneInstance.activeCamera = this.camera;
    }

    this.cameraInitialized = true;
    this.applyEagleView(target);
  }

  async initialize(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;

    if (this.engine && this.scene) {
      return;
    }

    const canUseRealEngine = typeof document !== 'undefined' && !!canvas && !!canvas.getContext;
    this.engine = canUseRealEngine ? new BABYLON.Engine(canvas, true) : new BABYLON.NullEngine();
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.96, 0.97, 0.99, 1);

    this.camera = new BABYLON.ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 6000, BABYLON.Vector3.Zero(), this.scene);
    this.camera.upVector = new BABYLON.Vector3(0, 0, 1);
    this.camera.attachControl(canvas, true);
    this.camera.setTarget(BABYLON.Vector3.Zero());
    this.camera.lowerRadiusLimit = 400;
    this.camera.upperRadiusLimit = 20000;
    this.scene.activeCamera = this.camera;

    this.freeCamera = new BABYLON.FreeCamera('cornerCamera', BABYLON.Vector3.Zero(), this.scene);
    this.freeCamera.upVector = new BABYLON.Vector3(0, 0, 1);
    this.freeCamera.minZ = 0.1;
    this.freeCamera.maxZ = 50000;
    this.freeCamera.inputs.clear();

    this.camera.onViewMatrixChangedObservable.add(() => this.handleCameraChanged());
    this.freeCamera.onViewMatrixChangedObservable.add(() => this.handleCameraChanged());

    this.scene.onPointerObservable.add((pointerInfo) => this.handleCornerPointer(pointerInfo));

    this.reportZoomChange(this.camera.radius);

    const hemisphericLight = new BABYLON.HemisphericLight('ambientLight', new BABYLON.Vector3(0, 1, 0), this.scene);
    hemisphericLight.intensity = 0.6;
    hemisphericLight.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);

    const directionalLight = new BABYLON.DirectionalLight('directionalLight', new BABYLON.Vector3(-0.5, -0.7, -0.8), this.scene);
    directionalLight.intensity = 0.5;
    directionalLight.diffuse = new BABYLON.Color3(0.95, 0.95, 0.98);

    const ssao = new BABYLON.SSAO2RenderingPipeline('ssao', this.scene, { ssaoRatio: 1, blurRatio: 1 }, [this.camera, this.freeCamera]);
    ssao.radius = 4;
    ssao.totalStrength = 0.6;
    ssao.base = 0.1;
    ssao.expensiveBlur = true;
    ssao.samples = 16;
    ssao.maxZ = 15000;

    const ground = BABYLON.MeshBuilder.CreateGround('ground', { width: 12000, height: 12000 }, this.scene);
    ground.position.x = 6000;
    ground.position.y = 6000;
    ground.position.z = -10;
    ground.material = new BABYLON.StandardMaterial('groundMaterial', this.scene);
    (ground.material as BABYLON.StandardMaterial).diffuseColor = new BABYLON.Color3(0.88, 0.9, 0.92);

    this.createBaseAxes(this.scene);

    this.engine.runRenderLoop(() => {
      this.scene?.render();
    });
  }

  private createBaseAxes(sceneInstance: BABYLON.Scene): void {
    if (!SHOW_AXIS) {
      return;
    }

    const axisLength = 900;
    const axisThickness = 8;
    const axisLift = 6;

    const origin = new BABYLON.Vector3(0, axisLift, 0);
    const xAxis = BABYLON.MeshBuilder.CreateBox('baseAxis_x', { width: axisLength, height: axisThickness, depth: axisThickness }, sceneInstance);
    xAxis.position = origin.add(new BABYLON.Vector3(axisLength / 2, 0, 0));
    const xMaterial = new BABYLON.StandardMaterial('baseAxisMaterial_x', sceneInstance);
    xMaterial.diffuseColor = new BABYLON.Color3(0.9, 0.25, 0.25);
    xAxis.material = xMaterial;

    const yAxis = BABYLON.MeshBuilder.CreateBox('baseAxis_y', { width: axisThickness, height: axisLength, depth: axisThickness }, sceneInstance);
    yAxis.position = origin.add(new BABYLON.Vector3(0, axisLength / 2, 0));
    const yMaterial = new BABYLON.StandardMaterial('baseAxisMaterial_y', sceneInstance);
    yMaterial.diffuseColor = new BABYLON.Color3(0.25, 0.45, 0.95);
    yAxis.material = yMaterial;

    const zAxis = BABYLON.MeshBuilder.CreateBox('baseAxis_z', { width: axisThickness, height: axisThickness, depth: axisLength }, sceneInstance);
    zAxis.position = origin.add(new BABYLON.Vector3(0, 0, axisLength / 2));
    const zMaterial = new BABYLON.StandardMaterial('baseAxisMaterial_z', sceneInstance);
    zMaterial.diffuseColor = new BABYLON.Color3(0.25, 0.75, 0.35);
    zAxis.material = zMaterial;

    if (typeof OffscreenCanvas !== 'undefined') {
      const xLabel = BABYLON.MeshBuilder.CreatePlane('baseAxisLabel_x', { width: 120, height: 60 }, sceneInstance);
      xLabel.position = origin.add(new BABYLON.Vector3(axisLength + 110, 30, 0));
      xLabel.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
      const xTexture = new BABYLON.DynamicTexture('baseAxisLabelTexture_x', { width: 256, height: 128 }, sceneInstance, false);
      xTexture.hasAlpha = true;
      xTexture.drawText('X', null, 92, 'bold 96px sans-serif', '#d82f2f', 'transparent', true);
      const xLabelMaterial = new BABYLON.StandardMaterial('baseAxisLabelMaterial_x', sceneInstance);
      xLabelMaterial.diffuseTexture = xTexture;
      xLabelMaterial.emissiveColor = BABYLON.Color3.White();
      xLabelMaterial.opacityTexture = xTexture;
      xLabel.material = xLabelMaterial;

      const yLabel = BABYLON.MeshBuilder.CreatePlane('baseAxisLabel_y', { width: 120, height: 60 }, sceneInstance);
      yLabel.position = origin.add(new BABYLON.Vector3(0, axisLength + 110, 0));
      yLabel.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
      const yTexture = new BABYLON.DynamicTexture('baseAxisLabelTexture_y', { width: 256, height: 128 }, sceneInstance, false);
      yTexture.hasAlpha = true;
      yTexture.drawText('Y', null, 92, 'bold 96px sans-serif', '#2f62d8', 'transparent', true);
      const yLabelMaterial = new BABYLON.StandardMaterial('baseAxisLabelMaterial_y', sceneInstance);
      yLabelMaterial.diffuseTexture = yTexture;
      yLabelMaterial.emissiveColor = BABYLON.Color3.White();
      yLabelMaterial.opacityTexture = yTexture;
      yLabel.material = yLabelMaterial;

      const zLabel = BABYLON.MeshBuilder.CreatePlane('baseAxisLabel_z', { width: 120, height: 60 }, sceneInstance);
      zLabel.position = origin.add(new BABYLON.Vector3(0, 30, axisLength + 110));
      zLabel.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
      const zTexture = new BABYLON.DynamicTexture('baseAxisLabelTexture_z', { width: 256, height: 128 }, sceneInstance, false);
      zTexture.hasAlpha = true;
      zTexture.drawText('Z', null, 92, 'bold 96px sans-serif', '#2f9e44', 'transparent', true);
      const zLabelMaterial = new BABYLON.StandardMaterial('baseAxisLabelMaterial_z', sceneInstance);
      zLabelMaterial.diffuseTexture = zTexture;
      zLabelMaterial.emissiveColor = BABYLON.Color3.White();
      zLabelMaterial.opacityTexture = zTexture;
      zLabel.material = zLabelMaterial;
    }
  }

  private createSceneLabel(
    sceneInstance: BABYLON.Scene,
    name: string,
    text: string,
    position: BABYLON.Vector3,
    color: string,
  ): BABYLON.Mesh | null {
    if (typeof OffscreenCanvas === 'undefined') {
      return null;
    }

    const label = BABYLON.MeshBuilder.CreatePlane(name, { width: 220, height: 84 }, sceneInstance);
    label.position = position;
    label.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
    label.isPickable = false;

    const texture = new BABYLON.DynamicTexture(`${name}Texture`, { width: 512, height: 192 }, sceneInstance, false);
    texture.hasAlpha = true;
    texture.drawText(text, null, 132, 'bold 84px sans-serif', color, 'transparent', true);

    const material = new BABYLON.StandardMaterial(`${name}Material`, sceneInstance);
    material.diffuseTexture = texture;
    material.emissiveColor = BABYLON.Color3.White();
    material.opacityTexture = texture;
    material.disableLighting = true;
    label.material = material;

    return label;
  }

  private createRoomGrid(target: Scene['rooms'][0], sceneInstance: BABYLON.Scene): void {
    const z = -9.5;
    const gridLines: BABYLON.Vector3[][] = [];

    for (let x = 0; x <= target.dimensions.width; x += this.gridSpacing) {
      gridLines.push([
        new BABYLON.Vector3(x, 0, z),
        new BABYLON.Vector3(x, target.dimensions.depth, z),
      ]);
    }

    for (let y = 0; y <= target.dimensions.depth; y += this.gridSpacing) {
      gridLines.push([
        new BABYLON.Vector3(0, y, z),
        new BABYLON.Vector3(target.dimensions.width, y, z),
      ]);
    }

    const grid = BABYLON.MeshBuilder.CreateLineSystem('sceneGrid_xy', { lines: gridLines }, sceneInstance);
    grid.color = new BABYLON.Color3(0.55, 0.65, 0.72);
    grid.visibility = this.gridAlpha;

    const originDot = BABYLON.MeshBuilder.CreateSphere('sceneOrigin_dot', { diameter: 50 }, sceneInstance);
    originDot.position = new BABYLON.Vector3(0, 0, z + 45);
    const originMaterial = new BABYLON.StandardMaterial('sceneOriginMaterial_dot', sceneInstance);
    originMaterial.diffuseColor = new BABYLON.Color3(0.08, 0.8, 0.2);
    originMaterial.emissiveColor = new BABYLON.Color3(0.45, 0.34, 0.03);
    originDot.material = originMaterial;
  }

  render(scene: Scene): void {
    const sceneInstance = this.scene;
    const engineInstance = this.engine;

    if (!sceneInstance || !engineInstance) {
      return;
    }

    this.renderGeneration += 1;

    sceneInstance.transformNodes
      .filter((node) => node.name.startsWith('object_'))
      .forEach((node) => node.dispose(false, true));

    sceneInstance.meshes
      .filter((mesh) => {
        const isSceneMesh = ['wall_', 'door_', 'window_', 'object_', 'sceneGuide_', 'sceneGrid_', 'sceneOrigin_', 'sceneLabel_', 'cameraMarker_'].some((prefix) => mesh.name.startsWith(prefix));
        return isSceneMesh || mesh.name === 'roomBox';
      })
      .forEach((mesh) => mesh.dispose());

    sceneInstance.materials
      .filter((material) => material.name.startsWith('wallMaterial_') || material.name.startsWith('wallInteriorMaterial_') || material.name.startsWith('wallExteriorMaterial_') || material.name.startsWith('doorMaterial_') || material.name.startsWith('sceneLabelMaterial_') || material.name.startsWith('cameraMarkerMaterial_'))
      .forEach((material) => material.dispose());

    const target = scene.rooms[0];
    if (!target) {
      return;
    }

    // Compute openings and wall segments for debugging
    const openings: Opening[] = target.doors.map((d) => ({
      id: d.id,
      wallId: '', // to be resolved later
      position: 0,
      width: d.width,
      bottom: 0,
      height: d.height,
      type: 'door',
    } as Opening));

    // We'll implement helper functions to find host wall
    const dot = (a: BABYLON.Vector3, b: BABYLON.Vector3) => a.x * b.x + a.y * b.y + a.z * b.z;
    const length = (a: BABYLON.Vector3) => Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    const projectOntoSegment = (start: BABYLON.Vector3, end: BABYLON.Vector3, point: BABYLON.Vector3) => {
      const seg = end.subtract(start);
      const segLen2 = dot(seg, seg);
      if (segLen2 === 0) return { t: 0, proj: start };
      const t = dot(point.subtract(start), seg) / segLen2;
      const clamped = Math.max(0, Math.min(1, t));
      const proj = start.add(seg.scale(clamped));
      return { t: clamped, proj };
    };

    // map each door/window to a host wall and normalized position
    const resolvedOpenings: Opening[] = [];
    const findHostWall = (center: BABYLON.Vector3): { wall?: Wall; t?: number; proj?: BABYLON.Vector3 } => {
      let best: Wall | undefined;
      let bestDist = Infinity;
      let bestT: number | undefined;
      let bestProj: BABYLON.Vector3 | undefined;
      for (const wall of target.walls) {
        const start = new BABYLON.Vector3(wall.start.x, wall.start.y, wall.start.z);
        const end = new BABYLON.Vector3(wall.end.x, wall.end.y, wall.end.z);
        const { t, proj } = projectOntoSegment(start, end, center);
        const d = length(center.subtract(proj));
        if (d < bestDist && t >= -1e-9 && t <= 1 + 1e-9) {
          best = wall;
          bestDist = d;
          bestT = t;
          bestProj = proj;
        }
      }
      return { wall: best, t: bestT, proj: bestProj };
    };

    // doors
    for (const d of target.doors) {
      const center = new BABYLON.Vector3((d.start.x + d.end.x) / 2, (d.start.y + d.end.y) / 2, (d.start.z + d.end.z) / 2);
      const { wall, t, proj } = findHostWall(center);
      if (!wall) {
        console.debug('[wall-seg-debug] door', d.id, 'no host wall found, center', center);
        continue;
      }
      const start = new BABYLON.Vector3(wall.start.x, wall.start.y, wall.start.z);
      const end = new BABYLON.Vector3(wall.end.x, wall.end.y, wall.end.z);
      const seg = end.subtract(start);
      const wallLen = length(seg);
      const position = t ?? 0;
      resolvedOpenings.push({ id: d.id, wallId: wall.id, position, width: d.width, bottom: 0, height: d.height, type: 'door' });
      console.debug('[wall-seg-debug] door', d.id, 'host', wall.id, 't', t, 'proj', proj, 'position', position, 'wallLen', wallLen);
    }

    // windows
    for (const w of target.windows) {
      const center = new BABYLON.Vector3((w.start.x + w.end.x) / 2, (w.start.y + w.end.y) / 2, (w.start.z + w.end.z) / 2);
      const { wall, t, proj } = findHostWall(center);
      if (!wall) {
        console.debug('[wall-seg-debug] window', w.id, 'no host wall found, center', center);
        continue;
      }
      const start = new BABYLON.Vector3(wall.start.x, wall.start.y, wall.start.z);
      const end = new BABYLON.Vector3(wall.end.x, wall.end.y, wall.end.z);
      const seg = end.subtract(start);
      const wallLen = length(seg);
      const position = t ?? 0;
      const bottom = w.sillHeight ?? 0;
      resolvedOpenings.push({ id: w.id, wallId: wall.id, position, width: w.width, bottom, height: w.height, type: 'window' });
      console.debug('[wall-seg-debug] window', w.id, 'host', wall.id, 't', t, 'proj', proj, 'position', position, 'wallLen', wallLen, 'bottom', bottom);
    }

    const segments = generateWallSegments(target.walls, resolvedOpenings);
    console.debug('[wall-seg-debug] resolvedOpenings', resolvedOpenings);
    console.debug('[wall-seg-debug] segments', segments);

    const roomCenter = new BABYLON.Vector3(target.dimensions.width / 2, target.dimensions.depth / 2, 0);

    this.currentRoomCenter = roomCenter;

    this.ensureCamera(new BABYLON.Vector3(roomCenter.x, roomCenter.y, 0), sceneInstance);
    this.createRoomGrid(target, sceneInstance);
    this.createSceneGuides(target, sceneInstance);
    this.createCornerCameras(target, sceneInstance);

    target.walls.forEach((wall) => {
      const wallCenter = new BABYLON.Vector3((wall.start.x + wall.end.x) / 2, (wall.start.y + wall.end.y) / 2, wall.height + 120);
      const roomOffset = new BABYLON.Vector3(roomCenter.x - wallCenter.x, roomCenter.y - wallCenter.y, 0);
      const offsetLength = roomOffset.length();
      const labelPosition = offsetLength > 0 ? wallCenter.add(roomOffset.scale(60 / offsetLength)) : wallCenter;
      this.createSceneLabel(sceneInstance, `sceneLabel_wall_${wall.id}`, wall.id, labelPosition, '#1f2937');
    });

    resolvedOpenings.forEach((opening) => {
      const hostWall = target.walls.find((wall) => wall.id === opening.wallId);
      if (!hostWall) {
        return;
      }

      const wallStart = new BABYLON.Vector3(hostWall.start.x, hostWall.start.y, hostWall.start.z);
      const wallEnd = new BABYLON.Vector3(hostWall.end.x, hostWall.end.y, hostWall.end.z);
      const wallVector = wallEnd.subtract(wallStart);
      const wallLength = wallVector.length();
      if (wallLength === 0) {
        return;
      }

      const wallDirection = wallVector.scale(1 / wallLength);
      const openingCenter = wallStart.add(wallVector.scale(opening.position));
      const openingMidZ = opening.bottom + opening.height / 4;
      const openingLabelBase = new BABYLON.Vector3(openingCenter.x, openingCenter.y, openingMidZ);
      const roomOffset = new BABYLON.Vector3(roomCenter.x - openingLabelBase.x, roomCenter.y - openingLabelBase.y, 0);
      const offsetLength = roomOffset.length();
      const labelPosition = offsetLength > 0 ? openingLabelBase.add(roomOffset.scale(55 / offsetLength)) : openingLabelBase;
      const labelText = opening.type === 'door' ? opening.id : opening.id;

      this.createSceneLabel(
        sceneInstance,
        `sceneLabel_${opening.type}_${opening.id}`,
        labelText,
        labelPosition,
        opening.type === 'door' ? '#92400e' : '#065f46',
      );
    });

    // Render wall segments using the same vertex construction as full walls
    // so start/end coordinates remain exact after rotation.
    // Build a map of wall IDs to walls for UV coordinate calculation
    this.wallById.clear();
    target.walls.forEach((w) => this.wallById.set(w.id, w));

    const runs = groupContiguousWallSegments(segments, this.wallById);
    runs.forEach((run) => this.buildWallRunMesh(sceneInstance, run, roomCenter));

    target.objects.forEach((object) => this.buildObject(object));

    this.syncWallTransparency(sceneInstance, this.currentRoomCenter ?? roomCenter);

    engineInstance.resize();
  }

  private buildWallRunMesh(
    sceneInstance: BABYLON.Scene,
    run: WallRun,
    roomCenter: BABYLON.Vector3,
  ): void {
    const start = new BABYLON.Vector3(run.start.x, run.start.y, run.start.z);
    const end = new BABYLON.Vector3(run.end.x, run.end.y, run.end.z);
    const direction = end.subtract(start);
    const segmentLength = direction.length();
    if (segmentLength === 0) return;

    const normalizedDirection = direction.normalize();
    const sideways = new BABYLON.Vector3(-normalizedDirection.y, normalizedDirection.x, 0).scale(this.wallThickness / 2);
    const up = new BABYLON.Vector3(0, 0, run.top - run.bottom);

    const p0 = start.add(sideways).add(new BABYLON.Vector3(0, 0, run.bottom));
    const p1 = end.add(sideways).add(new BABYLON.Vector3(0, 0, run.bottom));
    const p2 = end.subtract(sideways).add(new BABYLON.Vector3(0, 0, run.bottom));
    const p3 = start.subtract(sideways).add(new BABYLON.Vector3(0, 0, run.bottom));
    const p4 = p0.add(up);
    const p5 = p1.add(up);
    const p6 = p2.add(up);
    const p7 = p3.add(up);

    const mesh = new BABYLON.Mesh(`wall_${run.id}`, sceneInstance);
    const sourcePoints = [p0, p1, p2, p3, p4, p5, p6, p7];

    // Calculate UVs from the run's position along the original wall so the
    // texture continues seamlessly across the merged run.
    const wall = this.wallById.get(run.wallId);
    let uStart = 0;
    let uEnd = 1;
    let vStart = 0;
    let vEnd = 1;
    let runStartDist = 0;
    let runEndDist = 0;
    let wallLength = 0;
    if (wall) {
      const wallStart = new BABYLON.Vector3(wall.start.x, wall.start.y, wall.start.z);
      const wallEnd = new BABYLON.Vector3(wall.end.x, wall.end.y, wall.end.z);
      const wallVec = wallEnd.subtract(wallStart);
      wallLength = wallVec.length();
      if (wallLength > 0) {
        const wallDir = wallVec.scale(1 / wallLength);
        runStartDist = BABYLON.Vector3.Dot(start.subtract(wallStart), wallDir);
        runEndDist = BABYLON.Vector3.Dot(end.subtract(wallStart), wallDir);
        uStart = runStartDist / wallLength;
        uEnd = runEndDist / wallLength;
        vStart = run.bottom / wall.height;
        vEnd = run.top / wall.height;
      }
    }

    const sourceUvs = [
      [uStart, vStart],
      [uEnd, vStart],
      [uEnd, vStart],
      [uStart, vStart],
      [uStart, vEnd],
      [uEnd, vEnd],
      [uEnd, vEnd],
      [uStart, vEnd],
    ];

    const sidewaysNormal = sideways.normalize();
    const wallMidpoint = start.add(end).scale(0.5);
    const toRoom = roomCenter.subtract(wallMidpoint);
    const plusSideIsInterior = BABYLON.Vector3.Dot(sidewaysNormal, toRoom) > 0;

    const interiorIndex = 0;
    const exteriorIndex = 1;
    const isInteriorWall = run.type === 'interior';
    // +sideways face (p0/p1) faces the room when plusSideIsInterior is true
    const plusFaceIndex = isInteriorWall || plusSideIsInterior ? interiorIndex : exteriorIndex;
    // -sideways face (p2/p3) is the opposite side
    const minusFaceIndex = isInteriorWall || !plusSideIsInterior ? interiorIndex : exteriorIndex;

    // Header/sill runs sit inside an opening span. Their start/end faces are
    // internal when they meet the adjacent full-height pieces, so omit those
    // caps to avoid coincident coplanar faces (z-fighting).
    const isPartialHeight = run.bottom > 1e-9 || run.top < (wall?.height ?? run.top) - 1e-9;
    const omitStart = !!wall && isPartialHeight && runStartDist > 1e-9;
    const omitEnd = !!wall && isPartialHeight && runEndDist < wallLength - 1e-9;

    const faces: { indices: number[]; materialIndex: number }[] = [
      { indices: [4, 5, 6, 4, 6, 7], materialIndex: interiorIndex },
      { indices: [0, 2, 1, 0, 3, 2], materialIndex: interiorIndex },
    ];
    if (!omitStart) {
      faces.push({ indices: [0, 4, 7, 0, 7, 3], materialIndex: interiorIndex });
    }
    if (!omitEnd) {
      faces.push({ indices: [1, 2, 6, 1, 6, 5], materialIndex: interiorIndex });
    }
    faces.push({ indices: [0, 1, 5, 0, 5, 4], materialIndex: plusFaceIndex });
    faces.push({ indices: [3, 7, 6, 3, 6, 2], materialIndex: minusFaceIndex });

    const flatPositions: number[] = [];
    const flatNormals: number[] = [];
    const flatUvs: number[] = [];
    const flatIndices: number[] = [];

    for (const face of faces) {
      for (let t = 0; t < 2; t++) {
        const i0 = face.indices[t * 3];
        const i1 = face.indices[t * 3 + 1];
        const i2 = face.indices[t * 3 + 2];
        const edge1 = sourcePoints[i1].subtract(sourcePoints[i0]);
        const edge2 = sourcePoints[i2].subtract(sourcePoints[i0]);
        const normal = BABYLON.Vector3.Cross(edge1, edge2).normalize();

        const baseIndex = flatPositions.length / 3;
        for (const vertexIndex of [i0, i1, i2]) {
          const point = sourcePoints[vertexIndex];
          const uv = sourceUvs[vertexIndex];
          flatPositions.push(point.x, point.y, point.z);
          flatNormals.push(normal.x, normal.y, normal.z);
          flatUvs.push(uv[0], uv[1]);
        }

        flatIndices.push(baseIndex, baseIndex + 1, baseIndex + 2);
      }
    }

    const vertexData = new BABYLON.VertexData();
    vertexData.positions = flatPositions;
    vertexData.indices = flatIndices;
    vertexData.normals = flatNormals;
    vertexData.uvs = flatUvs;
    vertexData.applyToMesh(mesh);

    mesh.metadata = {
      wallId: run.wallId,
      start: run.start,
      end: run.end,
      thickness: this.wallThickness,
      height: run.top,
      type: run.type,
    };

    const makeWallMaterial = (name: string, textureUrl: string) => {
      const mat = new BABYLON.StandardMaterial(name, sceneInstance);
      mat.diffuseTexture = new BABYLON.Texture(textureUrl, sceneInstance);
      mat.diffuseTexture.hasAlpha = false;
      mat.emissiveColor = BABYLON.Color3.White();
      mat.diffuseColor = BABYLON.Color3.White();
      mat.specularColor = BABYLON.Color3.Black();
      mat.disableLighting = false;
      mat.alpha = 1;
      return mat;
    };

    const interiorMaterial = makeWallMaterial(`wallInteriorMaterial_${run.id}`, interiorWallTexture);
    const exteriorMaterial = makeWallMaterial(`wallExteriorMaterial_${run.id}`, exteriorWallTexture);

    const multiMaterial = new BABYLON.MultiMaterial(`wallMaterial_${run.id}`, sceneInstance);
    multiMaterial.subMaterials = [interiorMaterial, exteriorMaterial];
    mesh.material = multiMaterial;

    mesh.releaseSubMeshes();
    const totalVertices = flatPositions.length / 3;
    let indexOffset = 0;
    for (const face of faces) {
      new BABYLON.SubMesh(face.materialIndex, 0, totalVertices, indexOffset, 6, mesh);
      indexOffset += 6;
    }

    mesh.metadata.exteriorMaterial = exteriorMaterial;
    mesh.metadata.interiorMaterial = interiorMaterial;
  }

  private createSceneGuides(target: Scene['rooms'][0], sceneInstance: BABYLON.Scene): void {
    const guideWidth = 18;
    const guideDepth = 18;
    const guideHeight = target.dimensions.height;
    const guideOffset = 24;
    const centerY = target.dimensions.depth / 2;

    const leftGuide = BABYLON.MeshBuilder.CreateBox('sceneGuide_left', { width: guideWidth, height: guideDepth, depth: guideHeight }, sceneInstance);
    leftGuide.position = new BABYLON.Vector3(guideOffset, centerY, guideHeight / 2);
    const leftMaterial = new BABYLON.StandardMaterial('sceneGuideMaterial_left', sceneInstance);
    leftMaterial.diffuseColor = new BABYLON.Color3(0.82, 0.28, 0.28);
    leftMaterial.emissiveColor = new BABYLON.Color3(0.12, 0.03, 0.03);
    leftGuide.material = leftMaterial;

    const rightGuide = BABYLON.MeshBuilder.CreateBox('sceneGuide_right', { width: guideWidth, height: guideDepth, depth: guideHeight }, sceneInstance);
    rightGuide.position = new BABYLON.Vector3(target.dimensions.width - guideOffset, centerY, guideHeight / 2);
    const rightMaterial = new BABYLON.StandardMaterial('sceneGuideMaterial_right', sceneInstance);
    rightMaterial.diffuseColor = new BABYLON.Color3(0.28, 0.58, 0.88);
    rightMaterial.emissiveColor = new BABYLON.Color3(0.03, 0.06, 0.12);
    rightGuide.material = rightMaterial;
  }

  private createCornerCameras(target: Scene['rooms'][0], sceneInstance: BABYLON.Scene): void {
    const roomWidth = target.dimensions.width;
    const roomDepth = target.dimensions.depth;
    const center = new BABYLON.Vector3(roomWidth / 2, roomDepth / 2, 0);
    const eyeHeight = target.dimensions.height * CORNER_CAMERA_HEIGHT_FACTOR;

    this.roomHeight = target.dimensions.height;

    const rawCorners = [
      { x: 0, y: 0 },
      { x: roomWidth, y: 0 },
      { x: roomWidth, y: roomDepth },
      { x: 0, y: roomDepth },
    ];

    const corners = rawCorners.map((corner, index) => {
      const dx = center.x - corner.x;
      const dy = center.y - corner.y;
      const length = Math.hypot(dx, dy) || 1;
      return {
        id: `corner_${index}`,
        x: corner.x + (dx / length) * CORNER_CAMERA_INSET,
        y: corner.y + (dy / length) * CORNER_CAMERA_INSET,
      };
    });

    corners.forEach((corner) => {
      const body = BABYLON.MeshBuilder.CreateBox(
        `cameraMarker_${corner.id}`,
        { width: 100, height: 80, depth: 60 },
        sceneInstance,
      );
      body.position = new BABYLON.Vector3(corner.x, corner.y, eyeHeight);
      body.rotation.z = Math.atan2(center.y - corner.y, center.x - corner.x);

      const bodyMaterial = new BABYLON.StandardMaterial(`cameraMarkerMaterial_${corner.id}`, sceneInstance);
      bodyMaterial.diffuseColor = new BABYLON.Color3(0.95, 0.55, 0.1);
      bodyMaterial.emissiveColor = new BABYLON.Color3(0.4, 0.2, 0.02);
      bodyMaterial.specularColor = BABYLON.Color3.Black();
      body.material = bodyMaterial;

      const lens = BABYLON.MeshBuilder.CreateSphere(
        `cameraLens_${corner.id}`,
        { diameter: 36 },
        sceneInstance,
      );
      lens.parent = body;
      lens.position = new BABYLON.Vector3(55, 0, 0);
      lens.isPickable = false;
      const lensMaterial = new BABYLON.StandardMaterial(`cameraMarkerMaterial_${corner.id}_lens`, sceneInstance);
      lensMaterial.diffuseColor = new BABYLON.Color3(0.08, 0.1, 0.13);
      lensMaterial.emissiveColor = new BABYLON.Color3(0.2, 0.5, 0.9);
      lensMaterial.specularColor = BABYLON.Color3.Black();
      lens.material = lensMaterial;

      body.actionManager = new BABYLON.ActionManager(sceneInstance);
      body.actionManager.registerAction(
        new BABYLON.ExecuteCodeAction(BABYLON.ActionManager.OnPickTrigger, () => {
          this.focusCornerCamera(corner);
        }),
      );
    });
  }

  private syncWallTransparency(
    sceneInstance: BABYLON.Scene,
    roomCenter: BABYLON.Vector3
  ): void {
    const camera = sceneInstance.activeCamera;

    if (!camera) {
      return;
    }

    const cameraPosition = camera.globalPosition ?? camera.position;

    const wallMeshes = sceneInstance.meshes.filter(
      (mesh) =>
        mesh.name.startsWith('wall_') &&
        mesh.metadata?.wallId &&
        mesh.metadata?.start &&
        mesh.metadata?.end
    );

    const meshesByWall = new Map<string, BABYLON.AbstractMesh[]>();
    wallMeshes.forEach((mesh) => {
      const wallId = mesh.metadata.wallId as string;
      const group = meshesByWall.get(wallId) ?? [];
      group.push(mesh);
      meshesByWall.set(wallId, group);
    });

    meshesByWall.forEach((meshes, wallId) => {
      const wall = this.wallById.get(wallId);

      if (!wall) {
        return;
      }

      const start = new BABYLON.Vector3(
        wall.start.x,
        wall.start.y,
        wall.start.z
      );

      const end = new BABYLON.Vector3(
        wall.end.x,
        wall.end.y,
        wall.end.z
      );

      const direction = end.subtract(start);
      direction.z = 0;

      const wallLength = direction.length();

      if (wallLength < 0.001) {
        return;
      }

      const midpoint = start.add(end).scale(0.5);

      const wallNormal = new BABYLON.Vector3(
        -direction.y / wallLength,
        direction.x / wallLength,
        0
      );

      const roomDirection = roomCenter.subtract(midpoint);
      roomDirection.z = 0;

      const cameraDirection = cameraPosition.subtract(midpoint);
      cameraDirection.z = 0;

      const roomSide = BABYLON.Vector3.Dot(wallNormal, roomDirection);
      const cameraSide = BABYLON.Vector3.Dot(wallNormal, cameraDirection);

      // A wall obstructs the room when the camera is on the wall's outward
      // face (the side opposite the room interior). Comparing the 2D signs in
      // the plan plane keeps the result independent of zoom distance and of how
      // many segments the wall was split into.
      const cameraIsOutside = roomSide * cameraSide < 0;

      meshes.forEach((mesh) => {
        const exteriorMaterial = mesh.metadata.exteriorMaterial as
          | BABYLON.StandardMaterial
          | null;

        const interiorMaterial = mesh.metadata.interiorMaterial as
          | BABYLON.StandardMaterial
          | null;

        if (!exteriorMaterial || !interiorMaterial) {
          return;
        }

        // Interior faces are always opaque; keep them out of the alpha-blend
        // pass so the depth buffer handles them correctly.
        interiorMaterial.transparencyMode =
          BABYLON.Material.MATERIAL_OPAQUE;
        interiorMaterial.alpha = 1;

        // Only the outward-facing side fades, and only when it obstructs the
        // camera. Otherwise it stays opaque.
        exteriorMaterial.transparencyMode = cameraIsOutside
          ? BABYLON.Material.MATERIAL_ALPHABLEND
          : BABYLON.Material.MATERIAL_OPAQUE;
        exteriorMaterial.alpha = cameraIsOutside ? 0.35 : 1;
      });
    });
  }

  private buildObject(object: SceneObject): void {
    if (!this.scene) {
      return;
    }

    const modelUrl = MODEL_URLS[object.type];
    if (modelUrl) {
      void this.buildModelObject(object, modelUrl);
      return;
    }

    const mesh = BABYLON.MeshBuilder.CreateBox(
      `object_${object.id}`,
      {
        width: object.dimensions.width,
        height: object.dimensions.height,
        depth: object.dimensions.depth,
      },
      this.scene,
    );

    mesh.position = new BABYLON.Vector3(object.position.x, object.position.y, object.position.z + object.dimensions.height / 2);
    mesh.rotation = new BABYLON.Vector3(object.rotation.x, object.rotation.y, object.rotation.z);

    const material = new BABYLON.StandardMaterial(`material_${object.id}`, this.scene);
    material.diffuseColor = new BABYLON.Color3(0.55, 0.7, 0.9);
    if (object.material?.color) {
      const color = BABYLON.Color3.FromHexString(object.material.color);
      material.diffuseColor = color;
    }
    mesh.material = material;
  }

  private async buildModelObject(object: SceneObject, modelUrl: string): Promise<void> {
    const sceneInstance = this.scene;

    if (!sceneInstance) {
      return;
    }

    const generation = this.renderGeneration;

    try {
      const container = await this.loadModelContainer(modelUrl, sceneInstance);

      if (!this.scene || generation !== this.renderGeneration) {
        return;
      }

      this.placeModelInstance(container, object, this.scene);
    } catch (error) {
      console.warn(`[renderer] failed to load model for ${object.id}`, error);
    }
  }

  private loadModelContainer(modelUrl: string, sceneInstance: BABYLON.Scene): Promise<BABYLON.AssetContainer> {
    const existing = this.modelContainers.get(modelUrl);
    if (existing) {
      return existing;
    }

    const pending = BABYLON.LoadAssetContainerAsync(modelUrl, sceneInstance).catch((error: unknown) => {
      this.modelContainers.delete(modelUrl);
      throw error;
    });

    this.modelContainers.set(modelUrl, pending);
    return pending;
  }

  private placeModelInstance(container: BABYLON.AssetContainer, object: SceneObject, sceneInstance: BABYLON.Scene): void {
    const instance = container.instantiateModelsToScene((name) => `object_${object.id}__${name}`, true);

    const pivot = new BABYLON.TransformNode(`object_${object.id}`, sceneInstance);
    const oriented = new BABYLON.TransformNode(`object_${object.id}__oriented`, sceneInstance);
    oriented.parent = pivot;
    instance.rootNodes.forEach((node) => {
      node.parent = oriented;
    });

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (const mesh of oriented.getChildMeshes()) {
      mesh.computeWorldMatrix(true);
      const box = mesh.getBoundingInfo().boundingBox;
      minX = Math.min(minX, box.minimumWorld.x);
      minY = Math.min(minY, box.minimumWorld.y);
      minZ = Math.min(minZ, box.minimumWorld.z);
      maxX = Math.max(maxX, box.maximumWorld.x);
      maxY = Math.max(maxY, box.maximumWorld.y);
      maxZ = Math.max(maxZ, box.maximumWorld.z);
    }

    oriented.scaling = new BABYLON.Vector3(
      object.dimensions.width / (maxX - minX),
      object.dimensions.height / (maxY - minY),
      object.dimensions.depth / (maxZ - minZ),
    );
    oriented.rotation = new BABYLON.Vector3(MODEL_UP_TO_SCENE_UP, 0, 0);
    oriented.position = new BABYLON.Vector3(
      -((minX + maxX) / 2) * oriented.scaling.x,
      ((minZ + maxZ) / 2) * oriented.scaling.z,
      -minY * oriented.scaling.y,
    );

    pivot.rotation = new BABYLON.Vector3(object.rotation.x, object.rotation.y, object.rotation.z);
    pivot.position = new BABYLON.Vector3(object.position.x, object.position.y, object.position.z);
  }

  dispose(): void {
    this.engine?.stopRenderLoop();
    this.scene?.dispose();
    this.engine?.dispose();
    this.scene = null;
    this.engine = null;
    this.modelContainers.forEach((pending) => {
      void pending.then((container) => container.dispose()).catch(() => undefined);
    });
    this.modelContainers.clear();
  }
}
