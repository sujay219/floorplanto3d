import * as BABYLON from 'babylonjs';
import interiorWallTexture from '../../textures/wall_white.png';

type WallCap = 'top' | 'bottom' | 'start' | 'end';

interface WallSpec {
  name: string;
  start: BABYLON.Vector3;
  end: BABYLON.Vector3;
  bottom: number;
  top: number;
  thickness: number;
  omitCaps?: WallCap[];
}

export class BabylonRenderer2 {
  private engine: BABYLON.Engine | null = null;
  private scene: BABYLON.Scene | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private camera: BABYLON.ArcRotateCamera | null = null;
  private zoomChangeListeners = new Set<(radius: number) => void>();
  private lastReportedRadius: number | null = null;
  private readonly defaultCameraRadius = 1200;
  private wallThickness = 20;

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

  onZoomChange(listener: (radius: number) => void): () => void {
    this.zoomChangeListeners.add(listener);
    if (this.camera) {
      listener(this.getRadius());
    }

    return () => {
      this.zoomChangeListeners.delete(listener);
    };
  }

  setWallThickness(thickness: number): number {
    const nextThickness = Math.min(400, Math.max(20, thickness));
    this.wallThickness = nextThickness;
    return nextThickness;
  }

  private reportZoomChange(radius: number): void {
    if (this.lastReportedRadius === radius) {
      return;
    }

    this.lastReportedRadius = radius;
    this.zoomChangeListeners.forEach((listener) => listener(radius));
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

    this.camera = new BABYLON.ArcRotateCamera('debugCamera', -Math.PI / 4, Math.PI / 3, 1200, BABYLON.Vector3.Zero(), this.scene);
    this.camera.upVector = new BABYLON.Vector3(0, 0, 1);
    this.camera.attachControl(canvas, true);
    this.camera.setTarget(new BABYLON.Vector3(200, 0, 120));
    this.camera.lowerRadiusLimit = 200;
    this.camera.upperRadiusLimit = 8000;
    this.scene.activeCamera = this.camera;

    const hemisphericLight = new BABYLON.HemisphericLight('debugAmbientLight', new BABYLON.Vector3(0, 0, 1), this.scene);
    hemisphericLight.intensity = 0.7;
    hemisphericLight.diffuse = new BABYLON.Color3(0.9, 0.92, 0.95);

    const directionalLight = new BABYLON.DirectionalLight('debugDirectionalLight', new BABYLON.Vector3(-0.5, -0.7, -0.8), this.scene);
    directionalLight.intensity = 0.5;
    directionalLight.diffuse = new BABYLON.Color3(0.95, 0.95, 0.98);

    this.engine.runRenderLoop(() => {
      this.scene?.render();
    });
  }

  render(_scene?: unknown): void {
    const sceneInstance = this.scene;
    if (!sceneInstance) {
      return;
    }

    sceneInstance.meshes
      .filter((mesh) => mesh.name.startsWith('debugWall_'))
      .forEach((mesh) => mesh.dispose());

    sceneInstance.materials
      .filter((material) => material.name.startsWith('debugWall_'))
      .forEach((material) => material.dispose());

    // Two vertically stacked walls sharing the horizontal edge at z = 120.
    // A second pair runs perpendicular from one end so the vertical joint and
    // the corner joint can be compared side by side.
    const thickness = this.wallThickness;
    const walls: WallSpec[] = [
      {
        name: 'debugWall_lower',
        start: new BABYLON.Vector3(0, 0, 0),
        end: new BABYLON.Vector3(400, 0, 0),
        bottom: 0,
        top: 120,
        thickness,
        omitCaps: ['top'],
      },
      {
        name: 'debugWall_upper',
        start: new BABYLON.Vector3(0, 0, 0),
        end: new BABYLON.Vector3(400, 0, 0),
        bottom: 120,
        top: 240,
        thickness,
        omitCaps: ['bottom'],
      },
      {
        name: 'debugWall_perp_lower',
        start: new BABYLON.Vector3(400, 0, 0),
        end: new BABYLON.Vector3(400, 300, 0),
        bottom: 0,
        top: 120,
        thickness,
        omitCaps: ['top'],
      },
      {
        name: 'debugWall_perp_upper',
        start: new BABYLON.Vector3(400, 0, 0),
        end: new BABYLON.Vector3(400, 300, 0),
        bottom: 120,
        top: 240,
        thickness,
        omitCaps: ['bottom'],
      },
    ];

    walls.forEach((wall) => this.buildWall(sceneInstance, wall));
  }

  private buildWall(sceneInstance: BABYLON.Scene, wall: WallSpec): void {
    const start = wall.start;
    const end = wall.end;
    const direction = end.subtract(start);
    const length = direction.length();
    if (length === 0) {
      return;
    }

    const normalizedDirection = direction.normalize();
    const sideways = new BABYLON.Vector3(-normalizedDirection.y, normalizedDirection.x, 0).scale(wall.thickness / 2);
    const up = new BABYLON.Vector3(0, 0, wall.top - wall.bottom);

    const p0 = start.add(sideways).add(new BABYLON.Vector3(0, 0, wall.bottom));
    const p1 = end.add(sideways).add(new BABYLON.Vector3(0, 0, wall.bottom));
    const p2 = end.subtract(sideways).add(new BABYLON.Vector3(0, 0, wall.bottom));
    const p3 = start.subtract(sideways).add(new BABYLON.Vector3(0, 0, wall.bottom));
    const p4 = p0.add(up);
    const p5 = p1.add(up);
    const p6 = p2.add(up);
    const p7 = p3.add(up);

    const positions = [
      p0.x, p0.y, p0.z,
      p1.x, p1.y, p1.z,
      p2.x, p2.y, p2.z,
      p3.x, p3.y, p3.z,
      p4.x, p4.y, p4.z,
      p5.x, p5.y, p5.z,
      p6.x, p6.y, p6.z,
      p7.x, p7.y, p7.z,
    ];

    const capFaces: Record<WallCap, number[]> = {
      top: [4, 5, 6, 4, 6, 7],
      bottom: [0, 2, 1, 0, 3, 2],
      start: [0, 4, 7, 0, 7, 3],
      end: [1, 2, 6, 1, 6, 5],
    };

    const sideFaces = [
      0, 1, 5, 0, 5, 4,
      3, 7, 6, 3, 6, 2,
    ];

    const omitted = new Set(wall.omitCaps ?? []);
    const indices = [
      ...(omitted.has('top') ? [] : capFaces.top),
      ...(omitted.has('bottom') ? [] : capFaces.bottom),
      ...(omitted.has('start') ? [] : capFaces.start),
      ...(omitted.has('end') ? [] : capFaces.end),
      ...sideFaces,
    ];

    const normals: number[] = [];
    BABYLON.VertexData.ComputeNormals(positions, indices, normals);

    const uvs = [
      0, 1, // p0
      1, 1, // p1
      1, 1, // p2
      0, 1, // p3
      0, 0, // p4
      1, 0, // p5
      1, 0, // p6
      0, 0, // p7
    ];

    const mesh = new BABYLON.Mesh(wall.name, sceneInstance);
    const vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    vertexData.normals = normals;
    vertexData.uvs = uvs;
    vertexData.applyToMesh(mesh);
    mesh.convertToFlatShadedMesh();

    const material = new BABYLON.StandardMaterial(`${wall.name}Material`, sceneInstance);
    material.diffuseTexture = new BABYLON.Texture(interiorWallTexture, sceneInstance);
    material.diffuseTexture.hasAlpha = false;
    material.diffuseColor = BABYLON.Color3.White();
    material.specularColor = BABYLON.Color3.Black();
    material.disableLighting = false;
    mesh.material = material;

    mesh.metadata = {
      start: { x: wall.start.x, y: wall.start.y, z: wall.start.z },
      end: { x: wall.end.x, y: wall.end.y, z: wall.end.z },
      bottom: wall.bottom,
      top: wall.top,
      thickness: wall.thickness,
    };
  }

  dispose(): void {
    this.engine?.stopRenderLoop();
    this.scene?.dispose();
    this.engine?.dispose();
    this.scene = null;
    this.engine = null;
    this.camera = null;
    this.canvas = null;
  }
}
