import React, { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { BabylonRenderer } from '@home-ai/renderer';
import type { Scene } from '@home-ai/scene-schema';
import { developmentScene } from './mock-scene';
import { processFloorPlanImage } from './floorplan';
import { Detect2DPage } from './Detect2DPage';
import './styles.css';

type Status = 'idle' | 'processing' | 'ready' | 'error';

const BED_DIMENSIONS = { width: 1100, depth: 760, height: 340 };
const BED_WALL_ID = 'wall_002';

const bedPlacement = (room: Scene['rooms'][number]) => {
  const centered = {
    position: { x: room.dimensions.width / 2, y: room.dimensions.depth / 2, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
  };
  const wall = room.walls.find((candidate) => candidate.id === BED_WALL_ID);

  if (!wall) {
    return centered;
  }

  const dx = wall.end.x - wall.start.x;
  const dy = wall.end.y - wall.start.y;
  const length = Math.hypot(dx, dy);

  if (length < 0.001) {
    return centered;
  }

  const midX = (wall.start.x + wall.end.x) / 2;
  const midY = (wall.start.y + wall.end.y) / 2;
  let normalX = -dy / length;
  let normalY = dx / length;
  const towardRoom = (room.dimensions.width / 2 - midX) * normalX + (room.dimensions.depth / 2 - midY) * normalY;

  if (towardRoom < 0) {
    normalX = -normalX;
    normalY = -normalY;
  }

  const offset = wall.thickness / 2 + BED_DIMENSIONS.depth / 2;

  return {
    position: { x: midX + normalX * offset, y: midY + normalY * offset, z: 0 },
    rotation: { x: 0, y: 0, z: Math.atan2(dy, dx) },
  };
};

const withBedInFirstRoom = (source: Scene): Scene => {
  const [firstRoom, ...otherRooms] = source.rooms;

  if (!firstRoom || firstRoom.objects.some((object) => object.id === 'bed_01')) {
    return source;
  }

  return {
    ...source,
    rooms: [
      {
        ...firstRoom,
        objects: [
          ...firstRoom.objects,
          {
            id: 'bed_01',
            type: 'bed',
            name: 'Bed',
            ...bedPlacement(firstRoom),
            dimensions: BED_DIMENSIONS,
          },
        ],
      },
      ...otherRooms,
    ],
  };
};

const App = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<BabylonRenderer | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [zoom, setZoom] = useState<number>(4200);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [cameraMode, setCameraMode] = useState<'eagle' | 'corner'>('eagle');

  useEffect(() => {
    let cancelled = false;

    const loadExample = async () => {
      setStatus('processing');
      setError(null);

      try {
        const response = await fetch('/windows.png');
        if (!response.ok) {
          throw new Error(`Failed to fetch example floor plan (${response.status})`);
        }
        const blob = await response.blob();
        const file = new File([blob], 'windows.png', { type: blob.type || 'image/png' });
        const nextScene = await processFloorPlanImage(file);
        if (!cancelled) {
          setScene(withBedInFirstRoom(nextScene));
          setStatus('ready');
        }
      } catch (err) {
        if (!cancelled) {
          setScene(withBedInFirstRoom(developmentScene));
          setStatus('error');
          setError(err instanceof Error ? err.message : 'Failed to load the example floor plan.');
        }
      }
    };

    void loadExample();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas || rendererRef.current) {
      return;
    }

    const renderer = new BabylonRenderer();
    rendererRef.current = renderer;
    renderer.setWallThickness(20);
    const unsubscribeZoom = renderer.onZoomChange((radius) => {
      setZoom(radius);
    });
    const unsubscribeCameraView = renderer.onCameraViewChange(setCameraMode);

    void renderer.initialize(canvas).then(() => {
      setZoom(renderer.getRadius());
    });

    return () => {
      unsubscribeZoom();
      unsubscribeCameraView();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (scene) {
      rendererRef.current?.render(scene);
    }
  }, [scene]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file) {
      return;
    }

    setError(null);
    setStatus('processing');

    try {
      const nextScene = await processFloorPlanImage(file);
      setScene(nextScene);
      setStatus('ready');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to process the floor plan.');
    }
  };

  const handleReset = () => {
    setError(null);
    setStatus('idle');
    setScene(withBedInFirstRoom(developmentScene));
  };

  const handleResetCamera = () => {
    rendererRef.current?.resetToEagleView();
  };

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Home AI</h1>
          <p className="tagline">Floor plan to 3D</p>
        </div>
        <div className="actions">
          <button type="button" className="btn btn-primary" onClick={handleUploadClick} disabled={status === 'processing'}>
            Upload floor plan
          </button>
          <button type="button" className="btn" onClick={handleReset} disabled={status === 'processing'}>
            Reset scene
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleResetCamera}
            disabled={cameraMode === 'eagle' || status === 'processing'}
          >
            Reset camera
          </button>
          <a className="btn" href="/detect2d">
            2D Detect
          </a>
          <div className={`status status-${status}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            {status === 'processing' ? 'Processing floor plan…' : null}
            {status === 'ready' ? 'Floor plan ready' : null}
            {status === 'idle' ? 'Sample scene loaded.' : null}
            {status === 'error' ? 'Processing failed' : null}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(event) => void handleFileChange(event)}
          />
        </div>
      </header>
      {error ? (
        <div className="alert" role="alert">
          <span>{error}</span>
          <span>Upload another floor plan image to try again.</span>
        </div>
      ) : null}
      <div className="workspace">
        <aside className="rail">
          <span className="rail-label">Zoom</span>
          <input
            aria-label="Zoom"
            className="zoom-slider"
            type="range"
            min={400}
            max={8000}
            step={1}
            value={zoom}
            onChange={(event) => {
              const nextZoom = Number(event.currentTarget.value);
              setZoom(nextZoom);
              rendererRef.current?.setZoom(nextZoom);
            }}
          />
          <output className="zoom-value">{Math.round(zoom)}</output>
        </aside>
        <section className="viewport">
          <canvas ref={canvasRef} aria-label="3D scene viewport" />
          <footer className="statusbar">
            <span>
              <strong>{scene ? scene.rooms.length : 0}</strong> {scene && scene.rooms.length === 1 ? 'room' : 'rooms'}
            </span>
            <span>{cameraMode === 'eagle' ? 'Eagle view' : 'Corner view'}</span>
          </footer>
        </section>
      </div>
    </main>
  );
};

const path = window.location.pathname.replace(/\/+$/, '') || '/';
const Page = path === '/detect2d' ? Detect2DPage : App;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Page />
  </React.StrictMode>,
);
