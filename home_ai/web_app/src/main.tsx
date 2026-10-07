import React, { useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { BabylonRenderer } from '@home-ai/renderer';
import type { Scene } from '@home-ai/scene-schema';
import { developmentScene } from './mock-scene';
import { processFloorPlanImage } from './floorplan';

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
    <main style={{ padding: '1.5rem', fontFamily: 'sans-serif', maxWidth: '1240px', margin: '0 auto' }}>
      <h1>Home AI</h1>
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <button onClick={handleUploadClick} disabled={status === 'processing'}>
          Upload Floor Plan
        </button>
        <button onClick={handleReset} disabled={status === 'processing'}>
          Reset
        </button>
        <button onClick={handleResetCamera} disabled={cameraMode === 'eagle' || status === 'processing'}>
          Reset to Eagle View
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(event) => void handleFileChange(event)}
        />
        {status === 'processing' ? <span style={{ color: '#1d4ed8' }}>Processing floor plan…</span> : null}
        {status === 'ready' ? <span style={{ color: '#15803d' }}>Floor plan loaded.</span> : null}
      </div>
      {error ? <p style={{ color: 'crimson' }}>{error}</p> : null}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'stretch' }}>
        <aside
          style={{
            width: '92px',
            padding: '1rem 0.75rem',
            borderRadius: '16px',
            border: '1px solid #d1d5db',
            background: 'linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)',
            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'space-between',
            minHeight: '520px',
          }}
        >
          <div style={{ fontSize: '0.75rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: '#4b5563' }}>
            Zoom
          </div>
          <input
            aria-label="Zoom"
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
            style={{
              width: '28px',
              height: '420px',
              writingMode: 'vertical-rl',
              direction: 'rtl',
              WebkitAppearance: 'slider-vertical',
              accentColor: '#3b82f6',
              cursor: 'ns-resize',
            }}
          />
          <div style={{ fontSize: '1.35rem', fontWeight: 700, color: '#111827', lineHeight: 1 }}>
            {Math.round(zoom)}
          </div>
        </aside>
        <section style={{ display: 'flex', flex: 1, flexDirection: 'column', gap: '0.75rem' }}>
          <canvas
            ref={canvasRef}
            style={{
              width: '100%',
              height: '570px',
              borderRadius: '12px',
              border: '1px solid #d1d5db',
              background: '#e5e7eb',
              display: 'block',
              flex: 1,
            }}
          />
        </section>
      </div>
      <div style={{ marginTop: '0.75rem' }}>
        <strong>Scene status:</strong> {scene ? `${scene.rooms.length} room(s)` : 'loading...'}
      </div>
    </main>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
