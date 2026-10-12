import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchWallDetector } from './wallDetector';
import type { WallDetectorResult } from './wallDetector';

type Status = 'idle' | 'processing' | 'ready' | 'error';

interface DrawInfo {
  count: number;
  erased: number;
  total: number;
  canvas: string;
  mask: string;
  time: string;
}

const BINS = 20;
const CHART_HEIGHT = 60;

interface HistogramControlProps {
  label: string;
  values: number[];
  threshold: number;
  onThresholdChange: (value: number) => void;
}

const HistogramControl = ({ label, values, threshold, onThresholdChange }: HistogramControlProps) => {
  const max = useMemo(() => (values.length ? Math.max(...values) : 1), [values]);
  const counts = useMemo(() => {
    const bins = new Array<number>(BINS).fill(0);
    for (const value of values) {
      const index = Math.min(BINS - 1, Math.floor((value / max) * BINS));
      bins[index] += 1;
    }
    return bins;
  }, [values, max]);

  const maxCount = Math.max(...counts, 1);
  const binWidth = 100 / BINS;
  const thresholdX = Math.min(100, (threshold / max) * 100);

  return (
    <div className="wall-detector-histogram">
      <div className="wall-detector-histogram-head">
        <span className="wall-detector-histogram-label">{label}</span>
        <span className="wall-detector-histogram-value">≥ {threshold.toFixed(1)} px</span>
      </div>
      <svg
        className="wall-detector-histogram-chart"
        viewBox={`0 0 100 ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${label} distribution`}
      >
        {counts.map((count, index) => {
          const height = (count / maxCount) * CHART_HEIGHT;
          const binStart = (index / BINS) * max;
          return (
            <rect
              key={index}
              x={index * binWidth + 0.2}
              y={CHART_HEIGHT - height}
              width={binWidth - 0.4}
              height={height}
              fill={binStart >= threshold ? 'var(--accent)' : 'var(--border-strong)'}
            />
          );
        })}
        <line
          x1={thresholdX}
          x2={thresholdX}
          y1={0}
          y2={CHART_HEIGHT}
          stroke="var(--accent-strong)"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <input
        className="wall-detector-slider"
        type="range"
        min={0}
        max={max}
        step={Math.max(max / 200, 0.1)}
        value={threshold}
        aria-label={`${label} threshold`}
        onChange={(event) => onThresholdChange(Number(event.currentTarget.value))}
        onInput={(event) => onThresholdChange(Number(event.currentTarget.value))}
      />
    </div>
  );
};

export const WallDetectorPage = () => {
  const [result, setResult] = useState<WallDetectorResult | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [minThickness, setMinThickness] = useState(0);
  const [minLength, setMinLength] = useState(0);
  const [maskImage, setMaskImage] = useState<HTMLImageElement | null>(null);
  const [drawInfo, setDrawInfo] = useState<DrawInfo>({
    count: 0,
    erased: 0,
    total: 0,
    canvas: '–',
    mask: '–',
    time: '–',
  });
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus('processing');
    fetchWallDetector()
      .then((payload) => {
        if (cancelled) return;
        setResult(payload);
        setStatus('ready');
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Decode the combined mask once; the threshold effect below then redraws
  // synchronously on every slider change (no per-tick image decode).
  useEffect(() => {
    if (!result) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) setMaskImage(image);
    };
    image.src = result.images.combined.data;
    return () => {
      cancelled = true;
    };
  }, [result]);

  const thicknessValues = useMemo(
    () => (result ? result.segments.map((segment) => segment.thickness) : []),
    [result],
  );
  const lengthValues = useMemo(
    () => (result ? result.segments.map((segment) => segment.length) : []),
    [result],
  );
  const shownCount = useMemo(
    () =>
      result
        ? result.segments.filter(
            (segment) => segment.thickness >= minThickness && segment.length >= minLength,
          ).length
        : 0,
    [result, minThickness, minLength],
  );

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const time = new Date().toLocaleTimeString();
    if (!canvas || !maskImage || !result) {
      const info = {
        erased: 0,
        total: 0,
        canvas: canvas ? 'ok' : 'missing',
        mask: maskImage ? `${maskImage.width}×${maskImage.height}` : 'missing',
        time,
      };
      console.warn('[wall-detector] draw skipped', info);
      setDrawInfo((previous) => ({ ...info, count: previous.count + 1 }));
      return;
    }

    canvas.width = maskImage.width;
    canvas.height = maskImage.height;
    const context = canvas.getContext('2d');
    if (!context) return;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(maskImage, 0, 0);

    // The mask is white strokes on black, so painting failing strokes solid
    // black removes them visually. With no threshold nothing is painted and
    // the result is the combined mask exactly. Tiny components can carry a
    // degenerate outline, so those are painted over their bounding box, and
    // every outline is both filled and stroked so hairline strokes vanish.
    let erased = 0;
    context.fillStyle = '#000000';
    context.strokeStyle = '#000000';
    context.lineWidth = 2;
    context.lineJoin = 'round';
    for (const segment of result.segments) {
      if (segment.thickness >= minThickness && segment.length >= minLength) continue;
      erased += 1;
      const polygon = segment.polygon;
      if (polygon.length < 3) {
        const { x, y, width, height } = segment.bbox;
        context.fillRect(x - 1, y - 1, width + 2, height + 2);
        continue;
      }
      context.beginPath();
      polygon.forEach((point, index) => {
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.fill();
      context.stroke();
    }

    const info = {
      erased,
      total: result.segments.length,
      canvas: `${canvas.width}×${canvas.height}`,
      mask: `${maskImage.width}×${maskImage.height}`,
      time,
    };
    console.log('[wall-detector] draw', { ...info, minThickness, minLength });
    setDrawInfo((previous) => ({ ...info, count: previous.count + 1 }));
  }, [result, maskImage, minThickness, minLength]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Wall detector</h1>
          <p className="tagline">real-time.png · thickness &amp; length thresholds · build 5</p>
        </div>
        <div className="actions">
          <a className="btn" href="/detect2d">
            2D Detect
          </a>
          <a className="btn" href="/">
            3D Viewer
          </a>
          <div className={`status status-${status}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            {status === 'processing' ? 'Detecting walls…' : null}
            {status === 'ready' ? 'Wall detection ready' : null}
            {status === 'idle' ? 'Loading the real-time example…' : null}
            {status === 'error' ? 'Wall detection failed' : null}
          </div>
        </div>
      </header>
      {error ? (
        <div className="alert" role="alert">
          {error}
        </div>
      ) : null}
      {result ? (
        <section className="wall-detector">
          <div className="wall-detector-images">
            <figure className="detect2d-view">
              <figcaption>Phase 1 · normalized.png</figcaption>
              <img
                className="detect2d-image"
                src={result.images.normalized.data}
                alt="Normalized floor plan"
              />
            </figure>
            <figure className="detect2d-view">
              <figcaption>Step B · 04_combined.png</figcaption>
              <img
                className="detect2d-image"
                src={result.images.combined.data}
                alt="Combined horizontal and vertical wall structures"
              />
            </figure>
          </div>
          <div className="wall-detector-histograms">
            <HistogramControl
              label="Wall thickness"
              values={thicknessValues}
              threshold={minThickness}
              onThresholdChange={setMinThickness}
            />
            <HistogramControl
              label="Wall length"
              values={lengthValues}
              threshold={minLength}
              onThresholdChange={setMinLength}
            />
          </div>
          <figure className="detect2d-view wall-detector-result">
            <figcaption>
              Result · {shownCount} of {result.segments.length} walls above both thresholds
            </figcaption>
            <canvas ref={canvasRef} aria-label="Filtered wall result" />
            <footer className="statusbar">
              <span>
                <strong>{shownCount}</strong> of <strong>{result.segments.length}</strong> walls
                shown
              </span>
              <span>
                thickness ≥ {minThickness.toFixed(1)} px · length ≥ {minLength.toFixed(1)} px
              </span>
              <button type="button" className="btn" onClick={redraw}>
                Redraw result
              </button>
            </footer>
            <footer className="statusbar wall-detector-debug">
              <span>
                build 5 · draw #{drawInfo.count} · painted black {drawInfo.erased}/
                {drawInfo.total} failing walls
              </span>
              <span>
                canvas {drawInfo.canvas} · mask {drawInfo.mask} · last draw {drawInfo.time}
              </span>
            </footer>
          </figure>
        </section>
      ) : null}
    </main>
  );
};
