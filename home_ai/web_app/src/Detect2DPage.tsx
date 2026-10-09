import React, { useRef, useState } from 'react';
import { detectFloorPlan2D, detectRectangles } from './detect2d';
import type { Detect2DReport, Detect2DResult, RectanglesResult } from './detect2d';

type Status = 'idle' | 'processing' | 'ready' | 'error';

const formatNumber = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined ? '–' : value.toFixed(digits);

const ReportPanel = ({ report }: { report: Detect2DReport }) => {
  const { original, normalized, inspection } = report;

  return (
    <aside className="rail detect2d-report">
      <span className="rail-label">Phase 1 report</span>
      <div className="detect2d-report-body">
        <div className="detect2d-report-row">
          <span>Original</span>
          <span>
            {original.width} × {original.height} px · {original.mode} · {original.format ?? 'unknown'}
          </span>
        </div>
        <div className="detect2d-report-row">
          <span>Normalized</span>
          <span>
            {normalized.width} × {normalized.height} px · {normalized.mode}
            {normalized.resized ? ` · scaled ${normalized.scale}` : ''}
          </span>
        </div>
        <div className="detect2d-report-row">
          <span>Background</span>
          <span>
            level {formatNumber(inspection.background.level)} ({inspection.background.polarity}) ·
            gradient {formatNumber(inspection.background.gradient_range)}
          </span>
        </div>
        <div className="detect2d-report-row">
          <span>Contrast</span>
          <span>
            {formatNumber(inspection.contrast.low)}–{formatNumber(inspection.contrast.high)} · median{' '}
            {formatNumber(inspection.contrast.median)} · RMS {inspection.contrast.rms_contrast}
          </span>
        </div>
        <div className="detect2d-report-row">
          <span>Noise σ</span>
          <span>
            {formatNumber(inspection.noise.sigma, 2)} → {formatNumber(inspection.noise.sigma_normalized, 2)}{' '}
            after normalization
          </span>
        </div>
        <div className="detect2d-report-row">
          <span>Ink stroke width</span>
          <span>
            {formatNumber(inspection.wall_thickness.median_px)} px median (p25–p75{' '}
            {formatNumber(inspection.wall_thickness.p25_px)}–{formatNumber(inspection.wall_thickness.p75_px)}) ·{' '}
            {inspection.wall_thickness.sample_count} runs
          </span>
        </div>
      </div>
    </aside>
  );
};

const RelationshipList = ({
  label,
  pairs,
}: {
  label: string;
  pairs: { text: string }[];
}) => {
  if (pairs.length === 0) {
    return null;
  }

  return (
    <p className="detect2d-relationships">
      <strong>{label}:</strong> {pairs.slice(0, 6).map((pair) => pair.text).join(' · ')}
      {pairs.length > 6 ? ` · +${pairs.length - 6} more` : ''}
    </p>
  );
};

const Phase2Results = ({ result }: { result: RectanglesResult }) => {
  const { summary, duplicates, nested, parameters } = result.report;

  return (
    <section className="detect2d-phase2">
      <div className="detect2d-phase2-head">
        <h2>Phase 2 · rectangle candidates</h2>
        <div className="detect2d-chips">
          <span className="detect2d-chip">
            {summary.total_rectangles} candidates ({summary.drawn_rectangles} drawn)
          </span>
          <span className="detect2d-chip detect2d-chip-small">small {summary.by_category.small}</span>
          <span className="detect2d-chip detect2d-chip-medium">medium {summary.by_category.medium}</span>
          <span className="detect2d-chip detect2d-chip-large">large {summary.by_category.large}</span>
          <span className="detect2d-chip">{summary.duplicate_count} duplicates</span>
          <span className="detect2d-chip">{summary.nested_count} nested</span>
        </div>
        <RelationshipList
          label="Obvious duplicates"
          pairs={duplicates.map((pair) => ({
            text: `${pair.duplicate} = ${pair.representative} (IoU ${pair.iou.toFixed(2)})`,
          }))}
        />
        <RelationshipList
          label="Nested"
          pairs={nested.map((pair) => ({
            text: `${pair.inner} in ${pair.outer} (${(pair.containment * 100).toFixed(0)}%)`,
          }))}
        />
        <details className="detect2d-params">
          <summary>Detection parameters</summary>
          <dl>
            {Object.entries(parameters).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{Array.isArray(value) ? value.join(', ') : String(value)}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>
      <img
        className="detect2d-image"
        src={result.images.overlay.data}
        alt="Rectangle candidates with IDs drawn over the original floor plan"
      />
      <div className="detect2d-view-grid">
        <figure className="detect2d-view">
          <figcaption>small candidates</figcaption>
          <img src={result.images.small.data} alt="Small rectangle candidates" />
        </figure>
        <figure className="detect2d-view">
          <figcaption>medium candidates</figcaption>
          <img src={result.images.medium.data} alt="Medium rectangle candidates" />
        </figure>
        <figure className="detect2d-view">
          <figcaption>large candidates</figcaption>
          <img src={result.images.large.data} alt="Large rectangle candidates" />
        </figure>
      </div>
    </section>
  );
};

const Detect2DPage = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [result, setResult] = useState<Detect2DResult | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [rectResult, setRectResult] = useState<RectanglesResult | null>(null);
  const [rectStatus, setRectStatus] = useState<Status>('idle');
  const [rectError, setRectError] = useState<string | null>(null);

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
    setSourceFile(file);
    setRectResult(null);
    setRectStatus('idle');
    setRectError(null);

    try {
      const detected = await detectFloorPlan2D(file);
      setResult(detected);
      setStatus('ready');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to normalize the floor plan.');
    }
  };

  const handleDetectRectangles = async () => {
    if (!sourceFile) {
      return;
    }

    setRectError(null);
    setRectStatus('processing');

    try {
      const detected = await detectRectangles(sourceFile);
      setRectResult(detected);
      setRectStatus('ready');
    } catch (err) {
      setRectStatus('error');
      setRectError(err instanceof Error ? err.message : 'Failed to detect rectangles.');
    }
  };

  const report = result?.report;

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Home AI</h1>
          <p className="tagline">Phase 1 · image normalization</p>
        </div>
        <div className="actions">
          <button type="button" className="btn btn-primary" onClick={handleUploadClick} disabled={status === 'processing'}>
            Upload floor plan
          </button>
          <a className="btn" href="/">
            3D Viewer
          </a>
          <div className={`status status-${status}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            {status === 'processing' ? 'Normalizing…' : null}
            {status === 'ready' ? 'Normalization ready' : null}
            {status === 'idle' ? 'Upload a floor plan to begin.' : null}
            {status === 'error' ? 'Normalization failed' : null}
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
        {report ? <ReportPanel report={report} /> : null}
        <section className="viewport">
          {result ? (
            <img
              className="detect2d-comparison"
              src={result.images.comparison.data}
              alt="Original and normalized floor plan side by side"
            />
          ) : (
            <div className="detect2d-empty">
              {status === 'processing' ? 'Normalizing…' : 'Upload a floor plan to see the Phase 1 result.'}
            </div>
          )}
          <footer className="statusbar">
            <span>
              <strong>Phase 1</strong> · normalization only (no thresholding, walls or rooms yet)
            </span>
            <span>
              {report
                ? `${report.original.width} × ${report.original.height} px · ${report.original.mode} · ${report.original.format ?? 'unknown'}`
                : '–'}
            </span>
          </footer>
        </section>
      </div>
      {rectResult ? <Phase2Results result={rectResult} /> : null}
      <footer className="detect2d-bottombar">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void handleDetectRectangles()}
          disabled={!sourceFile || rectStatus === 'processing'}
        >
          {rectStatus === 'processing' ? 'Detecting rectangles…' : 'Detect rectangles'}
        </button>
        <span className="detect2d-bottombar-hint">
          {rectStatus === 'error' && rectError
            ? rectError
            : 'Phase 2 · rectangle candidates on the normalized image (nothing is removed)'}
        </span>
      </footer>
    </main>
  );
};

export { Detect2DPage };
