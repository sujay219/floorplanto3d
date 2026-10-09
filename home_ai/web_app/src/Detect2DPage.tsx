import React, { useRef, useState } from 'react';
import { detectFloorPlan2D } from './detect2d';
import type { Detect2DReport, Detect2DResult } from './detect2d';

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

const Detect2DPage = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [result, setResult] = useState<Detect2DResult | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);

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
      const detected = await detectFloorPlan2D(file);
      setResult(detected);
      setStatus('ready');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to normalize the floor plan.');
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
    </main>
  );
};

export { Detect2DPage };
