// The uploaded study as the model sees it: three views side by side with one slice scrubber, plus what was found in the files.
import { useEffect, useState } from "react";
import { sliceUrl } from "../api.js";
import { PLANES, PLANE_LABEL, describeFlag } from "../findings.js";

const LEVEL_TEXT = { 1: "fluid-sensitive series", 2: "fallback series (no fluid-sensitive one found)" };

export default function ScanViewer({ upload, onReplace, disabled }) {
  const n = upload.n_slices;
  const [index, setIndex] = useState(Math.floor(n / 2));
  const planes = Object.fromEntries(upload.planes.map((p) => [p.plane, p]));

  useEffect(() => {
    // Preload every slice so scrubbing is instant.
    for (const plane of PLANES) {
      if (!planes[plane]?.present) continue;
      for (let i = 0; i < n; i++) new Image().src = sliceUrl(upload.upload_id, plane, i);
    }
  }, [upload.upload_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = (delta) => setIndex((i) => Math.min(n - 1, Math.max(0, i + delta)));

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Knee scan</h2>
        <button className="link-button" onClick={onReplace} disabled={disabled}>Use a different scan</button>
      </div>

      <div className="viewer">
        {PLANES.map((plane) => {
          const info = planes[plane];
          return (
            <figure key={plane} className={`view ${info?.present ? "" : "missing"}`}
              onWheel={(e) => info?.present && step(e.deltaY > 0 ? 1 : -1)}>
              {info?.present ? (
                <img src={sliceUrl(upload.upload_id, plane, index)} alt={`${PLANE_LABEL[plane]}, slice ${index + 1} of ${n}`} draggable="false" />
              ) : (
                <div className="view-empty">Not in this upload</div>
              )}
              <figcaption>
                <strong>{PLANE_LABEL[plane]}</strong> <span className="muted">{plane}</span>
              </figcaption>
            </figure>
          );
        })}
      </div>

      <div className="scrubber">
        <label htmlFor="slice">Slice {index + 1} of {n}</label>
        <input id="slice" type="range" min="0" max={n - 1} value={index} onChange={(e) => setIndex(Number(e.target.value))} />
      </div>

      {upload.quality_flags.length > 0 && (
        <ul className="flags" aria-label="Image quality notes">
          {upload.quality_flags.map((flag) => <li key={flag}>{describeFlag(flag)}</li>)}
        </ul>
      )}

      <details className="series-details">
        <summary>
          {upload.n_images} images read
          {upload.n_skipped > 0 && `, ${upload.n_skipped} files skipped`}
          {upload.laterality !== "unknown" && `, ${upload.laterality} knee`}
        </summary>
        <table>
          <thead>
            <tr><th>View</th><th>Series used</th><th>Slices</th></tr>
          </thead>
          <tbody>
            {upload.planes.map((p) => (
              <tr key={p.plane}>
                <td>{PLANE_LABEL[p.plane]}</td>
                <td>{p.present ? <>{p.description || p.series_folder}<br /><span className="muted small">{LEVEL_TEXT[p.level]}</span></> : <span className="muted">none</span>}</td>
                <td className="num">{p.present ? p.n_original_slices : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {upload.series.length > 3 && (
          <p className="muted small">Other series in the upload: {upload.series.filter((s) => !s.used).map((s) => s.description || s.folder).join("; ")}</p>
        )}
        {upload.skipped_examples.length > 0 && (
          <p className="muted small">Skipped, for example: {upload.skipped_examples.join("; ")}</p>
        )}
      </details>
    </div>
  );
}