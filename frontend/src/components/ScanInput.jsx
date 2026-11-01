// Getting a scan in: drop or pick DICOM files, a study folder or a .zip; or pick a study the Model Service already has.
import { useEffect, useRef, useState } from "react";
import { getStudies } from "../api.js";

// Walks dropped folders (Chrome, Edge, Firefox, Safari all support webkitGetAsEntry).
async function collectDropped(dataTransfer) {
  const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((file) => ({ file, name: file.name }));
  const out = [];
  async function walk(entry, prefix) {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ file, name: prefix + file.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
  }
  for (const entry of entries) await walk(entry, "");
  return out;
}

const fromInput = (list) => [...list].map((file) => ({ file, name: file.webkitRelativePath || file.name }));

export default function ScanInput({ upload, onFiles, onDatasetStudy, health }) {
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);
  const folderInput = useRef(null);
  const busy = upload.state === "sending" || upload.state === "reading";

  async function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    const files = await collectDropped(e.dataTransfer);
    if (files.length) onFiles(files);
  }

  return (
    <div className="panel">
      <h2>Knee scan</h2>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${busy ? "busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {busy ? (
          <UploadProgress upload={upload} />
        ) : (
          <>
            <svg viewBox="0 0 64 64" className="drop-art" aria-hidden="true">
              <rect x="10" y="6" width="44" height="52" rx="4" />
              <path d="M26 16v12l6 6v14M38 16v12l-6 6" />
            </svg>
            <p className="drop-title">Drop the MRI study here</p>
            <p className="muted">DICOM files (.dcm), the study folder, or a .zip of it. All three views (side, front, top-down) give the best result.</p>
            <div className="drop-actions">
              <button className="secondary" onClick={() => fileInput.current.click()}>Choose files or .zip</button>
              <button className="secondary" onClick={() => folderInput.current.click()}>Choose a folder</button>
            </div>
          </>
        )}
        <input ref={fileInput} type="file" multiple hidden accept=".dcm,.zip,.DCM,application/dicom,application/zip,*/*"
          onChange={(e) => e.target.files.length && onFiles(fromInput(e.target.files))} />
        <input ref={folderInput} type="file" hidden webkitdirectory="" directory=""
          onChange={(e) => e.target.files.length && onFiles(fromInput(e.target.files))} />
      </div>
      {upload.state === "error" && (
        <p className="error-text" role="alert">{upload.message}</p>
      )}
      <DatasetPicker onPick={onDatasetStudy} health={health} />
    </div>
  );
}

function UploadProgress({ upload }) {
  if (upload.state === "reading") {
    return (
      <div className="progress-block">
        <p className="drop-title">Reading the scan…</p>
        <p className="muted">Sorting {upload.count} files into views and preparing the slices the model reads.</p>
        <div className="bar indeterminate"><span /></div>
      </div>
    );
  }
  return (
    <div className="progress-block">
      <p className="drop-title">Uploading {upload.count} file{upload.count === 1 ? "" : "s"}</p>
      <div className="bar"><span style={{ width: `${Math.round(upload.progress * 100)}%` }} /></div>
      <p className="muted">{Math.round(upload.progress * 100)}%</p>
    </div>
  );
}

function DatasetPicker({ onPick, health }) {
  const [open, setOpen] = useState(false);
  const [studies, setStudies] = useState(null);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open || studies) return;
    getStudies().then(setStudies).catch((e) => setError(e.message));
  }, [open, studies]);

  const anyId = studies?.any_study_id_accepted;
  const ids = studies?.study_ids ?? [];

  return (
    <details className="dataset" open={open} onToggle={(e) => setOpen(e.target.open)}>
      <summary>Use a study from the RSNA dataset instead</summary>
      {error && <p className="error-text">{error}</p>}
      {studies && !anyId && ids.length === 0 && (
        <p className="muted">The Model Service has no dataset studies in {health?.config?.model_mode ?? "this"} mode.</p>
      )}
      {studies && (anyId || ids.length > 0) && (
        <form
          className="dataset-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (value.trim()) onPick(value.trim());
          }}
        >
          <label htmlFor="study-id">Study ID</label>
          {ids.length > 0 ? (
            <select id="study-id" value={value} onChange={(e) => setValue(e.target.value)}>
              <option value="">Choose one of {ids.length} studies</option>
              {ids.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          ) : (
            <input id="study-id" value={value} onChange={(e) => setValue(e.target.value)} placeholder="1.2.826.0.1.3680043.8.498…" />
          )}
          <button className="secondary" type="submit" disabled={!value.trim()}>Use this study</button>
          {anyId && <p className="muted small">Stub mode accepts any ID and returns demo scores.</p>}
        </form>
      )}
    </details>
  );
}