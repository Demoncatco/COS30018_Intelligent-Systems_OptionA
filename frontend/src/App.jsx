// Main screen: scan on the left (upload, viewer, patient details), analysis on the right (live progress, report).
import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, getSession, startSession, streamEvents, uploadFiles } from "./api.js";
import Header from "./components/Header.jsx";
import ScanInput from "./components/ScanInput.jsx";
import ScanViewer from "./components/ScanViewer.jsx";
import PatientForm from "./components/PatientForm.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";

const EMPTY_PATIENT = { age: "", sex: "unknown", symptoms: "", injury_mechanism: "" };

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [source, setSource] = useState(null); // {kind: "upload", upload} | {kind: "dataset", studyId}
  const [upload, setUpload] = useState({ state: "idle" }); // idle | sending | reading | error
  const [patient, setPatient] = useState(EMPTY_PATIENT);
  const [mode, setMode] = useState("multi_agent");
  const [run, setRun] = useState(null); // {sessionId, mode, events, status, detail, error}
  const closeStream = useRef(null);

  const refreshHealth = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setHealthError("");
      })
      .catch((e) => setHealthError(e.message));
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = setInterval(refreshHealth, 30000);
    return () => clearInterval(timer);
  }, [refreshHealth]);

  useEffect(() => () => closeStream.current?.(), []);

  async function handleFiles(files) {
    setRun(null);
    setUpload({ state: "sending", progress: 0, count: files.length });
    try {
      const summary = await uploadFiles(files, (progress) =>
        setUpload(progress >= 1 ? { state: "reading", count: files.length } : { state: "sending", progress, count: files.length }),
      );
      setSource({ kind: "upload", upload: summary });
      setUpload({ state: "idle" });
    } catch (e) {
      setUpload({ state: "error", message: e.message });
    }
  }

  function reset() {
    closeStream.current?.();
    setSource(null);
    setRun(null);
    setUpload({ state: "idle" });
  }

  async function analyse() {
    const studyId = source.kind === "upload" ? source.upload.upload_id : source.studyId;
    const age = patient.age === "" ? null : Number(patient.age);
    const body = { ...patient, age };
    closeStream.current?.();
    setRun({ mode, events: [], status: "pending", detail: null, error: "" });
    try {
      const session = await startSession({ studyId, patient: body, mode });
      setRun((r) => ({ ...r, sessionId: session.session_id, status: session.status }));
      closeStream.current = streamEvents(session.session_id, {
        onEvent: (event) =>
          setRun((r) => {
            if (r.events.some((e) => e.seq === event.seq)) return r;
            const status = event.kind === "status" && event.data?.status ? event.data.status : r.status;
            return { ...r, events: [...r.events, event], status };
          }),
        onEnd: async () => {
          try {
            const detail = await getSession(session.session_id);
            setRun((r) => ({ ...r, detail, status: detail.status }));
          } catch (e) {
            setRun((r) => ({ ...r, error: e.message }));
          }
        },
        onError: () => setRun((r) => (r.detail ? r : { ...r, reconnecting: true })),
      });
    } catch (e) {
      setRun((r) => ({ ...r, status: "failed", error: e.message }));
    }
  }

  const running = run && !run.detail && !run.error;
  const ready = Boolean(source) && !running;

  return (
    <div className="app">
      <Header health={health} error={healthError} onRetry={refreshHealth} />
      <main className="layout">
        <section className="scan-column" aria-label="Scan">
          {source?.kind === "upload" ? (
            <ScanViewer upload={source.upload} onReplace={reset} disabled={running} />
          ) : source?.kind === "dataset" ? (
            <div className="panel dataset-chosen">
              <h2>Dataset study</h2>
              <p className="study-id">{source.studyId}</p>
              <p className="muted">Scans from the dataset have no preview here; they are scored from the Model Service's own copy.</p>
              <button className="link-button" onClick={reset} disabled={running}>Choose a different scan</button>
            </div>
          ) : (
            <ScanInput
              upload={upload}
              onFiles={handleFiles}
              onDatasetStudy={(studyId) => setSource({ kind: "dataset", studyId })}
              health={health}
            />
          )}

          <PatientForm value={patient} onChange={setPatient} disabled={running} />

          <div className="panel run-bar">
            <fieldset className="mode" disabled={running}>
              <legend>How to analyse</legend>
              <label className={mode === "multi_agent" ? "selected" : ""}>
                <input type="radio" name="mode" value="multi_agent" checked={mode === "multi_agent"} onChange={() => setMode("multi_agent")} />
                <span>
                  <strong>Agent team</strong>
                  <small>Diagnostic, recommendation and coordinator agents check each other and cite sources.</small>
                </span>
              </label>
              <label className={mode === "baseline" ? "selected" : ""}>
                <input type="radio" name="mode" value="baseline" checked={mode === "baseline"} onChange={() => setMode("baseline")} />
                <span>
                  <strong>Single call (baseline)</strong>
                  <small>One model call, no tools or checks. Faster; used for comparison.</small>
                </span>
              </label>
            </fieldset>
            <button className="primary" onClick={analyse} disabled={!ready}>
              {running ? "Analysing…" : run?.detail ? "Analyse again" : "Analyse scan"}
            </button>
            {!source && <p className="hint">Add a scan first.</p>}
          </div>
        </section>

        <section className="analysis-column" aria-label="Analysis" aria-live="polite">
          <AnalysisPanel run={run} health={health} hasScan={Boolean(source)} />
        </section>
      </main>
    </div>
  );
}
