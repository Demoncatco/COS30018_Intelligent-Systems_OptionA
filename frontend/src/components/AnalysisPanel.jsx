// Right-hand side: what happens before, during and after an analysis run.
import AgentProgress from "./AgentProgress.jsx";
import Report from "./Report.jsx";
import TraceLog from "./TraceLog.jsx";

export default function AnalysisPanel({ run, health, hasScan }) {
  if (!run) {
    return (
      <div className="panel empty-state">
        <h2>Results</h2>
        <p>{hasScan ? "Press Analyse scan when you're ready." : "Add a knee MRI on the left to get started."}</p>
        <p className="muted">
          You'll see the likely problems ranked by probability, how sure the system is, what to do next, and the trusted
          sources the advice comes from. Each agent's steps show here live while it works.
        </p>
      </div>
    );
  }

  if (run.error && !run.sessionId) {
    return (
      <div className="panel">
        <h2>Results</h2>
        <p className="error-text" role="alert">The analysis could not start: {run.error}</p>
      </div>
    );
  }

  if (!run.detail) {
    return (
      <div className="panel">
        <AgentProgress run={run} />
        {run.error && <p className="error-text" role="alert">{run.error}</p>}
        <TraceLog events={run.events} live />
      </div>
    );
  }

  return (
    <div className="panel report-panel">
      <Report detail={run.detail} health={health} />
      <TraceLog events={run.events.length ? run.events : []} sessionId={run.sessionId} />
    </div>
  );
}