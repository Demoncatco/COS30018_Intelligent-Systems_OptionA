// Live status of each agent, worked out from the trace events as they stream in.

const LANES = {
  multi_agent: [
    { agent: "coordinator", name: "Coordinator", role: "Plans the run and settles disagreements" },
    { agent: "diagnostic", name: "Diagnostic agent", role: "Reads the scan with the detection model", tool: "run_diagnostic" },
    { agent: "recommendation", name: "Recommendation agent", role: "Checks trusted sources and writes advice", tool: "run_recommendation" },
  ],
  baseline: [
    { agent: "model_service", name: "Detection model", role: "Scores the scan" },
    { agent: "baseline", name: "Single model call", role: "Writes the report in one step" },
  ],
};

function laneState(lane, events, finished) {
  const own = events.filter((e) => e.agent === lane.agent);
  if (own.some((e) => e.kind === "error") && own.at(-1)?.kind === "error") return "error";
  if (lane.agent === "coordinator" || lane.agent === "baseline") return finished ? "done" : events.length ? "working" : "waiting";
  if (lane.tool) {
    const calls = events.filter((e) => e.agent === "coordinator" && e.kind === "tool_call" && e.data?.tool === lane.tool);
    if (!calls.length) return "waiting";
    const lastCall = calls.at(-1).seq;
    const answered = own.some((e) => e.kind === "tool_result" && e.seq > lastCall);
    return answered ? "done" : "working";
  }
  return own.some((e) => e.kind === "tool_result") ? "done" : own.length ? "working" : "waiting";
}

const STATE_TEXT = { waiting: "Waiting", working: "Working", done: "Done", error: "Problem" };

export default function AgentProgress({ run }) {
  const lanes = LANES[run.mode] ?? LANES.multi_agent;
  const rechecks = run.events.filter((e) => e.agent === "coordinator" && e.kind === "tool_call" && e.data?.tool === "run_diagnostic").length - 1;
  const latest = run.events.at(-1);

  return (
    <div className="progress">
      <h2>Analysing</h2>
      <ol className="lanes">
        {lanes.map((lane) => {
          const state = laneState(lane, run.events, false);
          const last = run.events.filter((e) => e.agent === lane.agent).at(-1);
          return (
            <li key={lane.agent} className={`lane ${state}`}>
              <span className="lane-mark" aria-hidden="true" />
              <div>
                <strong>{lane.name}</strong>
                <span className="lane-state">{STATE_TEXT[state]}</span>
                <p className="muted small">{state === "waiting" || !last ? lane.role : last.message}</p>
              </div>
            </li>
          );
        })}
      </ol>
      {rechecks > 0 && <p className="muted small">The recommendation agent asked for {rechecks} recheck{rechecks > 1 ? "s" : ""} of the scan.</p>}
      {run.reconnecting && <p className="muted small">Reconnecting to the live trace…</p>}
      {!latest && <p className="muted small">Waiting for a free worker…</p>}
    </div>
  );
}