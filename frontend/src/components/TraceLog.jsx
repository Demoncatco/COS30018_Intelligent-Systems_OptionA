// Every step the agents took, newest last. Open while running, collapsed once the report is in.
import { useEffect, useRef } from "react";

const AGENT_NAME = {
  coordinator: "Coordinator", diagnostic: "Diagnostic", recommendation: "Recommendation",
  model_service: "Model", baseline: "Baseline",
};

export default function TraceLog({ events, live = false, sessionId }) {
  const end = useRef(null);
  useEffect(() => {
    if (live) end.current?.scrollIntoView({ block: "nearest" });
  }, [events.length, live]);

  if (!events.length && !live) return null;
  const start = events[0]?.timestamp ?? 0;

  return (
    <details className="trace" open={live}>
      <summary>
        Agent trace <span className="muted">({events.length} steps{sessionId ? `, session ${sessionId}` : ""})</span>
      </summary>
      <ol className="trace-list">
        {events.map((e) => (
          <li key={e.seq} className={`trace-item ${e.agent} ${e.kind}`}>
            <span className="t">{(e.timestamp - start).toFixed(1)}s</span>
            <span className="who">{AGENT_NAME[e.agent] ?? e.agent}</span>
            <span className="what">
              {e.message}
              {e.latency_ms != null && e.tokens_out != null && (
                <span className="muted small"> ({Math.round(e.latency_ms)} ms, {(e.tokens_in ?? 0) + e.tokens_out} tokens)</span>
              )}
            </span>
          </li>
        ))}
        <li ref={end} aria-hidden="true" className="trace-end" />
      </ol>
    </details>
  );
}