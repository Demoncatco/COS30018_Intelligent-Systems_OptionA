// Top bar with the system's readiness: how scans are scored and whether the local LLMs are up.
export default function Header({ health, error, onRetry }) {
  let scoring = { text: "Checking…", tone: "idle" };
  let agents = { text: "Checking…", tone: "idle" };

  if (error) {
    scoring = { text: "Backend offline", tone: "bad" };
    agents = null;
  } else if (health) {
    const ms = health.model_service ?? {};
    const uploads = ms.uploads;
    if (health.scoring?.source === "local") scoring = { text: "Trained model (Diagnostic Agent)", tone: "good" };
    else if (ms.status === "unreachable") scoring = { text: "Model Service offline", tone: "bad" };
    else if (uploads === "stub" || health.config?.model_mode === "stub") scoring = { text: "Demo scores (stub model)", tone: "warn" };
    else if (uploads === "model") scoring = { text: `Trained model (${ms.model_version ?? "checkpoint"})`, tone: "good" };
    else if (uploads === "unavailable") scoring = { text: "Uploads need a checkpoint", tone: "warn" };
    else scoring = { text: ms.status === "ok" ? "Model Service ready" : "Model Service not ready", tone: ms.status === "ok" ? "good" : "bad" };

    const ollama = health.ollama ?? {};
    if (ollama.ready) agents = { text: "Language models ready", tone: "good" };
    else if (ollama.ollama === "ok") agents = { text: "Language models not pulled", tone: "bad" };
    else agents = { text: "Ollama offline", tone: "bad" };
  }

  return (
    <header className="topbar">
      <div className="brand">
        <svg viewBox="0 0 32 32" aria-hidden="true" className="logo">
          <rect width="32" height="32" rx="7" />
          <path d="M11 5v9l4 4v9M21 5v9l-4 4" />
        </svg>
        <div>
          <h1>Knee MRI Reader</h1>
          <p>Checks a knee MRI for 12 common problems and explains what they may mean.</p>
        </div>
      </div>
      <ul className="status-list" aria-label="System status">
        <li className={`status ${scoring.tone}`} title={error || health?.scoring?.detail || health?.model_service?.detail || ""}>
          <span className="dot" aria-hidden="true" />
          {scoring.text}
        </li>
        {agents && (
          <li className={`status ${agents.tone}`} title={health?.ollama?.error || ""}>
            <span className="dot" aria-hidden="true" />
            {agents.text}
          </li>
        )}
        {error && (
          <li>
            <button className="link-button" onClick={onRetry}>Check again</button>
          </li>
        )}
      </ul>
    </header>
  );
}