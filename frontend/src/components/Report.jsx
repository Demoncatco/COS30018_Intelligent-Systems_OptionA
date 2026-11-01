// The final report: the answer first (what the scan may show and how urgent it is), then the evidence behind it.
import { useState } from "react";
import { FINDING_INFO, PLANES, PLANE_SHORT, URGENCY, pct } from "../findings.js";

const STATUS_NOTE = {
  partial: "Part of the analysis did not finish, so this report is incomplete.",
  inconclusive: "The scan could not be read well enough for a reliable result.",
  failed: "The analysis failed and no report could be produced.",
};

// "Joint effusion" -> "joint effusion" mid-sentence; acronyms and names keep their capitals.
function phrase(label) {
  const first = label.split(" ")[0];
  return first === first.toUpperCase() || first.endsWith("'s") ? label : label[0].toLowerCase() + label.slice(1);
}

function joinNames(names) {
  if (names.length > 3) return `${names.slice(0, 3).join(", ")} and ${names.length - 3} more`;
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

export default function Report({ detail, health }) {
  const report = detail.final_report ?? {};
  const threshold = health?.config?.report_threshold ?? 0.5;
  const reported = report.findings ?? [];
  const status = detail.status;
  // Local Diagnostic Agent model means real scores, whatever mode the Model Service is in
  const local = health?.scoring?.source === "local";
  const stub = !local && (health?.config?.model_mode === "stub" || health?.model_service?.uploads === "stub");
  const urgency = URGENCY[report.urgency];
  const names = reported.map((f) => phrase(FINDING_INFO[f.name]?.label ?? f.name));

  return (
    <article className="report">
      <header className="report-head">
        <h2>{status === "failed" ? "No result" : "What the scan may show"}</h2>
        <p className="produced-by muted small">
          {detail.produced_by === "baseline" ? "Single model call (baseline)" : "Agent team"}
          {report.total_latency_ms > 0 && `, ${(report.total_latency_ms / 1000).toFixed(1)} s of model time`}
          {report.total_tokens > 0 && `, ${report.total_tokens.toLocaleString()} tokens`}
        </p>
      </header>

      {stub && (
        <p className="notice warn">
          Demo scores: the Model Service is in stub mode, so these numbers are random and are not a real reading of this scan.
        </p>
      )}
      {STATUS_NOTE[status] && <p className={`notice ${status === "failed" ? "bad" : "warn"}`}>{STATUS_NOTE[status]}</p>}
      {report.user_action && <p className="notice action"><strong>What to do: </strong>{report.user_action}</p>}

      {status !== "failed" && (
        <div className="headline">
          <p className="headline-text">
            {reported.length
              ? <>Signs of <strong>{joinNames(names)}</strong>.</>
              : detail.scores ? "No problem scored above the reporting threshold." : "No findings were produced."}
          </p>
          <div className="headline-meta">
            {urgency && <span className={`urgency ${urgency.tone}`}>{urgency.label}</span>}
            {report.overall_confidence != null && (
              <span className="confidence">System confidence {pct(report.overall_confidence)}</span>
            )}
          </div>
        </div>
      )}

      <FindingsChart scores={detail.scores} reported={reported} threshold={threshold} />

      {report.impression && (
        <section>
          <h3>First impression from the scan</h3>
          <p>{report.impression}</p>
        </section>
      )}

      {report.recommendation_text && (
        <section>
          <h3>What this may mean and next steps</h3>
          <RichText text={report.recommendation_text} />
        </section>
      )}

      {report.evidence_status === "no_credible_source_found" && (
        <p className="notice warn">No trusted medical source was found for this advice, so it is limited to general guidance.</p>
      )}

      {report.citations?.length > 0 && (
        <section>
          <h3>Sources</h3>
          <ol className="citations">
            {report.citations.map((c, i) => <li key={i}><Citation text={c} /></li>)}
          </ol>
        </section>
      )}

      {(report.disagreements?.length > 0 || report.recheck_rounds > 0) && (
        <section>
          <h3>How the agents checked each other</h3>
          {report.recheck_rounds > 0 && (
            <p>The scan was re-examined {report.recheck_rounds} time{report.recheck_rounds > 1 ? "s" : ""} after the recommendation agent raised concerns.</p>
          )}
          {report.disagreements?.length > 0 && (
            <>
              <p>These points were still unresolved at the end:</p>
              <ul>{report.disagreements.map((d, i) => <li key={i}>{d}</li>)}</ul>
            </>
          )}
        </section>
      )}

      {report.errors?.length > 0 && (
        <details className="errors">
          <summary>{report.errors.length} problem{report.errors.length > 1 ? "s" : ""} during the run</summary>
          <ul>{report.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </details>
      )}

      <p className="disclaimer">{report.disclaimer || "These results are not a diagnosis. Please consult a qualified clinician."}</p>
    </article>
  );
}

function FindingsChart({ scores, reported, threshold }) {
  const [showAll, setShowAll] = useState(false);
  const byName = Object.fromEntries(reported.map((f) => [f.name, f]));
  const rows = scores
    ? Object.entries(scores).map(([name, p]) => ({ name, p, ...byName[name] })).sort((a, b) => b.p - a.p)
    : reported.map((f) => ({ ...f, p: f.probability }));
  if (!rows.length) return null;

  const above = rows.filter((r) => r.p >= threshold);
  const visible = showAll ? rows : rows.slice(0, Math.max(above.length, 3));
  const hidden = rows.length - visible.length;

  return (
    <section className="chart" aria-label="Probability of each finding">
      <div className="chart-scale" aria-hidden="true">
        <span />
        <span className="scale-track"><span>0%</span><span>100%</span></span>
        <span />
      </div>
      <ul className="chart-rows">
        {visible.map((r) => {
          const info = FINDING_INFO[r.name] ?? { label: r.name, detail: "" };
          const on = r.p >= threshold;
          return (
            <li key={r.name} className={`chart-row ${on ? "on" : ""}`}>
              <div className="chart-label">
                <span className="finding-name">{info.label}</span>
                {r.low_confidence && <span className="tag">low confidence</span>}
                <span className="finding-detail muted small">{info.detail}</span>
              </div>
              <div className="chart-bar" role="img" aria-label={`${info.label}: ${pct(r.p)}`}>
                <span className="fill" style={{ width: `${r.p * 100}%` }} />
                <span className="threshold" style={{ left: `${threshold * 100}%` }} />
                {r.per_plane && PLANES.filter((pl) => pl in r.per_plane).map((pl) => (
                  <span key={pl} className="plane-mark" style={{ left: `${r.per_plane[pl] * 100}%` }}
                    title={`${pl}: ${pct(r.per_plane[pl])}`}>{PLANE_SHORT[pl]}</span>
                ))}
              </div>
              <span className="chart-value">{pct(r.p)}</span>
            </li>
          );
        })}
      </ul>
      <div className="chart-foot">
        <p className="muted small">
          The line on each bar marks the {pct(threshold)} reporting threshold.
          {reported.some((r) => r.per_plane && Object.keys(r.per_plane).length) &&
            " Letters show each view's own score (S side, C front, A top-down); spread-out letters mean the views disagree."}
        </p>
        {hidden > 0 && <button className="link-button" onClick={() => setShowAll(true)}>Show the other {hidden} findings</button>}
        {showAll && rows.length > 3 && <button className="link-button" onClick={() => setShowAll(false)}>Show fewer</button>}
      </div>
    </section>
  );
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function Citation({ text }) {
  const url = text.match(/https?:\/\/\S+/)?.[0]?.replace(/[).,]+$/, "");
  if (!url) return text;
  const host = hostOf(url);
  const label = text.replace(url, "").replace(/[\s\-–:|()]+$/, "").replace(/^[\s\-–:|()]+/, "");
  return (
    <a href={url} target="_blank" rel="noreferrer noopener">
      {label || host}
      {label && <span className="muted small"> {host}</span>}
    </a>
  );
}

// Paragraphs, "- " bullets and **bold**, which is all the agents produce.
function RichText({ text }) {
  const blocks = text.trim().split(/\n\s*\n/);
  const inline = (s) => s.split(/(\*\*[^*]+\*\*)/).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part);
  return blocks.map((block, i) => {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.every((l) => /^[-*•]\s/.test(l))) {
      return <ul key={i}>{lines.map((l, j) => <li key={j}>{inline(l.replace(/^[-*•]\s+/, ""))}</li>)}</ul>;
    }
    return <p key={i}>{lines.map((l, j) => <span key={j}>{j > 0 && <br />}{inline(l.replace(/^#+\s*/, ""))}</span>)}</p>;
  });
}