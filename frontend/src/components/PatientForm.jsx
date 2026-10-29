// Optional patient details; the Recommendation Agent uses them to tailor advice.
export default function PatientForm({ value, onChange, disabled }) {
  const set = (key) => (e) => onChange({ ...value, [key]: e.target.value });
  return (
    <details className="panel patient">
      <summary>
        <h2>About the patient</h2>
        <span className="muted">Optional. Helps tailor the advice.</span>
      </summary>
      <fieldset disabled={disabled} className="patient-grid">
        <label>
          Age
          <input type="number" min="0" max="120" inputMode="numeric" value={value.age} onChange={set("age")} />
        </label>
        <label>
          Sex
          <select value={value.sex} onChange={set("sex")}>
            <option value="unknown">Prefer not to say</option>
            <option value="female">Female</option>
            <option value="male">Male</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label className="wide">
          Symptoms
          <textarea rows="2" maxLength="1000" value={value.symptoms} onChange={set("symptoms")}
            placeholder="e.g. pain on the inner side, swelling, knee gives way" />
        </label>
        <label className="wide">
          How it happened
          <input maxLength="500" value={value.injury_mechanism} onChange={set("injury_mechanism")}
            placeholder="e.g. twisted while landing during netball" />
        </label>
      </fieldset>
    </details>
  );
}