// Backend API calls. Same origin in production (served by FastAPI), proxied by Vite in development.

async function request(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!res.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

export const getHealth = () => request("/health");
export const getStudies = () => request("/studies");
export const getSession = (id) => request(`/sessions/${id}`);
export const getReport = (id) => request(`/sessions/${id}/report`);
export const getUpload = (id) => request(`/uploads/${id}`);
export const sliceUrl = (id, plane, index) => `/uploads/${id}/slices/${plane}/${index}.png`;

export function startSession({ studyId, patient, mode }) {
  const path = mode === "baseline" ? "/baseline" : "/sessions";
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ study_id: studyId, patient_metadata: patient }),
  });
}

// XHR rather than fetch so the user sees upload progress on large studies.
export function uploadFiles(files, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const { file, name } of files) form.append("files", file, name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/uploads");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.upload.onload = () => onProgress(1);
    xhr.onload = () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error page */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Could not reach the backend. Is it running on port 8000?"));
    xhr.send(form);
  });
}

// Live trace over Server-Sent Events; returns a function that closes the stream.
export function streamEvents(sessionId, { onEvent, onEnd, onError }) {
  const source = new EventSource(`/sessions/${sessionId}/stream`);
  source.addEventListener("agent_event", (e) => onEvent(JSON.parse(e.data)));
  source.addEventListener("end", (e) => {
    source.close();
    onEnd(JSON.parse(e.data));
  });
  source.onerror = () => onError?.();
  return () => source.close();
}