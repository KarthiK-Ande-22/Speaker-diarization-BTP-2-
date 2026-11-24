import type { ModelChoicesInfo, Models, RecordingDetail, RecordingSummary, RunningStatus } from "./types";

async function json<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error((body as { error?: string }).error ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export const getStatus = () =>
  fetch("/api/status").then((r) => json<{ running: RunningStatus | null }>(r)).then((body) => body.running);

export const listRecordings = () => fetch("/api/recordings").then((r) => json<RecordingSummary[]>(r));

export const getRecording = (id: string) => fetch(`/api/recordings/${id}`).then((r) => json<RecordingDetail>(r));

export const runRemaining = (id: string) =>
  fetch(`/api/recordings/${id}/run`, { method: "POST" }).then((r) => json<{ started: boolean }>(r));

export const getModelChoices = () => fetch("/api/models").then((r) => json<ModelChoicesInfo>(r));

/** Analyse the same file again with other models; returns the new recording's id. */
export const rerunWith = (id: string, models: Models, keyterms: string) =>
  fetch(`/api/recordings/${id}/rerun`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models, keyterms }),
  })
    .then((r) => json<{ id: string }>(r))
    .then((body) => body.id);

export const getLog = (id: string, stage: string) =>
  fetch(`/api/recordings/${id}/log/${stage}`).then((r) => r.text());

export const fileUrl = (id: string, kind: "audio" | "clean" | "people") => `/api/recordings/${id}/file/${kind}`;

/** Upload with progress (fetch cannot report upload progress). */
export function uploadRecording(form: FormData, onProgress: (fraction: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/recordings");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      let body: { id?: string; error?: string } = {};
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // not JSON: handled below
      }
      if (request.status < 300 && body.id) resolve(body.id);
      else reject(new Error(body.error ?? `Upload failed (${request.status})`));
    };
    request.onerror = () => reject(new Error("Upload failed: the server could not be reached."));
    request.send(form);
  });
}
