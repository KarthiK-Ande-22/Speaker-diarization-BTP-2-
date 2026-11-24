import { FileCheck2, Upload } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";
import { getModelChoices, uploadRecording } from "../api";
import { percent } from "../format";
import type { ModelChoicesInfo, ModelPart, Models } from "../types";
import ModelChoices from "./ModelChoices";

/** The home page: one large area to choose a recording and start the pipeline. */
export default function UploadCard({ onStarted }: { onStarted: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [language, setLanguage] = useState("");
  const [choices, setChoices] = useState<ModelChoicesInfo | null>(null);
  const [models, setModels] = useState<Models | null>(null);
  const [keyterms, setKeyterms] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getModelChoices()
      .then((info) => {
        setChoices(info);
        setModels(info.defaults);
      })
      .catch((err: Error) => setError(err.message));
  }, []);
  const cloud =
    !!choices &&
    !!models &&
    (Object.keys(models) as ModelPart[]).some((part) => choices.models[part].options.find((o) => o.key === models[part])?.cloud);

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Choose a video or audio file first.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    if (start) form.append("start", start);
    if (end) form.append("end", end);
    if (language) form.append("language", language);
    if (models) form.append("models", JSON.stringify(models));
    if (keyterms) form.append("keyterms", keyterms);
    setError("");
    setProgress(0);
    try {
      onStarted(await uploadRecording(form, setProgress));
    } catch (err) {
      setError((err as Error).message);
      setProgress(null);
    }
  }

  return (
    <section className="upload-hero">
      <div className="upload-head">
        <h1>Analyse a consultation</h1>
        <p>
          Upload a recording of a doctor and a patient. The pipeline finds who spoke when, extracts speech,
          medical-term and video features, and decides who is the doctor.
        </p>
      </div>
      <form className="card upload" onSubmit={submit}>
        <div
          className={`dropzone${dragging ? " dragging" : ""}${file ? " chosen" : ""}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        >
          <input ref={input} type="file" accept="video/*,audio/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <span className="dropzone-icon">{file ? <FileCheck2 size={28} strokeWidth={1.75} /> : <Upload size={28} strokeWidth={1.75} />}</span>
          {file ? (
            <>
              <strong className="dropzone-title">{file.name}</strong>
              <span className="muted">{(file.size / 1e6).toFixed(1)} MB · click to choose another file</span>
            </>
          ) : (
            <>
              <strong className="dropzone-title">Drop a video or audio file here</strong>
              <span className="muted">or click to choose one · mp4, mov, mp3, wav, m4a</span>
            </>
          )}
        </div>

        <div className="options">
          <label>
            Start at (seconds)
            <input type="number" min="0" step="0.1" placeholder="from the beginning" value={start} onChange={(e) => setStart(e.target.value)} />
          </label>
          <label>
            End at (seconds)
            <input type="number" min="0" step="0.1" placeholder="to the end" value={end} onChange={(e) => setEnd(e.target.value)} />
          </label>
          <label>
            Language
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="">Detect automatically</option>
              <option value="en">English</option>
              <option value="te">Telugu</option>
            </select>
          </label>
        </div>

        {choices && models && (
          <ModelChoices info={choices} models={models} keyterms={keyterms} onModels={setModels} onKeyterms={setKeyterms} />
        )}

        {error && <p className="error">{error}</p>}
        <button className="button button-block" type="submit" disabled={progress !== null}>
          {progress === null ? "Start analysis" : `Uploading ${percent(progress)}`}
        </button>
        <p className="upload-note">
          {cloud
            ? "Runs on this computer except the ElevenLabs models chosen above. The last stage sends the extracted features to Sarvam."
            : "Runs on this computer. Only the last stage sends the extracted features (never the recording) to Sarvam."}
        </p>
      </form>
    </section>
  );
}
