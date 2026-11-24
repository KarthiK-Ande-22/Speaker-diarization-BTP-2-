import { Cloud } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";
import { fileUrl, getModelChoices, getRecording, rerunWith, runRemaining } from "../api";
import { dateTime, percent } from "../format";
import type { ModelChoicesInfo, Models, RecordingDetail, Roles, Speaker } from "../types";
import AudioPlayer from "./AudioPlayer";
import ModelChoices from "./ModelChoices";
import StageNav from "./StageNav";
import StagePanel from "./StagePanel";
import { SpeakerTag } from "./ui";

/** `chosen` = the stage in the address (the one you clicked); without it the running stage is shown. */
export default function RecordingView({ id, chosen }: { id: string; chosen: string | null }) {
  const [data, setData] = useState<RecordingDetail | null>(null);
  const [error, setError] = useState("");
  const [restart, setRestart] = useState(0);
  const [rerun, setRerun] = useState<{ info: ModelChoicesInfo; models: Models; keyterms: string } | null>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const choose = (stage: string | null) => (window.location.hash = `#/recording/${id}${stage ? `/${stage}` : ""}`);

  // Load the recording; while the pipeline is running, ask again every 2 seconds.
  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const load = () =>
      getRecording(id)
        .then((detail) => {
          if (stopped) return;
          setData(detail);
          if (detail.job.running) timer = window.setTimeout(load, 2000);
        })
        .catch((err: Error) => !stopped && setError(err.message));
    load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [id, restart]);
  useEffect(() => setRerun(null), [id]);

  if (!data) return <p className={error ? "error page-message" : "muted page-message"}>{error || "Loading…"}</p>;

  const { job, results } = data;
  const final = results.decision?.final;
  const doctor = final && final.decision !== "-" ? final.decision : null;
  const patient: Speaker | null = doctor ? (doctor === "A" ? "B" : "A") : null;
  const roles: Roles = doctor && patient ? { [doctor]: "Doctor", [patient]: "Patient" } : {};

  const running = job.stages.find((s) => s.status === "running");
  const lastDone = [...job.stages].reverse().find((s) => s.status === "done");
  const known = chosen && job.stages.some((s) => s.key === chosen) ? chosen : null;
  const selectedKey = known ?? running?.key ?? lastDone?.key ?? job.stages[0].key;
  const number = job.stages.findIndex((s) => s.key === selectedKey) + 1;
  const canRun = !job.running && job.stages.some((s) => s.status === "waiting" || s.status === "failed");

  const play = (seconds: number) => {
    if (!audio.current) return;
    audio.current.currentTime = seconds;
    void audio.current.play();
  };
  const openRerun = () =>
    getModelChoices()
      .then((info) => setRerun({ info, models: job.models, keyterms: job.keyterms.join(", ") }))
      .catch((err: Error) => setError(err.message));
  const startRerun = () =>
    rerun &&
    rerunWith(id, rerun.models, rerun.keyterms)
      .then((newId) => (window.location.hash = `#/recording/${newId}`))
      .catch((err: Error) => setError(err.message));
  const unchanged = rerun && JSON.stringify(rerun.models) === JSON.stringify(job.models) && rerun.keyterms === job.keyterms.join(", ");

  const part =
    job.start !== null || job.end !== null ? `${job.start ?? 0}–${job.end !== null ? `${job.end} s` : "end"}` : "whole file";

  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="side-block">
          <h1 className="rec-title" title={job.name}>
            {job.name.replace(/\.[a-z0-9]+$/i, "")}
          </h1>
          <p className="muted small">
            {dateTime(job.created)} · {part} · {job.has_video ? "video" : "audio"}
          </p>
        </div>

        {doctor && patient && final && (
          <dl className="verdict">
            <dt>Doctor</dt>
            <dd>
              <SpeakerTag speaker={doctor} />
            </dd>
            <dt>Patient</dt>
            <dd>
              <SpeakerTag speaker={patient} />
            </dd>
            <dt>Confidence</dt>
            <dd>
              <span className="mono">{percent(final.confidence)}</span> <span className="muted small">({final.agreement} agree)</span>
            </dd>
          </dl>
        )}

        {results.diarization?.audio && <AudioPlayer src={fileUrl(id, "audio")} audio={audio} />}

        <StageNav stages={job.stages} selected={selectedKey} onSelect={choose} />

        {canRun && (
          <button
            className="button button-block"
            onClick={() =>
              runRemaining(id)
                .then(() => {
                  choose(null);
                  setRestart((n) => n + 1);
                })
                .catch((err: Error) => setError(err.message))
            }
          >
            Run the remaining stages
          </button>
        )}

        <div className="side-block">
          <div className="side-heading">Models</div>
          <dl className="verdict models-used">
            {job.models_used.map((m) => (
              <Fragment key={m.part}>
                <dt>{m.title}</dt>
                <dd>
                  {m.label} {m.cloud && <Cloud size={13} className="cloud-icon" aria-label="cloud" />}
                </dd>
              </Fragment>
            ))}
            {job.keyterms.length > 0 && (
              <>
                <dt>Key terms</dt>
                <dd>{job.keyterms.join(", ")}</dd>
              </>
            )}
          </dl>
        </div>
        {rerun ? (
          <div className="rerun">
            <ModelChoices
              compact
              info={rerun.info}
              models={rerun.models}
              keyterms={rerun.keyterms}
              onModels={(models) => setRerun({ ...rerun, models })}
              onKeyterms={(keyterms) => setRerun({ ...rerun, keyterms })}
            />
            <p className="muted small">
              A new recording is made from the same file. Saved results that still apply (pyannote, Whisper, Scribe) are reused.
            </p>
            <div className="rerun-buttons">
              <button className="button" disabled={!!unchanged} onClick={startRerun} title={unchanged ? "Change a model first" : ""}>
                Start
              </button>
              <button className="link" onClick={() => setRerun(null)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button className="button button-outline button-block" disabled={job.running} onClick={openRerun}>
            Run again with other models
          </button>
        )}
        {error && <p className="error">{error}</p>}
      </aside>

      <StagePanel
        key={selectedKey}
        id={id}
        stage={job.stages[number - 1]}
        number={number}
        total={job.stages.length}
        results={results}
        roles={roles}
        onPlay={play}
      />
    </div>
  );
}
