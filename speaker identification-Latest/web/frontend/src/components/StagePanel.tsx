import { useEffect, useRef, useState } from "react";
import { getLog } from "../api";
import { duration } from "../format";
import { STAGE_INFO } from "../stages";
import type { Results, Roles, Stage } from "../types";
import { StageOutput, hasOutput } from "./Sections";
import { Elapsed, StatusBadge } from "./ui";

/** The stage's log, refreshed every 2 seconds while the stage runs and kept scrolled to the end. */
function LogView({ id, stage, live }: { id: string; stage: string; live: boolean }) {
  const [text, setText] = useState("");
  const box = useRef<HTMLPreElement>(null);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const load = () =>
      getLog(id, stage).then((log) => {
        if (stopped) return;
        setText(log);
        if (live) timer = window.setTimeout(load, 2000);
      });
    load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [id, stage, live]);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [text]);

  return (
    <pre ref={box} className="log">
      {text || (live ? "Starting …" : "(empty)")}
    </pre>
  );
}

/** The right side of the screen: one stage, with its output (when done) or its log. */
export default function StagePanel(props: {
  id: string;
  stage: Stage;
  number: number;
  total: number;
  results: Results;
  roles: Roles;
  onPlay: (seconds: number) => void;
}) {
  const { id, stage, number, total, results, roles, onPlay } = props;
  // While a stage runs, its log is shown (any output on disk is from an earlier run).
  const ready = stage.status !== "running" && hasOutput(stage.key, results);
  const [tab, setTab] = useState<"output" | "log">(ready ? "output" : "log");
  const info = STAGE_INFO[stage.key];

  // A stage that finishes while you watch its log switches to its output.
  useEffect(() => {
    if (ready) setTab("output");
  }, [ready]);

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-content panel-header-row">
          <div>
            <h2>
              <span className="panel-num">{number}.</span> {stage.title}
            </h2>
            <div className="panel-sub">
              <StatusBadge status={stage.status} />
              {(stage.status === "running" || stage.seconds !== undefined) && (
                <span>· {stage.status === "running" ? <Elapsed since={stage.started} /> : duration(stage.seconds!)}</span>
              )}
              <span>
                · stage {number} of {total}
              </span>
            </div>
          </div>
          <div className="tabs">
            <button className={`tab${tab === "output" ? " active" : ""}`} disabled={!ready} onClick={() => setTab("output")}>
              Output
            </button>
            <button className={`tab${tab === "log" ? " active" : ""}`} disabled={stage.status === "waiting"} onClick={() => setTab("log")}>
              Log
            </button>
          </div>
        </div>
      </header>
      <div className="panel-body panel-content">
        {info && <p className="panel-description">{info.description}</p>}
        {stage.note && stage.status !== "done" && <p className="note">{stage.note}</p>}
        {stage.status === "waiting" ? (
          <p className="empty">This stage starts when the earlier stages have finished.</p>
        ) : tab === "output" && ready ? (
          <StageOutput stageKey={stage.key} id={id} results={results} roles={roles} onPlay={onPlay} />
        ) : stage.status === "skipped" && !stage.seconds ? (
          <p className="empty">This stage was skipped{stage.note ? `: ${stage.note}` : ""}.</p>
        ) : (
          <LogView id={id} stage={stage.key} live={stage.status === "running"} />
        )}
      </div>
    </section>
  );
}
