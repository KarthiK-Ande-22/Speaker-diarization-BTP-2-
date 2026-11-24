import { duration } from "../format";
import type { Stage } from "../types";
import { Elapsed, StatusIcon } from "./ui";

function subtitle(stage: Stage) {
  if (stage.status === "running") return <>running · <Elapsed since={stage.started} /></>;
  if (stage.status === "done") return stage.seconds !== undefined ? `done · ${duration(stage.seconds)}` : "done";
  if (stage.status === "skipped") return `skipped${stage.note ? ` · ${stage.note}` : ""}`;
  if (stage.status === "failed") return "failed · see the log";
  return "waiting";
}

/** The pipeline stages on the left; click one to see its log or its output on the right. */
export default function StageNav({ stages, selected, onSelect }: { stages: Stage[]; selected: string; onSelect: (key: string) => void }) {
  const done = stages.filter((s) => s.status === "done").length;
  return (
    <nav className="stage-nav">
      <div className="side-heading">
        <span>Pipeline</span>
        <span className="muted">
          {done}/{stages.length} done
        </span>
      </div>
      {stages.map((stage, index) => (
        <button
          key={stage.key}
          className={`stage-item stage-${stage.status}${stage.key === selected ? " selected" : ""}`}
          onClick={() => onSelect(stage.key)}
        >
          <span className="stage-num">{index + 1}</span>
          <span className="stage-item-text">
            <span className="stage-item-title">{stage.title}</span>
            <span className="stage-item-sub">{subtitle(stage)}</span>
          </span>
          <StatusIcon status={stage.status} />
        </button>
      ))}
    </nav>
  );
}
