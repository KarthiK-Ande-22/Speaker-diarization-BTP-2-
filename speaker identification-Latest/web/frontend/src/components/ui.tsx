import { Check, Clock, LoaderCircle, Minus, TriangleAlert } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { duration } from "../format";
import type { Roles, Speaker, StageStatus } from "../types";

export function Card({ title, aside, children, className = "" }: { title?: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-header">
          {title && <h2>{title}</h2>}
          {aside && <div className="card-aside">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/** "Speaker A" (blue) or "Speaker B" (rust), with the role once it is known. */
export function SpeakerTag({ speaker, roles }: { speaker: string; roles?: Roles }) {
  if (speaker !== "A" && speaker !== "B") return <span className="muted">–</span>;
  const role = roles?.[speaker as Speaker];
  return (
    <span className={`speaker speaker-${speaker.toLowerCase()}`}>
      <span className="speaker-mark" />
      Speaker {speaker}
      {role && <span className="speaker-role">({role.toLowerCase()})</span>}
    </span>
  );
}

const STATUS_TEXT: Record<StageStatus | "partial", string> = {
  waiting: "waiting",
  running: "running",
  done: "done",
  failed: "failed",
  skipped: "skipped",
  partial: "not finished",
};

/** The status as a short coloured word. */
export function StatusBadge({ status }: { status: StageStatus | "partial" }) {
  return <span className={`status status-${status}`}>{STATUS_TEXT[status]}</span>;
}

/** Small icon for a stage's status: tick, spinner, warning, dash or clock. */
export function StatusIcon({ status, size = 15 }: { status: StageStatus; size?: number }) {
  const props = { size, strokeWidth: 2.2, className: `status-icon icon-${status}` };
  if (status === "done") return <Check {...props} />;
  if (status === "running") return <LoaderCircle {...props} />;
  if (status === "failed") return <TriangleAlert {...props} />;
  if (status === "skipped") return <Minus {...props} />;
  return <Clock {...props} />;
}

export function Stat({ label, value, note }: { label: ReactNode; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

/** Time since `since`, updated every second. */
export function Elapsed({ since }: { since?: string | null }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return since ? <>{duration(Math.max(0, (now - new Date(since).getTime()) / 1000))}</> : null;
}
