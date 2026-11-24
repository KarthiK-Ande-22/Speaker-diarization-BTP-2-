import { AudioLines, ChevronRight, Video } from "lucide-react";
import { useEffect, useState } from "react";
import { listRecordings } from "../api";
import { dateTime, duration, percent } from "../format";
import type { RecordingSummary } from "../types";
import { SpeakerTag, StatusBadge } from "./ui";

function RecordingRow({ item }: { item: RecordingSummary }) {
  const decided = item.doctor === "A" || item.doctor === "B";
  return (
    <a className="rec-row" href={`#/recording/${item.id}`}>
      <span className="rec-row-icon">{item.has_video ? <Video size={18} /> : <AudioLines size={18} />}</span>
      <span className="rec-row-main">
        <span className="rec-row-name">{item.name}</span>
        <span className="rec-row-meta">
          {dateTime(item.created)}
          {item.audio_s !== null && ` · ${duration(item.audio_s)}`}
          {item.models && ` · ${item.models}`}
          {item.status !== "done" && ` · ${item.stages_done} of ${item.stages} stages done`}
        </span>
      </span>
      <span className="rec-row-result">
        {decided && (
          <>
            Doctor <SpeakerTag speaker={item.doctor!} /> <span className="muted">{percent(item.confidence ?? 0)}</span>
          </>
        )}
      </span>
      <StatusBadge status={item.status} />
      <ChevronRight size={18} className="rec-row-arrow" />
    </a>
  );
}

/** Recordings as a simple list (newest first); `limit` shows only the most recent ones. */
export default function RecordingList({ limit }: { limit?: number }) {
  const [items, setItems] = useState<RecordingSummary[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const load = () =>
      listRecordings()
        .then((list) => {
          if (stopped) return;
          setItems(list);
          if (list.some((item) => item.status === "running")) timer = window.setTimeout(load, 4000);
        })
        .catch((err: Error) => !stopped && setError(err.message));
    load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!items) return <p className="muted">Loading…</p>;
  if (items.length === 0) return <p className="empty-card">No recordings analysed yet.</p>;
  return (
    <div className="rec-list">
      {items.slice(0, limit ?? items.length).map((item) => (
        <RecordingRow key={item.id} item={item} />
      ))}
    </div>
  );
}
