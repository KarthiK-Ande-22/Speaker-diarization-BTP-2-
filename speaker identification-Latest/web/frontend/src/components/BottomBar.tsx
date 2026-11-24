import { ArrowRight, CircleCheck, LoaderCircle, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { getStatus } from "../api";
import type { RunningStatus } from "../types";
import { Elapsed } from "./ui";

/** Always at the bottom of the screen: what the server is running now. */
export default function BottomBar({ currentId }: { currentId: string | null }) {
  const [running, setRunning] = useState<RunningStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const load = () =>
      getStatus()
        .then((status) => {
          if (stopped) return;
          setRunning(status);
          setOffline(false);
        })
        .catch(() => !stopped && setOffline(true))
        .finally(() => {
          if (!stopped) timer = window.setTimeout(load, 2000);
        });
    load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, []);

  if (offline) {
    return (
      <footer className="bottombar">
        <WifiOff size={15} className="bottombar-red" />
        The server is not reachable. Start it with <code>python web/server/app.py</code>
      </footer>
    );
  }
  if (!running) {
    return (
      <footer className="bottombar">
        <CircleCheck size={15} className="bottombar-quiet" />
        <span className="bottombar-quiet">Ready · no analysis running</span>
      </footer>
    );
  }
  return (
    <footer className="bottombar">
      <a className="bottombar-run" href={`#/recording/${running.id}`}>
        <LoaderCircle size={15} className="spin bottombar-red" />
        <strong>Running</strong>
        <span>{running.number ? `Stage ${running.number} of ${running.total} · ${running.title}` : "Starting the next stage"}</span>
        {running.started && (
          <span className="bottombar-quiet">
            <Elapsed since={running.started} />
          </span>
        )}
        <span className="bottombar-name">{running.name}</span>
        {currentId !== running.id && (
          <span className="bottombar-open">
            Open <ArrowRight size={14} />
          </span>
        )}
      </a>
    </footer>
  );
}
