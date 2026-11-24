import { Pause, Play } from "lucide-react";
import { useEffect, useState, type MouseEvent, type RefObject } from "react";
import { clock } from "../format";

/** A small player: play/pause, a progress bar you can click, and the time.
 *  The <audio> element is shared through `audio` so the timeline and transcript can jump to a moment. */
export default function AudioPlayer({ src, audio }: { src: string; audio: RefObject<HTMLAudioElement> }) {
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [length, setLength] = useState(0);

  useEffect(() => {
    const element = audio.current;
    if (!element) return;
    const update = () => {
      setTime(element.currentTime);
      setLength(Number.isFinite(element.duration) ? element.duration : 0);
      setPlaying(!element.paused);
    };
    const events = ["timeupdate", "play", "pause", "loadedmetadata", "ended"];
    events.forEach((name) => element.addEventListener(name, update));
    return () => events.forEach((name) => element.removeEventListener(name, update));
  }, [audio]);

  const toggle = () => {
    const element = audio.current;
    if (!element) return;
    if (element.paused) void element.play();
    else element.pause();
  };
  const seek = (event: MouseEvent<HTMLDivElement>) => {
    const element = audio.current;
    if (!element || !length) return;
    const box = event.currentTarget.getBoundingClientRect();
    element.currentTime = ((event.clientX - box.left) / box.width) * length;
  };

  return (
    <div className="player">
      <audio ref={audio} src={src} preload="metadata" />
      <button className="player-button" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
      </button>
      <div className="player-track" onClick={seek}>
        <span style={{ width: length ? `${(time / length) * 100}%` : "0%" }} />
      </div>
      <span className="player-time">
        {clock(time)} / {clock(length)}
      </span>
    </div>
  );
}
