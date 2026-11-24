// The output of each pipeline stage, shown on the right of the recording page.
import { Play } from "lucide-react";
import { useRef } from "react";
import { fileUrl } from "../api";
import { clock, duration, percent, value } from "../format";
import type { Results, Roles, Segment, Speaker } from "../types";
import AudioPlayer from "./AudioPlayer";
import FeatureTable from "./FeatureTable";
import { SpeakerTag, Stat } from "./ui";

const SPEAKERS: Speaker[] = ["A", "B"];

/** Which part of the results belongs to each stage. */
const RESULT_OF: Record<string, keyof Results> = {
  clean: "clean",
  diarization: "diarization",
  speech: "speech",
  transcription: "transcript",
  medical: "medical",
  video: "video",
  decision: "decision",
};

export const hasOutput = (stageKey: string, results: Results) => results[RESULT_OF[stageKey]] !== undefined;

interface OutputProps {
  stageKey: string;
  id: string;
  results: Results;
  roles: Roles;
  onPlay: (seconds: number) => void;
}

export function StageOutput({ stageKey, id, results, roles, onPlay }: OutputProps) {
  switch (stageKey) {
    case "clean":
      return <CleanAudio id={id} data={results.clean!} />;
    case "diarization":
      return <Diarization data={results.diarization!} roles={roles} onPlay={onPlay} />;
    case "speech":
      return <FeatureTable rows={results.speech!.features} roles={roles} />;
    case "transcription":
      return <Transcript data={results.transcript!} roles={roles} canPlay={!!results.diarization?.audio} onPlay={onPlay} />;
    case "medical":
      return <Medical data={results.medical!} roles={roles} />;
    case "video":
      return <Video id={id} data={results.video!} roles={roles} />;
    case "decision":
      return <Decision data={results.decision!} />;
    default:
      return null;
  }
}

// ---------------------------------------------------------------- clean audio (optional)

function CleanAudio({ id, data }: { id: string; data: NonNullable<Results["clean"]> }) {
  const audio = useRef<HTMLAudioElement>(null);
  return (
    <>
      <div className="stats">
        <Stat label="Original" value={`${data.original_s.toFixed(2)} s`} />
        <Stat label="Cleaned" value={`${data.cleaned_s.toFixed(2)} s`} note="must match the original so the times line up" />
      </div>
      <p className="muted small">
        Finding speakers and the transcript use this cleaned audio. Pitch, pauses and speed, and the player on the left, use the original.
      </p>
      <h3 className="subhead">Listen to the cleaned audio</h3>
      <AudioPlayer src={fileUrl(id, "clean")} audio={audio} />
    </>
  );
}

// ---------------------------------------------------------------- 1. who spoke when

function Timeline({ segments, total, roles, onPlay }: { segments: Segment[]; total: number; roles: Roles; onPlay: (t: number) => void }) {
  const step = total > 900 ? 180 : total > 300 ? 60 : 30;
  const ticks = Array.from({ length: Math.floor(total / step) + 1 }, (_, i) => i * step);
  return (
    <div className="timeline">
      {SPEAKERS.map((speaker) => (
        <div className="lane" key={speaker}>
          <div className="lane-label">
            <SpeakerTag speaker={speaker} roles={roles} />
          </div>
          <div className="lane-track">
            {segments
              .filter((s) => s.speaker === speaker)
              .map((s, i) => (
                <span
                  key={i}
                  className={`segment segment-${speaker.toLowerCase()}`}
                  style={{ left: `${(s.start / total) * 100}%`, width: `${Math.max(((s.end - s.start) / total) * 100, 0.25)}%` }}
                  title={`${clock(s.start)} – ${clock(s.end)} · click to listen`}
                  onClick={() => onPlay(s.start)}
                />
              ))}
          </div>
        </div>
      ))}
      <div className="lane">
        <div className="lane-label" />
        <div className="lane-track ticks">
          {ticks.map((t) => (
            <span key={t} style={{ left: `${(t / total) * 100}%` }}>
              {clock(t)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function Diarization({ data, roles, onPlay }: { data: NonNullable<Results["diarization"]>; roles: Roles; onPlay: (t: number) => void }) {
  const total = (data.talk_s.A ?? 0) + (data.talk_s.B ?? 0) || 1;
  return (
    <>
      <div className="stats">
        <Stat label="Analysed" value={duration(data.audio_s)} note={data.model ?? undefined} />
        {SPEAKERS.map((s) => (
          <Stat
            key={s}
            label={<SpeakerTag speaker={s} roles={roles} />}
            value={duration(data.talk_s[s] ?? 0)}
            note={`${percent((data.talk_s[s] ?? 0) / total)} of the talking · ${data.turns[s] ?? 0} turns`}
          />
        ))}
        <Stat label="Both talking at once" value={`${data.both_talking_s.toFixed(1)} s`} />
        {data.voice_recheck && (
          <Stat label="Voice re-check" value={`${data.voice_recheck.pieces_moved} moved`} note={`of ${data.voice_recheck.pieces_checked} pieces checked`} />
        )}
      </div>
      <h3 className="subhead">Timeline · click a block to listen</h3>
      <Timeline segments={data.segments} total={data.audio_s} roles={roles} onPlay={onPlay} />
    </>
  );
}

// ---------------------------------------------------------------- 3. transcript

function Transcript({ data, roles, canPlay, onPlay }: { data: NonNullable<Results["transcript"]>; roles: Roles; canPlay: boolean; onPlay: (t: number) => void }) {
  return (
    <>
      <p className="muted small">{data.length} turns · click a time to listen from there</p>
      <div className="transcript">
        {data.map((turn) => (
          <div key={turn.turn} className={`utterance utterance-${turn.speaker.toLowerCase()}`}>
            <div className="utterance-meta">
              <SpeakerTag speaker={turn.speaker} roles={roles} />
              {canPlay ? (
                <button className="link small play" onClick={() => onPlay(turn.start)} title="Listen from here">
                  <Play size={11} fill="currentColor" /> {clock(turn.start)}
                </button>
              ) : (
                <span className="muted small">{clock(turn.start)}</span>
              )}
            </div>
            <p>{turn.text}</p>
          </div>
        ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- 4. medical terms

function Medical({ data, roles }: { data: NonNullable<Results["medical"]>; roles: Roles }) {
  const groups = [
    { side: "doctor", label: "Treatment and test terms" },
    { side: "patient", label: "Symptom and history terms" },
  ];
  return (
    <>
      <div className="columns">
        {SPEAKERS.map((speaker) => (
          <div key={speaker} className="column">
            <SpeakerTag speaker={speaker} roles={roles} />
            {groups.map(({ side, label }) => {
              const terms = data.terms.filter((t) => t.speaker === speaker && t.side === side);
              return (
                <div key={side}>
                  <div className="muted small">{label}</div>
                  {terms.length ? (
                    <div className="chips">
                      {terms.map((t, i) => (
                        <span key={i} className="chip" title={t.question ? "said in a question" : undefined}>
                          {t.term}
                          <span className="chip-type">
                            {t.type}
                            {t.question ? " · asked" : ""}
                          </span>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className="muted small">none</div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <FeatureTable rows={data.features} roles={roles} />
    </>
  );
}

// ---------------------------------------------------------------- 5. video

function Video({ id, data, roles }: { id: string; data: NonNullable<Results["video"]>; roles: Roles }) {
  return (
    <>
      <div className="stats">
        <Stat label="Faces found" value={data.faces ?? "–"} />
        <Stat label="Face–voice match (on screen)" value={value(data.link.on_screen)} note="who is shown while each person talks" />
        <Stat label="Face–voice match (mouth)" value={value(data.link.mouth)} note="0.5 means no information" />
        <Stat label="Face–voice match (combined)" value={value(data.link.both)} />
      </div>
      {data.image && (
        <figure className="people">
          <img src={fileUrl(id, "people")} alt="Sample pictures of each person found in the video" />
          <figcaption className="muted small">Pictures of each person, labelled with the speaker whose voice they were matched to.</figcaption>
        </figure>
      )}
      <FeatureTable rows={data.features.filter((f) => f.feature !== "face_link_confidence")} roles={roles} />
      <p className="muted small">The image model's probabilities are not reliable on their own; compare the two speakers.</p>
    </>
  );
}

// ---------------------------------------------------------------- 6. decision

function Decision({ data }: { data: NonNullable<Results["decision"]> }) {
  const final = data.final;
  const doctor = final && final.decision !== "-" ? final.decision : null;
  return (
    <>
      {final && doctor ? (
        <>
          <table className="table summary-table">
            <tbody>
              <tr>
                <th>Doctor</th>
                <td>
                  <SpeakerTag speaker={doctor} />
                </td>
              </tr>
              <tr>
                <th>Patient</th>
                <td>
                  <SpeakerTag speaker={doctor === "A" ? "B" : "A"} />
                </td>
              </tr>
              <tr>
                <th>Confidence</th>
                <td>
                  <span className="mono strong">{percent(final.confidence)}</span>
                  <span className="bar">
                    <span style={{ width: percent(final.confidence) }} />
                  </span>
                  <span className="muted">
                    {final.agree} of {final.answers} answers agree,{" "}
                    {final.same_in_both_orders ? "the same in both speaker orders" : "but the answer changed with the speaker order"}
                  </span>
                </td>
              </tr>
              <tr>
                <th>Based on</th>
                <td>
                  {final.groups.join(", ")} <span className="muted">· {data.model}</span>
                </td>
              </tr>
            </tbody>
          </table>
          {final.reasons.length > 0 && (
            <section>
              <h3 className="subhead">Reasons given by the model</h3>
              <ol className="reasons">
                {final.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ol>
              <p className="muted small">These are the model's own words; it sometimes misreads a number, so check them against the feature tables.</p>
            </section>
          )}
        </>
      ) : (
        <p className="empty">The model could not decide.</p>
      )}
      <h3 className="subhead">What each feature group says on its own</h3>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Features given to the model</th>
              <th>Doctor</th>
              <th className="num">Confidence</th>
              <th className="num">Answers agreeing</th>
              <th>Same in both orders</th>
            </tr>
          </thead>
          <tbody>
            {data.combinations.map((c) => (
              <tr key={c.letters} className={c.letters === final?.letters ? "row-final" : undefined}>
                <td>{c.groups.join(" + ")}</td>
                <td>
                  <SpeakerTag speaker={c.decision} />
                </td>
                <td className="num">{percent(c.confidence)}</td>
                <td className="num">{c.agreement}</td>
                <td>{c.same_in_both_orders ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
