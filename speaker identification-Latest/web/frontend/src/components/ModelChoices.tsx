import { Cloud } from "lucide-react";
import type { ModelChoicesInfo, ModelPart, Models } from "../types";

interface Props {
  info: ModelChoicesInfo;
  models: Models;
  keyterms: string;
  onModels: (models: Models) => void;
  onKeyterms: (text: string) => void;
  compact?: boolean; // one column (the recording page's sidebar)
}

/** One choice per part of the pipeline. The first option (our own pipeline, on this computer) is the default. */
export default function ModelChoices({ info, models, keyterms, onModels, onKeyterms, compact }: Props) {
  const parts = Object.keys(info.models) as ModelPart[];
  const option = (part: ModelPart) => info.models[part].options.find((o) => o.key === models[part]);
  const cloud = parts.filter((part) => option(part)?.cloud);
  const usesScribe = Object.values(models).includes("scribe");

  return (
    <div className="models">
      <div className="models-head">
        <strong>Models</strong>
        <span className="muted small">Default: our own pipeline, on this computer</span>
      </div>
      <div className={`options models-grid${compact ? " compact" : ""}`}>
        {parts.map((part) => (
          <label key={part}>
            {info.models[part].title}
            <select value={models[part]} onChange={(e) => onModels({ ...models, [part]: e.target.value })}>
              {info.models[part].options.map((o) => (
                <option key={o.key} value={o.key} disabled={o.cloud && !info.elevenlabs}>
                  {o.label}
                  {o.cloud ? (info.elevenlabs ? " · cloud" : " · needs ELEVENLABS_API_KEY") : ""}
                </option>
              ))}
            </select>
            <span className="option-help">{info.models[part].help}</span>
          </label>
        ))}
      </div>
      {usesScribe && (
        <div className="options models-grid compact">
          <label>
            Key terms for ElevenLabs Scribe (optional)
            <input
              type="text"
              placeholder="e.g. metformin, salbutamol"
              value={keyterms}
              onChange={(e) => onKeyterms(e.target.value)}
            />
            <span className="option-help">comma-separated words it should spell correctly (adds 20 % to Scribe's price)</span>
          </label>
        </div>
      )}
      {cloud.length > 0 && (
        <p className="cloud-note">
          <Cloud size={14} /> The audio is sent to ElevenLabs for: {cloud.map((p) => info.models[p].title.toLowerCase()).join(", ")}.
        </p>
      )}
    </div>
  );
}
