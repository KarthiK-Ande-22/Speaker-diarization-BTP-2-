import { STAGE_INFO } from "../stages";

const TITLES: [string, string][] = [
  ["clean", "Clean audio"],
  ["diarization", "Who spoke when"],
  ["speech", "Speech features"],
  ["transcription", "Transcript"],
  ["medical", "Medical terms"],
  ["video", "Video features"],
  ["decision", "Role decision"],
];

/** The 7 stages explained. */
export default function HowItWorks() {
  return (
    <>
      <div className="page-head">
        <h1>How it works</h1>
        <p className="muted">
          A recorded consultation goes through seven stages. By default the first six run on this computer and extract
          features; the last one asks an LLM to decide, from those features, which speaker is the doctor. For the first
          stages you can choose an ElevenLabs model instead (in the cloud, shown with a dashed border) when you upload, or
          later with "Run again with other models" on a recording's page.
        </p>
      </div>
      <div className="how-grid">
        {TITLES.map(([key, title], index) => {
          const { icon: Icon, description, outputs, tools } = STAGE_INFO[key];
          return (
            <article key={key} className="card how-card">
              <div className="how-card-top">
                <span className="how-icon">
                  <Icon size={22} strokeWidth={1.75} />
                </span>
                <span className="tag">Stage {index + 1}</span>
              </div>
              <h3>{title}</h3>
              <p>{description}</p>
              <div className="how-tools">
                {tools.map((tool) => (
                  <span key={tool} className={`tag${tool.startsWith("ElevenLabs") ? " tag-cloud" : ""}`}>
                    {tool}
                  </span>
                ))}
              </div>
              <p className="how-outputs">
                <strong>Gives:</strong> {outputs}
              </p>
            </article>
          );
        })}
      </div>
    </>
  );
}
