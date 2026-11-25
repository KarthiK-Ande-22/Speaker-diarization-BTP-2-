// Shapes of the data sent by the Flask server (web/server/app.py).

export type Speaker = "A" | "B";
export type StageStatus = "waiting" | "running" | "done" | "failed" | "skipped";

export interface Stage {
  key: string;
  title: string;
  status: StageStatus;
  seconds?: number;
  started?: string;
  note?: string;
}

/** The parts of the pipeline whose model can be chosen. */
export type ModelPart = "clean" | "diarization" | "transcription" | "medical";
export type Models = Record<ModelPart, string>;

export interface ModelOption {
  key: string;
  label: string;
  short: string;
  cloud?: boolean;
}

/** GET /api/models: the choices of each part (the first is the default) and whether ElevenLabs is set up. */
export interface ModelChoicesInfo {
  models: Record<ModelPart, { title: string; help: string; options: ModelOption[] }>;
  defaults: Models;
  elevenlabs: boolean;
}

export interface Job {
  id: string;
  name: string;
  created: string;
  start: number | null;
  end: number | null;
  language: string | null;
  has_video: boolean;
  models: Models;
  keyterms: string[];
  models_used: { part: ModelPart; title: string; label: string; cloud: boolean }[];
  stages: Stage[];
  running: boolean;
}

export interface FeatureRow {
  feature: string;
  meaning: string;
  A: number | null;
  B: number | null;
  diff: number | null;
  expected: string; // "doctor", "patient", "decide from data" or "-"
  suggests: string; // "A", "B" or "-"
}

export interface Segment {
  speaker: Speaker;
  start: number;
  end: number;
}

export interface TranscriptTurn {
  turn: number;
  speaker: Speaker;
  start: number;
  end: number;
  text: string;
}

export interface MedicalTerm {
  speaker: Speaker;
  term: string;
  type: string;
  side: "doctor" | "patient";
  question: boolean;
}

export interface CombinationResult {
  letters: string;
  groups: string[];
  decision: Speaker | "-";
  confidence: number;
  agree: number;
  answers: number;
  agreement: string;
  same_in_both_orders: boolean | null; // null: only one speaker order was asked
  mean_stated_confidence: number;
  reasons: string[];
}

export interface Results {
  clean?: { original_s: number; cleaned_s: number };
  diarization?: {
    model: string | null;
    audio_s: number;
    talk_s: Partial<Record<Speaker, number>>;
    turns: Partial<Record<Speaker, number>>;
    both_talking_s: number;
    voice_recheck?: { pieces_checked: number; pieces_moved: number; seconds_moved: number } | null;
    audio: boolean;
    segments: Segment[];
  };
  speech?: { features: FeatureRow[] };
  transcript?: TranscriptTurn[];
  medical?: { features: FeatureRow[]; terms: MedicalTerm[] };
  video?: { features: FeatureRow[]; link: Record<string, number | null>; faces: number | null; image: boolean };
  decision?: {
    model: string;
    temperature: number;
    calls_per_combination?: number;
    answers_per_order?: number;
    combinations: CombinationResult[];
    final: CombinationResult | null;
  };
}

export interface RecordingDetail {
  job: Job;
  results: Results;
}

export interface RecordingSummary {
  id: string;
  name: string;
  created: string;
  status: "running" | "done" | "failed" | "partial";
  stages_done: number;
  stages: number;
  doctor: Speaker | "-" | null;
  confidence: number | null;
  audio_s: number | null;
  has_video: boolean;
  models: string;
  image: boolean;
}

export type Roles = Partial<Record<Speaker, "Doctor" | "Patient">>;

/** What the server is running now (null when idle). */
export interface RunningStatus {
  id: string;
  name: string;
  stage: string | null;
  title: string | null;
  started: string | null;
  number: number | null;
  total: number;
}
