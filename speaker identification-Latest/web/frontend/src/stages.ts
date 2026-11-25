import { AudioLines, Eraser, FileText, Scale, Stethoscope, Users, Video, type LucideIcon } from "lucide-react";

/** Icon and plain description of each pipeline stage (used by "How it works" and the stage panel).
 *  `tools`: the default models; "ElevenLabs ..." entries are optional choices on the upload form. */
export const STAGE_INFO: Record<string, { icon: LucideIcon; description: string; outputs: string; tools: string[] }> = {
  clean: {
    icon: Eraser,
    description:
      "Optional, off by default. The ElevenLabs Voice Isolator removes background noise. Finding speakers and the transcript then use the cleaned audio; pitch, pauses and speed keep the original.",
    outputs: "a cleaned copy of the audio, with the same timing",
    tools: ["ElevenLabs Voice Isolator"],
  },
  diarization: {
    icon: Users,
    description:
      "pyannote splits the recording into two speakers (Speaker A is whoever talks first). A voice re-check moves pieces that sound like the other person. ElevenLabs Scribe can be chosen instead; it gives one speaker per word and never marks both talking at once.",
    outputs: "turns, talking time, overlap and a timeline",
    tools: ["pyannote community-1", "voice re-check", "ElevenLabs Scribe"],
  },
  speech: {
    icon: AudioLines,
    description:
      "How each person talks: timing, pitch and speed, measured with Praat. No transcript is needed, so it works in any language.",
    outputs: "10 features (talk share, turn length, interruptions, initiative, response gap, pitch, rising endings, speed, pauses)",
    tools: ["Praat (parselmouth)"],
  },
  transcription: {
    icon: FileText,
    description:
      "Whisper medium writes down what was said, or ElevenLabs Scribe v2 Medical, which is tuned for clinical audio and can be given key terms such as medicine names. Every word is given to the speaker who is talking at that moment.",
    outputs: "the transcript, turn by turn",
    tools: ["Whisper medium", "ElevenLabs Scribe v2 Medical"],
  },
  medical: {
    icon: Stethoscope,
    description:
      "A biomedical NER model finds medicines, tests, treatments, symptoms and history in every sentence. ElevenLabs entity detection can be chosen instead: it finds medicines, conditions, procedures and tests, but no symptoms or history.",
    outputs: "medical terms per speaker and 4 features",
    tools: ["d4data biomedical NER", "NLTK", "ElevenLabs entity detection"],
  },
  video: {
    icon: Video,
    description: "Faces are found, grouped into two people and matched to the voices. An image model checks for a white coat, a stethoscope, and notes or a computer.",
    outputs: "3 features and how sure the face–voice match is",
    tools: ["YuNet", "SFace", "CLIP"],
  },
  decision: {
    icon: Scale,
    description:
      "Sarvam's LLM reads the extracted features (not the recording) and says who is the doctor. It is asked once for every combination of feature groups: each alone, each pair, and all three, which gives the final answer.",
    outputs: "doctor, patient, confidence, the model's reasons, and the answer for every combination",
    tools: ["Sarvam LLM"],
  },
};
