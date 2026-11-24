"""Step 2 - Speech features: the top 10 (definitions in feature_list.md).

Input : data/<id>/turns.csv, diarization.rttm, audio_16k.wav, info.json   (made by Step 1)
Output: data/<id>/turn_features.csv     one row per turn: pitch, rising end, syllables, pauses
        data/<id>/speaker_features.csv  the 10 features: value for A, value for B, A - B

Run (from the "speaker identification-Latest" folder):
    python 02_speech_features/speech_features.py        # every recording in recordings.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_diarization"))
from diarize import DATA, read_recordings, read_rttm  # noqa: E402

# Conversation timing
INTERRUPT_MIN_TURN_S = 1.0     # an interruption is a turn at least this long ...
INTERRUPT_MIN_OVERLAP_S = 0.2  # ... that starts while the other person talks on for at least this long
EXCHANGE_MAX_TURN_S = 5.0      # an exchange starts with a turn this short or shorter ...
EXCHANGE_MAX_GAP_S = 2.0       # ... answered within this many seconds ...
EXCHANGE_REPLY_RATIO = 2.0     # ... by a turn at least this many times longer
# Pitch
FRAME_S = 0.01                 # Praat measures every 10 ms
FIRST_PASS_HZ = (60, 600)      # pitch range of the first pass, before each speaker's own range is known
RISE_MIN_TURN_S = 1.0          # rising ends are measured on turns at least this long ...
RISE_WINDOW_S = 0.5            # ... over their last 0.5 s of voice ...
RISE_MIN_FRAMES = 10           # ... if it has at least 10 voiced frames
RISE_MIN_ST = 2.0              # "rising" = the fitted line climbs at least 2 semitones
# Speed (de Jong & Wempe 2009)
SILENCE_BELOW_PEAK_DB = 25     # silence = quieter than the speaker's 99th-percentile loudness by 25 dB
MIN_DIP_DB = 2.0               # a syllable peak must be 2 dB above the dip before it
MIN_PAUSE_S = 0.25             # shorter silences are not pauses

# Which role should have the HIGHER value (None = decide from data)
EXPECTED = {
    "talk_share": "doctor", "median_turn_s": "patient", "turn_length_cv": "doctor",
    "interruptions_per_min": "doctor", "initiative_share": "doctor", "response_gap_s": "patient",
    "pitch_variation_st": "patient", "rising_end_share": None,
    "articulation_rate": "doctor", "pause_rate_per_min": "patient",
}
SPEAKERS = ("A", "B")
OTHER = {"A": "B", "B": "A"}


def read_turns(path):
    with open(path) as f:
        return [{"turn": int(r["turn"]), "speaker": r["speaker"],
                 **{k: float(r[k]) for k in ("start", "end", "duration", "overlap_s")}} for r in csv.DictReader(f)]


def both_talking(segments):
    """Intervals where A and B talk at the same moment (from pyannote's segments)."""
    a = [(s, e) for spk, s, e in segments if spk == "A"]
    b = [(s, e) for spk, s, e in segments if spk == "B"]
    return [(max(s1, s2), min(e1, e2)) for s1, e1 in a for s2, e2 in b if min(e1, e2) > max(s1, s2)]


def inside(times, intervals):
    """True for every time that falls inside one of the intervals."""
    mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times < end)
    return mask


# ---------------------------------------------------------------- features 1-6: conversation timing

def timing_features(turns, minutes, bounds):
    """bounds[turn] = (first, last) moment the turn is actually loud. pyannote often gives the silence
    between two speakers to one of them, so gaps are measured from the sound, not from turn times."""
    own = {s: [t for t in turns if t["speaker"] == s] for s in SPEAKERS}
    total = sum(t["duration"] for t in turns)

    # Interruption: a turn of >= 1 s that starts while the other person keeps talking for >= 0.2 s
    interruptions = {s: sum(t["duration"] >= INTERRUPT_MIN_TURN_S and any(
        o["start"] < t["start"] and o["end"] - t["start"] >= INTERRUPT_MIN_OVERLAP_S for o in own[OTHER[s]])
        for t in own[s]) for s in SPEAKERS}

    # The "floor": turns that are not completely inside one of the other person's turns
    # (short replies said during the other's talk are left out), in time order.
    floor = [t for t in turns if not any(o["start"] <= t["start"] and t["end"] <= o["end"]
                                         for o in own[OTHER[t["speaker"]]])]
    gaps = {s: [] for s in SPEAKERS}
    exchange_starts = {s: 0 for s in SPEAKERS}
    for prev, cur in zip(floor, floor[1:]):
        if cur["speaker"] == prev["speaker"]:
            continue
        gap = bounds[cur["turn"]][0] - bounds[prev["turn"]][1]   # negative = started before the other stopped
        gaps[cur["speaker"]].append(gap)
        if (prev["duration"] <= EXCHANGE_MAX_TURN_S and gap <= EXCHANGE_MAX_GAP_S
                and cur["duration"] >= EXCHANGE_REPLY_RATIO * prev["duration"]):
            exchange_starts[prev["speaker"]] += 1
    n_exchanges = sum(exchange_starts.values())

    features = {}
    for s in SPEAKERS:
        d = np.array([t["duration"] for t in own[s]])
        features[s] = {
            "talk_share": d.sum() / total,
            "median_turn_s": np.median(d),
            "turn_length_cv": d.std() / d.mean(),
            "interruptions_per_min": interruptions[s] / minutes,
            "initiative_share": exchange_starts[s] / n_exchanges if n_exchanges else None,
            "response_gap_s": np.median(gaps[s]) if gaps[s] else None,
        }
    counts = {"interruptions": interruptions, "exchange_starts": exchange_starts,
              "speaker_changes": {s: len(gaps[s]) for s in SPEAKERS}}
    return features, counts


# ---------------------------------------------------------------- features 7-10: pitch and speed

def rising_end(times, semitones):
    """How many semitones the pitch climbs over the last 0.5 s of voice (None if too little voice)."""
    if len(times) == 0:
        return None
    last = times >= times[-1] - RISE_WINDOW_S
    if last.sum() < RISE_MIN_FRAMES:
        return None
    slope = np.polyfit(times[last], semitones[last], 1)[0]   # semitones per second
    return slope * RISE_WINDOW_S


def syllables_and_pauses(db, free, voiced, silence_db):
    """de Jong & Wempe on one turn's loudness curve (10 ms frames).
    Returns (syllables, pauses inside the turn, seconds of speaking without pauses)."""
    silent = (db < silence_db) & free                 # both-talking frames are never silent
    long_silences, start = [], None
    for i, is_silent in enumerate(np.append(silent, False)):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            if (i - start) * FRAME_S >= MIN_PAUSE_S:
                long_silences.append((start, i))
            start = None
    pauses = sum(1 for a, b in long_silences if a > 0 and b < len(db))   # speech before and after it
    in_silence = np.zeros(len(db), dtype=bool)
    for a, b in long_silences:
        in_silence[a:b] = True
    speaking_s = (free & ~in_silence).sum() * FRAME_S

    peaks = [j for j in range(1, len(db) - 1) if db[j] > db[j - 1] and db[j] >= db[j + 1] and db[j] > silence_db]
    syllables, previous = 0, 0
    for j in peaks:
        dip = db[previous:j].min() if j > previous else db[j]
        if db[j] - dip >= MIN_DIP_DB and free[j] and voiced[j]:
            syllables += 1
        previous = j
    return syllables, pauses, speaking_s


def sound_features(wav, turns, overlaps):
    sound = parselmouth.Sound(str(wav))
    intensity = sound.to_intensity(minimum_pitch=50.0, time_step=FRAME_S)
    i_t, i_db = intensity.xs(), intensity.values[0]
    i_free = ~inside(i_t, overlaps)

    first = sound.to_pitch(time_step=FRAME_S, pitch_floor=FIRST_PASS_HZ[0], pitch_ceiling=FIRST_PASS_HZ[1])
    f_t, f_hz = first.xs(), first.selected_array["frequency"]
    f_free = ~inside(f_t, overlaps)

    features, rows, bounds = {}, [], {}
    for s in SPEAKERS:
        own = [t for t in turns if t["speaker"] == s]
        spans = [(t["start"], t["end"]) for t in own]

        # Pass 2: track pitch again inside this speaker's own range
        q25, q75 = np.percentile(f_hz[inside(f_t, spans) & f_free & (f_hz > 0)], [25, 75])
        floor_hz, ceiling_hz = 0.75 * q25, 1.5 * q75
        pitch = sound.to_pitch(time_step=FRAME_S, pitch_floor=floor_hz, pitch_ceiling=ceiling_hz)
        p_t, p_hz = pitch.xs(), pitch.selected_array["frequency"]
        p_ok = ~inside(p_t, overlaps) & (p_hz > 0)                       # voiced, one speaker only
        median_hz = np.median(p_hz[inside(p_t, spans) & p_ok])
        semitones = np.where(p_hz > 0, 12 * np.log2(np.maximum(p_hz, 1e-9) / median_hz), np.nan)
        voiced_at = lambda times: p_hz[np.clip(np.searchsorted(p_t, times), 0, len(p_t) - 1)] > 0

        silence_db = np.percentile(i_db[inside(i_t, spans) & i_free], 99) - SILENCE_BELOW_PEAK_DB

        rises, total_syllables, total_pauses, total_speaking = [], 0, 0, 0.0
        for t in own:
            p_in = (p_t >= t["start"]) & (p_t < t["end"]) & p_ok
            rise = rising_end(p_t[p_in], semitones[p_in]) if t["duration"] >= RISE_MIN_TURN_S else None
            if rise is not None:
                rises.append(rise >= RISE_MIN_ST)

            i_in = (i_t >= t["start"]) & (i_t < t["end"])
            loud = i_t[i_in][i_db[i_in] >= silence_db]
            bounds[t["turn"]] = (float(loud[0]), float(loud[-1])) if len(loud) else (t["start"], t["end"])
            syl, pauses, speaking = syllables_and_pauses(i_db[i_in], i_free[i_in], voiced_at(i_t[i_in]), silence_db)
            total_syllables, total_pauses, total_speaking = total_syllables + syl, total_pauses + pauses, total_speaking + speaking

            rows.append({"turn": t["turn"], "speaker": s, "start": t["start"], "end": t["end"],
                         "duration": t["duration"], "overlap_s": t["overlap_s"],
                         "loud_from": round(bounds[t["turn"]][0], 2), "loud_to": round(bounds[t["turn"]][1], 2),
                         "median_pitch_hz": round(float(np.median(p_hz[p_in])), 1) if p_in.any() else "",
                         "end_rise_st": round(rise, 2) if rise is not None else "",
                         "rising": "" if rise is None else int(rise >= RISE_MIN_ST),
                         "syllables": syl, "pauses": pauses, "speaking_s": round(speaking, 2),
                         "syllables_per_s": round(syl / speaking, 2) if speaking > 0 else ""})

        turn_minutes = sum(t["duration"] for t in own) / 60
        features[s] = {
            "pitch_variation_st": float(np.std(semitones[inside(p_t, spans) & p_ok])),
            "rising_end_share": float(np.mean(rises)) if rises else None,
            "articulation_rate": total_syllables / total_speaking if total_speaking else None,
            "pause_rate_per_min": total_pauses / turn_minutes,
        }
        features[s]["_info"] = {"pitch_range_hz": [round(floor_hz), round(ceiling_hz)], "median_pitch_hz": round(median_hz, 1),
                                "rising_turns": f"{sum(rises)}/{len(rises)}", "syllables": total_syllables,
                                "pauses": total_pauses}
    rows.sort(key=lambda r: r["turn"])
    return features, rows, bounds


# ---------------------------------------------------------------- run and save

def points_to(feature, diff):
    """Which speaker this feature says is the doctor, using the expected direction."""
    if EXPECTED[feature] is None or diff is None or diff == 0:
        return "-"
    higher = "A" if diff > 0 else "B"
    return higher if EXPECTED[feature] == "doctor" else OTHER[higher]


def extract(rec_id):
    folder = DATA / rec_id
    info = json.loads((folder / "info.json").read_text())
    turns = read_turns(folder / "turns.csv")
    overlaps = both_talking(read_rttm(folder / "diarization.rttm"))

    sound, rows, bounds = sound_features(folder / "audio_16k.wav", turns, overlaps)
    timing, counts = timing_features(turns, info["audio_s"] / 60, bounds)

    with open(folder / "turn_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = []
    for feature in EXPECTED:
        a, b = ({**timing[s], **sound[s]}[feature] for s in SPEAKERS)
        diff = a - b if a is not None and b is not None else None
        table.append({"feature": feature, "A": a, "B": b, "A_minus_B": diff,
                      "expected_higher": EXPECTED[feature] or "decide from data", "doctor_by_this_feature": points_to(feature, diff)})
    with open(folder / "speaker_features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else ("" if v is None else v) for k, v in r.items()}
                          for r in table])

    show = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n=== {rec_id}: speech features ===")
    print(f"{'feature':24s}{'A':>9s}{'B':>9s}{'A-B':>9s}   expected higher   doctor by this feature")
    for r in table:
        print(f"{r['feature']:24s}{show(r['A']):>9s}{show(r['B']):>9s}{show(r['A_minus_B']):>9s}   "
              f"{r['expected_higher']:17s} {r['doctor_by_this_feature']}")
    print(f"\nCounts: interruptions {counts['interruptions']}, exchange starts {counts['exchange_starts']}, "
          f"speaker changes {counts['speaker_changes']}")
    for s in SPEAKERS:
        print(f"Speaker {s}: {sound[s]['_info']}")
    print(f"Saved {folder.relative_to(ROOT)}/speaker_features.csv and turn_features.csv")


def main():
    for row in read_recordings():
        extract(row["recording_id"])


if __name__ == "__main__":
    main()
