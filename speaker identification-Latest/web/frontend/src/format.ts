/** 75.4 -> "1:15" */
export const clock = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

/** 75.4 -> "1 min 15 s" */
export const duration = (seconds: number) =>
  seconds < 60 ? `${Math.round(seconds)} s` : `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;

/** A measured value with sensible precision; null -> "–". */
export const value = (v: number | null | undefined) => {
  if (v === null || v === undefined) return "–";
  if (Number.isInteger(v)) return String(v);
  const size = Math.abs(v);
  return v.toFixed(size >= 10 ? 1 : size >= 1 ? 2 : 3);
};

export const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

export const dateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export const capitalise = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
