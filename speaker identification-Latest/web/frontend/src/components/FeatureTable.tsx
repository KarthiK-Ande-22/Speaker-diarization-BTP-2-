import { capitalise, value } from "../format";
import type { FeatureRow, Roles } from "../types";
import { SpeakerTag } from "./ui";

/** One row per feature: what it means, the value for each speaker (the higher one in bold),
 *  which role is expected to be higher, and which speaker that expectation points to. */
export default function FeatureTable({ rows, roles }: { rows: FeatureRow[]; roles: Roles }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Feature</th>
            <th className="num">
              <SpeakerTag speaker="A" roles={roles} />
            </th>
            <th className="num">
              <SpeakerTag speaker="B" roles={roles} />
            </th>
            <th>Expected higher for</th>
            <th>Points to doctor</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const higher = row.A !== null && row.B !== null && row.A !== row.B ? (row.A > row.B ? "A" : "B") : null;
            return (
              <tr key={row.feature}>
                <td>
                  <div>{capitalise(row.meaning)}</div>
                  <div className="mono muted small">{row.feature}</div>
                </td>
                <td className={`num${higher === "A" ? " higher" : ""}`}>{value(row.A)}</td>
                <td className={`num${higher === "B" ? " higher" : ""}`}>{value(row.B)}</td>
                <td className="muted">{row.expected === "-" ? "–" : row.expected}</td>
                <td>
                  <SpeakerTag speaker={row.suggests} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
