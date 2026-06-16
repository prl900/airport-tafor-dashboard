import { useEffect, useState } from "react";
import { api } from "./api";

const fmt = (v, d = 2) => (v == null ? "—" : v.toFixed(d));
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);

// HSS is the honest "who forecasts the adverse event best" metric — rank by it.
export default function Comparison({ icao }) {
  const [rows, setRows] = useState(null);

  useEffect(() => {
    if (!icao) return;
    setRows(null);
    api.comparison(icao).then(setRows).catch(console.error);
  }, [icao]);

  if (!rows) return null;
  if (rows.length <= 1)
    return (
      <div className="chart-card">
        <h3>Forecaster comparison</h3>
        <div className="empty" style={{ marginTop: 8 }}>
          Run <code>wx compare</code> to score baseline forecasters against the official TAF.
        </div>
      </div>
    );

  const best = [...rows].sort((a, b) => (b.HSS || 0) - (a.HSS || 0))[0];

  return (
    <div className="chart-card">
      <h3>Forecaster comparison — can a candidate beat the official TAF?</h3>
      <table className="cmp">
        <thead>
          <tr>
            <th>Forecaster</th><th>Mean skill</th><th>POD</th><th>FAR</th>
            <th>HSS</th><th>n</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.profile} className={r.profile === "official" ? "official" : ""}>
              <td>
                {r.profile}
                {r.profile === best.profile && r.HSS > 0 && <span className="badge">best skill</span>}
              </td>
              <td>{fmt(r.mean_weighted_score, 3)}</td>
              <td>{pct(r.POD)}</td>
              <td>{pct(r.FAR)}</td>
              <td>{fmt(r.HSS)}</td>
              <td>{r.n?.toLocaleString?.() ?? r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="cmp-note">
        Mean skill can mislead: a climatology that always predicts the common (VFR) case
        scores high yet has POD≈0 — it never warns of IFR/LIFR. HSS rewards detecting the
        adverse event, which is what matters operationally.
      </p>
    </div>
  );
}
