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

  // Rank by Brier Skill Score — the probabilistic metric that credits hedging.
  const best = [...rows].sort((a, b) => (b.bss ?? -1) - (a.bss ?? -1))[0];

  return (
    <div className="chart-card">
      <h3>Forecaster comparison — can a candidate beat the official TAF?</h3>
      <table className="cmp">
        <thead>
          <tr>
            <th>Forecaster</th><th>Brier↓</th><th>BSS↑</th>
            <th>HSS</th><th>POD</th><th>FAR</th><th>n</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.profile} className={r.profile === "official" ? "official" : ""}>
              <td>
                {r.profile}
                {r.profile === best.profile && (r.bss ?? -1) > 0 && (
                  <span className="badge">best skill</span>
                )}
              </td>
              <td>{fmt(r.brier, 3)}</td>
              <td>{fmt(r.bss)}</td>
              <td>{fmt(r.HSS)}</td>
              <td>{pct(r.POD)}</td>
              <td>{pct(r.FAR)}</td>
              <td>{r.n?.toLocaleString?.() ?? r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="cmp-note">
        <b>Brier</b> scores the forecast <i>probability</i> of IFR-or-worse (lower is better);
        <b> BSS</b> is its skill vs climatology (&gt;0 = better). Unlike HSS's strict
        contingency — which counts every <code>PROB30/TEMPO</code> hedge as a false alarm — Brier
        credits a 30%-fog forecast that verifies ~30% of the time, so it's the fair way to ask
        whether a model beats the official TAF.
      </p>
    </div>
  );
}
