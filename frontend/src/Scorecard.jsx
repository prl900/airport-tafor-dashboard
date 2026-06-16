import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api } from "./api";

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const num = (v, u = "") => (v == null ? "—" : `${v}${u}`);

function Metric({ label, value, hint }) {
  return (
    <div className="metric" title={hint}>
      <div className="metric-val">{value}</div>
      <div className="metric-lbl">{label}</div>
    </div>
  );
}

export default function Scorecard({ icao }) {
  const [card, setCard] = useState(null);
  const leadRef = useRef(null);

  useEffect(() => {
    if (!icao) return;
    setCard(null);
    api.scorecard(icao).then(setCard).catch(console.error);
  }, [icao]);

  useEffect(() => {
    if (!card || !leadRef.current || !card.lead_curve.length) return;
    Plotly.react(
      leadRef.current,
      [
        {
          x: card.lead_curve.map((d) => d.lead_bucket),
          y: card.lead_curve.map((d) => d.mean_score),
          type: "bar",
          marker: { color: "#3a8ee6" },
        },
      ],
      {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#8b97a7", size: 11 },
        margin: { l: 40, r: 12, t: 6, b: 34 },
        xaxis: { title: "lead time (h)", gridcolor: "#2a3242" },
        yaxis: { title: "mean skill", gridcolor: "#2a3242", range: [0, 3] },
      },
      { displayModeBar: false, responsive: true }
    );
  }, [card]);

  if (!card) return <div className="loading">Loading verification…</div>;
  const s = card.skill;
  if (!s.n) return <div className="empty">No verification yet. Run <code>wx verify</code>.</div>;

  return (
    <div className="chart-card">
      <h3>TAF verification — IFR-or-worse event ({s.n.toLocaleString()} forecast-hours)</h3>
      <div className="metrics">
        <Metric label="POD" value={pct(s.POD)} hint="Probability of detection" />
        <Metric label="FAR" value={pct(s.FAR)} hint="False-alarm ratio" />
        <Metric label="CSI" value={pct(s.CSI)} hint="Critical success index" />
        <Metric label="HSS" value={s.HSS == null ? "—" : s.HSS.toFixed(2)} hint="Heidke skill score" />
        <Metric label="Bias" value={s.bias == null ? "—" : s.bias.toFixed(2)} hint="Frequency bias (>1 = over-forecast)" />
      </div>
      <div className="metrics" style={{ marginTop: 8 }}>
        <Metric label="Vis MAE" value={num(card.errors.vis_mae_m, " m")} />
        <Metric label="Ceil MAE" value={num(card.errors.ceiling_mae_ft, " ft")} />
        <Metric label="Wind MAE" value={num(card.errors.wind_mae_kt, " kt")} />
        <Metric label="Dir MAE" value={num(card.errors.dir_mae_deg, "°")} />
      </div>
      <div style={{ marginTop: 10 }}>
        <div style={{ color: "var(--muted)", fontSize: 12, margin: "0 0 4px 4px" }}>
          Skill by forecast lead time
        </div>
        <div ref={leadRef} style={{ height: 170 }} />
      </div>
    </div>
  );
}
