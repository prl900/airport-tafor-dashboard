import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { categoryColor } from "./api";

const CAT_ORDER = { LIFR: 0, IFR: 1, MVFR: 2, VFR: 3 };
const LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#8b97a7", size: 11 },
  margin: { l: 48, r: 14, t: 6, b: 30 },
  xaxis: { gridcolor: "#2a3242", color: "#8b97a7" },
  yaxis: { gridcolor: "#2a3242", color: "#8b97a7" },
  showlegend: true,
  legend: { orientation: "h", y: 1.15, font: { size: 10 } },
};
const CONFIG = { displayModeBar: false, responsive: true };

function useChart(buildTraces, buildLayout, deps) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(ref.current, buildTraces(), { ...LAYOUT, ...buildLayout() }, CONFIG);
  }, deps);
  return ref;
}

export default function StationCharts({ metar, taf }) {
  const obsT = metar.map((m) => m.observed_at);

  // 1. Flight-category timeline (obs as coloured step, TAF base as overlay line)
  const catRef = useChart(
    () => {
      const obs = {
        x: obsT,
        y: metar.map((m) => CAT_ORDER[m.flight_category] ?? null),
        mode: "markers",
        type: "scatter",
        name: "METAR",
        marker: {
          size: 6,
          color: metar.map((m) => categoryColor(m.flight_category)),
        },
      };
      // TAF prevailing (BASE + FM groups) drawn as a step line.
      const baseGroups = taf
        .flatMap((t) => t.groups)
        .filter((g) => g.group_type === "BASE" || g.group_type === "FM")
        .filter((g) => g.flight_category)
        .sort((a, b) => new Date(a.valid_from) - new Date(b.valid_from));
      const fcst = {
        x: baseGroups.map((g) => g.valid_from),
        y: baseGroups.map((g) => CAT_ORDER[g.flight_category] ?? null),
        mode: "lines",
        line: { shape: "hv", color: "#e6a23a", width: 1.5 },
        name: "TAF (prevailing)",
      };
      return [obs, fcst];
    },
    () => ({
      yaxis: {
        gridcolor: "#2a3242",
        tickmode: "array",
        tickvals: [0, 1, 2, 3],
        ticktext: ["LIFR", "IFR", "MVFR", "VFR"],
        range: [-0.4, 3.4],
      },
    }),
    [metar, taf]
  );

  // 2. Visibility + ceiling
  const visRef = useChart(
    () => [
      {
        x: obsT,
        y: metar.map((m) => m.vis_m),
        name: "Visibility (m)",
        mode: "lines",
        line: { color: "#3a8ee6", width: 1.3 },
      },
      {
        x: obsT,
        y: metar.map((m) => m.ceiling_ft),
        name: "Ceiling (ft)",
        yaxis: "y2",
        mode: "lines",
        line: { color: "#c45cd2", width: 1.3 },
      },
    ],
    () => ({
      yaxis: { title: "vis (m)", gridcolor: "#2a3242" },
      yaxis2: { title: "ceil (ft)", overlaying: "y", side: "right", showgrid: false },
    }),
    [metar]
  );

  // 3. Wind speed + gust
  const windRef = useChart(
    () => [
      {
        x: obsT,
        y: metar.map((m) => m.wind_spd_kt),
        name: "Wind (kt)",
        mode: "lines",
        line: { color: "#2ecc71", width: 1.3 },
      },
      {
        x: obsT,
        y: metar.map((m) => m.wind_gust_kt),
        name: "Gust (kt)",
        mode: "markers",
        marker: { color: "#e74c3c", size: 4 },
      },
    ],
    () => ({}),
    [metar]
  );

  return (
    <>
      <div className="chart-card">
        <h3>Flight category — observed vs forecast</h3>
        <div ref={catRef} style={{ height: 200 }} />
      </div>
      <div className="chart-card">
        <h3>Visibility &amp; ceiling</h3>
        <div ref={visRef} style={{ height: 200 }} />
      </div>
      <div className="chart-card">
        <h3>Wind</h3>
        <div ref={windRef} style={{ height: 180 }} />
      </div>
    </>
  );
}
