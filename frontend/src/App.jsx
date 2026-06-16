import { useEffect, useState } from "react";
import { api } from "./api";
import SpainMap from "./SpainMap";
import StationCharts from "./StationCharts";

export default function App() {
  const [stations, setStations] = useState([]);
  const [selected, setSelected] = useState(null);
  const [range, setRange] = useState({ start: "2023-01-01", end: "2023-02-01" });
  const [data, setData] = useState({ metar: [], taf: [] });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.stations().then((s) => {
      setStations(s);
      // Auto-select the first station that already has observations, so the
      // dashboard is immediately useful on load.
      const withData = s.find((st) => st.latest_category);
      if (withData) setSelected(withData.icao);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    Promise.all([
      api.metar(selected, range.start, range.end),
      api.taf(selected, range.start, range.end),
    ])
      .then(([metar, taf]) => setData({ metar, taf }))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selected, range]);

  const station = stations.find((s) => s.icao === selected);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Airport TAFOR Dashboard</h1>
        <span className="sub">Spain · METAR observations &amp; TAF forecasts</span>
        <span className="spacer" />
        <div className="range">
          <label>from</label>
          <input
            type="date"
            value={range.start}
            onChange={(e) => setRange((r) => ({ ...r, start: e.target.value }))}
          />
          <label>to</label>
          <input
            type="date"
            value={range.end}
            onChange={(e) => setRange((r) => ({ ...r, end: e.target.value }))}
          />
        </div>
      </header>

      <SpainMap stations={stations} selected={selected} onSelect={setSelected} />

      <div className="panel">
        {!selected && (
          <div className="empty">
            Select an airport on the map to see its METAR observations and TAF forecasts.
          </div>
        )}
        {station && (
          <>
            <div className="station-head">
              <span className="icao">{station.icao}</span>
              <h2>{station.name}</h2>
            </div>
            <div className="station-meta">
              {station.region} · {station.lat.toFixed(3)}, {station.lon.toFixed(3)} ·{" "}
              {data.metar.length} obs · {data.taf.length} TAFs in range
            </div>
            {loading ? (
              <div className="loading">Loading…</div>
            ) : (
              <StationCharts metar={data.metar} taf={data.taf} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
