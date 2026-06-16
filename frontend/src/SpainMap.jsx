import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { categoryColor } from "./api";

// Token-free raster style using OpenStreetMap tiles.
const STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

export default function SpainMap({ stations, selected, onSelect }) {
  const mapRef = useRef(null);
  const markersRef = useRef({});
  const [mapError, setMapError] = useState(false);

  useEffect(() => {
    let map;
    try {
      map = new maplibregl.Map({
        container: "map",
        style: STYLE,
        center: [-3.7, 40.2],
        zoom: 4.6,
      });
      map.addControl(new maplibregl.NavigationControl(), "top-right");
      mapRef.current = map;
    } catch (e) {
      // e.g. no WebGL (headless/no-GPU); degrade instead of crashing the app.
      console.error("Map init failed:", e);
      setMapError(true);
    }
    return () => map && map.remove();
  }, []);

  // (Re)draw markers when station data changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !stations.length) return;
    Object.values(markersRef.current).forEach((m) => m.remove());
    markersRef.current = {};
    stations.forEach((s) => {
      const el = document.createElement("div");
      el.className = "apt-marker";
      el.style.background = categoryColor(s.latest_category);
      el.title = `${s.icao} — ${s.name}`;
      el.addEventListener("click", () => onSelect(s.icao));
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([s.lon, s.lat])
        .addTo(map);
      markersRef.current[s.icao] = marker;
    });
  }, [stations]);

  // Reflect the selected station.
  useEffect(() => {
    Object.entries(markersRef.current).forEach(([icao, m]) => {
      m.getElement().classList.toggle("selected", icao === selected);
    });
  }, [selected]);

  return (
    <div className="map-wrap">
      <div id="map" />
      {mapError && (
        <div className="map-fallback">
          <p>Map needs WebGL (unavailable here). Pick an airport:</p>
          <div className="apt-list">
            {stations.map((s) => (
              <button
                key={s.icao}
                className={s.icao === selected ? "sel" : ""}
                style={{ borderLeftColor: categoryColor(s.latest_category) }}
                onClick={() => onSelect(s.icao)}
              >
                <b>{s.icao}</b> {s.name}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="legend" style={{ display: mapError ? "none" : "block" }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>Latest flight category</div>
        {["VFR", "MVFR", "IFR", "LIFR", null].map((c) => (
          <div className="row" key={String(c)}>
            <span className="dot" style={{ background: categoryColor(c) }} />
            <span>{c || "no data"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
