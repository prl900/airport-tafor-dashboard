// Thin API client. In dev, /api is proxied to the FastAPI backend (vite.config.js).
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  stations: () => get("/stations"),
  metar: (icao, start, end) =>
    get(`/stations/${icao}/metar?start=${start}&end=${end}`),
  taf: (icao, start, end) =>
    get(`/stations/${icao}/taf?start=${start}&end=${end}`),
  verificationSummary: () => get("/verification/summary"),
};

// Standard aviation flight-category colours.
export const CATEGORY_COLORS = {
  VFR: "#2ecc71",
  MVFR: "#3a8ee6",
  IFR: "#e74c3c",
  LIFR: "#c45cd2",
  null: "#5a6472",
};

export const categoryColor = (c) => CATEGORY_COLORS[c] || CATEGORY_COLORS.null;
