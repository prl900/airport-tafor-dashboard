-- DuckDB schema for the airport TAFOR verification dashboard.
-- Idempotent: safe to run repeatedly (CREATE ... IF NOT EXISTS).
--
-- Performance note: DuckDB is columnar/OLAP and its secondary indexes (the ART
-- structures backing PRIMARY KEY / UNIQUE) are a known weak spot for bulk loads
-- (per-row maintenance + slow ON CONFLICT). The high-volume tables therefore
-- carry NO PK/UNIQUE constraints; `id` is just a sequence-backed BIGINT used for
-- joins (hash joins need no index), and idempotency is enforced by anti-join
-- dedup at insert time (see wx/db/repositories.py).

INSTALL spatial;
LOAD spatial;

-- ---------------------------------------------------------------------------
-- Master station list (seeded from wx.config.AIRPORTS by `wx initdb`)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stations (
    icao         VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    lat          DOUBLE  NOT NULL,
    lon          DOUBLE  NOT NULL,
    elevation_m  INTEGER,
    region       VARCHAR
);

-- ---------------------------------------------------------------------------
-- Raw, immutable, source-tagged messages (re-parseable without re-download).
-- Dedup key (anti-join): (icao, observed_at|issued_at, source).
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_raw_metar START 1;
CREATE TABLE IF NOT EXISTS raw_metar (
    id           BIGINT DEFAULT nextval('seq_raw_metar'),
    icao         VARCHAR NOT NULL,
    observed_at  TIMESTAMPTZ NOT NULL,
    raw_text     VARCHAR NOT NULL,
    source       VARCHAR NOT NULL,           -- 'iem' | 'ogimet' | 'aemet'
    ingested_at  TIMESTAMPTZ NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS seq_raw_taf START 1;
CREATE TABLE IF NOT EXISTS raw_taf (
    id           BIGINT DEFAULT nextval('seq_raw_taf'),
    icao         VARCHAR NOT NULL,
    issued_at    TIMESTAMPTZ NOT NULL,
    valid_from   TIMESTAMPTZ,
    valid_to     TIMESTAMPTZ,
    raw_text     VARCHAR NOT NULL,
    source       VARCHAR NOT NULL,           -- 'ogimet' | 'aemet'
    ingested_at  TIMESTAMPTZ NOT NULL
);

-- ---------------------------------------------------------------------------
-- Parsed METAR observations. Dedup key: (raw_metar_id).
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_metar_obs START 1;
CREATE TABLE IF NOT EXISTS metar_obs (
    id              BIGINT DEFAULT nextval('seq_metar_obs'),
    raw_metar_id    BIGINT NOT NULL,
    icao            VARCHAR NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    wind_dir_deg    INTEGER,
    wind_spd_kt     DOUBLE,
    wind_gust_kt    DOUBLE,
    vis_m           DOUBLE,
    temp_c          DOUBLE,
    dewpoint_c      DOUBLE,
    qnh_hpa         DOUBLE,
    ceiling_ft      INTEGER,
    flight_category VARCHAR,                  -- VFR | MVFR | IFR | LIFR
    clouds          JSON,
    weather         JSON
);

-- ---------------------------------------------------------------------------
-- Parsed TAF: header + one row per change group. Dedup key: (raw_taf_id).
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_taf_forecast START 1;
CREATE TABLE IF NOT EXISTS taf_forecast (
    id           BIGINT DEFAULT nextval('seq_taf_forecast'),
    raw_taf_id   BIGINT NOT NULL,
    icao         VARCHAR NOT NULL,
    issued_at    TIMESTAMPTZ NOT NULL,
    valid_from   TIMESTAMPTZ,
    valid_to     TIMESTAMPTZ
);

CREATE SEQUENCE IF NOT EXISTS seq_taf_group START 1;
CREATE TABLE IF NOT EXISTS taf_group (
    id               BIGINT DEFAULT nextval('seq_taf_group'),
    taf_forecast_id  BIGINT NOT NULL,
    group_type       VARCHAR NOT NULL,        -- BASE|FM|BECMG|TEMPO|PROB30|PROB40|PROB_TEMPO
    probability      INTEGER,
    valid_from       TIMESTAMPTZ,
    valid_to         TIMESTAMPTZ,
    wind_dir_deg     INTEGER,
    wind_spd_kt      DOUBLE,
    wind_gust_kt     DOUBLE,
    vis_m            DOUBLE,
    ceiling_ft       INTEGER,
    flight_category  VARCHAR,
    clouds           JSON,
    weather          JSON
);

-- ---------------------------------------------------------------------------
-- Materialised hourly expected-state (output of verification/timeline.py)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS taf_expected_hourly (
    taf_forecast_id  BIGINT NOT NULL,
    icao             VARCHAR NOT NULL,
    valid_hour       TIMESTAMPTZ NOT NULL,
    prevailing       JSON,
    tempo            JSON,
    prob             JSON
);

-- ---------------------------------------------------------------------------
-- Verification results, hourly granularity. Dedup key:
-- (taf_forecast_id, valid_hour, scoring_profile).
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_verification_hourly START 1;
CREATE TABLE IF NOT EXISTS verification_hourly (
    id               BIGINT DEFAULT nextval('seq_verification_hourly'),
    taf_forecast_id  BIGINT NOT NULL,
    icao             VARCHAR NOT NULL,
    valid_hour       TIMESTAMPTZ NOT NULL,
    lead_time_h      INTEGER,
    scoring_profile  VARCHAR NOT NULL,        -- 'categorical'|'persistence'|'climatology'|'model:*'
    fcst_category    VARCHAR,
    obs_category     VARCHAR,
    category_outcome VARCHAR,                 -- hit|miss|false_alarm|correct_neg
    fcst_prob        DOUBLE,                   -- P(IFR-or-worse) for Brier scoring
    wind_err_kt      DOUBLE,
    dir_err_deg      DOUBLE,
    temp_err_c       DOUBLE,
    vis_err_m        DOUBLE,
    ceiling_err_ft   DOUBLE,
    weighted_score   DOUBLE
);

-- ---------------------------------------------------------------------------
-- NWP point series extracted per station from ERA5 (Phase 3).
-- Dedup key: (icao, valid_time, source).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nwp_point (
    icao          VARCHAR NOT NULL,
    valid_time    TIMESTAMPTZ NOT NULL,
    source        VARCHAR NOT NULL,           -- 'era5'
    wind10m_spd   DOUBLE,
    wind10m_dir   DOUBLE,
    gust          DOUBLE,
    t2m_c         DOUBLE,
    d2m_c         DOUBLE,
    tcc           DOUBLE,
    lcc           DOUBLE,
    mcc           DOUBLE,
    hcc           DOUBLE,
    cbh_m         DOUBLE,
    tp_mm         DOUBLE,
    mslp_hpa      DOUBLE
);
