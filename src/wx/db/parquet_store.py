"""Parquet-backed storage for the high-volume, append-only ``nwp_point`` data.

Why: DuckDB is excellent as the query engine (the causal ASOF joins in ``wx.ai.dataset``
depend on it), but a single ``.duckdb`` file is single-writer across processes — which
serialises ingest against training/serving and bites hardest under an operational real-time
feed. Storing ``nwp_point`` as partitioned Parquet that DuckDB reads as an external view
removes that bottleneck: each ingest batch (a backfill month, or an operational forecast
cycle) writes its own file under a hive partition — no DB lock, trivially parallel, and
appending is just dropping a file. Re-ingesting a key is idempotent via a latest-wins dedup
in the view (no read-modify-write).

The query side is unchanged: ``nwp_point`` is exposed as a DuckDB view with exactly the
table's columns, so ``build_samples`` / the API / the verifier keep working verbatim.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import duckdb
import pandas as pd

from wx.config import DATA_DIR

NWP_PARQUET_DIR = DATA_DIR / "nwp"

# Same columns as the nwp_point table (db/repositories.store_nwp_points), in order.
NWP_COLS = ["icao", "valid_time", "source", "ref_time", "step_h",
            "wind10m_spd", "wind10m_dir", "gust", "t2m_c", "d2m_c",
            "tcc", "lcc", "mcc", "hcc", "cbh_m", "tp_mm", "mslp_hpa",
            "cape_jkg", "blh_m", "tcwv_kgm2", "skt_c"]

# Dedup key (matches the anti-join key of the DuckDB-table path).
_KEY = ["icao", "valid_time", "source", "ref_time", "step_h"]


def store_nwp_points_parquet(records: list[dict], base_dir=None) -> int:
    """Append a batch of nwp_point records as a Parquet file under
    ``base_dir/source=.../year=.../month=.../<batch>_*.parquet``. Returns rows written.

    Append-only and lock-free: each call writes a uniquely-named file, so concurrent
    ingest jobs never contend. Idempotency is handled at read time by ``nwp_point_view_ddl``
    (latest ``_ingested_at`` wins per key), so re-running an ingest is safe."""
    if not records:
        return 0
    base_dir = str(base_dir or NWP_PARQUET_DIR)
    df = pd.DataFrame(records, columns=NWP_COLS).copy()
    vt = pd.to_datetime(df["valid_time"], utc=True)
    df["year"] = vt.dt.year
    df["month"] = vt.dt.month
    df["_ingested_at"] = datetime.now(timezone.utc)

    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC';")
        con.register("_pq_df", df)
        batch = uuid.uuid4().hex[:12]
        con.execute(
            f"COPY (SELECT * FROM _pq_df) TO '{base_dir}' "
            f"(FORMAT PARQUET, PARTITION_BY (source, year, month), "
            f"FILENAME_PATTERN '{batch}_{{i}}', OVERWRITE_OR_IGNORE)"
        )
    finally:
        con.close()
    return len(df)


def nwp_point_view_ddl(base_dir=None) -> str:
    """CREATE-OR-REPLACE VIEW SQL exposing the Parquet store as ``nwp_point`` with exactly
    the table's columns. Deduplicates to the latest ingest per key so re-ingested batches
    behave like an upsert. Hive partitioning lets DuckDB prune by source/year/month."""
    base_dir = str(base_dir or NWP_PARQUET_DIR)
    glob = f"{base_dir}/**/*.parquet"
    cols = ", ".join(NWP_COLS)
    part = ", ".join(_KEY)
    return f"""
    CREATE OR REPLACE VIEW nwp_point AS
    SELECT {cols} FROM (
        SELECT *, row_number() OVER (PARTITION BY {part} ORDER BY _ingested_at DESC) AS _rn
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)
    ) WHERE _rn = 1
    """


def register_nwp_view(con: duckdb.DuckDBPyConnection, base_dir=None) -> None:
    """Replace the ``nwp_point`` table with a Parquet-backed view on this connection.

    Drops the table if present (the schema still creates an empty one) and creates the view,
    so downstream SQL resolves ``nwp_point`` to the Parquet store transparently."""
    base_dir = base_dir or NWP_PARQUET_DIR
    # No files yet → nothing to view; leave the (empty) table in place to avoid a read error.
    from pathlib import Path
    if not any(Path(str(base_dir)).glob("**/*.parquet")):
        return
    con.execute("DROP TABLE IF EXISTS nwp_point")
    con.execute("DROP VIEW IF EXISTS nwp_point")
    con.execute(nwp_point_view_ddl(base_dir))
