"""Recreate verification_hourly with the fcst_prob column and re-score all profiles
(official + persistence + climatology) so probabilistic Brier/BSS is available."""

import time

from wx.db.connection import SCHEMA_PATH, connect
from wx.verification.bulk import run_profiles

con = connect()
con.execute("DROP TABLE IF EXISTS verification_hourly")
con.execute("DROP SEQUENCE IF EXISTS seq_verification_hourly")
con.execute(SCHEMA_PATH.read_text())          # recreates verification_hourly w/ fcst_prob
print("verification_hourly recreated with fcst_prob", flush=True)

t = time.time()
v = run_profiles(con, ["categorical"])
print(f"verify (official): {v} in {time.time()-t:.0f}s", flush=True)

t = time.time()
c = run_profiles(con, ["persistence", "climatology"])
print(f"compare baselines: {c} in {time.time()-t:.0f}s", flush=True)

total = con.execute("SELECT count(*) FROM verification_hourly").fetchone()[0]
print(f"verification_hourly total: {total:,}", flush=True)
con.close()
