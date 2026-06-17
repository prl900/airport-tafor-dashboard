"""Phase C — the model ladder: multi-task models + a Forecaster wrapper.

A model maps the causal feature frame to the multi-task targets (vis, ceiling,
has_ceiling, wind speed, wind dir sin/cos, flight category). `ModelForecaster`
rebuilds the identical causal features at inference time and emits an
`ExpectedHour` timeline, so a trained model is scored by the same verifier as the
official TAF — the apples-to-apples promotion test.

Ladder rungs share this code; only the estimator builder changes:
  linreg → rf → gbm (HistGradientBoosting) → mlp  (all sklearn; deep TS in Phase D)
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd

from wx.ai.dataset import build_inference_features, feature_columns
from wx.ai.generate import Forecaster
from wx.parsing.normalize import flight_category
from wx.verification.timeline import ExpectedHour

CAT_FEATURES = ["icao", "region"]
REG_TARGETS = ["y_vis_m", "y_ceiling_ft", "y_wspd", "y_wdir_sin", "y_wdir_cos"]
CLF_TARGETS = ["y_cat", "y_has_ceiling"]


def _estimators(rung: str):
    """Return (regressor_factory, classifier_factory) for a ladder rung."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    if rung == "linreg":
        return (lambda: LinearRegression(),
                lambda: LogisticRegression(max_iter=1000))
    if rung == "rf":
        # depth-capped + fewer trees: 7 multi-target forests on ~800k rows are slow
        # and memory-heavy at full depth (the earlier full run was killed here).
        return (lambda: RandomForestRegressor(n_estimators=100, max_depth=20,
                                              n_jobs=-1, random_state=0),
                lambda: RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                               class_weight="balanced", random_state=0))
    if rung == "gbm":
        # NO class_weight: balancing inflates P(adverse) to boost recall/HSS but
        # wrecks calibration (BSS went to -0.46). We instead calibrate probabilities
        # (isotonic on val) and recover HSS with a val-tuned decision threshold.
        return (lambda: HistGradientBoostingRegressor(random_state=0),
                lambda: HistGradientBoostingClassifier(random_state=0))
    if rung == "mlp":
        return (lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                                     early_stopping=True, random_state=0),
                lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                      early_stopping=True, random_state=0))
    raise ValueError(f"unknown rung {rung!r}")


def _make_preprocessor(numeric_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # keep_empty_features: don't drop all-NaN columns (e.g. ERA5 absent for a station)
    # so the transformed width is stable between fit and inference.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, CAT_FEATURES),
    ])


class MultiTaskModel:
    """Shared preprocessor + per-target estimators (regression + classification)."""

    def __init__(self, rung: str):
        self.rung = rung
        self.numeric_cols: list[str] = []
        self.pre = None
        self.reg = {}
        self.clf = {}
        # Calibration (fit on the val split): isotonic remap of raw P(adverse) ->
        # calibrated probability, plus the decision threshold that maximizes HSS.
        # Both None/0.5 until calibrated, so an uncalibrated model still works.
        self.calibrator = None
        self.adverse_threshold = 0.5

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "MultiTaskModel":
        reg_factory, clf_factory = _estimators(self.rung)
        self.numeric_cols = feature_columns(df)
        self.pre = _make_preprocessor(self.numeric_cols)
        X = self.pre.fit_transform(df[self.numeric_cols + CAT_FEATURES])
        # Targets can be missing (e.g. METARs with no reported vis/wind/category) —
        # fit each head on the rows where its own target is present.
        for t in REG_TARGETS:
            y = df[t].astype(float)
            mask = y.notna().to_numpy()
            m = reg_factory()
            m.fit(X[mask], y[mask])
            self.reg[t] = m
        for t in CLF_TARGETS:
            y = df[t]
            mask = y.notna().to_numpy()
            m = clf_factory()
            m.fit(X[mask], y[mask])
            self.clf[t] = m
        if val_df is not None and not val_df.empty:
            self._calibrate(val_df)
        return self

    def _calibrate(self, val_df: pd.DataFrame) -> None:
        """Fit isotonic calibration + HSS-optimal threshold on the val split.

        Decouples the two probabilistic jobs: isotonic gives well-calibrated
        P(adverse) for Brier/BSS, and the tuned threshold restores recall/HSS that
        a calibrated (unbalanced) classifier's argmax would otherwise lose."""
        from sklearn.isotonic import IsotonicRegression

        y = val_df["y_cat"]
        mask = y.notna().to_numpy()
        if not mask.any():
            return
        raw = self._raw_adverse_proba(val_df)[mask]
        events = y[mask].isin(("IFR", "LIFR")).to_numpy(int)
        if events.sum() == 0 or events.sum() == len(events):
            return  # need both classes present to calibrate / tune a threshold
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, events)
        self.calibrator = iso
        self.adverse_threshold = _best_hss_threshold(iso.predict(raw), events)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.pre.transform(df[self.numeric_cols + CAT_FEATURES])
        out = pd.DataFrame(index=df.index)
        for t, m in self.reg.items():
            out[t.replace("y_", "pred_")] = m.predict(X)
        for t, m in self.clf.items():
            out[t.replace("y_", "pred_")] = m.predict(X)
        return out

    def _raw_adverse_proba(self, df: pd.DataFrame):
        """Uncalibrated P(IFR-or-worse) from the category classifier's class probs."""
        X = self.pre.transform(df[self.numeric_cols + CAT_FEATURES])
        clf = self.clf["y_cat"]
        proba = clf.predict_proba(X)
        adverse_idx = [i for i, c in enumerate(clf.classes_) if c in ("IFR", "LIFR")]
        if not adverse_idx:
            return np.zeros(len(df))
        return proba[:, adverse_idx].sum(axis=1)

    def predict_adverse_proba(self, df: pd.DataFrame):
        """Calibrated P(IFR-or-worse) — the model's probabilistic forecast for the
        Brier/BSS benchmark. Applies the isotonic calibrator when fitted."""
        raw = self._raw_adverse_proba(df)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def predict_adverse_event(self, df: pd.DataFrame):
        """Binary IFR-or-worse decision at the val-tuned threshold — drives HSS."""
        return self.predict_adverse_proba(df) >= self.adverse_threshold

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MultiTaskModel":
        import joblib
        return joblib.load(path)


class ModelForecaster(Forecaster):
    """Wrap a trained MultiTaskModel as a Forecaster scored by the verifier."""

    def __init__(self, model: MultiTaskModel, name: str | None = None):
        self.model = model
        self.name = name or f"model:{model.rung}"

    def generate(self, con, icao, issued_at, valid_from, valid_to) -> list[ExpectedHour]:
        hours = _hourly(valid_from, valid_to)
        feats = build_inference_features(con, icao, issued_at, hours)
        if feats.empty:
            return []
        preds = self.model.predict(feats)
        # Calibrated adverse decision at the val-tuned threshold, kept consistent with
        # the BSS/HSS operating point (an unbalanced classifier's argmax under-calls
        # the rare adverse class).
        adverse = self.model.predict_adverse_event(feats)
        out = []
        for (_, row), (_, p), is_adv in zip(feats.iterrows(), preds.iterrows(), adverse):
            has_ceiling = p["pred_has_ceiling"] >= 0.5
            ceiling = float(p["pred_ceiling_ft"]) if has_ceiling else None
            vis = max(0.0, float(p["pred_vis_m"]))
            # trust the dedicated classifier head for category, but never let it
            # disagree with an obviously worse derived category, and honour the
            # tuned adverse threshold (downgrade to at least IFR when it fires).
            cat = _worse(p["pred_cat"], flight_category(ceiling, vis))
            if is_adv:
                cat = _worse(cat, "IFR")
            prevailing = {
                "vis_m": vis,
                "ceiling_ft": ceiling,
                "wind_spd_kt": max(0.0, float(p["pred_wspd"])),
                "wind_dir_deg": _dir_from_sincos(p["pred_wdir_sin"], p["pred_wdir_cos"]),
                "flight_category": cat,
            }
            out.append(ExpectedHour(row["valid_time"].to_pydatetime(), prevailing))
        return out


def _hourly(valid_from, valid_to):
    h = pd.Timestamp(valid_from).tz_convert("UTC").floor("h")
    end = pd.Timestamp(valid_to).tz_convert("UTC")
    out = []
    while h < end:
        out.append(h)
        h += timedelta(hours=1)
    return out


def _dir_from_sincos(s, c):
    if pd.isna(s) or pd.isna(c):
        return None
    return float(np.degrees(math.atan2(s, c)) % 360)


def _best_hss_threshold(probs, events) -> float:
    """Threshold on calibrated P(adverse) maximizing binary HSS over (probs, events).

    HSS = 2(H·CN − M·FA) / [(H+M)(M+CN) + (H+FA)(FA+CN)], swept over probability
    quantiles. Falls back to 0.5 if no candidate separates the classes."""
    probs = np.asarray(probs, dtype=float)
    events = np.asarray(events, dtype=int)
    cands = np.unique(np.quantile(probs, np.linspace(0.02, 0.98, 97)))
    pos, neg = int(events.sum()), int(len(events) - events.sum())
    best_t, best_hss = 0.5, -np.inf
    for t in cands:
        pred = probs >= t
        hits = int(np.sum(pred & (events == 1)))
        fa = int(np.sum(pred & (events == 0)))
        misses, cn = pos - hits, neg - fa
        denom = (hits + misses) * (misses + cn) + (hits + fa) * (fa + cn)
        if denom == 0:
            continue
        hss = 2.0 * (hits * cn - misses * fa) / denom
        if hss > best_hss:
            best_hss, best_t = hss, float(t)
    return best_t


_RANK = {"LIFR": 0, "IFR": 1, "MVFR": 2, "VFR": 3}


def _worse(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if _RANK.get(a, 3) <= _RANK.get(b, 3) else b
