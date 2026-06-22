"""Phase C+ — GPU multi-task MLP (PyTorch), drop-in for the model ladder.

``TorchMultiTaskModel`` exposes the SAME surface as the sklearn ``MultiTaskModel``
(``fit`` / ``predict`` / ``predict_adverse_proba`` / ``predict_adverse_event`` /
``save`` / ``load`` + ``calibrator`` / ``adverse_threshold``), so ``ModelForecaster``,
``train_and_evaluate`` and the promotion gate score it unchanged.

Differences from the sklearn rungs:
- a **shared trunk** feeds per-target heads (vis/ceiling/wind regression, flight-
  category + has-ceiling classification) — multi-task, not 7 independent estimators;
- a **class-aware** cross-entropy (inverse-frequency weights) handles the rare
  adverse classes during training, **then** the same isotonic calibration + val-tuned
  HSS threshold restore well-behaved probabilities (the lever that fixed gbm);
- trains on the GPU in minibatches, so it sidesteps the dense-matrix RAM ceiling.

The categorical features (icao/region) reuse the shared one-hot preprocessor from
``models.py`` — for 24 airports a one-hot is equivalent to a small embedding, so the
feature contract stays identical to the rest of the ladder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from wx.ai.dataset import feature_columns
from wx.ai.models import (
    CAT_FEATURES,
    CLF_TARGETS,
    REG_TARGETS,
    _best_hss_threshold,
    _make_preprocessor,
)

ADVERSE = ("IFR", "LIFR")


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_dense(X) -> np.ndarray:
    """ColumnTransformer may return a sparse matrix (one-hot); densify to float32."""
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _build_net(in_dim: int, n_cat: int, trunk=(256, 128), dropout: float = 0.2):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], in_dim
            for h in trunk:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.reg_head = nn.Linear(d, len(REG_TARGETS))   # standardized reg outputs
            self.cat_head = nn.Linear(d, n_cat)              # flight-category logits
            self.ceil_head = nn.Linear(d, 1)                 # has-ceiling logit

        def forward(self, x):
            z = self.trunk(x)
            return self.reg_head(z), self.cat_head(z), self.ceil_head(z).squeeze(-1)

    return _Net()


class TorchMultiTaskModel:
    """Shared-trunk multi-task MLP with the MultiTaskModel interface."""

    def __init__(self, rung: str = "mlp", trunk=(256, 128), dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 4096,
                 max_epochs: int = 80, patience: int = 8, seed: int = 0):
        self.rung = rung
        self.trunk = tuple(trunk)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        # fitted state
        self.numeric_cols: list[str] = []
        self.pre = None
        self.cat_classes: list[str] = []
        self.reg_mean = None          # per-reg-target standardization (np arrays)
        self.reg_std = None
        self._state_dict = None       # CPU state_dict for (un)pickling
        self._in_dim = None
        self._net = None              # live nn.Module (rebuilt lazily on first use)
        # calibration (fit on val) — identical contract to MultiTaskModel
        self.calibrator = None
        self.adverse_threshold = 0.5

    # --- feature / target prep ------------------------------------------------
    def _design_matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        cols = self.numeric_cols + CAT_FEATURES
        X = self.pre.fit_transform(df[cols]) if fit else self.pre.transform(df[cols])
        return _to_dense(X)

    def _reg_targets(self, df: pd.DataFrame):
        """(standardized Y, mask) for the regression heads; NaN targets masked out."""
        Y = np.column_stack([df[t].to_numpy(dtype=float) for t in REG_TARGETS])
        mask = ~np.isnan(Y)
        Ys = (Y - self.reg_mean) / self.reg_std
        Ys = np.where(mask, Ys, 0.0).astype(np.float32)
        return Ys, mask.astype(np.float32)

    # --- fit ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "TorchMultiTaskModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        dev = _device()

        self.numeric_cols = feature_columns(df)
        self.pre = _make_preprocessor(self.numeric_cols)
        X = self._design_matrix(df, fit=True)
        self._in_dim = X.shape[1]

        # regression standardization (per target, ignoring missing labels)
        raw = np.column_stack([df[t].to_numpy(dtype=float) for t in REG_TARGETS])
        self.reg_mean = np.nanmean(raw, axis=0)
        self.reg_std = np.nanstd(raw, axis=0)
        self.reg_std[self.reg_std == 0] = 1.0
        Yr, Mr = self._reg_targets(df)

        # flight-category targets + inverse-frequency class weights (class-aware loss)
        ycat = df["y_cat"]
        self.cat_classes = sorted(c for c in ycat.dropna().unique())
        cat_idx = {c: i for i, c in enumerate(self.cat_classes)}
        yc = ycat.map(cat_idx).to_numpy()                      # NaN -> nan (masked)
        cat_mask = ~np.isnan(yc)
        yc_filled = np.where(cat_mask, yc, 0).astype(np.int64)
        counts = np.array([(yc[cat_mask] == i).sum() for i in range(len(self.cat_classes))])
        w = counts.sum() / (len(counts) * np.maximum(counts, 1))
        class_w = torch.tensor(w, dtype=torch.float32, device=dev)

        yceil = df["y_has_ceiling"].to_numpy(dtype=np.float32)

        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Yr), torch.from_numpy(Mr),
            torch.from_numpy(yc_filled), torch.from_numpy(cat_mask.astype(np.float32)),
            torch.from_numpy(yceil),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        net = _build_net(self._in_dim, len(self.cat_classes), self.trunk, self.dropout).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")
        bce = nn.BCEWithLogitsLoss()

        # validation tensors for early stopping (the same split used for calibration)
        val_pack = None
        if val_df is not None and not val_df.empty:
            Xv = torch.from_numpy(self._design_matrix(val_df, fit=False)).to(dev)
            Yrv, Mrv = self._reg_targets(val_df)
            ycv = val_df["y_cat"].map(cat_idx).to_numpy()
            vcm = ~np.isnan(ycv)
            val_pack = (Xv, torch.from_numpy(Yrv).to(dev), torch.from_numpy(Mrv).to(dev),
                        torch.from_numpy(np.where(vcm, ycv, 0).astype(np.int64)).to(dev),
                        torch.from_numpy(vcm.astype(np.float32)).to(dev),
                        torch.from_numpy(val_df["y_has_ceiling"].to_numpy(np.float32)).to(dev))

        def losses(reg_out, cat_out, ceil_out, yr, mr, yc_, cm, ych):
            reg_l = (((reg_out - yr) ** 2) * mr).sum() / mr.sum().clamp_min(1.0)
            cat_l = (ce(cat_out, yc_) * cm).sum() / cm.sum().clamp_min(1.0)
            ceil_l = bce(ceil_out, ych)
            return reg_l + cat_l + ceil_l

        best_val, best_state, since = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            net.train()
            for xb, yr, mr, yc_, cm, ych in loader:
                xb, yr, mr = xb.to(dev), yr.to(dev), mr.to(dev)
                yc_, cm, ych = yc_.to(dev), cm.to(dev), ych.to(dev)
                opt.zero_grad()
                out = net(xb)
                loss = losses(*out, yr, mr, yc_, cm, ych)
                loss.backward()
                opt.step()
            if val_pack is not None:
                net.eval()
                with torch.no_grad():
                    vl = losses(*net(val_pack[0]), *val_pack[1:]).item()
                if vl < best_val - 1e-4:
                    best_val, since = vl, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    since += 1
                    if since >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)

        self._net = net
        self._state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if val_df is not None and not val_df.empty:
            self._calibrate(val_df)
        return self

    # --- inference ------------------------------------------------------------
    def _ensure_net(self):
        import torch

        if self._net is None:
            net = _build_net(self._in_dim, len(self.cat_classes), self.trunk, self.dropout)
            net.load_state_dict(self._state_dict)
            self._net = net.to(_device())
        return self._net

    def _forward(self, df: pd.DataFrame):
        """Return (reg_out_destandardized, cat_proba, has_ceiling_proba) as numpy."""
        import torch

        net = self._ensure_net()
        dev = _device()
        X = torch.from_numpy(self._design_matrix(df, fit=False)).to(dev)
        net.eval()
        outs_r, outs_c, outs_h = [], [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                r, c, h = net(X[i:i + self.batch_size])
                outs_r.append(r.cpu().numpy())
                outs_c.append(torch.softmax(c, dim=1).cpu().numpy())
                outs_h.append(torch.sigmoid(h).cpu().numpy())
        reg = np.vstack(outs_r) * self.reg_std + self.reg_mean
        return reg, np.vstack(outs_c), np.concatenate(outs_h)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        reg, cat_p, ceil_p = self._forward(df)
        out = pd.DataFrame(index=df.index)
        for j, t in enumerate(REG_TARGETS):
            out[t.replace("y_", "pred_")] = reg[:, j]
        out["pred_cat"] = [self.cat_classes[i] for i in cat_p.argmax(axis=1)]
        out["pred_has_ceiling"] = ceil_p
        return out

    def _raw_adverse_proba(self, df: pd.DataFrame):
        _, cat_p, _ = self._forward(df)
        idx = [i for i, c in enumerate(self.cat_classes) if c in ADVERSE]
        if not idx:
            return np.zeros(len(df))
        return cat_p[:, idx].sum(axis=1)

    def predict_adverse_proba(self, df: pd.DataFrame):
        raw = self._raw_adverse_proba(df)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def predict_adverse_event(self, df: pd.DataFrame):
        return self.predict_adverse_proba(df) >= self.adverse_threshold

    # --- calibration (mirrors MultiTaskModel._calibrate) ----------------------
    def _calibrate(self, val_df: pd.DataFrame) -> None:
        from sklearn.isotonic import IsotonicRegression

        y = val_df["y_cat"]
        mask = y.notna().to_numpy()
        if not mask.any():
            return
        raw = self._raw_adverse_proba(val_df)[mask]
        events = y[mask].isin(ADVERSE).to_numpy(int)
        if events.sum() == 0 or events.sum() == len(events):
            return
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, events)
        self.calibrator = iso
        self.adverse_threshold = _best_hss_threshold(iso.predict(raw), events)

    # --- device-safe (un)pickling --------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_net"] = None          # never pickle a live (possibly CUDA) module
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._net = None              # rebuilt lazily from _state_dict on first use

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TorchMultiTaskModel":
        import joblib

        return joblib.load(path)
