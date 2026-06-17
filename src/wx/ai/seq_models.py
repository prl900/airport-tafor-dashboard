"""Phase D — sequence/probabilistic model: a TFT-lite seq2seq on the GPU.

``SeqForecastModel`` consumes the windowed tensors from ``seq_dataset`` and predicts
the **full multi-horizon timeline** in one shot:

    station embedding + static  ─┐
    GRU encoder over past METAR ─┼─► decoder init state
    GRU decoder over known-future ERA5 covariates ─► per-horizon heads
        ├─ regression head  (vis / ceiling / wind, standardized)
        └─ category head    (VFR/MVFR/IFR/LIFR logits → P(adverse))

It keeps the lever that fixed the ladder: a class-aware cross-entropy during
training, then isotonic calibration + a val-tuned HSS threshold on the pooled
(sample, horizon) adverse probabilities. Losses are masked by ``y_mask`` so only
horizons with a real verifying obs contribute.

The probabilistic output (calibrated P(adverse) per horizon) is what Phase D adds
over the tabular ladder: it is the natural driver for PROB/TEMPO group generation
and is scored by Brier / BSS exactly like the rest of the benchmark.
"""

from __future__ import annotations

import numpy as np

from wx.ai.models import _best_hss_threshold
from wx.ai.seq_dataset import (
    FUTURE_FEATURES,
    PAST_FEATURES,
    STATIC_FEATURES,
    TARGET_REG,
)

ADVERSE_CODES = (0, 1)  # LIFR, IFR in seq_dataset.CAT_CODE
N_CAT = 4


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_net(n_stations, emb_dim, hidden, dropout):
    import torch
    import torch.nn as nn

    class _Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations, emb_dim)
            self.static = nn.Sequential(
                nn.Linear(len(STATIC_FEATURES) + emb_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout))
            self.encoder = nn.GRU(len(PAST_FEATURES), hidden, batch_first=True)
            self.dec_init = nn.Linear(hidden * 2, hidden)   # [enc_h ; static_ctx] -> h0
            self.decoder = nn.GRU(len(FUTURE_FEATURES), hidden, batch_first=True)
            self.reg_head = nn.Linear(hidden, len(TARGET_REG))
            self.cat_head = nn.Linear(hidden, N_CAT)

        def forward(self, x_past, x_future, x_static, sid):
            ctx = self.static(torch.cat([x_static, self.emb(sid)], dim=1))
            _, h_enc = self.encoder(x_past)                 # h_enc: (1, B, hidden)
            h0 = torch.tanh(self.dec_init(torch.cat([h_enc.squeeze(0), ctx], dim=1)))
            dec_out, _ = self.decoder(x_future, h0.unsqueeze(0))   # (B, H, hidden)
            return self.reg_head(dec_out), self.cat_head(dec_out)

    return _Seq2Seq()


def _build_lstm_net(n_stations, emb_dim, hidden, dropout, bidirectional, attention, n_heads):
    """A more capable seq2seq: (bi)LSTM encoder + cross-attention decoder.

    The decoder LSTM (seeded by the encoder's final state + static context) attends to
    the full encoded past at every horizon — so each forecast hour can look back at the
    most relevant part of the observed trajectory, not just a single squashed state."""
    import torch
    import torch.nn as nn

    dirs = 2 if bidirectional else 1

    class _LSTMSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations, emb_dim)
            self.static = nn.Sequential(
                nn.Linear(len(STATIC_FEATURES) + emb_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout))
            self.encoder = nn.LSTM(len(PAST_FEATURES), hidden, batch_first=True,
                                   bidirectional=bidirectional)
            self.dec_init_h = nn.Linear(hidden * dirs + hidden, hidden)
            self.dec_init_c = nn.Linear(hidden * dirs + hidden, hidden)
            self.decoder = nn.LSTM(len(FUTURE_FEATURES), hidden, batch_first=True)
            self.attn = (nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                               batch_first=True) if attention else None)
            self.enc_proj = nn.Linear(hidden * dirs, hidden) if attention else None
            self.combine = nn.Linear(hidden * 2, hidden) if attention else None
            self.reg_head = nn.Linear(hidden, len(TARGET_REG))
            self.cat_head = nn.Linear(hidden, N_CAT)

        def forward(self, x_past, x_future, x_static, sid):
            B = x_past.shape[0]
            ctx = self.static(torch.cat([x_static, self.emb(sid)], dim=1))
            enc_out, (h, c) = self.encoder(x_past)              # enc_out (B,K,hid*dirs)
            h_cat = h.transpose(0, 1).reshape(B, -1)            # (B, dirs*hid)
            init = torch.cat([h_cat, ctx], dim=1)
            h0 = torch.tanh(self.dec_init_h(init)).unsqueeze(0)
            c0 = torch.tanh(self.dec_init_c(init)).unsqueeze(0)
            dec_out, _ = self.decoder(x_future, (h0, c0))       # (B,H,hidden)
            if self.attn is not None:
                keys = self.enc_proj(enc_out)                   # (B,K,hidden)
                ctx_attn, _ = self.attn(dec_out, keys, keys)    # (B,H,hidden)
                dec_out = self.combine(torch.cat([dec_out, ctx_attn], dim=-1))
            return self.reg_head(dec_out), self.cat_head(dec_out)

    return _LSTMSeq2Seq()


class SeqForecastModel:
    """Seq2seq with a MultiTaskModel-style calibration contract.

    ``cell='gru'`` is the TFT-lite GRU baseline; ``cell='lstm'`` (optionally
    ``bidirectional`` + ``attention``) is the more capable cross-attention variant."""

    def __init__(self, rung: str = "seq2seq", emb_dim: int = 12, hidden: int = 160,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 256, max_epochs: int = 80, patience: int = 8,
                 cat_loss_weight: float = 3.0, cell: str = "gru",
                 bidirectional: bool = False, attention: bool = False, n_heads: int = 4,
                 seed: int = 0):
        self.rung = rung
        self.emb_dim, self.hidden, self.dropout = emb_dim, hidden, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience, self.seed = (
            batch_size, max_epochs, patience, seed)
        self.cell, self.bidirectional, self.attention, self.n_heads = (
            cell, bidirectional, attention, n_heads)
        # The category head drives P(adverse) (BSS/HSS); upweight it so the 5-way
        # regression MSE doesn't dominate the shared trunk's gradient.
        self.cat_loss_weight = cat_loss_weight
        self.n_stations = None
        self.reg_mean = self.reg_std = None
        self.class_w = None
        self._state_dict = None
        self._net = None
        self.calibrator = None
        self.adverse_threshold = 0.5

    # --- helpers --------------------------------------------------------------
    def _make_net(self):
        """Build the net for this rung's cell type (defaults preserve old pickles)."""
        if getattr(self, "cell", "gru") == "lstm":
            return _build_lstm_net(self.n_stations, self.emb_dim, self.hidden, self.dropout,
                                   getattr(self, "bidirectional", False),
                                   getattr(self, "attention", False),
                                   getattr(self, "n_heads", 4))
        return _build_net(self.n_stations, self.emb_dim, self.hidden, self.dropout)

    def _standardize_targets(self, batch):
        obs = batch.y_mask == 1
        Y = batch.y_reg.reshape(-1, len(TARGET_REG))
        m = obs.reshape(-1)
        self.reg_mean = Y[m].mean(axis=0)
        self.reg_std = Y[m].std(axis=0)
        self.reg_std[self.reg_std == 0] = 1.0

    def _tensors(self, batch, dev):
        import torch

        yr = (batch.y_reg - self.reg_mean) / self.reg_std
        return (torch.from_numpy(batch.x_past).float(),
                torch.from_numpy(batch.x_future).float(),
                torch.from_numpy(batch.x_static).float(),
                torch.from_numpy(batch.icao).long(),
                torch.from_numpy(yr).float(),
                torch.from_numpy(np.clip(batch.y_cat, 0, None)).long(),  # -1 -> 0 (masked)
                torch.from_numpy(batch.y_mask).float())

    # --- fit ------------------------------------------------------------------
    def fit(self, train, val=None) -> "SeqForecastModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        dev = _device()
        self.n_stations = int(max(train.icao.max(), 0) + 1)
        if val is not None:
            self.n_stations = max(self.n_stations, int(val.icao.max()) + 1)
        self._standardize_targets(train)

        # class-aware weights from observed training categories
        obs = train.y_mask == 1
        codes = train.y_cat[obs]
        counts = np.array([(codes == i).sum() for i in range(N_CAT)])
        w = counts.sum() / (N_CAT * np.maximum(counts, 1))
        self.class_w = w.astype(np.float32)
        class_w = torch.tensor(self.class_w, device=dev)

        tensors = self._tensors(train, dev)
        ds = TensorDataset(*tensors)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        net = self._make_net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")

        val_loader = None
        if val is not None and len(val.t0):
            val_loader = DataLoader(TensorDataset(*self._tensors(val, dev)),
                                    batch_size=self.batch_size, shuffle=False)

        def batch_loss(xp, xf, xs, sid, yr, yc, ym):
            reg_out, cat_out = net(xp, xf, sid=sid, x_static=xs)
            mm = ym.unsqueeze(-1)
            reg_l = (((reg_out - yr) ** 2) * mm).sum() / mm.sum().clamp_min(1.0)
            cl = ce(cat_out.reshape(-1, N_CAT), yc.reshape(-1)).reshape(yc.shape)
            cat_l = (cl * ym).sum() / ym.sum().clamp_min(1.0)
            return reg_l + self.cat_loss_weight * cat_l

        best, best_state, since = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            net.train()
            for batch in loader:
                batch = [t.to(dev) for t in batch]
                opt.zero_grad()
                loss = batch_loss(*batch)
                loss.backward()
                opt.step()
            if val_loader is not None:
                net.eval()
                with torch.no_grad():
                    vl = float(np.mean([batch_loss(*[t.to(dev) for t in b]).item()
                                        for b in val_loader]))
                if vl < best - 1e-4:
                    best, since = vl, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    since += 1
                    if since >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net
        self._state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if val is not None and len(val.t0):
            self._calibrate(val)
        return self

    # --- inference ------------------------------------------------------------
    def _ensure_net(self):
        if self._net is None:
            net = self._make_net()
            net.load_state_dict(self._state_dict)
            self._net = net.to(_device())
        return self._net

    def _forward(self, batch):
        """Per-(sample,horizon) (reg_destd, cat_proba) as numpy arrays."""
        import torch

        net = self._ensure_net()
        dev = _device()
        net.eval()
        xp = torch.from_numpy(batch.x_past).float()
        xf = torch.from_numpy(batch.x_future).float()
        xs = torch.from_numpy(batch.x_static).float()
        sid = torch.from_numpy(batch.icao).long()
        regs, cats = [], []
        with torch.no_grad():
            for i in range(0, len(batch.t0), 2048):
                sl = slice(i, i + 2048)
                r, c = net(xp[sl].to(dev), xf[sl].to(dev),
                           x_static=xs[sl].to(dev), sid=sid[sl].to(dev))
                regs.append(r.cpu().numpy())
                cats.append(torch.softmax(c, dim=-1).cpu().numpy())
        reg = np.concatenate(regs) * self.reg_std + self.reg_mean
        return reg, np.concatenate(cats)

    def _raw_adverse_proba(self, batch):
        _, cat_p = self._forward(batch)
        return cat_p[..., list(ADVERSE_CODES)].sum(axis=-1)     # (N, H)

    def predict_adverse_proba(self, batch):
        raw = self._raw_adverse_proba(batch)
        if self.calibrator is not None:
            shape = raw.shape
            return self.calibrator.predict(raw.reshape(-1)).reshape(shape)
        return raw

    def predict_adverse_event(self, batch):
        return self.predict_adverse_proba(batch) >= self.adverse_threshold

    # --- calibration (pooled over observed (sample,horizon) points) -----------
    def _calibrate(self, val) -> None:
        from sklearn.isotonic import IsotonicRegression

        raw = self._raw_adverse_proba(val)
        obs = val.y_mask == 1
        events = np.isin(val.y_cat, ADVERSE_CODES) & obs
        raw_o, ev_o = raw[obs], events[obs].astype(int)
        if ev_o.sum() == 0 or ev_o.sum() == len(ev_o):
            return
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_o, ev_o)
        self.calibrator = iso
        self.adverse_threshold = _best_hss_threshold(iso.predict(raw_o), ev_o)

    # --- device-safe pickling -------------------------------------------------
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_net"] = None
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._net = None

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib

        return joblib.load(path)
