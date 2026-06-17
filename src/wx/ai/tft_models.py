"""Phase D — Temporal Fusion Transformer (TFT) with quantile outputs.

A faithful-but-pragmatic TFT over the ``seq_dataset`` windows. It keeps the
distinctive TFT machinery — Gated Residual Networks (GRN), static-context
conditioning, a GRU encoder/decoder for local processing, **static enrichment**, and
**interpretable masked multi-head attention** over the horizon axis — and adds the
thing Phase D is really about: **quantile regression** (pinball loss) for
vis/ceiling/wind, giving a predictive *distribution* per horizon (the natural driver
for PROB/TEMPO groups), while retaining the calibrated P(adverse) category head so it
stays comparable to the ladder on Brier/BSS/HSS.

(For tractability the per-variable Variable Selection Networks are replaced by a joint
GRN input projection; everything else follows Lim et al. 2021.)

Drop-in for ``scripts/train_seq.py``: same interface as ``SeqForecastModel`` plus
``predict_quantiles``.
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
from wx.ai.seq_models import ADVERSE_CODES, N_CAT, _device

QUANTILES = (0.1, 0.5, 0.9)
MEDIAN_IDX = 1


def _build_tft(n_stations, emb_dim, hidden, n_heads, dropout, n_quantiles):
    import torch
    import torch.nn as nn

    class GRN(nn.Module):
        """Gated Residual Network: skip + GLU-gated nonlinear branch, optional context."""

        def __init__(self, in_dim, hid, out_dim=None, ctx_dim=None, drop=dropout):
            super().__init__()
            out_dim = out_dim or in_dim
            self.fc1 = nn.Linear(in_dim, hid)
            self.ctx = nn.Linear(ctx_dim, hid, bias=False) if ctx_dim else None
            self.fc2 = nn.Linear(hid, hid)
            self.glu = nn.Linear(hid, out_dim * 2)
            self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.drop = nn.Dropout(drop)

        def forward(self, x, c=None):
            h = self.fc1(x)
            if self.ctx is not None and c is not None:
                h = h + self.ctx(c if c.dim() == x.dim() else c.unsqueeze(1))
            h = self.fc2(torch.nn.functional.elu(h))
            a, b = self.glu(self.drop(h)).chunk(2, dim=-1)
            return self.norm(self.skip(x) + a * torch.sigmoid(b))

    class TFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations, emb_dim)
            sdim = len(STATIC_FEATURES) + emb_dim
            # static context vectors: variable-context (skip), enrichment, enc init
            self.grn_static = GRN(sdim, hidden, hidden)
            self.grn_enrich_ctx = GRN(sdim, hidden, hidden)
            self.grn_h0 = GRN(sdim, hidden, hidden)
            # per-timestep input projections (joint GRN in place of per-var VSN)
            self.past_proj = GRN(len(PAST_FEATURES), hidden, hidden)
            self.fut_proj = GRN(len(FUTURE_FEATURES), hidden, hidden)
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
            self.decoder = nn.GRU(hidden, hidden, batch_first=True)
            self.gate_lstm = nn.Linear(hidden, hidden * 2)        # GLU after enc/dec
            self.norm_lstm = nn.LayerNorm(hidden)
            self.enrich = GRN(hidden, hidden, hidden, ctx_dim=hidden)
            self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                              batch_first=True)
            self.gate_attn = nn.Linear(hidden, hidden * 2)
            self.norm_attn = nn.LayerNorm(hidden)
            self.grn_out = GRN(hidden, hidden, hidden)
            self.reg_head = nn.Linear(hidden, len(TARGET_REG) * n_quantiles)
            self.cat_head = nn.Linear(hidden, N_CAT)
            self.h = hidden

        def _glu(self, lin, x):
            a, b = lin(x).chunk(2, dim=-1)
            return a * torch.sigmoid(b)

        def forward(self, x_past, x_future, x_static, sid):
            B, K = x_past.shape[0], x_past.shape[1]
            H = x_future.shape[1]
            s = torch.cat([x_static, self.emb(sid)], dim=1)
            c_var, c_enr, c_h0 = self.grn_static(s), self.grn_enrich_ctx(s), self.grn_h0(s)

            past = self.past_proj(x_past, c_var)            # (B,K,hid)
            fut = self.fut_proj(x_future, c_var)            # (B,H,hid)
            h0 = c_h0.unsqueeze(0)
            enc_out, h_enc = self.encoder(past, h0)
            dec_out, _ = self.decoder(fut, h_enc)
            temporal = torch.cat([enc_out, dec_out], dim=1)  # (B,K+H,hid)
            proj_in = torch.cat([past, fut], dim=1)
            # gated skip over the GRU stack (LSTM gating in TFT)
            temporal = self.norm_lstm(proj_in + self._glu(self.gate_lstm, temporal))
            # static enrichment
            enriched = self.enrich(temporal, c_enr)
            # interpretable masked self-attention (causal over the time axis)
            L = K + H
            mask = torch.triu(torch.ones(L, L, device=temporal.device, dtype=torch.bool),
                              diagonal=1)
            att, _ = self.attn(enriched, enriched, enriched, attn_mask=mask)
            att = self.norm_attn(enriched + self._glu(self.gate_attn, att))
            out = self.grn_out(att)[:, K:, :]               # decoder (future) positions
            reg = self.reg_head(out).view(B, H, len(TARGET_REG), n_quantiles)
            return reg, self.cat_head(out)

    return TFT()


class TFTModel:
    """TFT with quantile regression + calibrated P(adverse). SeqForecastModel-compatible."""

    def __init__(self, rung: str = "tft", emb_dim: int = 12, hidden: int = 96,
                 n_heads: int = 4, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-5, batch_size: int = 128, max_epochs: int = 60,
                 patience: int = 8, cat_loss_weight: float = 3.0,
                 quantiles=QUANTILES, seed: int = 0):
        self.rung = rung
        self.emb_dim, self.hidden, self.n_heads, self.dropout = emb_dim, hidden, n_heads, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.cat_loss_weight, self.seed = cat_loss_weight, seed
        self.quantiles = tuple(quantiles)
        self.n_stations = None
        self.reg_mean = self.reg_std = self.class_w = None
        self._state_dict = self._net = None
        self.calibrator = None
        self.adverse_threshold = 0.5

    # --- prep -----------------------------------------------------------------
    def _standardize_targets(self, batch):
        m = (batch.y_mask == 1).reshape(-1)
        Y = batch.y_reg.reshape(-1, len(TARGET_REG))
        self.reg_mean = Y[m].mean(axis=0)
        self.reg_std = Y[m].std(axis=0)
        self.reg_std[self.reg_std == 0] = 1.0

    def _tensors(self, batch):
        import torch

        yr = (batch.y_reg - self.reg_mean) / self.reg_std
        return (torch.from_numpy(batch.x_past).float(),
                torch.from_numpy(batch.x_future).float(),
                torch.from_numpy(batch.x_static).float(),
                torch.from_numpy(batch.icao).long(),
                torch.from_numpy(yr).float(),
                torch.from_numpy(np.clip(batch.y_cat, 0, None)).long(),
                torch.from_numpy(batch.y_mask).float())

    # --- fit ------------------------------------------------------------------
    def fit(self, train, val=None) -> "TFTModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        dev = _device()
        self.n_stations = int(train.icao.max()) + 1
        if val is not None and len(val.t0):
            self.n_stations = max(self.n_stations, int(val.icao.max()) + 1)
        self._standardize_targets(train)

        obs = train.y_mask == 1
        counts = np.array([(train.y_cat[obs] == i).sum() for i in range(N_CAT)])
        self.class_w = (counts.sum() / (N_CAT * np.maximum(counts, 1))).astype(np.float32)
        class_w = torch.tensor(self.class_w, device=dev)
        qs = torch.tensor(self.quantiles, device=dev).view(1, 1, 1, -1)

        loader = DataLoader(TensorDataset(*self._tensors(train)),
                            batch_size=self.batch_size, shuffle=True)
        val_loader = (DataLoader(TensorDataset(*self._tensors(val)),
                                 batch_size=self.batch_size, shuffle=False)
                      if val is not None and len(val.t0) else None)

        net = _build_tft(self.n_stations, self.emb_dim, self.hidden, self.n_heads,
                         self.dropout, len(self.quantiles)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")

        def loss_fn(xp, xf, xs, sid, yr, yc, ym):
            reg, cat = net(xp, xf, xs, sid)                  # reg: (B,H,T,Q)
            err = yr.unsqueeze(-1) - reg                     # pinball / quantile loss
            pin = torch.max(qs * err, (qs - 1) * err)        # (B,H,T,Q)
            mm = ym.unsqueeze(-1).unsqueeze(-1)
            reg_l = (pin * mm).sum() / mm.sum().clamp_min(1.0)
            cl = ce(cat.reshape(-1, N_CAT), yc.reshape(-1)).reshape(yc.shape)
            cat_l = (cl * ym).sum() / ym.sum().clamp_min(1.0)
            return reg_l + self.cat_loss_weight * cat_l

        best, best_state, since = float("inf"), None, 0
        for _ in range(self.max_epochs):
            net.train()
            for b in loader:
                b = [t.to(dev) for t in b]
                opt.zero_grad()
                loss_fn(*b).backward()
                opt.step()
            if val_loader is not None:
                net.eval()
                with torch.no_grad():
                    vl = float(np.mean([loss_fn(*[t.to(dev) for t in b]).item()
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
            net = _build_tft(self.n_stations, self.emb_dim, self.hidden, self.n_heads,
                             self.dropout, len(self.quantiles))
            net.load_state_dict(self._state_dict)
            self._net = net.to(_device())
        return self._net

    def _raw(self, batch):
        """(quantile reg (N,H,T,Q) destandardized, cat proba (N,H,N_CAT))."""
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
            for i in range(0, len(batch.t0), 1024):
                sl = slice(i, i + 1024)
                r, c = net(xp[sl].to(dev), xf[sl].to(dev), xs[sl].to(dev), sid[sl].to(dev))
                regs.append(r.cpu().numpy())
                cats.append(torch.softmax(c, dim=-1).cpu().numpy())
        reg = np.concatenate(regs)
        reg = reg * self.reg_std[None, None, :, None] + self.reg_mean[None, None, :, None]
        return reg, np.concatenate(cats)

    def _forward(self, batch):
        """Point prediction = median quantile, for the shared eval driver."""
        reg_q, cat_p = self._raw(batch)
        return reg_q[..., MEDIAN_IDX], cat_p

    def predict_quantiles(self, batch):
        """Per-horizon predictive distribution: (N, H, len(TARGET_REG), len(quantiles))."""
        return self._raw(batch)[0]

    def _raw_adverse_proba(self, batch):
        return self._raw(batch)[1][..., list(ADVERSE_CODES)].sum(axis=-1)

    def predict_adverse_proba(self, batch):
        raw = self._raw_adverse_proba(batch)
        if self.calibrator is not None:
            return self.calibrator.predict(raw.reshape(-1)).reshape(raw.shape)
        return raw

    def predict_adverse_event(self, batch):
        return self.predict_adverse_proba(batch) >= self.adverse_threshold

    def _calibrate(self, val) -> None:
        from sklearn.isotonic import IsotonicRegression

        raw = self._raw_adverse_proba(val)
        obs = val.y_mask == 1
        events = (np.isin(val.y_cat, ADVERSE_CODES) & obs)[obs].astype(int)
        raw_o = raw[obs]
        if events.sum() == 0 or events.sum() == len(events):
            return
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_o, events)
        self.calibrator = iso
        self.adverse_threshold = _best_hss_threshold(iso.predict(raw_o), events)

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
