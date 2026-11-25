"""Deep SAD training pipeline.

Objective (Ruff et al., 2020, Eq. 6) for ``n`` unlabeled and ``m`` labeled
samples with semi-supervised labels ``y~ in {+1 (normal), -1 (anomaly)}``::

    min_W   1/(n+m) * sum_i      ||phi(x_i; W) - c||^2                 (unlabeled)
          + eta/(n+m) * sum_j ( ||phi(x~_j; W) - c||^2 ) ^ y~_j        (labeled)
          + lambda/2 * sum_l ||W_l||_F^2                               (weight decay)

Labeled anomalies therefore contribute the *inverse* distance and are pushed
away from the hypersphere centre ``c``; the anomaly score is ``||phi(x)-c||^2``.
"""

from __future__ import annotations

import io
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import AutoEncoder, Encoder, count_parameters

# callback(stage, epoch, n_epochs, loss)   stage in {"pretrain", "train"}
ProgressCB = Callable[[str, int, int, float], None]


@dataclass
class DeepSADConfig:
    hidden: tuple = (128, 64)
    rep_dim: int = 32
    eta: float = 1.0                 # weight of the labeled-sample term
    lr: float = 1e-3
    weight_decay: float = 1e-6       # lambda / 2 in the paper's notation
    pretrain_epochs: int = 30
    epochs: int = 50
    batch_size: int = 128
    eps: float = 1e-6                # numerical stability of the inverse distance
    center_eps: float = 0.1          # |c_k| is pushed away from 0 (paper, Sec. 3)
    grad_clip: float = 5.0           # robustness addition (not in the paper)
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DeepSAD:
    """Deep SAD anomaly detector (fit / score / save / load)."""

    def __init__(self, in_dim: int, cfg: Optional[DeepSADConfig] = None):
        self.cfg = cfg or DeepSADConfig()
        self.in_dim = in_dim
        set_seed(self.cfg.seed)
        self.ae = AutoEncoder(in_dim, self.cfg.hidden, self.cfg.rep_dim)
        self.encoder = Encoder(in_dim, self.cfg.hidden, self.cfg.rep_dim)
        self.center: Optional[torch.Tensor] = None
        self.pretrained = False
        self.history: dict[str, list[float]] = {"pretrain": [], "train": []}
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------ utils
    @property
    def n_parameters(self) -> int:
        return count_parameters(self.encoder)

    def _loader(self, *arrays: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
        n = len(tensors[0])
        if n < 2:
            raise ValueError("Need at least 2 samples (BatchNorm).")
        bs = min(self.cfg.batch_size, n)
        # drop a trailing batch of size 1 (BatchNorm cannot train on it)
        drop_last = shuffle and n > bs and (n % bs == 1)
        return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle, drop_last=drop_last)

    # --------------------------------------------------------------- stage 1
    def pretrain(self, X: np.ndarray, y_semi: Optional[np.ndarray] = None,
                 progress_cb: Optional[ProgressCB] = None) -> list[float]:
        """Autoencoder pretraining.  Labeled anomalies are excluded so the
        autoencoder does not learn to reconstruct known anomalies."""
        t0 = time.time()
        mask = np.ones(len(X), bool) if y_semi is None else (y_semi != -1)
        loader = self._loader(X[mask], shuffle=True)
        E = self.cfg.pretrain_epochs
        opt = torch.optim.Adam(self.ae.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[max(1, int(0.8 * E))], gamma=0.1)
        self.history["pretrain"] = []
        for epoch in range(1, E + 1):
            self.ae.train()
            total, cnt = 0.0, 0
            for (xb,) in loader:
                opt.zero_grad()
                recon = self.ae(xb)
                loss = ((recon - xb) ** 2).sum(dim=1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.cfg.grad_clip)
                opt.step()
                total += loss.item() * len(xb)
                cnt += len(xb)
            sched.step()
            self.history["pretrain"].append(total / cnt)
            if progress_cb:
                progress_cb("pretrain", epoch, E, total / cnt)
        # copy pretrained encoder weights into the network that Deep SAD trains
        self.encoder.load_state_dict(self.ae.encoder.state_dict())
        self.pretrained = E > 0
        self.timings["pretrain"] = time.time() - t0
        return self.history["pretrain"]

    # --------------------------------------------------------------- stage 2
    @torch.no_grad()
    def init_center(self, X: np.ndarray, y_semi: Optional[np.ndarray] = None) -> torch.Tensor:
        """c = mean of initial embeddings; coordinates too close to 0 are moved
        to +-eps, otherwise the network could output them trivially."""
        mask = np.ones(len(X), bool) if y_semi is None else (y_semi != -1)
        self.encoder.eval()
        z = self._embed(X[mask])
        c = z.mean(dim=0)
        e = self.cfg.center_eps
        c[(c.abs() < e) & (c < 0)] = -e
        c[(c.abs() < e) & (c >= 0)] = e
        self.center = c
        return c

    # --------------------------------------------------------------- stage 3
    def fit(self, X: np.ndarray, y_semi: np.ndarray,
            progress_cb: Optional[ProgressCB] = None) -> list[float]:
        """Train with the Deep SAD objective.

        ``y_semi``:  0 = unlabeled, +1 = labeled normal, -1 = labeled anomaly.
        """
        t0 = time.time()
        y_semi = np.asarray(y_semi)
        if self.center is None:
            self.init_center(X, y_semi)
        c = self.center
        loader = self._loader(X, y_semi, shuffle=True)
        E = self.cfg.epochs
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[max(1, int(0.8 * E))], gamma=0.1)
        self.history["train"] = []
        for epoch in range(1, E + 1):
            self.encoder.train()
            total, cnt = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                dist = ((self.encoder(xb) - c) ** 2).sum(dim=1)
                losses = torch.where(yb == 0, dist, self.cfg.eta * ((dist + self.cfg.eps) ** yb))
                loss = losses.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.grad_clip)
                opt.step()
                total += loss.item() * len(xb)
                cnt += len(xb)
            sched.step()
            self.history["train"].append(total / cnt)
            if progress_cb:
                progress_cb("train", epoch, E, total / cnt)
        self.timings["train"] = time.time() - t0
        return self.history["train"]

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def _embed(self, X: np.ndarray) -> torch.Tensor:
        self.encoder.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        out = [self.encoder(X_t[i:i + 4096]) for i in range(0, len(X_t), 4096)]
        return torch.cat(out, dim=0)

    def embed(self, X: np.ndarray) -> np.ndarray:
        return self._embed(X).numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score ||phi(x) - c||^2 (higher = more anomalous)."""
        if self.center is None:
            raise RuntimeError("Model is not trained yet.")
        z = self._embed(X)
        return ((z - self.center) ** 2).sum(dim=1).numpy()

    # ------------------------------------------------------------- variants
    def spawn(self, **cfg_overrides) -> "DeepSAD":
        """New untrained-head model that *shares this model's pretrained
        autoencoder* – lets ablations (eta, #labels, Deep SVDD baseline) skip
        repeating stage 1."""
        m = DeepSAD(self.in_dim, replace(self.cfg, **cfg_overrides))
        m.ae.load_state_dict(self.ae.state_dict())
        m.encoder.load_state_dict(self.ae.encoder.state_dict())
        m.pretrained = self.pretrained
        m.history["pretrain"] = list(self.history["pretrain"])
        return m

    # ----------------------------------------------------------- persistence
    def to_bytes(self, extra: Optional[dict] = None) -> bytes:
        buf = io.BytesIO()
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "in_dim": self.in_dim,
                "encoder": self.encoder.state_dict(),
                "center": self.center,
                "history": self.history,
                "extra": extra or {},
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> tuple["DeepSAD", dict]:
        ckpt = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        cfg_dict = ckpt["cfg"]
        cfg_dict["hidden"] = tuple(cfg_dict["hidden"])
        model = cls(ckpt["in_dim"], DeepSADConfig(**cfg_dict))
        model.encoder.load_state_dict(ckpt["encoder"])
        model.center = ckpt["center"]
        model.history = ckpt.get("history", model.history)
        model.pretrained = True
        return model, ckpt.get("extra", {})
