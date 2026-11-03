"""CPU post-hoc restrictions of two modern Learning-to-Defer surrogates.

The deployed predictor remains frozen.  Its two class logits are fixed to
log(1-p) and log(p); only an additional defer logit is learned from EcoTrust's
signal bank.  This makes the comparison isolate the deferral rule while using
the published surrogate losses:

* Mozannar--Sontag (ICML 2020), softmax L_CE (Eq. 6).
* Verma--Nalisnick (ICML 2022), one-vs-all logistic loss (Eq. 8).

These are explicitly post-hoc restrictions, not claims to reproduce the
papers' joint classifier/rejector training setting.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler


def simulated_expert_predictions(y, accuracy=0.90, seed=42):
    """Binary expert demonstrations with class-independent error."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    correct = rng.random(len(y)) < accuracy
    return np.where(correct, y, 1 - y).astype(int)


def ms_surrogate_loss(outputs, expert_predictions, y, alpha=1.0):
    """Mozannar--Sontag L_CE^alpha, using natural logarithms."""
    y = y.long()
    expert_correct = (expert_predictions.long() == y).float()
    log_probs = F.log_softmax(outputs, dim=1)
    idx = torch.arange(len(y), device=outputs.device)
    class_weight = alpha * expert_correct + (1.0 - expert_correct)
    return (-expert_correct * log_probs[:, -1]
            - class_weight * log_probs[idx, y]).mean()


def ova_surrogate_loss(outputs, expert_predictions, y):
    """Verma--Nalisnick OvA logistic surrogate (Eq. 8)."""
    y = y.long()
    expert_correct = (expert_predictions.long() == y).float()
    class_logits, defer_logit = outputs[:, :-1], outputs[:, -1]
    idx = torch.arange(len(y), device=outputs.device)

    # phi(f)=log(1+exp(-f)); phi(-f)=log(1+exp(f)).
    l_true = F.softplus(-class_logits[idx, y])
    l_other = F.softplus(class_logits).sum(dim=1) - F.softplus(class_logits[idx, y])
    l_defer_wrong = F.softplus(defer_logit)
    l_defer_correct = F.softplus(-defer_logit)
    return (l_true + l_other + l_defer_wrong
            + expert_correct * (l_defer_correct - l_defer_wrong)).mean()


class _DeferHead(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class PostHocL2D:
    """Learn a defer logit while keeping the backend class logits frozen."""

    def __init__(self, method="ms", seed=42, epochs=300, lr=0.02, alpha=1.0):
        if method not in {"ms", "ova"}:
            raise ValueError("method must be 'ms' or 'ova'")
        self.method = method
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.model = None
        self.threshold_ = 0.0
        self.fit_diagnostics_ = {}

    @staticmethod
    def _class_logits(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.column_stack([np.log1p(-p), np.log(p)])

    def _outputs(self, X, p):
        z = torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)
        class_logits = torch.as_tensor(self._class_logits(p), dtype=torch.float32)
        with torch.no_grad():
            defer = self.model(z).unsqueeze(1)
        return torch.cat([class_logits, defer], dim=1)

    def fit(self, X, p, y, expert_predictions):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        expert_predictions = np.asarray(expert_predictions, dtype=int)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        z = torch.as_tensor(self.scaler.fit_transform(X), dtype=torch.float32)
        class_logits = torch.as_tensor(self._class_logits(p), dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        m_t = torch.as_tensor(expert_predictions, dtype=torch.long)
        self.model = _DeferHead(z.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        first_loss = None
        for _ in range(self.epochs):
            opt.zero_grad()
            outputs = torch.cat([class_logits, self.model(z).unsqueeze(1)], dim=1)
            if self.method == "ms":
                loss = ms_surrogate_loss(outputs, m_t, y_t, alpha=self.alpha)
            else:
                loss = ova_surrogate_loss(outputs, m_t, y_t)
            if first_loss is None:
                first_loss = float(loss.detach())
            loss.backward()
            opt.step()

        self.fit_diagnostics_ = {
            "method": self.method,
            "n_train": int(len(y)),
            "expert_accuracy_train": round(float((expert_predictions == y).mean()), 4),
            "loss_initial": round(first_loss, 5),
            "loss_final": round(float(loss.detach()), 5),
        }
        return self

    def defer_score(self, X, p):
        outputs = self._outputs(X, p)
        if self.method == "ms":
            probs = F.softmax(outputs, dim=1).numpy()
            predicted = (np.asarray(p) > 0.5).astype(int)
            return probs[:, -1] - probs[np.arange(len(predicted)), predicted]
        probs = torch.sigmoid(outputs).numpy()
        return probs[:, -1] - probs[:, :-1].max(axis=1)

    def estimated_expert_correctness(self, X, p):
        outputs = self._outputs(X, p)
        if self.method == "ova":
            return torch.sigmoid(outputs[:, -1]).numpy()
        probs = F.softmax(outputs, dim=1).numpy()
        return np.clip(probs[:, -1] / np.clip(1.0 - probs[:, -1], 1e-6, None), 0.0, 1.0)

    def tune_cost_threshold(self, X_cal, p_cal, y_cal, cost):
        scores = self.defer_score(X_cal, p_cal)
        y_cal = np.asarray(y_cal, dtype=int)
        pred = (np.asarray(p_cal) > 0.5).astype(int)
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
        candidates = np.concatenate(([-np.inf], candidates, [np.inf]))
        best = (np.inf, np.inf)
        for threshold in candidates:
            defer = scores > threshold
            keep = ~defer
            fn = ((pred[keep] == 0) & (y_cal[keep] == 1)).sum()
            fp = ((pred[keep] == 1) & (y_cal[keep] == 0)).sum()
            total = cost.c_fn * fn + cost.c_fp * fp + cost.c_h * defer.sum()
            # Tie-break towards fewer reviews.
            candidate = (float(total), int(defer.sum()))
            if candidate < best:
                best = candidate
                self.threshold_ = float(threshold)
        self.fit_diagnostics_["threshold"] = self.threshold_
        self.fit_diagnostics_["calibration_cost_total"] = best[0]
        self.fit_diagnostics_["calibration_escalations"] = best[1]
        return self

    def escalate(self, X, p):
        return self.defer_score(X, p) > self.threshold_

