# model_lstm.py
# Bidirectional-ready LSTM for thunderstorm end prediction.
# Takes a sequence of the last N lightning strikes as input.

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

class _StormLSTMNet(nn.Module):
    """
    LSTM network that processes a sequence of lightning strike features
    and outputs a single probability (storm ended?).

    Architecture:
        Input (seq_len, n_features)
            → LSTM layers with dropout
            → Attention pooling over time steps
            → FC head → Sigmoid
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, bidirectional: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.num_directions

        # Temporal attention: learn which time steps matter most
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # No sigmoid here — BCEWithLogitsLoss handles it numerically
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        logits : (batch,)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*dirs)

        # Attention pooling
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )                                    # (batch, seq_len)
        context = (attn_weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, hidden*dirs)

        logits = self.classifier(context).squeeze(-1)   # (batch,)
        return logits


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    Scikit-learn-style wrapper around _StormLSTMNet.
    """

    def __init__(self):
        cfg = CFG.lstm
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[_StormLSTMNet] = None
        self.is_fitted = False

    def _build_net(self, input_size: int) -> None:
        cfg = self.cfg
        self.net = _StormLSTMNet(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.info(f"LSTM architecture: {self.net}")
        logger.info(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,    # (n, seq_len, features)
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMModel":

        torch.manual_seed(self.cfg.random_state)
        input_size = X_train.shape[2]
        self._build_net(input_size)

        # Positive class weight to handle imbalance
        pos_weight = torch.tensor([self.cfg.pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        best_val_ap = -1.0
        best_state = None
        patience_counter = 0

        logger.info(f"Training LSTM on {self.device}...")
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            if val_loader is not None:
                val_loss, val_ap = self._val_epoch(val_loader, criterion, y_val)
                scheduler.step(val_loss)
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | "
                    f"Val AP: {val_ap:.4f}"
                )

                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch} (best AP: {best_val_ap:.4f})")
                    break
            else:
                logger.info(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        self.net.eval()
        loader = self._make_loader(X, shuffle=False)
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.net(batch_x.to(self.device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "cfg": self.cfg,
            "input_size": next(iter(self.net.lstm.parameters())).shape[1],
        }, path)
        logger.info(f"LSTM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        checkpoint = torch.load(path, map_location="cpu")
        instance = cls()
        instance._build_net(checkpoint["input_size"])
        instance.net.load_state_dict(checkpoint["state_dict"])
        instance.is_fitted = True
        logger.info(f"LSTM model loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                     shuffle: bool = False) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_t, y_t)
        else:
            dataset = TensorDataset(X_t)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def _train_epoch(self, loader: DataLoader, criterion, optimizer) -> float:
        self.net.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            optimizer.zero_grad()
            logits = self.net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        return total_loss / len(loader.dataset)

    def _val_epoch(self, loader: DataLoader, criterion, y_true: np.ndarray
                   ) -> Tuple[float, float]:
        self.net.eval()
        total_loss = 0.0
        all_probs = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_x)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(all_probs)
        ap = average_precision_score(y_true, probs)
        return total_loss / len(loader.dataset), ap
