"""
Lightweight CNN baseline for RAVEN 3x3 visual reasoning classification.

Design goals:
- Pure PyTorch, CPU-friendly, simple architecture.
- Compatible with variable image resolution via adaptive pooling.
- Robust runtime checks with clear error diagnostics.
- Built-in training loss anomaly monitor.
- Safe weight save/load utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_WEIGHT_PATH = Path("experiments/weights/cnn_baseline.pth")
DEFAULT_LOG_PATH = Path("experiments/logs/cnn_baseline_train.log")


def _setup_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create a timestamped logger for training/runtime alerts."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cnn_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


LOGGER = _setup_logger()


class ModelInputError(ValueError):
    """Raised when model input is malformed or invalid."""


class ModelForwardError(RuntimeError):
    """Raised when forward pass fails with actionable details."""


def validate_panels_input(panels: torch.Tensor, expect_panels: int = 16) -> None:
    """
    Validate model input tensor from dataloader.

    Expected:
    - shape: [B, 16, 1, H, W]
    - dtype: float32/float64/float16/bfloat16
    - finite values only
    """
    where = "validate_panels_input"
    if panels is None:
        raise ModelInputError(f"[{where}] Input is None.")
    if not isinstance(panels, torch.Tensor):
        raise ModelInputError(
            f"[{where}] Input type must be torch.Tensor, got {type(panels)}."
        )
    if panels.ndim != 5:
        raise ModelInputError(
            f"[{where}] Expected ndim=5 ([B,16,1,H,W]), got ndim={panels.ndim}, "
            f"shape={tuple(panels.shape)}."
        )
    bsz, num_panels, channels, height, width = panels.shape
    if bsz <= 0:
        raise ModelInputError(f"[{where}] Batch size must be > 0, got {bsz}.")
    if num_panels != expect_panels:
        raise ModelInputError(
            f"[{where}] Expected num_panels={expect_panels}, got {num_panels}."
        )
    if channels != 1:
        raise ModelInputError(
            f"[{where}] Expected channels=1 (grayscale), got {channels}."
        )
    if height <= 0 or width <= 0:
        raise ModelInputError(
            f"[{where}] Invalid spatial size HxW={height}x{width}; must be >0."
        )
    if not torch.is_floating_point(panels):
        raise ModelInputError(
            f"[{where}] Expected floating tensor, got dtype={panels.dtype}. "
            "Please convert input to float."
        )
    if not torch.isfinite(panels).all():
        raise ModelInputError(f"[{where}] Input contains NaN or Inf values.")


class PanelEncoder(nn.Module):
    """
    Lightweight CNN encoder for one grayscale panel.

    Input:  [N, 1, H, W]
    Output: [N, feature_dim]
    """

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Adaptive pooling makes model resolution-agnostic.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.flatten(1)


class CNNBaseline(nn.Module):
    """
    Lightweight RAVEN 3x3 classifier.

    Expected input:
      - panels: [B, 16, 1, H, W]
          first 8 = context panels
          last 8 = candidate answer panels
    Output:
      - logits: [B, 8]
      - probs:  [B, 8] (softmax probabilities)
    """

    def __init__(self, num_choices: int = 8, feature_dim: int = 128) -> None:
        super().__init__()
        self.num_choices = num_choices
        self.feature_dim = feature_dim

        self.encoder = PanelEncoder(feature_dim=feature_dim)
        self.choice_head = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
        )
        self._validate_model_structure()

    def _validate_model_structure(self) -> None:
        """Check architecture-level assumptions at initialization."""
        where = "CNNBaseline._validate_model_structure"
        if self.num_choices != 8:
            # RAVEN standard is 8 candidates; enforce consistency.
            raise ValueError(
                f"[{where}] num_choices must be 8 for RAVEN 3x3, got {self.num_choices}."
            )
        if not isinstance(self.choice_head[-1], nn.Linear):
            raise ValueError(f"[{where}] Invalid head: final layer is not Linear.")
        if self.choice_head[-1].out_features != 1:
            raise ValueError(
                f"[{where}] Head out_features must be 1 per candidate, got "
                f"{self.choice_head[-1].out_features}."
            )

        # Dry-run check to ensure output dims are correct.
        try:
            with torch.no_grad():
                dummy = torch.zeros(2, 16, 1, 80, 80, dtype=torch.float32)
                logits, probs = self.forward(dummy)
                if logits.shape != (2, self.num_choices):
                    raise ValueError(
                        f"[{where}] logits shape mismatch: expected "
                        f"(2, {self.num_choices}), got {tuple(logits.shape)}."
                    )
                if probs.shape != (2, self.num_choices):
                    raise ValueError(
                        f"[{where}] probs shape mismatch: expected "
                        f"(2, {self.num_choices}), got {tuple(probs.shape)}."
                    )
        except Exception as exc:
            raise ValueError(f"[{where}] structure dry-run failed: {exc}") from exc

    def forward(self, panels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with defensive exception handling.

        Returns:
            logits: [B, 8]
            probs:  [B, 8]
        """
        where = "CNNBaseline.forward"
        try:
            validate_panels_input(panels, expect_panels=16)

            bsz = panels.size(0)
            flat_panels = panels.reshape(bsz * 16, 1, panels.size(-2), panels.size(-1))
            panel_features = self.encoder(flat_panels).reshape(bsz, 16, self.feature_dim)

            context_feat = panel_features[:, :8, :].mean(dim=1)  # [B, D]
            choice_feat = panel_features[:, 8:, :]  # [B, 8, D]

            # Score each candidate with context-aware feature composition.
            context_expand = context_feat.unsqueeze(1).expand(-1, self.num_choices, -1)
            abs_diff = torch.abs(choice_feat - context_expand)
            pair_feat = torch.cat([context_expand, choice_feat, abs_diff], dim=-1)

            logits = self.choice_head(pair_feat).squeeze(-1)  # [B, 8]
            probs = F.softmax(logits, dim=-1)
            return logits, probs
        except (ModelInputError, ValueError, RuntimeError) as exc:
            raise ModelForwardError(
                f"[{where}] Forward failed. reason={exc.__class__.__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelForwardError(
                f"[{where}] Unexpected forward error: {exc.__class__.__name__}: {exc}"
            ) from exc


class LossMonitor:
    """
    Monitor training loss stability and stop on anomalies.

    Trigger rules:
    - Loss is NaN/Inf.
    - Loss spikes beyond `spike_factor * previous_loss`.
    """

    def __init__(
        self,
        spike_factor: float = 5.0,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> None:
        if spike_factor <= 1.0:
            raise ValueError("spike_factor must be > 1.0")
        self.spike_factor = spike_factor
        self.prev_loss: Optional[float] = None
        self.logger = _setup_logger(log_path)

    def check(self, loss_value: float, step: int) -> None:
        """Validate one loss value, raise RuntimeError on anomalies."""
        where = "LossMonitor.check"
        if loss_value is None:
            msg = f"[{where}] loss is None at step={step}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        if not torch.isfinite(torch.tensor(loss_value)):
            msg = f"[{where}] loss is NaN/Inf at step={step}, loss={loss_value}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        if self.prev_loss is not None and loss_value > self.prev_loss * self.spike_factor:
            msg = (
                f"[{where}] Loss spike detected at step={step}: "
                f"prev={self.prev_loss:.6f}, curr={loss_value:.6f}, "
                f"threshold={self.prev_loss * self.spike_factor:.6f}"
            )
            self.logger.error(msg)
            raise RuntimeError(msg)
        self.prev_loss = float(loss_value)
        self.logger.info("step=%d | loss=%.6f", step, loss_value)


def save_weights(
    model: nn.Module,
    path: Path = DEFAULT_WEIGHT_PATH,
    extra: Optional[Dict] = None,
) -> None:
    """Safely save model weights and optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    LOGGER.info("Weights saved to %s", path)


def load_weights(model: nn.Module, path: Path = DEFAULT_WEIGHT_PATH, strict: bool = True) -> None:
    """
    Safely load weights with integrity and compatibility checks.
    """
    where = "load_weights"
    if not path.exists():
        raise FileNotFoundError(f"[{where}] Weight file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"[{where}] Weight file is empty: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"[{where}] Failed to read weight file: {exc}") from exc

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(
            f"[{where}] Invalid checkpoint format. Required key 'state_dict' missing."
        )
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(f"[{where}] Empty or invalid state_dict in checkpoint.")

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"[{where}] State dict incompatible with current model structure: {exc}"
        ) from exc

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"[{where}] strict=True but key mismatch detected. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    LOGGER.info("Weights loaded from %s | strict=%s", path, strict)


def training_step(
    model: CNNBaseline,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    monitor: LossMonitor,
    step: int,
    device: torch.device | str = "cpu",
) -> float:
    """
    One robust training step with loss anomaly monitoring.

    Expected batch keys:
      - panels: [B,16,1,H,W], float
      - label:  [B], long
    """
    where = "training_step"
    if "panels" not in batch or "label" not in batch:
        raise KeyError(f"[{where}] batch must contain 'panels' and 'label'.")

    panels = batch["panels"]
    labels = batch["label"]

    if not isinstance(labels, torch.Tensor):
        raise ModelInputError(f"[{where}] label must be torch.Tensor, got {type(labels)}")
    if labels.ndim != 1:
        raise ModelInputError(f"[{where}] label must be 1D [B], got shape={tuple(labels.shape)}")
    if labels.dtype not in (torch.int64, torch.long):
        raise ModelInputError(
            f"[{where}] label dtype must be torch.long/int64, got {labels.dtype}"
        )
    if not torch.isfinite(labels.float()).all():
        raise ModelInputError(f"[{where}] label contains NaN/Inf.")
    if labels.numel() == 0:
        raise ModelInputError(f"[{where}] label tensor is empty.")

    model.train()
    panels = panels.to(device)
    labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(panels)
    loss = criterion(logits, labels)
    loss_value = float(loss.detach().cpu().item())
    monitor.check(loss_value=loss_value, step=step)

    loss.backward()
    optimizer.step()
    return loss_value

