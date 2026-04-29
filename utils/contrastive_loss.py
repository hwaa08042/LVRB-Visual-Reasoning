"""
Contrastive learning module for RAVEN-style visual reasoning training.

This file is designed as a plug-in utility:
- No mandatory change to existing training loop.
- Can be called optionally in train_cnn.py.
- Robust fallback behavior when pair-building or loss computation fails.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger("contrastive_loss")


def setup_logger(log_path: Path = Path("experiments/logs/contrastive_loss.log")) -> None:
    """Configure timestamped logger for contrastive diagnostics."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def extract_rule_ids_from_paths(paths: Sequence[str]) -> List[str]:
    """
    Parse rule id from sample paths.

    Strategy:
      - Use parent directory name as rule tag (common RAVEN organization).
      - If parse fails, fallback to 'unknown_rule'.
    """
    rules: List[str] = []
    for p in paths:
        try:
            rules.append(Path(str(p)).parent.name or "unknown_rule")
        except Exception:
            rules.append("unknown_rule")
    return rules


def validate_feature_tensor(features: torch.Tensor, where: str) -> None:
    """Validate embedding tensor before contrastive operations."""
    if features is None:
        raise ValueError(f"[{where}] features is None.")
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"[{where}] features must be torch.Tensor, got {type(features)}.")
    if features.ndim != 2:
        raise ValueError(
            f"[{where}] features must be [B,D], got ndim={features.ndim}, shape={tuple(features.shape)}."
        )
    if features.shape[0] <= 1:
        raise ValueError(f"[{where}] batch size must be > 1 for pair building.")
    if features.shape[1] <= 0:
        raise ValueError(f"[{where}] feature dim must be > 0.")
    if not torch.isfinite(features).all():
        raise ValueError(f"[{where}] features contain NaN/Inf.")


def validate_model_feature_dim(model: nn.Module, features: torch.Tensor) -> None:
    """
    Verify compatibility between model feature dimension and incoming embeddings.
    """
    if hasattr(model, "feature_dim"):
        model_dim = int(getattr(model, "feature_dim"))
        feat_dim = int(features.shape[1])
        if model_dim != feat_dim:
            raise ValueError(
                "Embedding dim mismatch: model.feature_dim="
                f"{model_dim}, features.shape[1]={feat_dim}."
            )


@dataclass
class PairBuildResult:
    pos_pairs: List[Tuple[int, int]]
    neg_pairs: List[Tuple[int, int]]
    adjusted: bool
    adjust_note: str


def build_pairs_by_rules(
    rule_ids: Sequence[str],
    max_pairs: int = 2048,
) -> PairBuildResult:
    """
    Build positive/negative pairs by rule identity.

    Positive pair: same rule.
    Negative pair: different rule.

    Robust adjustment:
      - If pos or neg count is 0, auto-expand logic by fallback pairing.
    """
    n = len(rule_ids)
    pos_pairs: List[Tuple[int, int]] = []
    neg_pairs: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if rule_ids[i] == rule_ids[j]:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))
            if len(pos_pairs) + len(neg_pairs) >= max_pairs:
                break
        if len(pos_pairs) + len(neg_pairs) >= max_pairs:
            break

    adjusted = False
    note = "未触发样本对调整。"
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        adjusted = True
        # Fallback strategy:
        # create synthetic balanced pairs by index distance.
        pos_pairs = []
        neg_pairs = []
        half = max(1, n // 2)
        for i in range(n - 1):
            # pseudo positive: adjacent
            pos_pairs.append((i, i + 1))
            # pseudo negative: far index
            j = (i + half) % n
            if j == i:
                j = (j + 1) % n
            a, b = min(i, j), max(i, j)
            if a != b:
                neg_pairs.append((a, b))
            if len(pos_pairs) >= max_pairs // 2 and len(neg_pairs) >= max_pairs // 2:
                break
        note = (
            "检测到正/负样本对数量为 0，已自动扩大样本范围并启用索引回退配对策略。"
        )
        LOGGER.warning(note)

    return PairBuildResult(
        pos_pairs=pos_pairs[:max_pairs],
        neg_pairs=neg_pairs[:max_pairs],
        adjusted=adjusted,
        adjust_note=note,
    )


def cosine_contrastive_loss(
    features: torch.Tensor,
    pos_pairs: Sequence[Tuple[int, int]],
    neg_pairs: Sequence[Tuple[int, int]],
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Simple cosine contrastive loss.

    - Positive objective: maximize cosine similarity => loss_pos = 1 - cos
    - Negative objective: constrain cosine <= margin
      loss_neg = relu(cos - margin)
    """
    where = "cosine_contrastive_loss"
    validate_feature_tensor(features, where)
    if margin < -1.0 or margin > 1.0:
        raise ValueError(f"[{where}] margin must be in [-1,1], got {margin}.")
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        raise ValueError(f"[{where}] pos_pairs/neg_pairs must be non-empty.")

    z = F.normalize(features, p=2, dim=1)

    pos_i = torch.tensor([i for i, _ in pos_pairs], device=z.device, dtype=torch.long)
    pos_j = torch.tensor([j for _, j in pos_pairs], device=z.device, dtype=torch.long)
    neg_i = torch.tensor([i for i, _ in neg_pairs], device=z.device, dtype=torch.long)
    neg_j = torch.tensor([j for _, j in neg_pairs], device=z.device, dtype=torch.long)

    pos_sim = (z[pos_i] * z[pos_j]).sum(dim=1)
    neg_sim = (z[neg_i] * z[neg_j]).sum(dim=1)

    loss_pos = (1.0 - pos_sim).mean()
    loss_neg = F.relu(neg_sim - margin).mean()
    loss = loss_pos + loss_neg

    if not torch.isfinite(loss):
        raise ValueError(f"[{where}] loss is NaN/Inf.")
    return loss


@dataclass
class ContrastiveStepInfo:
    contrastive_loss_value: float
    pos_pairs: int
    neg_pairs: int
    used_fallback: bool
    note: str


class ContrastiveLossMonitor:
    """
    Monitor contrastive loss dynamics and pair distribution.
    """

    def __init__(self, spike_factor: float = 5.0) -> None:
        if spike_factor <= 1.0:
            raise ValueError("spike_factor must be > 1.0")
        self.spike_factor = spike_factor
        self.prev_loss: Optional[float] = None
        self.epoch_loss_values: List[float] = []
        self.epoch_pos_pairs: List[int] = []
        self.epoch_neg_pairs: List[int] = []

    def on_step(self, loss_value: float) -> None:
        """Step-level anomaly warning."""
        if not math.isfinite(loss_value):
            LOGGER.warning("对比损失异常: NaN/Inf。建议降低 contrastive_weight 或检查样本对构造。")
            return
        if self.prev_loss is not None and loss_value > self.prev_loss * self.spike_factor:
            LOGGER.warning(
                "对比损失突增: prev=%.6f curr=%.6f。建议降低 contrastive_weight。",
                self.prev_loss,
                loss_value,
            )
        self.prev_loss = loss_value
        self.epoch_loss_values.append(loss_value)

    def on_epoch_pair_stats(self, pos_pairs: int, neg_pairs: int) -> None:
        self.epoch_pos_pairs.append(pos_pairs)
        self.epoch_neg_pairs.append(neg_pairs)

    def summarize_epoch(self) -> Dict[str, float]:
        """Epoch summary + imbalance suggestion."""
        mean_loss = (
            float(sum(self.epoch_loss_values) / len(self.epoch_loss_values))
            if self.epoch_loss_values
            else 0.0
        )
        pos_total = int(sum(self.epoch_pos_pairs))
        neg_total = int(sum(self.epoch_neg_pairs))
        ratio = (pos_total / max(1, neg_total)) if neg_total > 0 else float("inf")

        if pos_total == 0 or neg_total == 0 or ratio > 3.0 or ratio < (1 / 3):
            LOGGER.warning(
                "样本对分布失衡: pos_total=%d neg_total=%d ratio=%.4f。建议增加 batch size "
                "或启用按规则采样。",
                pos_total,
                neg_total,
                ratio,
            )

        summary = {
            "contrastive_loss_mean": mean_loss,
            "pos_pairs_total": float(pos_total),
            "neg_pairs_total": float(neg_total),
            "pos_neg_ratio": float(ratio) if math.isfinite(ratio) else 1e9,
        }

        # Reset for next epoch.
        self.epoch_loss_values.clear()
        self.epoch_pos_pairs.clear()
        self.epoch_neg_pairs.clear()
        return summary


class ContrastiveLossAdapter:
    """
    Plug-in adapter to compute robust contrastive loss with fallback.

    Integration in training:
      extra_loss, info = adapter.compute(
          model=model,
          features=embeddings,  # [B,D]
          paths=batch["path"],
          ce_loss=ce_loss,       # optional fallback target
      )
      total_loss = ce_loss + contrastive_weight * extra_loss
    """

    def __init__(
        self,
        margin: float = 0.2,
        max_pairs: int = 2048,
    ) -> None:
        self.margin = margin
        self.max_pairs = max_pairs
        self.monitor = ContrastiveLossMonitor(spike_factor=5.0)

    def compute(
        self,
        model: nn.Module,
        features: torch.Tensor,
        paths: Sequence[str],
        ce_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ContrastiveStepInfo]:
        """
        Robust compute entrypoint.

        Fallback policy:
          - On any exception, if ce_loss is provided, return ce_loss.detach()*0 (no extra penalty).
          - Otherwise return zero tensor scalar on same device.
        """
        where = "ContrastiveLossAdapter.compute"
        device = features.device if isinstance(features, torch.Tensor) else torch.device("cpu")
        fallback_loss = (
            ce_loss.detach() * 0.0
            if isinstance(ce_loss, torch.Tensor)
            else torch.zeros((), device=device, dtype=torch.float32)
        )

        try:
            validate_feature_tensor(features, where)
            validate_model_feature_dim(model, features)

            if len(paths) != int(features.shape[0]):
                raise ValueError(
                    f"[{where}] len(paths)={len(paths)} does not match batch size={features.shape[0]}."
                )

            rule_ids = extract_rule_ids_from_paths(paths)
            pair_result = build_pairs_by_rules(rule_ids, max_pairs=self.max_pairs)
            loss = cosine_contrastive_loss(
                features=features,
                pos_pairs=pair_result.pos_pairs,
                neg_pairs=pair_result.neg_pairs,
                margin=self.margin,
            )

            loss_value = float(loss.detach().cpu().item())
            self.monitor.on_step(loss_value)
            self.monitor.on_epoch_pair_stats(
                pos_pairs=len(pair_result.pos_pairs), neg_pairs=len(pair_result.neg_pairs)
            )

            return loss, ContrastiveStepInfo(
                contrastive_loss_value=loss_value,
                pos_pairs=len(pair_result.pos_pairs),
                neg_pairs=len(pair_result.neg_pairs),
                used_fallback=False,
                note=pair_result.adjust_note,
            )
        except Exception as exc:
            LOGGER.warning(
                "对比损失计算失败，自动切换回普通交叉熵流程。reason=%s", exc
            )
            return fallback_loss, ContrastiveStepInfo(
                contrastive_loss_value=0.0,
                pos_pairs=0,
                neg_pairs=0,
                used_fallback=True,
                note="contrastive_fallback_to_ce",
            )

