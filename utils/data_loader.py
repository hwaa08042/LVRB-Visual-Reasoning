"""
Robust RAVEN data loader for 3x3 visual reasoning classification.

Features:
- Scan and validate NPZ samples under data/raven.
- Filter invalid/corrupted/empty samples with reason logging.
- Split dataset into train/val/test = 70%/10%/20%.
- Build PyTorch DataLoaders compatible with model/cnn_baseline.py.
- Output dataset statistics report including class/rule distribution.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("raven_data_loader")
DEFAULT_TRAIN_RULES = ("Center", "Single", "Outlier")
DEFAULT_OOD_RULES = ("Distribute", "Merge", "Progresion")


def setup_logger(log_path: Path = Path("experiments/logs/data_loader.log")) -> None:
    """Set a timestamped logger for data-loading diagnostics."""
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


def _progress(idx: int, total: int, width: int = 30) -> None:
    """Simple terminal progress bar."""
    if total <= 0:
        return
    ratio = idx / total
    n_done = int(width * ratio)
    bar = "#" * n_done + "-" * (width - n_done)
    print(f"\rLoading RAVEN: [{bar}] {idx}/{total}", end="", flush=True)
    if idx == total:
        print()


def _safe_to_int(value: np.ndarray) -> int:
    """Convert scalar-like numpy value to int with explicit checks."""
    try:
        return int(np.array(value).item())
    except Exception as exc:
        raise ValueError(f"label conversion failed: {exc}") from exc


def _extract_rule_name(npz_path: Path, npz_data: np.lib.npyio.NpzFile) -> str:
    """
    Extract rule/structure tag for monitoring.
    Priority:
      1) npz key 'structure'
      2) parent directory name
    """
    if "structure" in npz_data:
        structure = npz_data["structure"]
        try:
            return str(np.array(structure).item())
        except Exception:
            return str(structure)
    return npz_path.parent.name


def _validate_npz(path: Path) -> Tuple[bool, str, np.ndarray | None, int | None, str]:
    """
    Validate one NPZ sample.

    Expected:
      - keys: image, target
      - image shape: (16, H, W)
      - label range: 0..7
    """
    if not path.exists():
        return False, "missing_file", None, None, "unknown"
    if path.stat().st_size == 0:
        return False, "empty_file", None, None, "unknown"

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return False, f"npz_load_failed:{exc.__class__.__name__}", None, None, "unknown"

    if "image" not in data or "target" not in data:
        return False, "missing_image_or_target", None, None, _extract_rule_name(path, data)

    image = data["image"]
    rule = _extract_rule_name(path, data)

    if image is None or image.size == 0:
        return False, "empty_image", None, None, rule
    if image.ndim != 3 or image.shape[0] != 16:
        return False, f"invalid_shape:{tuple(image.shape)}", None, None, rule

    try:
        label = _safe_to_int(data["target"])
    except ValueError as exc:
        return False, f"invalid_label:{exc}", None, None, rule

    if not (0 <= label <= 7):
        return False, f"label_out_of_range:{label}", None, None, rule

    return True, "ok", image, label, rule


def _normalize_panel(panel: np.ndarray) -> np.ndarray:
    """Normalize panel values to float32 [0,1]."""
    panel = panel.astype(np.float32)
    if panel.max() > 1.0:
        panel = panel / 255.0
    return panel


def _resize_panel(panel: np.ndarray, image_size: int, output_channels: int) -> np.ndarray:
    """
    Resize one panel and normalize channel layout.

    output_channels:
      - 1: grayscale [1,H,W] (recommended for cnn_baseline)
      - 3: pseudo-RGB [3,H,W] by channel replication
    """
    panel = _normalize_panel(panel)
    panel_u8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(panel_u8, mode="L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0  # [H,W]
    if output_channels == 1:
        return arr[None, :, :]
    if output_channels == 3:
        return np.stack([arr, arr, arr], axis=0)
    raise ValueError(f"output_channels must be 1 or 3, got {output_channels}")


@dataclass
class SampleMeta:
    path: Path
    label: int
    rule: str


@dataclass
class DatasetStats:
    total_files: int
    valid_samples: int
    filtered_samples: int
    filtered_reasons: Dict[str, int]
    class_distribution: Dict[int, int]
    rule_distribution: Dict[str, int]
    warnings: List[str]


def scan_raven_samples(data_root: Path) -> Tuple[List[SampleMeta], DatasetStats]:
    """Scan and validate all NPZ files under data root."""
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    npz_files = sorted(data_root.rglob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No NPZ files found under: {data_root}")

    metas: List[SampleMeta] = []
    filtered_reasons: Dict[str, int] = {}
    class_counter: Counter = Counter()
    rule_counter: Counter = Counter()

    for idx, file_path in enumerate(npz_files, start=1):
        ok, reason, _, label, rule = _validate_npz(file_path)
        if ok and label is not None:
            metas.append(SampleMeta(path=file_path, label=label, rule=rule))
            class_counter[label] += 1
            rule_counter[rule] += 1
        else:
            filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
            LOGGER.warning("Skip sample: %s | reason=%s", file_path, reason)
        _progress(idx, len(npz_files))

    warnings: List[str] = []
    valid_count = len(metas)
    for cls_id in range(8):
        class_counter.setdefault(cls_id, 0)

    if valid_count > 0:
        for cls_id, count in sorted(class_counter.items()):
            ratio = count / valid_count
            if ratio < 0.05:
                msg = (
                    f"样本分布失衡警告: class={cls_id}, count={count}, ratio={ratio:.2%} (<5%)"
                )
                warnings.append(msg)
                LOGGER.warning(msg)

    stats = DatasetStats(
        total_files=len(npz_files),
        valid_samples=valid_count,
        filtered_samples=len(npz_files) - valid_count,
        filtered_reasons=dict(filtered_reasons),
        class_distribution=dict(sorted(class_counter.items())),
        rule_distribution=dict(sorted(rule_counter.items(), key=lambda x: x[0])),
        warnings=warnings,
    )
    return metas, stats


class RavenDataset(Dataset):
    """
    Robust RAVEN dataset.

    Return keys (fully compatible with cnn_baseline and train script):
      - context: [8,C,H,W]
      - choices: [8,C,H,W]
      - panels: [16,C,H,W]
      - label: scalar long
      - path: str
    """

    def __init__(
        self,
        samples: Sequence[SampleMeta],
        image_size: int = 224,
        output_channels: int = 1,
    ) -> None:
        if len(samples) == 0:
            raise ValueError("RavenDataset requires non-empty samples.")
        self.samples = list(samples)
        self.image_size = image_size
        self.output_channels = output_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        ok, reason, image, label, _ = _validate_npz(sample.path)
        if not ok or image is None or label is None:
            # Runtime-safe fallback: expose details for upper-level try/except.
            raise RuntimeError(f"Failed to load sample at {sample.path}: {reason}")

        context = np.stack(
            [
                _resize_panel(panel, image_size=self.image_size, output_channels=self.output_channels)
                for panel in image[:8]
            ],
            axis=0,
        )
        choices = np.stack(
            [
                _resize_panel(panel, image_size=self.image_size, output_channels=self.output_channels)
                for panel in image[8:16]
            ],
            axis=0,
        )
        panels = np.concatenate([context, choices], axis=0)

        return {
            "context": torch.from_numpy(context).float(),
            "choices": torch.from_numpy(choices).float(),
            "panels": torch.from_numpy(panels).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(sample.path),
        }


def _split_samples(
    samples: Sequence[SampleMeta],
    seed: int = 42,
) -> Tuple[List[SampleMeta], List[SampleMeta], List[SampleMeta]]:
    """Split into train/val/test = 70/10/20 with stratification when possible."""
    if len(samples) < 5:
        raise RuntimeError("Not enough valid samples for 70/10/20 split; need at least 5.")

    labels = [s.label for s in samples]
    label_counts = Counter(labels)
    can_stratify = len(label_counts) > 1 and min(label_counts.values()) >= 2
    stratify = labels if can_stratify else None

    # First split: train 70% / temp 30%.
    train_samples, temp_samples = train_test_split(
        list(samples),
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    temp_labels = [s.label for s in temp_samples]
    temp_counts = Counter(temp_labels)
    can_stratify_temp = len(temp_counts) > 1 and min(temp_counts.values()) >= 2
    stratify_temp = temp_labels if can_stratify_temp else None

    # Second split: val 10% / test 20% from original => val:test = 1:2 in temp.
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=2.0 / 3.0,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )
    return train_samples, val_samples, test_samples


def print_dataset_report(stats: DatasetStats) -> None:
    """Print robust dataset report for monitoring."""
    LOGGER.info("========== RAVEN 数据集统计报告 ==========")
    LOGGER.info("总文件数: %d", stats.total_files)
    LOGGER.info("有效样本数: %d", stats.valid_samples)
    LOGGER.info("过滤样本数: %d", stats.filtered_samples)
    LOGGER.info("过滤原因统计: %s", stats.filtered_reasons if stats.filtered_reasons else "{}")
    LOGGER.info("类别分布(0~7): %s", stats.class_distribution)
    LOGGER.info("规则分布: %s", stats.rule_distribution if stats.rule_distribution else "{}")
    if stats.warnings:
        for msg in stats.warnings:
            LOGGER.warning(msg)
    LOGGER.info("========================================")


def build_raven_dataloaders(
    data_root: str | Path = "data/raven",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    output_channels: int = 1,
    seed: int = 42,
    strict_rule_split: bool = True,
    train_rules: Sequence[str] = DEFAULT_TRAIN_RULES,
    ood_rules: Sequence[str] = DEFAULT_OOD_RULES,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetStats]:
    """
    Build train/val/test dataloaders.

    Returns:
      train_loader, val_loader, test_loader, dataset_stats
    """
    setup_logger()
    data_root = Path(data_root)
    samples, stats = scan_raven_samples(data_root=data_root)
    print_dataset_report(stats)

    if len(samples) == 0:
        raise RuntimeError("No valid samples after filtering.")

    if strict_rule_split:
        train_rule_set = {r.lower() for r in train_rules}
        ood_rule_set = {r.lower() for r in ood_rules}
        overlap = train_rule_set.intersection(ood_rule_set)
        if overlap:
            raise RuntimeError(f"Train/OOD rule overlap detected: {sorted(overlap)}")

        train_pool = [s for s in samples if s.rule.lower() in train_rule_set]
        ood_test_samples = [s for s in samples if s.rule.lower() in ood_rule_set]
        if len(train_pool) < 5:
            raise RuntimeError(
                "Not enough train-rule samples for strict OOD split. "
                f"rules={list(train_rules)}, count={len(train_pool)}"
            )
        if len(ood_test_samples) == 0:
            raise RuntimeError(
                "No OOD-rule samples found for strict OOD split. "
                f"rules={list(ood_rules)}"
            )

        # Split only ID pool into train/val (90/10), keep OOD test fixed.
        train_samples, val_samples, _id_test_unused = _split_samples(samples=train_pool, seed=seed)
        test_samples = ood_test_samples
        LOGGER.info(
            "Strict OOD split | train_rules=%s ood_rules=%s | train=%d val=%d ood_test=%d",
            list(train_rules),
            list(ood_rules),
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
    else:
        train_samples, val_samples, test_samples = _split_samples(samples=samples, seed=seed)
    LOGGER.info("Split result | train=%d val=%d test=%d", len(train_samples), len(val_samples), len(test_samples))

    train_set = RavenDataset(train_samples, image_size=image_size, output_channels=output_channels)
    val_set = RavenDataset(val_samples, image_size=image_size, output_channels=output_channels)
    test_set = RavenDataset(test_samples, image_size=image_size, output_channels=output_channels)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, stats


def write_rule_distribution_report(
    data_root: str | Path = "data/raven",
    train_rules: Sequence[str] = DEFAULT_TRAIN_RULES,
    ood_rules: Sequence[str] = DEFAULT_OOD_RULES,
    out_path: str | Path = "experiments/results/rule_distribution_report.txt",
) -> None:
    """Write academic-style rule distribution report for strict OOD setting."""
    setup_logger()
    samples, stats = scan_raven_samples(Path(data_root))
    train_rule_set = {r.lower() for r in train_rules}
    ood_rule_set = {r.lower() for r in ood_rules}

    train_counter = Counter(s.rule for s in samples if s.rule.lower() in train_rule_set)
    ood_counter = Counter(s.rule for s in samples if s.rule.lower() in ood_rule_set)
    overlap = train_rule_set.intersection(ood_rule_set)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "RAVEN Rule Distribution Report\n"
        "==============================\n"
        f"Train rules: {list(train_rules)}\n"
        f"OOD test rules: {list(ood_rules)}\n"
        f"Rule overlap: {sorted(overlap)}\n\n"
        f"Total valid samples: {stats.valid_samples}\n"
        f"Train-rule sample counts: {dict(train_counter)}\n"
        f"OOD-rule sample counts: {dict(ood_counter)}\n"
    )
    out_path.write_text(text, encoding="utf-8")
    LOGGER.info("Rule distribution report saved to %s", out_path)

