"""
RAVEN visual reasoning dataset loader (3x3 RPM classification, engineering-ready).

Usage:
    python raven_dataloader.py --data-root data/raven --batch-size 16 --image-size 80
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("raven_loader")
DEFAULT_TRAIN_RULES = ("Center", "Single", "Outlier")
DEFAULT_OOD_RULES = ("Distribute", "Merge", "Progresion")


def setup_logger(log_dir: Path) -> None:
    """Configure timestamped, readable console + file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "raven_dataloader.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def _progress_bar(idx: int, total: int, width: int = 32) -> None:
    """Simple progress bar without extra dependencies."""
    if total <= 0:
        return
    ratio = idx / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    print(f"\rScanning NPZ: [{bar}] {idx}/{total}", end="", flush=True)
    if idx == total:
        print()


def collect_raven_files(data_root: Path) -> List[Path]:
    """Collect all npz files under data root recursively."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    files = sorted(data_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {data_root}")
    return files


def _normalize_panel(panel: np.ndarray) -> np.ndarray:
    """Normalize panel pixels into [0, 1] float32."""
    panel = panel.astype(np.float32)
    if panel.max() > 1.0:
        panel = panel / 255.0
    return panel


def _resize_panel_to_tensor(panel: np.ndarray, image_size: int) -> np.ndarray:
    """Resize one grayscale panel and return [1, H, W]."""
    panel = _normalize_panel(panel)
    panel_uint8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(panel_uint8, mode="L")
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    resized = np.asarray(pil, dtype=np.float32) / 255.0
    return resized[None, :, :]


def _safe_load_npz(npz_path: Path) -> Tuple[bool, str, np.ndarray | None, int | None]:
    """
    Safe npz read with robust checks.

    Returns:
      - valid flag
      - reason code
      - image array if valid
      - label if valid
    """
    if not npz_path.exists():
        return False, "missing_file", None, None
    if npz_path.stat().st_size == 0:
        return False, "empty_file", None, None

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        return False, f"load_failed:{exc.__class__.__name__}", None, None

    if "image" not in data or "target" not in data:
        return False, "missing_image_or_target", None, None

    image = data["image"]
    try:
        label = int(np.array(data["target"]).item())
    except Exception:
        return False, "invalid_target_scalar", None, None

    # Strict shape validation: must be exactly (16, H, W).
    if image.ndim != 3 or image.shape[0] != 16:
        return False, f"invalid_shape:{tuple(image.shape)}", None, None

    # Strict label validation: must be 0~7.
    if not (0 <= label <= 7):
        return False, f"invalid_label:{label}", None, None

    return True, "ok", image, label


def _extract_rule_from_path(npz_path: Path) -> str:
    """Infer rule from parent folder name for strict OOD split."""
    return npz_path.parent.name


@dataclass
class ScanReport:
    total_samples: int
    valid_samples: int
    skipped_samples: int
    class_distribution: Dict[int, int]
    skipped_reasons: Dict[str, int]
    warnings: List[str]


def scan_and_validate_dataset(npz_paths: Sequence[Path]) -> Tuple[List[Path], List[int], ScanReport]:
    """Scan dataset, keep only valid samples, and build a quality report."""
    valid_paths: List[Path] = []
    valid_labels: List[int] = []
    skipped_reasons: Dict[str, int] = {}

    total = len(npz_paths)
    for i, path in enumerate(npz_paths, start=1):
        ok, reason, _, label = _safe_load_npz(path)
        if ok:
            valid_paths.append(path)
            valid_labels.append(label if label is not None else -1)
        else:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            LOGGER.warning("Skip invalid sample: %s | reason=%s", path, reason)
        _progress_bar(i, total)

    class_distribution = {cls_id: 0 for cls_id in range(8)}
    for lb in valid_labels:
        class_distribution[lb] += 1

    warnings: List[str] = []
    valid_count = len(valid_paths)
    if valid_count > 0:
        for cls_id, cnt in class_distribution.items():
            ratio = cnt / valid_count
            if ratio < 0.05:
                msg = (
                    f"类别不平衡警告: class={cls_id}, count={cnt}, ratio={ratio:.2%} (<5%)"
                )
                warnings.append(msg)
                LOGGER.warning(msg)

    report = ScanReport(
        total_samples=total,
        valid_samples=valid_count,
        skipped_samples=total - valid_count,
        class_distribution=class_distribution,
        skipped_reasons=skipped_reasons,
        warnings=warnings,
    )
    return valid_paths, valid_labels, report


class Raven3x3Dataset(Dataset):
    """
    RAVEN 3x3 classification dataset.

    Return dict (training-script aligned):
      - context: [8, 1, H, W]
      - choices: [8, 1, H, W]
      - panels: [16, 1, H, W]
      - label: scalar long
      - path: str
    """

    def __init__(self, file_paths: Sequence[Path], image_size: int = 80) -> None:
        if not file_paths:
            raise ValueError("file_paths is empty.")
        self.file_paths = list(file_paths)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        ok, reason, image, label = _safe_load_npz(path)
        if not ok or image is None or label is None:
            raise RuntimeError(f"Runtime load failure at {path}, reason={reason}")

        context = image[:8]
        choices = image[8:16]

        context_resized = np.stack(
            [_resize_panel_to_tensor(p, self.image_size) for p in context], axis=0
        )
        choices_resized = np.stack(
            [_resize_panel_to_tensor(p, self.image_size) for p in choices], axis=0
        )
        panels = np.concatenate([context_resized, choices_resized], axis=0)

        return {
            "context": torch.from_numpy(context_resized).float(),
            "choices": torch.from_numpy(choices_resized).float(),
            "panels": torch.from_numpy(panels).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


def _check_ood_split_risk(train_labels: Sequence[int], test_labels: Sequence[int]) -> List[str]:
    """Warn if any class is absent in train or test split."""
    risks: List[str] = []
    train_set = set(train_labels)
    test_set = set(test_labels)
    for cls_id in range(8):
        if cls_id not in train_set:
            risks.append(f"OOD 划分风险警告: 训练集缺失 class={cls_id}")
        if cls_id not in test_set:
            risks.append(f"OOD 划分风险警告: 测试集缺失 class={cls_id}")
    return risks


def split_valid_samples(
    valid_paths: Sequence[Path],
    valid_labels: Sequence[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Path], List[Path], List[int], List[int]]:
    """Split valid paths into train/test; stratify when possible."""
    if not valid_paths:
        raise RuntimeError("No valid RAVEN sample available for split.")

    labels_np = np.array(valid_labels)
    _, counts = np.unique(labels_np, return_counts=True)
    can_stratify = len(np.unique(labels_np)) > 1 and counts.min() >= 2
    stratify = labels_np if can_stratify else None

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        list(valid_paths),
        list(valid_labels),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train_paths, test_paths, train_labels, test_labels


def split_by_rules_strict(
    valid_paths: Sequence[Path],
    train_rules: Sequence[str] = DEFAULT_TRAIN_RULES,
    ood_rules: Sequence[str] = DEFAULT_OOD_RULES,
) -> Tuple[List[Path], List[Path], List[int], List[int]]:
    """Strict ID/OOD split by official rule folders (no overlap allowed)."""
    train_rule_set = {r.lower() for r in train_rules}
    ood_rule_set = {r.lower() for r in ood_rules}
    overlap = train_rule_set.intersection(ood_rule_set)
    if overlap:
        raise RuntimeError(f"Train/OOD rule overlap: {sorted(overlap)}")

    train_paths = [p for p in valid_paths if _extract_rule_from_path(p).lower() in train_rule_set]
    test_paths = [p for p in valid_paths if _extract_rule_from_path(p).lower() in ood_rule_set]
    if len(train_paths) == 0 or len(test_paths) == 0:
        raise RuntimeError(
            f"Strict rule split failed: train={len(train_paths)} test={len(test_paths)} "
            f"train_rules={list(train_rules)} ood_rules={list(ood_rules)}"
        )

    train_labels = [int(np.array(np.load(p, allow_pickle=True)["target"]).item()) for p in train_paths]
    test_labels = [int(np.array(np.load(p, allow_pickle=True)["target"]).item()) for p in test_paths]
    return train_paths, test_paths, train_labels, test_labels


def print_report(report: ScanReport) -> None:
    """Print structured dataset report."""
    LOGGER.info("========== 数据集自检报告 ==========")
    LOGGER.info("样本总数: %d", report.total_samples)
    LOGGER.info("有效样本数: %d", report.valid_samples)
    LOGGER.info("跳过样本数: %d", report.skipped_samples)
    LOGGER.info("类别分布(0~7): %s", report.class_distribution)
    if report.skipped_reasons:
        LOGGER.info("跳过原因统计: %s", report.skipped_reasons)
    if report.warnings:
        for msg in report.warnings:
            LOGGER.warning(msg)
    LOGGER.info("===================================")


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    image_size: int = 80,
    strict_rule_split: bool = True,
) -> Tuple[DataLoader, DataLoader, ScanReport, List[str]]:
    """Build train/test dataloaders for RAVEN 3x3 task."""
    data_root = Path(data_root)
    all_paths = collect_raven_files(data_root)
    valid_paths, valid_labels, report = scan_and_validate_dataset(all_paths)
    if strict_rule_split:
        train_paths, test_paths, train_labels, test_labels = split_by_rules_strict(valid_paths)
    else:
        train_paths, test_paths, train_labels, test_labels = split_valid_samples(
            valid_paths, valid_labels, test_size=test_size, random_state=random_state
        )

    split_warnings = _check_ood_split_risk(train_labels, test_labels)
    for warning in split_warnings:
        LOGGER.warning(warning)

    train_set = Raven3x3Dataset(train_paths, image_size=image_size)
    test_set = Raven3x3Dataset(test_paths, image_size=image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
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
    return train_loader, test_loader, report, split_warnings


def self_check(train_loader: DataLoader, report: ScanReport, split_warnings: Sequence[str]) -> bool:
    """
    Self-check criterion:
      pass only if no error/warning and no shape mismatch.
    """
    if len(train_loader.dataset) == 0:
        LOGGER.error("自检失败: 训练集为空。")
        return False

    batch = next(iter(train_loader))
    LOGGER.info("Batch keys: %s", list(batch.keys()))
    LOGGER.info("context shape: %s", tuple(batch["context"].shape))
    LOGGER.info("choices shape: %s", tuple(batch["choices"].shape))
    LOGGER.info("panels shape:  %s", tuple(batch["panels"].shape))
    LOGGER.info("label shape:   %s", tuple(batch["label"].shape))

    required_keys = {"context", "choices", "panels", "label", "path"}
    key_ok = set(batch.keys()) == required_keys
    shape_ok = (
        batch["context"].ndim == 5
        and batch["choices"].ndim == 5
        and batch["panels"].ndim == 5
        and batch["context"].shape[1:] == (8, 1, train_loader.dataset.image_size, train_loader.dataset.image_size)
        and batch["choices"].shape[1:] == (8, 1, train_loader.dataset.image_size, train_loader.dataset.image_size)
        and batch["panels"].shape[1:] == (16, 1, train_loader.dataset.image_size, train_loader.dataset.image_size)
        and batch["label"].ndim == 1
    )

    warning_count = len(report.warnings) + len(split_warnings)
    if not key_ok:
        LOGGER.error("自检失败: 输出字段不匹配。")
        return False
    if not shape_ok:
        LOGGER.error("自检失败: 张量维度不匹配。")
        return False
    if warning_count > 0:
        LOGGER.error("自检失败: 检测到 %d 条警告。", warning_count)
        return False

    LOGGER.info("自检通过: 无报错、无警告、无维度不匹配。")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAVEN 3x3 DataLoader builder")
    parser.add_argument("--data-root", type=str, default="data/raven", help="relative data root path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--log-dir", type=str, default="experiments/logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(Path(args.log_dir))

    train_loader, test_loader, report, split_warnings = build_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    print_report(report)

    LOGGER.info("Train samples: %d", len(train_loader.dataset))
    LOGGER.info("Test samples:  %d", len(test_loader.dataset))
    passed = self_check(train_loader, report, split_warnings)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
