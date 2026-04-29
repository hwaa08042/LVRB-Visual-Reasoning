"""
Perturbation robustness test for RAVEN CNN baseline.

Implemented perturbations:
1) Gaussian noise
2) Random color/intensity jitter
3) Random spatial shift
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "experiments/.mplconfig")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn_baseline import CNNBaseline, load_weights
from utils.data_loader import DEFAULT_OOD_RULES, RavenDataset, SampleMeta, scan_raven_samples


LOG_PATH = Path("experiments/logs/robustness_test.log")
REPORT_PATH = Path("experiments/results/robustness_report.txt")
PLOT_PATH = Path("experiments/results/robustness_plot.png")
WEIGHT_PATH = Path("experiments/weights/cnn_baseline.pth")

LOGGER = logging.getLogger("robustness_test")


def setup_logger() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def progress_bar(step: int, total: int, prefix: str = "", width: int = 30) -> None:
    if total <= 0:
        return
    ratio = step / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    print(f"\r{prefix}[{bar}] {step}/{total}", end="", flush=True)
    if step == total:
        print()


def build_ood_test_subset(samples: Sequence[SampleMeta]) -> List[SampleMeta]:
    """Use strict OOD rules subset for robustness evaluation."""
    ood_set = {r.lower() for r in DEFAULT_OOD_RULES}
    ood_samples = [s for s in samples if s.rule.lower() in ood_set]
    if len(ood_samples) == 0:
        raise RuntimeError(
            f"No OOD-rule samples found. expected rules={list(DEFAULT_OOD_RULES)}"
        )
    return ood_samples


def validate_noise_std(std: float) -> float:
    if std <= 0 or std > 1.0:
        LOGGER.warning("非法噪声强度 std=%s，自动改为默认 0.1", std)
        return 0.1
    return std


def validate_jitter_strength(strength: float) -> float:
    if strength <= 0 or strength > 1.0:
        LOGGER.warning("非法颜色扰动强度 strength=%s，自动改为默认 0.2", strength)
        return 0.2
    return strength


def validate_shift_ratio(shift_ratio: float) -> float:
    if shift_ratio < 0 or shift_ratio > 0.5:
        LOGGER.warning("非法偏移幅度 shift_ratio=%s，自动改为默认 0.1", shift_ratio)
        return 0.1
    return shift_ratio


def validate_panels_tensor(panels: torch.Tensor, where: str) -> None:
    if not isinstance(panels, torch.Tensor):
        raise ValueError(f"[{where}] panels must be torch.Tensor, got {type(panels)}")
    if panels.ndim != 5:
        raise ValueError(f"[{where}] panels ndim must be 5, got {panels.ndim}")
    if panels.shape[2] != 1:
        raise ValueError(f"[{where}] expected channel=1, got {panels.shape[2]}")
    if panels.shape[3] <= 0 or panels.shape[4] <= 0:
        raise ValueError(f"[{where}] invalid H/W: {panels.shape[3:]}")


def perturb_gaussian_noise(panels: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise and clamp to [0,1]."""
    where = "perturb_gaussian_noise"
    validate_panels_tensor(panels, where)
    noise = torch.randn_like(panels) * std
    out = torch.clamp(panels + noise, 0.0, 1.0)
    return out


def perturb_color_jitter(panels: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Random intensity jitter on grayscale images.
    Equivalent to random brightness + contrast-like scaling.
    """
    where = "perturb_color_jitter"
    validate_panels_tensor(panels, where)
    bsz, n, c, h, w = panels.shape
    scale = 1.0 + (torch.rand((bsz, n, 1, 1, 1), device=panels.device) * 2 - 1) * strength
    shift = (torch.rand((bsz, n, 1, 1, 1), device=panels.device) * 2 - 1) * strength
    out = torch.clamp(panels * scale + shift, 0.0, 1.0)
    if out.shape != (bsz, n, c, h, w):
        raise RuntimeError(f"[{where}] shape changed unexpectedly: {out.shape}")
    return out


def perturb_random_shift(panels: torch.Tensor, shift_ratio: float) -> torch.Tensor:
    """Random integer translation by zero-padding and cropping."""
    where = "perturb_random_shift"
    validate_panels_tensor(panels, where)
    bsz, n, c, h, w = panels.shape
    max_dx = max(1, int(h * shift_ratio))
    max_dy = max(1, int(w * shift_ratio))
    out = torch.empty_like(panels)

    for bi in range(bsz):
        for pi in range(n):
            img = panels[bi, pi]  # [1,H,W]
            dx = int(torch.randint(-max_dx, max_dx + 1, (1,)).item())
            dy = int(torch.randint(-max_dy, max_dy + 1, (1,)).item())

            # Padding order: (left, right, top, bottom)
            padded = F.pad(img, (max_dy, max_dy, max_dx, max_dx), mode="constant", value=0.0)
            x0 = max_dx + dx
            y0 = max_dy + dy
            crop = padded[:, x0 : x0 + h, y0 : y0 + w]
            if crop.shape != img.shape:
                raise RuntimeError(
                    f"[{where}] crop shape mismatch. expect={img.shape}, got={crop.shape}"
                )
            out[bi, pi] = crop
    return out


@dataclass
class TestMetrics:
    name: str
    params: Dict[str, float]
    clean_acc: float
    perturbed_acc: float
    drop_abs: float
    anomalies: int


def safe_accuracy(correct: int, total: int) -> float:
    try:
        if total <= 0:
            raise ZeroDivisionError("total=0")
        value = correct / total
        if not math.isfinite(value):
            raise ValueError("NaN/Inf")
        return value
    except Exception as exc:
        LOGGER.error("准确率计算异常: %s，回退为 0.0", exc)
        return 0.0


@torch.no_grad()
def evaluate_with_optional_perturb(
    model: CNNBaseline,
    loader: DataLoader,
    device: torch.device,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    timeout_ms: float = 1500.0,
) -> Tuple[float, int]:
    """
    Evaluate model accuracy.
    Returns:
      accuracy, anomaly_count
    """
    model.eval()
    total = 0
    correct = 0
    anomaly_count = 0

    for step, batch in enumerate(loader, start=1):
        progress_bar(step, len(loader), prefix="Robustness Eval ")
        paths = batch.get("path", [])
        if not isinstance(paths, list):
            paths = list(paths) if paths is not None else []
        try:
            panels = batch["panels"].to(device)
            labels = batch["label"].to(device)
            validate_panels_tensor(panels, "evaluate_with_optional_perturb")

            if perturb_fn is not None:
                try:
                    panels = perturb_fn(panels)
                except Exception as exc:
                    anomaly_count += int(labels.size(0))
                    LOGGER.error(
                        "扰动异常，跳过该 batch。reason=%s sample_paths=%s",
                        exc,
                        paths[:3],
                    )
                    continue

            t0 = time.perf_counter()
            logits, probs = model(panels)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > timeout_ms:
                anomaly_count += int(labels.size(0))
                LOGGER.warning(
                    "推理超时异常: %.2fms > %.2fms, sample_paths=%s",
                    elapsed_ms,
                    timeout_ms,
                    paths[:3],
                )

            # Probability anomaly: all probs <= 0 for a sample.
            zero_prob_mask = (probs <= 0).all(dim=1)
            if zero_prob_mask.any():
                idxs = torch.where(zero_prob_mask)[0].tolist()
                anomaly_count += len(idxs)
                LOGGER.warning("概率异常样本数=%d, sample_paths=%s", len(idxs), paths[:3])

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
        except Exception as exc:
            anomaly_count += 1
            LOGGER.error("推理异常: %s | sample_paths=%s", exc, paths[:3])
            continue

    return safe_accuracy(correct, total), anomaly_count


def save_report(metrics: Sequence[TestMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranking = sorted(metrics, key=lambda x: x.drop_abs, reverse=True)

    lines = [
        "RAVEN Robustness Test Report",
        "===========================",
    ]
    for m in metrics:
        lines.extend(
            [
                f"Perturbation: {m.name}",
                f"Parameters: {m.params}",
                f"Clean accuracy: {m.clean_acc * 100:.4f}%",
                f"Perturbed accuracy: {m.perturbed_acc * 100:.4f}%",
                f"Accuracy drop: {m.drop_abs * 100:.4f}%",
                f"Anomaly samples: {m.anomalies}",
                "",
            ]
        )

    lines.append("Robustness ranking (worst -> best):")
    for i, m in enumerate(ranking, start=1):
        lines.append(f"{i}. {m.name} | drop={m.drop_abs * 100:.4f}%")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("鲁棒性报告已保存: %s", out_path)


def save_plot(metrics: Sequence[TestMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [m.name for m in metrics]
    clean = [m.clean_acc * 100 for m in metrics]
    perturbed = [m.perturbed_acc * 100 for m in metrics]

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, clean, width=width, label="Clean")
    plt.bar(x + width / 2, perturbed, width=width, label="Perturbed")
    plt.xticks(x, names, rotation=10)
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    LOGGER.info("鲁棒性对比图已保存: %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAVEN perturbation robustness test")
    parser.add_argument("--data_root", type=str, default="data/raven")
    parser.add_argument("--weights", type=str, default=str(WEIGHT_PATH))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--jitter_strength", type=float, default=0.2)
    parser.add_argument("--shift_ratio", type=float, default=0.1)
    parser.add_argument("--timeout_ms", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    device = torch.device("cpu")

    noise_std = validate_noise_std(args.noise_std)
    jitter_strength = validate_jitter_strength(args.jitter_strength)
    shift_ratio = validate_shift_ratio(args.shift_ratio)
    timeout_ms = args.timeout_ms if args.timeout_ms > 0 else 1500.0

    # 1) Load model and validate checkpoint integrity.
    model = CNNBaseline(num_choices=8, feature_dim=128).to(device)
    try:
        load_weights(model, path=Path(args.weights), strict=True)
    except Exception as exc:
        LOGGER.error("模型加载失败: %s", exc)
        LOGGER.error(
            "请检查: 1) 权重路径是否正确; 2) 文件是否非空; 3) 权重与当前模型结构是否兼容。"
        )
        raise SystemExit(1)

    # 2) Build strict OOD test loader.
    samples, _stats = scan_raven_samples(Path(args.data_root))
    test_samples = build_ood_test_subset(samples)
    if len(test_samples) == 0:
        LOGGER.error("测试集为空，无法执行鲁棒性测试。")
        raise SystemExit(1)

    test_loader = DataLoader(
        RavenDataset(test_samples, image_size=args.image_size, output_channels=1),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=False,
    )

    # 3) Evaluate clean baseline once.
    clean_acc, clean_anomalies = evaluate_with_optional_perturb(
        model=model,
        loader=test_loader,
        device=device,
        perturb_fn=None,
        timeout_ms=timeout_ms,
    )
    LOGGER.info("Clean accuracy: %.4f%%", clean_acc * 100)

    perturb_configs = [
        (
            "gaussian_noise",
            {"noise_std": noise_std},
            lambda x: perturb_gaussian_noise(x, std=noise_std),
        ),
        (
            "color_jitter",
            {"jitter_strength": jitter_strength},
            lambda x: perturb_color_jitter(x, strength=jitter_strength),
        ),
        (
            "random_shift",
            {"shift_ratio": shift_ratio},
            lambda x: perturb_random_shift(x, shift_ratio=shift_ratio),
        ),
    ]

    metrics: List[TestMetrics] = []
    for name, params, fn in perturb_configs:
        LOGGER.info("开始测试扰动: %s | params=%s", name, params)
        pert_acc, pert_anomalies = evaluate_with_optional_perturb(
            model=model,
            loader=test_loader,
            device=device,
            perturb_fn=fn,
            timeout_ms=timeout_ms,
        )
        drop_abs = max(0.0, clean_acc - pert_acc)
        metric = TestMetrics(
            name=name,
            params=params,
            clean_acc=clean_acc,
            perturbed_acc=pert_acc,
            drop_abs=drop_abs,
            anomalies=clean_anomalies + pert_anomalies,
        )
        metrics.append(metric)

        LOGGER.info(
            "%s | clean=%.4f%% perturbed=%.4f%% drop=%.4f%%",
            name,
            clean_acc * 100,
            pert_acc * 100,
            drop_abs * 100,
        )

        # AI monitoring: warning when drop > 50%.
        if drop_abs > 0.5:
            LOGGER.warning("鲁棒性不足警告: 扰动 %s 导致准确率下降超过 50%%", name)

    # 4) Reasonability checks.
    for m in metrics:
        if m.perturbed_acc > m.clean_acc:
            LOGGER.warning(
                "结果合理性提示: 扰动后准确率高于扰动前 (%s: %.4f%% > %.4f%%)。"
                "可能原因: 样本随机性、模型欠拟合、扰动实现偏弱。",
                m.name,
                m.perturbed_acc * 100,
                m.clean_acc * 100,
            )
        if m.perturbed_acc < 0 or m.perturbed_acc > 1:
            LOGGER.warning(
                "结果异常: %s 的扰动后准确率超出范围 [0,1]。请检查评估流程。", m.name
            )

    save_report(metrics, REPORT_PATH)
    save_plot(metrics, PLOT_PATH)

    ranking = sorted(metrics, key=lambda x: x.drop_abs, reverse=True)
    LOGGER.info("鲁棒性影响排名（高->低）: %s", [f"{m.name}:{m.drop_abs*100:.2f}%" for m in ranking])


if __name__ == "__main__":
    main()

