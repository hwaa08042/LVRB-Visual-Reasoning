"""
OOD generalization test for RAVEN CNN baseline.

Goals:
- Load trained model from experiments/weights/cnn_baseline.pth
- Build seen-rule and unseen-rule test subsets
- Evaluate accuracy for both subsets and compare
- Monitor inference speed and abnormal predictions
- Save report + bar plot to experiments/results
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "experiments/.mplconfig")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn_baseline import CNNBaseline, load_weights
from utils.data_loader import (
    DEFAULT_OOD_RULES,
    DEFAULT_TRAIN_RULES,
    RavenDataset,
    SampleMeta,
    scan_raven_samples,
    setup_logger as setup_data_logger,
)


LOG_PATH = Path("experiments/logs/generalization_test.log")
WEIGHT_PATH = Path("experiments/weights/cnn_baseline.pth")
REPORT_PATH = Path("experiments/results/generalization_report.txt")
PLOT_PATH = Path("experiments/results/generalization_plot.png")

LOGGER = logging.getLogger("generalization_test")


def setup_logger() -> None:
    """Configure timestamped logger."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def progress_bar(step: int, total: int, prefix: str = "", width: int = 30) -> None:
    """Simple progress bar for evaluation."""
    if total <= 0:
        return
    ratio = step / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    print(f"\r{prefix}[{bar}] {step}/{total}", end="", flush=True)
    if step == total:
        print()


@dataclass
class EvalAnomaly:
    path: str
    reason: str
    detail: str


@dataclass
class EvalResult:
    name: str
    total: int
    correct: int
    accuracy: float
    avg_batch_time_ms: float
    avg_sample_time_ms: float
    anomalies: List[EvalAnomaly]


def _safe_accuracy(correct: int, total: int, tag: str) -> float:
    """Safe accuracy computation with fallback."""
    try:
        if total <= 0:
            raise ZeroDivisionError("total is 0")
        acc = correct / total
        if not math.isfinite(acc):
            raise ValueError("accuracy is NaN/Inf")
        return acc
    except Exception as exc:
        LOGGER.error("Accuracy fallback for %s due to %s. Using 0.0.", tag, exc)
        return 0.0


def build_strict_id_ood_subsets(samples: Sequence[SampleMeta]) -> Tuple[List[SampleMeta], List[SampleMeta]]:
    """Strict academic split: ID from train-rules, OOD from unseen-rules."""
    train_set = {r.lower() for r in DEFAULT_TRAIN_RULES}
    ood_set = {r.lower() for r in DEFAULT_OOD_RULES}
    overlap = train_set.intersection(ood_set)
    if overlap:
        raise RuntimeError(f"Train/OOD rule overlap: {sorted(overlap)}")
    id_samples = [s for s in samples if s.rule.lower() in train_set]
    ood_samples = [s for s in samples if s.rule.lower() in ood_set]
    if len(id_samples) == 0 or len(ood_samples) == 0:
        raise RuntimeError(
            f"Strict ID/OOD split invalid: id={len(id_samples)} ood={len(ood_samples)}. "
            "Please ensure rule folders follow RAVEN official types."
        )
    return id_samples, ood_samples


def evaluate_subset(
    model: CNNBaseline,
    loader: DataLoader,
    name: str,
    device: torch.device,
    slow_batch_threshold_ms: float = 500.0,
) -> EvalResult:
    """Evaluate one subset with speed + abnormal inference monitoring."""
    model.eval()
    total = 0
    correct = 0
    batch_times: List[float] = []
    anomalies: List[EvalAnomaly] = []

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            progress_bar(step, len(loader), prefix=f"Eval {name:7s} ")
            t0 = time.perf_counter()
            try:
                panels = batch["panels"].to(device)
                labels = batch["label"].to(device)
                paths = batch.get("path", [])
                if not isinstance(paths, list):
                    paths = list(paths) if paths is not None else []

                logits, probs = model(panels)
                preds = torch.argmax(logits, dim=1)

                batch_correct = int((preds == labels).sum().item())
                batch_total = int(labels.size(0))
                total += batch_total
                correct += batch_correct

                # Abnormal prediction monitoring.
                # Softmax theoretically > 0, but numerical underflow can yield zeros.
                zero_prob_mask = (probs <= 0).all(dim=1)
                if zero_prob_mask.any():
                    idxs = torch.where(zero_prob_mask)[0].tolist()
                    for i in idxs[:5]:
                        p = paths[i] if i < len(paths) else "unknown_path"
                        anomalies.append(
                            EvalAnomaly(
                                path=str(p),
                                reason="zero_probability",
                                detail="all class probabilities are <= 0",
                            )
                        )
            except Exception as exc:
                anomalies.append(
                    EvalAnomaly(
                        path="batch_level",
                        reason="inference_exception",
                        detail=str(exc),
                    )
                )
                LOGGER.error("Inference error on %s subset: %s", name, exc)
                continue
            finally:
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                batch_times.append(elapsed_ms)
                if elapsed_ms > slow_batch_threshold_ms:
                    anomalies.append(
                        EvalAnomaly(
                            path="batch_level",
                            reason="slow_inference",
                            detail=f"batch_time_ms={elapsed_ms:.2f} > {slow_batch_threshold_ms:.2f}",
                        )
                    )

    accuracy = _safe_accuracy(correct, total, tag=name)
    avg_batch = float(np.mean(batch_times)) if batch_times else 0.0
    avg_sample = (avg_batch / loader.batch_size) if loader.batch_size and loader.batch_size > 0 else 0.0
    return EvalResult(
        name=name,
        total=total,
        correct=correct,
        accuracy=accuracy,
        avg_batch_time_ms=avg_batch,
        avg_sample_time_ms=avg_sample,
        anomalies=anomalies,
    )


def validate_result_range(result: EvalResult) -> List[str]:
    """Validate result reasonability and return warning hints."""
    hints: List[str] = []
    if result.accuracy < 0.0 or result.accuracy > 1.0:
        hints.append(
            f"{result.name} 准确率异常（{result.accuracy:.6f}），可能原因：模型未训练好、数据划分错误、标签异常。"
        )
    return hints


def save_plot(id_acc: float, ood_acc: float, out_path: Path) -> None:
    """Save simple bar chart comparing ID/OOD accuracy."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["ID Rules", "OOD Rules"]
    values = [id_acc * 100.0, ood_acc * 100.0]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.title("OOD Generalization Accuracy")
    plt.ylim(0, 100)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_report(
    id_result: EvalResult,
    ood_result: EvalResult,
    extra_hints: Sequence[str],
    out_path: Path,
) -> None:
    """Generate and save strict OOD report text."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abs_drop = (id_result.accuracy - ood_result.accuracy) * 100.0
    rel_drop = (abs_drop / max(1e-8, id_result.accuracy * 100.0)) * 100.0 if id_result.accuracy > 0 else 0.0

    anomaly_counter = defaultdict(int)
    for a in id_result.anomalies + ood_result.anomalies:
        anomaly_counter[a.reason] += 1

    report = (
        "RAVEN OOD Generalization Report\n"
        "===============================\n"
        f"ID rules: {list(DEFAULT_TRAIN_RULES)}\n"
        f"OOD rules: {list(DEFAULT_OOD_RULES)}\n"
        f"ID test size: {id_result.total}\n"
        f"OOD test size: {ood_result.total}\n"
        f"ID accuracy: {id_result.accuracy * 100.0:.4f}%\n"
        f"OOD accuracy: {ood_result.accuracy * 100.0:.4f}%\n"
        f"Absolute drop (ID - OOD): {abs_drop:.4f}%\n"
        f"Relative drop: {rel_drop:.4f}%\n\n"
        "Inference speed:\n"
        f"- id avg batch time: {id_result.avg_batch_time_ms:.4f} ms\n"
        f"- ood avg batch time: {ood_result.avg_batch_time_ms:.4f} ms\n"
        f"- id avg sample time: {id_result.avg_sample_time_ms:.4f} ms\n"
        f"- ood avg sample time: {ood_result.avg_sample_time_ms:.4f} ms\n\n"
        "Anomaly statistics:\n"
        f"- total anomalies: {len(id_result.anomalies) + len(ood_result.anomalies)}\n"
        f"- by reason: {dict(anomaly_counter)}\n\n"
        "Reasonability checks:\n"
        + "".join([f"- {hint}\n" for hint in extra_hints])
    )
    out_path.write_text(report, encoding="utf-8")
    LOGGER.info("Generalization report saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD generalization test for RAVEN CNN baseline")
    parser.add_argument("--data_root", type=str, default="data/raven")
    parser.add_argument("--weights", type=str, default=str(WEIGHT_PATH))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slow_batch_threshold_ms", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    setup_logger()
    setup_data_logger()  # keep shared data pipeline logs enabled
    args = parse_args()
    device = torch.device("cpu")

    # 1) Model loading with robust checks and actionable hint.
    model = CNNBaseline(num_choices=8, feature_dim=128).to(device)
    weight_path = Path(args.weights)
    try:
        load_weights(model, path=weight_path, strict=True)
        LOGGER.info("Model loaded successfully from %s", weight_path)
    except Exception as exc:
        LOGGER.error("模型加载失败: %s", exc)
        LOGGER.error(
            "解决建议: 1) 确认权重路径正确；2) 确认文件非空且包含 state_dict；"
            "3) 确认模型结构与训练时一致；4) 先运行 train_cnn.py 生成权重。"
        )
        raise SystemExit(1)

    # 2) Dataset loading and strict ID/OOD split.
    samples, _stats = scan_raven_samples(Path(args.data_root))
    id_samples, ood_samples = build_strict_id_ood_subsets(samples)

    LOGGER.info(
        "Strict subset sizes | id=%d ood=%d total=%d",
        len(id_samples),
        len(ood_samples),
        len(samples),
    )

    if len(id_samples) == 0 and len(ood_samples) == 0:
        LOGGER.error("测试子集为空，无法执行泛化评估。请检查数据划分与数据完整性。")
        raise SystemExit(1)

    # 3) Build loaders for both subsets.
    id_loader = DataLoader(
        RavenDataset(id_samples, image_size=args.image_size, output_channels=1)
        if len(id_samples) > 0
        else [],
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=False,
    )
    ood_loader = DataLoader(
        RavenDataset(ood_samples, image_size=args.image_size, output_channels=1)
        if len(ood_samples) > 0
        else [],
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=False,
    )

    # 4) Evaluate with safety fallback.
    if len(id_samples) > 0:
        id_result = evaluate_subset(
            model=model,
            loader=id_loader,
            name="id",
            device=device,
            slow_batch_threshold_ms=args.slow_batch_threshold_ms,
        )
    else:
        id_result = EvalResult("id", 0, 0, 0.0, 0.0, 0.0, [])
        LOGGER.warning("ID subset empty, id accuracy defaulted to 0.0")

    if len(ood_samples) > 0:
        ood_result = evaluate_subset(
            model=model,
            loader=ood_loader,
            name="ood",
            device=device,
            slow_batch_threshold_ms=args.slow_batch_threshold_ms,
        )
    else:
        ood_result = EvalResult("ood", 0, 0, 0.0, 0.0, 0.0, [])
        LOGGER.warning("OOD subset empty, ood accuracy defaulted to 0.0")

    # 5) Result sanity checks.
    hints = []
    hints.extend(validate_result_range(id_result))
    hints.extend(validate_result_range(ood_result))
    if id_result.accuracy > 0 and ood_result.accuracy > id_result.accuracy:
        hints.append(
            "OOD accuracy higher than ID. Possible reasons: tiny sample size, noisy labels, or underfitted model."
        )
    if not hints:
        hints.append("结果范围校验通过。")

    # 6) Persist report and plot.
    save_report(
        id_result=id_result,
        ood_result=ood_result,
        extra_hints=hints,
        out_path=REPORT_PATH,
    )
    save_plot(id_result.accuracy, ood_result.accuracy, out_path=PLOT_PATH)
    LOGGER.info("Generalization plot saved to %s", PLOT_PATH)

    # 7) Console summary.
    LOGGER.info("ID accuracy:   %.4f%%", id_result.accuracy * 100.0)
    LOGGER.info("OOD accuracy: %.4f%%", ood_result.accuracy * 100.0)
    LOGGER.info(
        "Absolute drop (ID - OOD): %.4f%%",
        (id_result.accuracy - ood_result.accuracy) * 100.0,
    )


if __name__ == "__main__":
    main()

