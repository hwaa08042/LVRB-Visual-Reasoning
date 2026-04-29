"""
Unified experiment visualization utility.

This script integrates results from:
- Generalization test   (experiments/results/generalization_report.txt)
- Robustness test       (experiments/results/robustness_report.txt)
- Contrastive effect    (auto-detected contrastive result/report files)

Outputs are saved under:
  experiments/results/
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "experiments/.mplconfig")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("visualization")


def setup_logger(log_path: Path) -> None:
    """Configure readable timestamped logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


@dataclass
class GeneralizationResult:
    seen_acc: Optional[float] = None  # percentage (seen or ID)
    unseen_acc: Optional[float] = None  # percentage (unseen or OOD)
    acc_drop: Optional[float] = None


@dataclass
class RobustnessItem:
    perturbation: str
    clean_acc: Optional[float]  # percentage
    perturbed_acc: Optional[float]
    acc_drop: Optional[float]


@dataclass
class ContrastiveResult:
    baseline_acc: Optional[float] = None  # percentage
    contrastive_acc: Optional[float] = None
    improvement: Optional[float] = None


def safe_read_text(path: Path) -> Optional[str]:
    """Read text safely; return None on missing/invalid file."""
    if not path.exists():
        LOGGER.warning("结果文件缺失，已跳过: %s", path)
        return None
    if path.stat().st_size == 0:
        LOGGER.warning("结果文件为空，已跳过: %s", path)
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("结果文件读取失败，已跳过: %s | reason=%s", path, exc)
        return None


def _to_float_safe(value: str) -> Optional[float]:
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def parse_generalization_report(text: str) -> GeneralizationResult:
    """Parse generalization report text robustly."""
    result = GeneralizationResult()
    seen = re.search(r"Seen rules accuracy:\s*([0-9.]+)%", text)
    unseen = re.search(r"Unseen rules accuracy:\s*([0-9.]+)%", text)
    drop = re.search(r"Accuracy drop .*:\s*([0-9.\-]+)%", text)
    # Strict ID/OOD format compatibility.
    id_acc = re.search(r"ID accuracy:\s*([0-9.]+)%", text)
    ood_acc = re.search(r"OOD accuracy:\s*([0-9.]+)%", text)
    abs_drop = re.search(r"Absolute drop .*:\s*([0-9.\-]+)%", text)
    if seen:
        result.seen_acc = _to_float_safe(seen.group(1))
    if unseen:
        result.unseen_acc = _to_float_safe(unseen.group(1))
    if drop:
        result.acc_drop = _to_float_safe(drop.group(1))
    if id_acc:
        result.seen_acc = _to_float_safe(id_acc.group(1))
    if ood_acc:
        result.unseen_acc = _to_float_safe(ood_acc.group(1))
    if abs_drop:
        result.acc_drop = _to_float_safe(abs_drop.group(1))
    return result


def parse_robustness_report(text: str) -> List[RobustnessItem]:
    """Parse robustness report text into per-perturbation rows."""
    items: List[RobustnessItem] = []
    blocks = text.split("Perturbation:")
    for block in blocks[1:]:
        name_match = re.match(r"\s*([^\n]+)", block)
        clean_match = re.search(r"Clean accuracy:\s*([0-9.]+)%", block)
        pert_match = re.search(r"Perturbed accuracy:\s*([0-9.]+)%", block)
        drop_match = re.search(r"Accuracy drop:\s*([0-9.]+)%", block)
        if not name_match:
            continue
        items.append(
            RobustnessItem(
                perturbation=name_match.group(1).strip(),
                clean_acc=_to_float_safe(clean_match.group(1)) if clean_match else None,
                perturbed_acc=_to_float_safe(pert_match.group(1)) if pert_match else None,
                acc_drop=_to_float_safe(drop_match.group(1)) if drop_match else None,
            )
        )
    return items


def parse_contrastive_results(results_dir: Path) -> ContrastiveResult:
    """
    Parse contrastive-effect summary from potential files.

    Supported files (auto-detect):
    - contrastive_report.txt
    - contrastive_train_report.txt + train_report.txt
    - baseline_train_report.txt + contrastive_train_report.txt
    """
    result = ContrastiveResult()
    preferred = [
        results_dir / "contrastive_report.txt",
        results_dir / "contrastive_train_report.txt",
        results_dir / "contrastive_comparison.txt",
    ]
    # Case 0: strict comparison report.
    strict_cmp = safe_read_text(preferred[2])
    if strict_cmp:
        b = re.search(r"Baseline:\s*[\s\S]*?- OOD acc:\s*([0-9.]+)%", strict_cmp)
        c = re.search(r"Contrastive:\s*[\s\S]*?- OOD acc:\s*([0-9.]+)%", strict_cmp)
        if b:
            result.baseline_acc = _to_float_safe(b.group(1))
        if c:
            result.contrastive_acc = _to_float_safe(c.group(1))
        if result.baseline_acc is not None and result.contrastive_acc is not None:
            result.improvement = result.contrastive_acc - result.baseline_acc
            return result

    baseline_candidates = [
        results_dir / "train_report.txt",
        results_dir / "baseline_train_report.txt",
    ]

    # Case 1: direct contrastive report with baseline/contrastive accuracy.
    direct = safe_read_text(preferred[0])
    if direct:
        b = re.search(r"Baseline accuracy:\s*([0-9.]+)%", direct)
        c = re.search(r"Contrastive accuracy:\s*([0-9.]+)%", direct)
        if b:
            result.baseline_acc = _to_float_safe(b.group(1))
        if c:
            result.contrastive_acc = _to_float_safe(c.group(1))
        if result.baseline_acc is not None and result.contrastive_acc is not None:
            result.improvement = result.contrastive_acc - result.baseline_acc
            return result

    # Case 2: compare best validation accuracy from two train reports.
    contrastive_text = safe_read_text(preferred[1])
    baseline_text = None
    for p in baseline_candidates:
        baseline_text = safe_read_text(p)
        if baseline_text:
            break

    if contrastive_text and baseline_text:
        c_match = re.search(r"Best validation accuracy:\s*([0-9.]+)", contrastive_text)
        b_match = re.search(r"Best validation accuracy:\s*([0-9.]+)", baseline_text)
        if c_match and b_match:
            c_acc = _to_float_safe(c_match.group(1))
            b_acc = _to_float_safe(b_match.group(1))
            if c_acc is not None and b_acc is not None:
                result.baseline_acc = b_acc * 100.0
                result.contrastive_acc = c_acc * 100.0
                result.improvement = result.contrastive_acc - result.baseline_acc
    return result


def fmt_pct(value: Optional[float]) -> str:
    """Format percentage with 2 decimals."""
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}%"


def build_unified_table(
    gen: GeneralizationResult,
    rob: Sequence[RobustnessItem],
    con: ContrastiveResult,
) -> List[Dict[str, str]]:
    """Build unified comparison table rows."""
    rows: List[Dict[str, str]] = []

    rows.append(
        {
            "Experiment": "Generalization",
            "Metric": "Seen/ID Accuracy",
            "Value": fmt_pct(gen.seen_acc),
            "Note": "Seen rules or ID rules test set",
        }
    )
    rows.append(
        {
            "Experiment": "Generalization",
            "Metric": "Unseen/OOD Accuracy",
            "Value": fmt_pct(gen.unseen_acc),
            "Note": "Unseen rules or OOD rules test set",
        }
    )
    rows.append(
        {
            "Experiment": "Generalization",
            "Metric": "Accuracy Drop",
            "Value": fmt_pct(gen.acc_drop),
            "Note": "Seen - Unseen",
        }
    )

    for item in rob:
        rows.append(
            {
                "Experiment": "Robustness",
                "Metric": f"{item.perturbation} Drop",
                "Value": fmt_pct(item.acc_drop),
                "Note": f"Clean={fmt_pct(item.clean_acc)}, Perturbed={fmt_pct(item.perturbed_acc)}",
            }
        )

    rows.append(
        {
            "Experiment": "Contrastive",
            "Metric": "Baseline Accuracy",
            "Value": fmt_pct(con.baseline_acc),
            "Note": "Before contrastive",
        }
    )
    rows.append(
        {
            "Experiment": "Contrastive",
            "Metric": "Contrastive Accuracy",
            "Value": fmt_pct(con.contrastive_acc),
            "Note": "After contrastive",
        }
    )
    rows.append(
        {
            "Experiment": "Contrastive",
            "Metric": "Improvement",
            "Value": fmt_pct(con.improvement),
            "Note": "Contrastive - Baseline",
        }
    )
    return rows


def save_table(rows: Sequence[Dict[str, str]], results_dir: Path) -> Tuple[Path, Path]:
    """Save unified table into CSV and markdown."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_comparison_table.csv"
    md_path = results_dir / "experiment_comparison_table.md"

    headers = ["Experiment", "Metric", "Value", "Note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = ["| Experiment | Metric | Value | Note |", "|---|---|---|---|"]
    for r in rows:
        md_lines.append(f"| {r['Experiment']} | {r['Metric']} | {r['Value']} | {r['Note']} |")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path


def plot_bar_chart(
    gen: GeneralizationResult, rob: Sequence[RobustnessItem], con: ContrastiveResult, out_path: Path
) -> None:
    """Draw summary bar chart."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels: List[str] = []
    values: List[float] = []

    if gen.seen_acc is not None:
        labels.append("Gen Seen")
        values.append(gen.seen_acc)
    if gen.unseen_acc is not None:
        labels.append("Gen Unseen")
        values.append(gen.unseen_acc)
    for item in rob:
        if item.perturbed_acc is not None:
            labels.append(item.perturbation)
            values.append(item.perturbed_acc)
    if con.contrastive_acc is not None:
        labels.append("Contrastive")
        values.append(con.contrastive_acc)
    if con.baseline_acc is not None:
        labels.append("Baseline")
        values.append(con.baseline_acc)

    if len(values) == 0:
        LOGGER.warning("无可用数据，跳过柱状图绘制。")
        return

    try:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, values)
        plt.ylabel("Accuracy (%)")
        plt.title("Experiment Comparison")
        plt.ylim(0, 100)
        plt.xticks(rotation=20)
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as exc:
        LOGGER.warning("柱状图绘制失败，已跳过: %s", exc)


def plot_line_chart(rob: Sequence[RobustnessItem], out_path: Path) -> None:
    """Draw robustness trend line chart over perturbations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [x.perturbation for x in rob if x.perturbed_acc is not None]
    pert = [x.perturbed_acc for x in rob if x.perturbed_acc is not None]
    clean = [x.clean_acc for x in rob if x.perturbed_acc is not None]

    if len(names) == 0:
        LOGGER.warning("无鲁棒性数据，跳过折线图绘制。")
        return

    try:
        plt.figure(figsize=(8, 5))
        plt.plot(names, pert, marker="o", label="Perturbed Accuracy")
        if clean and all(v is not None for v in clean):
            plt.plot(names, clean, marker="s", linestyle="--", label="Clean Accuracy")
        plt.ylabel("Accuracy (%)")
        plt.title("Robustness Trend by Perturbation")
        plt.ylim(0, 100)
        plt.xticks(rotation=15)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as exc:
        LOGGER.warning("折线图绘制失败，已跳过: %s", exc)


def validate_results(
    gen: GeneralizationResult, rob: Sequence[RobustnessItem], con: ContrastiveResult
) -> List[str]:
    """Validate ranges and consistency; return warnings."""
    warns: List[str] = []

    def in_range(v: Optional[float], name: str) -> None:
        if v is None:
            return
        if v < 0 or v > 100:
            warns.append(f"{name} 超出合理范围 [0,100]: {v:.4f}")

    in_range(gen.seen_acc, "Generalization Seen Accuracy")
    in_range(gen.unseen_acc, "Generalization Unseen Accuracy")
    in_range(gen.acc_drop, "Generalization Drop")
    in_range(con.baseline_acc, "Contrastive Baseline Accuracy")
    in_range(con.contrastive_acc, "Contrastive Accuracy")

    if gen.seen_acc is not None and gen.unseen_acc is not None and gen.acc_drop is not None:
        expected = gen.seen_acc - gen.unseen_acc
        if abs(expected - gen.acc_drop) > 1.0:
            warns.append(
                f"泛化结果数值不一致: drop={gen.acc_drop:.2f}, expected={expected:.2f}"
            )

    for item in rob:
        in_range(item.clean_acc, f"{item.perturbation} clean")
        in_range(item.perturbed_acc, f"{item.perturbation} perturbed")
        in_range(item.acc_drop, f"{item.perturbation} drop")
        if (
            item.clean_acc is not None
            and item.perturbed_acc is not None
            and item.acc_drop is not None
            and abs(max(0.0, item.clean_acc - item.perturbed_acc) - item.acc_drop) > 1.0
        ):
            warns.append(f"鲁棒性结果不一致: {item.perturbation} 的 drop 与差值不匹配")
    return warns


def build_conclusion(
    gen: GeneralizationResult, rob: Sequence[RobustnessItem], con: ContrastiveResult
) -> List[str]:
    """Generate core findings for README-friendly summary."""
    conclusions: List[str] = []

    if gen.acc_drop is not None:
        if gen.acc_drop > 20:
            conclusions.append("泛化能力不足：未见规则准确率下降较明显（>20%）。")
        else:
            conclusions.append("泛化表现相对稳定：见过/未见规则差距可控。")

    valid_rob = [x for x in rob if x.acc_drop is not None]
    if valid_rob:
        worst = sorted(valid_rob, key=lambda x: x.acc_drop if x.acc_drop is not None else -1, reverse=True)[0]
        conclusions.append(f"最敏感扰动类型：`{worst.perturbation}`（准确率下降 {worst.acc_drop:.2f}%）。")

    if con.improvement is not None:
        if con.improvement > 0:
            conclusions.append(f"对比学习带来提升：+{con.improvement:.2f}%。")
        elif con.improvement < 0:
            conclusions.append(f"对比学习暂未提升：{con.improvement:.2f}%。建议调权重或采样策略。")
        else:
            conclusions.append("对比学习与基线持平。")
    return conclusions


def save_visualization_report(
    warnings: Sequence[str], conclusions: Sequence[str], out_path: Path
) -> None:
    """Save final visualization quality report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Visualization Validation Report", "============================", ""]
    lines.append("Data/Chart Validation:")
    if warnings:
        for w in warnings:
            lines.append(f"- [WARN] {w}")
    else:
        lines.append("- All checks passed.")

    lines.append("")
    lines.append("Core Conclusions:")
    if conclusions:
        for c in conclusions:
            lines.append(f"- {c}")
    else:
        lines.append("- 暂无结论（可用结果不足）。")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment visualization")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(results_dir / "visualization.log")

    # 1) Load/parse all result files safely.
    gen_text = safe_read_text(results_dir / "generalization_report.txt")
    rob_text = safe_read_text(results_dir / "robustness_report.txt")

    gen_result = parse_generalization_report(gen_text) if gen_text else GeneralizationResult()
    rob_result = parse_robustness_report(rob_text) if rob_text else []
    con_result = parse_contrastive_results(results_dir)

    # 2) Build and save unified comparison table.
    rows = build_unified_table(gen_result, rob_result, con_result)
    csv_path, md_path = save_table(rows, results_dir)
    LOGGER.info("实验对比表已保存: %s, %s", csv_path, md_path)

    # 3) Draw charts with safe guards.
    bar_path = results_dir / "experiment_comparison_bar.png"
    line_path = results_dir / "experiment_comparison_line.png"
    plot_bar_chart(gen_result, rob_result, con_result, bar_path)
    plot_line_chart(rob_result, line_path)
    LOGGER.info("可视化图表输出完成: %s, %s", bar_path, line_path)

    # 4) Post-validation and summary report.
    warnings = validate_results(gen_result, rob_result, con_result)
    conclusions = build_conclusion(gen_result, rob_result, con_result)
    vis_report = results_dir / "visualization_report.txt"
    save_visualization_report(warnings, conclusions, vis_report)
    LOGGER.info("可视化报告已保存: %s", vis_report)

    # Console highlights
    if warnings:
        LOGGER.warning("检测到 %d 条可视化/数据一致性告警。详见 visualization_report.txt", len(warnings))
    else:
        LOGGER.info("可视化校验通过，未发现明显一致性异常。")


if __name__ == "__main__":
    main()

