"""
Global AI code health checker for LVRB project.

Run:
    python utils/code_check.py
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import py_compile
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


SEVERITY_CRITICAL = "严重"
SEVERITY_WARNING = "警告"
SEVERITY_SUGGESTION = "建议"


RED = "\033[91m"
RESET = "\033[0m"


@dataclass
class Issue:
    severity: str
    title: str
    detail: str
    fix: str


def project_root() -> Path:
    """Resolve project root from this file location."""
    return Path(__file__).resolve().parent.parent


def collect_python_files(root: Path) -> List[Path]:
    """Collect all python files under project root."""
    return sorted(root.rglob("*.py"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def check_core_files(root: Path) -> List[Issue]:
    """Check required core files existence."""
    issues: List[Issue] = []
    required = [
        root / "raven_dataloader.py",
        root / "model/cnn_baseline.py",
        root / "train_cnn.py",
        root / "experiments/generalization_test.py",
        root / "experiments/robustness_test.py",
    ]
    for f in required:
        if not f.exists():
            issues.append(
                Issue(
                    severity=SEVERITY_CRITICAL,
                    title="核心文件缺失",
                    detail=f"缺失文件: {rel(f, root)}",
                    fix="补齐该文件或恢复到仓库标准结构。",
                )
            )
    return issues


def check_python_syntax(py_files: Sequence[Path], root: Path) -> List[Issue]:
    """Compile every .py file to detect syntax errors."""
    issues: List[Issue] = []
    for file_path in py_files:
        try:
            py_compile.compile(str(file_path), doraise=True)
        except Exception as exc:
            issues.append(
                Issue(
                    severity=SEVERITY_CRITICAL,
                    title="Python 语法错误",
                    detail=f"{rel(file_path, root)}: {exc}",
                    fix="修复语法错误后重新运行 code_check。",
                )
            )
    return issues


def _parse_imports(file_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Return:
      - absolute top-level module names
      - relative import module names
    """
    abs_imports: Set[str] = set()
    rel_imports: Set[str] = set()
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                if node.module:
                    rel_imports.add(node.module.split(".")[0])
            elif node.module:
                abs_imports.add(node.module.split(".")[0])
    return abs_imports, rel_imports


def _is_local_module(module_name: str, root: Path) -> bool:
    """Check whether module exists locally in project."""
    return (root / f"{module_name}.py").exists() or (root / module_name).is_dir()


def check_imports(py_files: Sequence[Path], root: Path) -> List[Issue]:
    """Check import resolvability and relative-import risks."""
    issues: List[Issue] = []
    stdlib_like = {
        "os",
        "sys",
        "io",
        "re",
        "math",
        "time",
        "csv",
        "argparse",
        "logging",
        "dataclasses",
        "typing",
        "pathlib",
        "traceback",
        "subprocess",
        "importlib",
        "collections",
        "py_compile",
        "ast",
    }
    third_party_allow = {"torch", "numpy", "PIL", "sklearn", "matplotlib"}

    for file_path in py_files:
        try:
            abs_mods, rel_mods = _parse_imports(file_path)
        except Exception as exc:
            issues.append(
                Issue(
                    severity=SEVERITY_WARNING,
                    title="导入扫描失败",
                    detail=f"{rel(file_path, root)}: AST 解析失败: {exc}",
                    fix="检查文件编码/语法并重试。",
                )
            )
            continue

        for mod in abs_mods:
            if mod in stdlib_like:
                continue
            if _is_local_module(mod, root):
                continue
            if importlib.util.find_spec(mod) is not None:
                continue
            if mod in third_party_allow:
                issues.append(
                    Issue(
                        severity=SEVERITY_WARNING,
                        title="第三方依赖可能缺失",
                        detail=f"{rel(file_path, root)} 导入 `{mod}` 但当前环境未解析。",
                        fix=f"安装依赖并重试，例如: pip install {mod.lower()}",
                    )
                )
            else:
                issues.append(
                    Issue(
                        severity=SEVERITY_WARNING,
                        title="模块导入可能错误",
                        detail=f"{rel(file_path, root)} 导入 `{mod}` 未找到。",
                        fix="检查模块名是否拼写正确，或确认 PYTHONPATH/项目结构。",
                    )
                )

        if rel_mods:
            # 当前项目以脚本执行为主，过多相对导入容易运行失败。
            issues.append(
                Issue(
                    severity=SEVERITY_SUGGESTION,
                    title="存在相对导入",
                    detail=f"{rel(file_path, root)} 包含相对导入: {sorted(rel_mods)}",
                    fix="若作为脚本直接运行，建议改为绝对导入并统一入口。",
                )
            )
    return issues


def check_hardcoded_paths(py_files: Sequence[Path], root: Path) -> List[Issue]:
    """Detect forbidden absolute hardcoded paths."""
    issues: List[Issue] = []
    patterns = [
        re.compile(r"/Users/[A-Za-z0-9._-]+"),
        re.compile(r"/User/[A-Za-z0-9._-]+"),
        re.compile(r"[A-Za-z]:\\\\"),
    ]
    for file_path in py_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for p in patterns:
            match = p.search(text)
            if match:
                # Ignore placeholder examples used inside checker docs/pattern comments.
                if match.group(0) in {"/Users/xxx", "/User/xxx"}:
                    continue
                issues.append(
                    Issue(
                        severity=SEVERITY_WARNING,
                        title="检测到疑似硬编码绝对路径",
                        detail=f"{rel(file_path, root)} 命中: {match.group(0)}",
                        fix="改为相对路径（如 data/raven, experiments/results）。",
                    )
                )
                break
    return issues


def check_dataset_exists(root: Path) -> List[Issue]:
    """Check dataset folder presence."""
    issues: List[Issue] = []
    data_dir = root / "data/raven"
    if not data_dir.exists():
        issues.append(
            Issue(
                severity=SEVERITY_WARNING,
                title="数据集目录不存在",
                detail="未找到 data/raven",
                fix="创建目录并放置 RAVEN .npz 数据，例如 `mkdir -p data/raven`。",
            )
        )
    return issues


def check_model_data_compatibility(root: Path) -> List[Issue]:
    """Check key shape/channel assumptions between loader and model."""
    issues: List[Issue] = []
    try:
        model_text = (root / "model/cnn_baseline.py").read_text(encoding="utf-8")
        loader_text = (root / "utils/data_loader.py").read_text(encoding="utf-8")
    except Exception as exc:
        return [
            Issue(
                severity=SEVERITY_WARNING,
                title="模型/数据兼容性检查跳过",
                detail=f"读取脚本失败: {exc}",
                fix="确认相关文件存在并可读取。",
            )
        ]

    # Heuristic checks for current project conventions.
    need_panels_16 = "expect_panels=16" in model_text or "num_panels != 16" in model_text
    channel_1 = "channels=1" in model_text or "Expected channels=1" in model_text
    loader_returns_panels = '"panels":' in loader_text and "np.concatenate([context, choices]" in loader_text
    loader_output_channels_1 = "output_channels: int = 1" in loader_text

    if need_panels_16 and not loader_returns_panels:
        issues.append(
            Issue(
                severity=SEVERITY_CRITICAL,
                title="模型/数据不兼容",
                detail="模型要求 16 panels，但数据加载脚本可能未返回 panels。",
                fix="确保 DataLoader 输出 `panels` 字段，shape 为 [B,16,C,H,W]。",
            )
        )
    if channel_1 and not loader_output_channels_1:
        issues.append(
            Issue(
                severity=SEVERITY_WARNING,
                title="通道默认值可能不兼容",
                detail="模型要求单通道，但 loader 默认 output_channels 可能不是 1。",
                fix="将 loader 默认 output_channels 设为 1，或训练时显式传参。",
            )
        )

    # Optional real-sample check when dataset exists and numpy is available.
    data_dir = root / "data/raven"
    if data_dir.exists() and np is not None:
        sample = next(iter(sorted(data_dir.rglob("*.npz"))), None)
        if sample is not None:
            try:
                data = np.load(sample, allow_pickle=True)
                image = data["image"]
                if image.ndim != 3 or image.shape[0] != 16:
                    issues.append(
                        Issue(
                            severity=SEVERITY_WARNING,
                            title="样本 shape 与模型约定不一致",
                            detail=f"{rel(sample, root)} image.shape={image.shape}",
                            fix="确认数据预处理将样本统一为 (16,H,W)。",
                        )
                    )
            except Exception as exc:
                issues.append(
                    Issue(
                        severity=SEVERITY_WARNING,
                        title="样本兼容性检查失败",
                        detail=f"读取示例数据失败: {rel(sample, root)} | {exc}",
                        fix="检查 npz 是否损坏，或修复数据字段 image/target。",
                    )
                )
    return issues


def check_try_except_presence(py_files: Sequence[Path], root: Path) -> List[Issue]:
    """Check whether scripts contain try-except protection."""
    issues: List[Issue] = []
    for file_path in py_files:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:
            continue
        has_try = any(isinstance(node, ast.Try) for node in ast.walk(tree))
        if not has_try:
            issues.append(
                Issue(
                    severity=SEVERITY_SUGGESTION,
                    title="缺少 try-except 保护",
                    detail=f"{rel(file_path, root)} 未检测到 try-except。",
                    fix="为关键 I/O、训练、推理路径增加异常捕获和日志。",
                )
            )
    return issues


def check_core_scripts_runnable(root: Path) -> List[Issue]:
    """Smoke test core scripts with --help (non-destructive)."""
    issues: List[Issue] = []
    scripts = [
        root / "train_cnn.py",
        root / "experiments/generalization_test.py",
        root / "experiments/robustness_test.py",
        root / "utils/visualization.py",
    ]
    for script in scripts:
        if not script.exists():
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(root),
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                issues.append(
                    Issue(
                        severity=SEVERITY_CRITICAL,
                        title="核心脚本可能无法运行",
                        detail=(
                            f"{rel(script, root)} --help 返回码={result.returncode} | "
                            f"stderr={result.stderr[:300].strip()}"
                        ),
                        fix="先修复导入/参数解析错误，再进行训练与评估。",
                    )
                )
        except Exception as exc:
            issues.append(
                Issue(
                    severity=SEVERITY_CRITICAL,
                    title="核心脚本运行检查失败",
                    detail=f"{rel(script, root)} 执行异常: {exc}",
                    fix="检查 Python 环境、依赖、脚本入口与命令参数。",
                )
            )
    return issues


def group_by_severity(issues: Sequence[Issue]) -> Dict[str, List[Issue]]:
    grouped: Dict[str, List[Issue]] = {
        SEVERITY_CRITICAL: [],
        SEVERITY_WARNING: [],
        SEVERITY_SUGGESTION: [],
    }
    for i in issues:
        grouped.setdefault(i.severity, []).append(i)
    return grouped


def build_report(root: Path, py_files: Sequence[Path], issues: Sequence[Issue]) -> str:
    """Build full markdown-style report text."""
    grouped = group_by_severity(issues)
    critical = grouped.get(SEVERITY_CRITICAL, [])
    warning = grouped.get(SEVERITY_WARNING, [])
    suggestion = grouped.get(SEVERITY_SUGGESTION, [])

    run_ok = len(critical) == 0
    conclusion = "项目可以正常运行（通过核心检查）。" if run_ok else "项目当前不可稳定运行（存在严重问题）。"

    lines: List[str] = []
    lines.append("LVRB 全局 AI 代码校验报告")
    lines.append("========================")
    lines.append("")
    lines.append("1) 项目结构检查")
    lines.append(f"- 项目根目录: {root}")
    lines.append(f"- Python 文件总数: {len(py_files)}")
    lines.append("")
    lines.append("2) 文件完整性")
    lines.append("- 已检查核心文件与实验脚本存在性。")
    lines.append("")
    lines.append("3) 语法检查")
    lines.append("- 已对全部 .py 执行 py_compile。")
    lines.append("")
    lines.append("4) 路径合规检查")
    lines.append("- 已扫描硬编码绝对路径（如 /Users/xxx）。")
    lines.append("")
    lines.append("5) 模型/数据兼容性检查")
    lines.append("- 已检查模型输入约定与数据输出字段一致性。")
    lines.append("")
    lines.append("6) 问题分级统计")
    lines.append(f"- 严重: {len(critical)}")
    lines.append(f"- 警告: {len(warning)}")
    lines.append(f"- 建议: {len(suggestion)}")
    lines.append("")

    for level in [SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_SUGGESTION]:
        items = grouped.get(level, [])
        lines.append(f"{level}问题明细:")
        if not items:
            lines.append("- 无")
        else:
            for idx, item in enumerate(items, start=1):
                lines.append(f"- [{idx}] {item.title}")
                lines.append(f"  - 详情: {item.detail}")
                lines.append(f"  - 修复建议: {item.fix}")
        lines.append("")

    lines.append("7) 最终结论")
    lines.append(f"- {conclusion}")
    return "\n".join(lines)


def print_summary(issues: Sequence[Issue]) -> None:
    """Print short summary, highlight critical findings in red."""
    grouped = group_by_severity(issues)
    critical = grouped.get(SEVERITY_CRITICAL, [])
    warning = grouped.get(SEVERITY_WARNING, [])
    suggestion = grouped.get(SEVERITY_SUGGESTION, [])

    print("=== LVRB Code Check Summary ===")
    print(f"严重: {len(critical)} | 警告: {len(warning)} | 建议: {len(suggestion)}")
    if critical:
        print(f"{RED}发现严重问题：核心脚本可能无法运行，请立即修复。{RESET}")
        for i in critical[:5]:
            print(f"{RED}- {i.title}: {i.detail}{RESET}")
    else:
        print("未发现严重问题，项目具备运行条件。")


def run_checks() -> Tuple[Path, List[Issue], str]:
    root = project_root()
    py_files = collect_python_files(root)
    issues: List[Issue] = []

    issues.extend(check_core_files(root))
    issues.extend(check_python_syntax(py_files, root))
    issues.extend(check_imports(py_files, root))
    issues.extend(check_hardcoded_paths(py_files, root))
    issues.extend(check_dataset_exists(root))
    issues.extend(check_model_data_compatibility(root))
    issues.extend(check_try_except_presence(py_files, root))
    issues.extend(check_core_scripts_runnable(root))

    report_text = build_report(root, py_files, issues)
    return root, issues, report_text


def main() -> None:
    try:
        root, issues, report_text = run_checks()
        report_path = root / "experiments/logs/code_check_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print_summary(issues)
        print(f"报告已输出: {report_path}")
    except Exception as exc:
        # Global fail-safe to keep one-command usability.
        print(f"{RED}code_check 执行失败: {exc}{RESET}")
        tb = traceback.format_exc()
        root = project_root()
        report_path = root / "experiments/logs/code_check_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "LVRB 全局 AI 代码校验报告\n"
            "========================\n"
            "执行异常（严重）\n"
            f"- 错误: {exc}\n\n"
            "Traceback:\n"
            f"{tb}\n",
            encoding="utf-8",
        )
        print(f"异常报告已输出: {report_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

