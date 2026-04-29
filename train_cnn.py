"""
Train script for model/cnn_baseline.py on RAVEN 3x3 task.

Key features:
- Full train/validation loop.
- CrossEntropyLoss + Adam optimizer.
- Robust parameter/data/runtime anomaly handling.
- Gradient anomaly monitor (exploding/vanishing norms).
- Overfitting warning monitor.
- Auto-save checkpoint on interruption.
- Final training report generation.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from model.cnn_baseline import CNNBaseline, LossMonitor, save_weights
from utils.contrastive_loss import ContrastiveLossAdapter, setup_logger as setup_contrastive_logger
from utils.data_loader import DatasetStats, build_raven_dataloaders


LOG_PATH = Path("experiments/logs/train_cnn_contrastive.log")
WEIGHT_DIR = Path("experiments/weights")
RESULT_PATH = Path("experiments/results/train_report.txt")


LOGGER = logging.getLogger("train_cnn")


def setup_logger(log_path: Path = LOG_PATH) -> None:
    """Configure console+file logger with timestamps."""
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


def progress_bar(step: int, total: int, prefix: str = "", width: int = 30) -> None:
    """Simple no-dependency progress bar for CPU training."""
    if total <= 0:
        return
    ratio = step / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    print(f"\r{prefix}[{bar}] {step}/{total}", end="", flush=True)
    if step == total:
        print()


def _safe_positive_int(name: str, value: int, default: int) -> int:
    """Fallback invalid integer params to default with warning."""
    if value is None or value <= 0:
        LOGGER.warning("Invalid %s=%s. Fallback to default=%d.", name, value, default)
        return default
    return value


def _safe_positive_float(name: str, value: float, default: float) -> float:
    """Fallback invalid float params to default with warning."""
    if value is None or value <= 0:
        LOGGER.warning("Invalid %s=%s. Fallback to default=%f.", name, value, default)
        return default
    return value


@dataclass
class TrainConfig:
    data_root: str = "data/raven"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    image_size: int = 224
    num_workers: int = 0
    seed: int = 42
    contrastive_weight: float = 0.3
    grad_explode_norm: float = 1_000.0
    grad_vanish_norm: float = 1e-10

    def sanitize(self) -> "TrainConfig":
        self.epochs = _safe_positive_int("epochs", self.epochs, 50)
        self.batch_size = _safe_positive_int("batch_size", self.batch_size, 32)
        self.num_workers = max(0, self.num_workers)
        self.image_size = _safe_positive_int("image_size", self.image_size, 224)
        self.lr = _safe_positive_float("lr", self.lr, 1e-3)
        self.contrastive_weight = _safe_positive_float(
            "contrastive_weight", self.contrastive_weight, 0.3
        )
        self.grad_explode_norm = _safe_positive_float(
            "grad_explode_norm", self.grad_explode_norm, 1_000.0
        )
        self.grad_vanish_norm = _safe_positive_float(
            "grad_vanish_norm", self.grad_vanish_norm, 1e-10
        )
        return self


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1
    overfit_warning_count: int = 0
    lr_history: List[float] = field(default_factory=list)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 norm of gradients."""
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_sq += grad_norm * grad_norm
    return math.sqrt(total_sq)


def check_grad_anomaly(grad_norm: float, cfg: TrainConfig, epoch: int, step: int) -> None:
    """Monitor gradient explosion/vanishing and log warnings."""
    if not math.isfinite(grad_norm):
        raise RuntimeError(
            f"Gradient anomaly at epoch={epoch}, step={step}: grad_norm is NaN/Inf ({grad_norm})."
        )
    if grad_norm > cfg.grad_explode_norm:
        raise RuntimeError(
            "Gradient explosion anomaly at epoch="
            f"{epoch}, step={step}: grad_norm={grad_norm:.6f} > {cfg.grad_explode_norm:.6f}"
        )
    if grad_norm < cfg.grad_vanish_norm:
        LOGGER.warning(
            "Gradient vanishing warning at epoch=%d step=%d: grad_norm=%.12f < %.12f",
            epoch,
            step,
            grad_norm,
            cfg.grad_vanish_norm,
        )


def _extract_batch(batch: Dict, phase: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Validate and extract required fields from batch dict."""
    if "panels" not in batch:
        raise KeyError(f"[{phase}] missing key 'panels' in batch.")
    if "label" not in batch:
        raise KeyError(f"[{phase}] missing key 'label' in batch.")

    panels = batch["panels"]
    labels = batch["label"]
    paths = batch.get("path", [])

    if not isinstance(paths, list):
        # DataLoader collation may yield tuple.
        paths = list(paths) if paths is not None else []

    if labels is None or (isinstance(labels, torch.Tensor) and labels.numel() == 0):
        raise ValueError(f"[{phase}] empty labels in batch, paths={paths[:3]}")

    return panels, labels, paths


def run_one_epoch_train(
    model: CNNBaseline,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_monitor: LossMonitor,
    contrastive_adapter: ContrastiveLossAdapter,
    contrastive_weight: float,
    cfg: TrainConfig,
    epoch: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch with robust exception filtering."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    valid_steps = 0

    for step, batch in enumerate(loader, start=1):
        progress_bar(step, len(loader), prefix=f"Train E{epoch:03d} ")
        try:
            panels, labels, paths = _extract_batch(batch, phase="train")
            panels = panels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(panels)
            ce_loss = criterion(logits, labels)
            bsz = panels.size(0)
            flat_panels = panels.reshape(bsz * 16, 1, panels.size(-2), panels.size(-1))
            panel_features = model.encoder(flat_panels).reshape(bsz, 16, model.feature_dim)
            # Sample-level embedding for contrastive pairing.
            sample_features = panel_features.mean(dim=1)  # [B, D]
            contrastive_loss, contrastive_info = contrastive_adapter.compute(
                model=model,
                features=sample_features,
                paths=paths,
                ce_loss=ce_loss,
            )
            loss = ce_loss + contrastive_weight * contrastive_loss
            loss_value = float(loss.detach().cpu().item())

            # Loss anomaly monitoring.
            loss_monitor.check(loss_value=loss_value, step=(epoch * 100000 + step))

            loss.backward()

            grad_norm = compute_grad_norm(model)
            check_grad_anomaly(grad_norm, cfg, epoch, step)

            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct = int((pred == labels).sum().item())
                batch_size = int(labels.size(0))

            total_loss += loss_value * batch_size
            total_correct += correct
            total_count += batch_size
            valid_steps += 1
            if step == len(loader):
                summary = contrastive_adapter.monitor.summarize_epoch()
                LOGGER.info(
                    "Contrastive epoch summary | mean=%.6f pos_pairs=%d neg_pairs=%d ratio=%.4f",
                    summary["contrastive_loss_mean"],
                    int(summary["pos_pairs_total"]),
                    int(summary["neg_pairs_total"]),
                    summary["pos_neg_ratio"],
                )

        except Exception as exc:
            msg = str(exc).lower()
            if "nan" in msg or "inf" in msg or "gradient explosion anomaly" in msg:
                raise RuntimeError(
                    f"Critical train anomaly at epoch={epoch}, step={step}: {exc}"
                ) from exc
            LOGGER.error(
                "Train batch skipped at epoch=%d step=%d | reason=%s | sample_paths=%s",
                epoch,
                step,
                exc,
                paths[:3] if "paths" in locals() else [],
            )
            continue

    if total_count == 0 or valid_steps == 0:
        raise RuntimeError("All training batches failed or were filtered.")
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def run_one_epoch_val(
    model: CNNBaseline,
    loader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one validation epoch with robust exception filtering."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    valid_steps = 0

    for step, batch in enumerate(loader, start=1):
        progress_bar(step, len(loader), prefix=f"Val   E{epoch:03d} ")
        try:
            panels, labels, paths = _extract_batch(batch, phase="val")
            panels = panels.to(device)
            labels = labels.to(device)

            logits, _ = model(panels)
            loss = criterion(logits, labels)
            loss_value = float(loss.detach().cpu().item())

            pred = torch.argmax(logits, dim=1)
            correct = int((pred == labels).sum().item())
            batch_size = int(labels.size(0))

            total_loss += loss_value * batch_size
            total_correct += correct
            total_count += batch_size
            valid_steps += 1
        except Exception as exc:
            LOGGER.error(
                "Val batch skipped at epoch=%d step=%d | reason=%s | sample_paths=%s",
                epoch,
                step,
                exc,
                paths[:3] if "paths" in locals() else [],
            )
            continue

    if total_count == 0 or valid_steps == 0:
        raise RuntimeError("All validation batches failed or were filtered.")
    return total_loss / total_count, total_correct / total_count


def detect_overfitting(history: TrainHistory, current_epoch: int) -> bool:
    """
    Trigger overfitting warning if for last 3 epochs:
    - val acc not improving
    - val loss not decreasing
    """
    if current_epoch < 3:
        return False
    val_acc_recent = history.val_acc[-3:]
    val_loss_recent = history.val_loss[-3:]

    non_improve_acc = val_acc_recent[2] <= val_acc_recent[1] <= val_acc_recent[0]
    non_decrease_loss = val_loss_recent[2] >= val_loss_recent[1] >= val_loss_recent[0]
    return non_improve_acc and non_decrease_loss


def write_train_report(
    history: TrainHistory,
    duration_sec: float,
    dataset_stats: DatasetStats,
    out_path: Path = RESULT_PATH,
) -> None:
    """Generate and save final training report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_train_loss = min(history.train_loss) if history.train_loss else float("nan")
    best_val_loss = min(history.val_loss) if history.val_loss else float("nan")
    first_train_loss = history.train_loss[0] if history.train_loss else float("nan")
    first_val_loss = history.val_loss[0] if history.val_loss else float("nan")
    last_train_loss = history.train_loss[-1] if history.train_loss else float("nan")
    last_val_loss = history.val_loss[-1] if history.val_loss else float("nan")

    report = (
        "RAVEN CNN Training Report\n"
        "========================\n"
        f"Training duration (sec): {duration_sec:.2f}\n"
        f"Best validation accuracy: {history.best_val_acc:.6f}\n"
        f"Best epoch: {history.best_epoch}\n"
        f"Overfitting warnings: {history.overfit_warning_count}\n\n"
        f"LR key nodes: first={history.lr_history[0] if history.lr_history else float('nan'):.8f}, "
        f"last={history.lr_history[-1] if history.lr_history else float('nan'):.8f}\n\n"
        "Loss curve key nodes:\n"
        f"- first_train_loss: {first_train_loss:.6f}\n"
        f"- best_train_loss: {best_train_loss:.6f}\n"
        f"- last_train_loss: {last_train_loss:.6f}\n"
        f"- first_val_loss: {first_val_loss:.6f}\n"
        f"- best_val_loss: {best_val_loss:.6f}\n"
        f"- last_val_loss: {last_val_loss:.6f}\n\n"
        "Dataset stats snapshot:\n"
        f"- total_files: {dataset_stats.total_files}\n"
        f"- valid_samples: {dataset_stats.valid_samples}\n"
        f"- filtered_samples: {dataset_stats.filtered_samples}\n"
        f"- class_distribution: {dataset_stats.class_distribution}\n"
        f"- rule_distribution: {dataset_stats.rule_distribution}\n"
    )
    out_path.write_text(report, encoding="utf-8")
    LOGGER.info("Training report saved to %s", out_path)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _reduce_lr(optimizer: torch.optim.Optimizer, factor: float = 0.5, min_lr: float = 1e-6) -> float:
    current = _current_lr(optimizer)
    new_lr = max(min_lr, current * factor)
    _set_optimizer_lr(optimizer, new_lr)
    return new_lr


def verify_checkpoint_loadability(
    model: CNNBaseline,
    checkpoint_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    """Validate checkpoint loading and output shape compatibility."""
    from model.cnn_baseline import load_weights

    load_weights(model, path=checkpoint_path, strict=True)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 16, 1, image_size, image_size, dtype=torch.float32, device=device)
        logits, probs = model(dummy)
    if logits.shape != (2, 8) or probs.shape != (2, 8):
        raise RuntimeError(
            "Checkpoint I/O shape mismatch: "
            f"logits={tuple(logits.shape)}, probs={tuple(probs.shape)}, expected=(2,8)"
        )
    LOGGER.info("Checkpoint verification passed: %s", checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN baseline on RAVEN")
    parser.add_argument("--data_root", type=str, default="data/raven")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive_weight", type=float, default=0.3)
    parser.add_argument("--grad_explode_norm", type=float, default=1_000.0)
    parser.add_argument("--grad_vanish_norm", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        contrastive_weight=args.contrastive_weight,
        grad_explode_norm=args.grad_explode_norm,
        grad_vanish_norm=args.grad_vanish_norm,
    ).sanitize()

    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, _test_loader, dataset_stats = build_raven_dataloaders(
            data_root=cfg.data_root,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            image_size=cfg.image_size,
            output_channels=1,  # Keep compatible with CNNBaseline input channels.
            seed=cfg.seed,
        )
    except Exception as exc:
        LOGGER.error("Data loading failed: %s", exc)
        raise

    model = CNNBaseline(num_choices=8, feature_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    loss_monitor = LossMonitor(spike_factor=5.0, log_path=LOG_PATH)
    setup_contrastive_logger(Path("experiments/logs/contrastive_loss.log"))
    contrastive_adapter = ContrastiveLossAdapter(margin=0.2, max_pairs=2048)

    history = TrainHistory()
    start_time = time.time()
    nan_restart_budget = 2

    try:
        for epoch in range(1, cfg.epochs + 1):
            LOGGER.info("========== Epoch %d/%d ==========", epoch, cfg.epochs)
            history.lr_history.append(_current_lr(optimizer))

            while True:
                try:
                    train_loss, train_acc = run_one_epoch_train(
                        model=model,
                        loader=train_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        loss_monitor=loss_monitor,
                        contrastive_adapter=contrastive_adapter,
                        contrastive_weight=cfg.contrastive_weight,
                        cfg=cfg,
                        epoch=epoch,
                        device=device,
                    )
                    break
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if ("nan" in msg or "inf" in msg) and nan_restart_budget > 0:
                        nan_restart_budget -= 1
                        new_lr = _reduce_lr(optimizer, factor=0.5)
                        LOGGER.warning(
                            "检测到 NaN/Inf，自动降低学习率并重启当前 epoch。new_lr=%.8f, remaining_restarts=%d",
                            new_lr,
                            nan_restart_budget,
                        )
                        continue
                    raise
            val_loss, val_acc = run_one_epoch_val(
                model=model,
                loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
            )

            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            LOGGER.info(
                "Epoch %d result | train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            # Save best checkpoint.
            if val_acc > history.best_val_acc:
                history.best_val_acc = val_acc
                history.best_epoch = epoch
                save_weights(
                    model,
                    path=WEIGHT_DIR / "cnn_contrastive.pth",
                    extra={"epoch": epoch, "val_acc": val_acc, "contrastive_weight": cfg.contrastive_weight},
                )

            if detect_overfitting(history, epoch):
                history.overfit_warning_count += 1
                new_lr = _reduce_lr(optimizer, factor=0.7)
                LOGGER.warning(
                    "过拟合警告: 连续 3 个 epochs 验证准确率不提升且 loss 不下降。"
                    "已自动降低学习率至 %.8f（当前模型无 dropout）。",
                    new_lr,
                )

            # Save checkpoint for every epoch.
            save_weights(
                model,
                path=WEIGHT_DIR / f"cnn_contrastive_epoch_{epoch:03d}.pth",
                extra={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": _current_lr(optimizer),
                    "contrastive_weight": cfg.contrastive_weight,
                },
            )

        # Save final model.
        save_weights(
            model,
            path=WEIGHT_DIR / "cnn_contrastive_last.pth",
            extra={
                "epoch": cfg.epochs,
                "best_val_acc": history.best_val_acc,
                "contrastive_weight": cfg.contrastive_weight,
            },
        )

    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user (KeyboardInterrupt). Saving checkpoint...")
        save_weights(model, path=WEIGHT_DIR / "cnn_contrastive_interrupt.pth", extra={"reason": "keyboard_interrupt"})
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            LOGGER.error("Training interrupted by OOM. Saving checkpoint...")
            save_weights(model, path=WEIGHT_DIR / "cnn_contrastive_interrupt.pth", extra={"reason": "oom"})
        else:
            LOGGER.error("Runtime error during training: %s", exc)
            save_weights(model, path=WEIGHT_DIR / "cnn_contrastive_interrupt.pth", extra={"reason": "runtime_error"})
            raise
    finally:
        duration = time.time() - start_time
        write_train_report(history=history, duration_sec=duration, dataset_stats=dataset_stats, out_path=RESULT_PATH)
        LOGGER.info("Training finished in %.2f seconds", duration)

    # Validate that saved weights can be loaded and model I/O shapes match.
    best_ckpt = WEIGHT_DIR / "cnn_contrastive.pth"
    last_ckpt = WEIGHT_DIR / "cnn_contrastive_last.pth"
    if best_ckpt.exists():
        verify_checkpoint_loadability(model, best_ckpt, cfg.image_size, device)
    if last_ckpt.exists():
        verify_checkpoint_loadability(model, last_ckpt, cfg.image_size, device)


if __name__ == "__main__":
    main()

