"""plots.py — Plotting individual experiment curves and comparison figures.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = result["cfg"]
    hist = result["history"]
    epochs = hist["epoch"]
    best_epoch = result["summary"].get("best_epoch", epochs[-1] if epochs else 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)

    # 1. Loss curves
    ax0 = axes[0]
    ax0.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    ax0.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax0.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax0.set_title("Loss vs Epoch", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend()

    # 2. Accuracy & Macro-F1
    ax1 = axes[1]
    ax1.plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    if "val_macro_f1" in hist and len(hist["val_macro_f1"]) > 0:
        ax1.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", lw=2)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax1.set_title("Validation Metrics vs Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 3. Gradient Norm (before clipping)
    ax2 = axes[2]
    ax2.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#9467bd", lw=2)
    if cfg.get("clip_norm") is not None:
        ax2.axhline(cfg["clip_norm"], color="red", linestyle=":", alpha=0.8, label=f"Clip c={cfg['clip_norm']}")
    ax2.set_title("Gradient L2 Norm vs Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Grad Norm")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Overall title
    cfg_str = f"opt={cfg.get('optimizer')}, lr={cfg.get('lr')}, bs={cfg.get('batch')}, init={cfg.get('init')}, clip={cfg.get('clip_norm')}, drop={cfg.get('dropout')}"
    fig.suptitle(f"[{cfg.get('exp_id', 'exp')}] {cfg.get('description', '')}\n({cfg_str})",
                 fontsize=12, fontweight="bold", y=1.03)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        if metric in hist and len(hist[metric]) > 0:
            epochs = hist["epoch"]
            ax.plot(epochs, hist[metric], label=f"{exp_id}", lw=2)

    metric_name_map = {
        "val_loss": "Validation Loss",
        "val_macro_f1": "Validation Macro-F1",
        "val_acc": "Validation Accuracy",
        "train_loss": "Train Loss",
        "grad_norm": "Gradient Norm (pre-clip)",
    }
    y_label = metric_name_map.get(metric, metric)
    ax.set_title(title if title else f"So sánh {y_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
