"""train.py — Training pipeline, evaluation, and experiment runner.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.03,                   # Chọn bằng validation
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str, reduction: str = "mean") -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    else:
        raise ValueError(f"Hàm mất mát không hỗ trợ: {loss_name}. Chọn 'ce' hoặc 'mse'.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    preds = []
    n = len(X)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name=loss_name, reduction="sum")
        total_loss += float(loss.item())
        preds.append(logits.argmax(dim=1))

    all_preds = torch.cat(preds, dim=0).cpu().numpy()
    y_true = y.cpu().numpy()

    avg_loss = total_loss / n
    acc = float((all_preds == y_true).mean())

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true, all_preds), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    device = data["X_tr"].device

    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(
        hidden=hidden,
        dropout=float(cfg.get("dropout", 0.0)),
        init=cfg.get("init", "he")
    ).to(device)

    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số {count_params(model)} không khớp EXPECTED_PARAMS {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    is_cuda = (device.type == "cuda")
    use_amp = is_cuda and (precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (is_cuda and precision == "fp16") else None

    # Đo loss bước 0 trên val TRƯỚC bước cập nhật đầu tiên
    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_eval["loss"]

    # Đánh giá train loss trên tập con cố định 50 000 mẫu để tăng tốc và công bằng
    n_tr_eval = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:n_tr_eval]
    y_tr_eval = data["y_tr"][:n_tr_eval]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_acc = 0.0
    best_macro_f1 = 0.0
    best_state = None
    diverged = False

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])

    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch"])
    clip_norm = cfg.get("clip_norm")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        step_grad_norms = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name=cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name=cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            step_grad_norms.append(gn)

        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        if diverged:
            print(f"[{cfg['exp_id']}] CẢNH BÁO: Loss phân kỳ (NaN/Inf) tại epoch {epoch}!")
            break

        # Đánh giá cuối epoch (ở chế độ eval)
        tr_res = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(step_grad_norms)) if step_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        # Lưu checkpoint có val_loss tốt nhất
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_acc = val_res["acc"]
            best_macro_f1 = val_res["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Tổng kết summary
    final_tr_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    avg_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0
    peak_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if is_cuda else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_tr_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_acc),
        "val_macro_f1": float(best_macro_f1),
        "time_per_epoch_s": round(avg_epoch_time, 2),
        "peak_mem_MB": round(peak_mem, 1),
        "diverged": diverged,
    }

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(path, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán vào {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(hidden=hidden, dropout=0.0, init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
