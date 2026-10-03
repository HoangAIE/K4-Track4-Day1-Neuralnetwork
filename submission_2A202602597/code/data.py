"""data.py — Data loading, splitting, standardization, and batching.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file npz tại {processed_dir}. Hãy chạy scripts/split_data.py trước.")

    tr_data = np.load(train_path)
    ev_data = np.load(eval_path)

    X_train_full = tr_data["X"].astype(np.float32)
    y_train_full = tr_data["y"].astype(np.int64)

    X_eval = ev_data["X"].astype(np.float32)
    y_eval = ev_data["y"].astype(np.int64)
    eval_row_id = ev_data["row_id"].astype(np.int64)

    assert X_train_full.shape == (464809, 54), f"Shape X_train_full={X_train_full.shape} khác (464809, 54)"
    assert y_train_full.shape == (464809,), f"Shape y_train_full={y_train_full.shape} khác (464809,)"
    assert X_eval.shape == (116203, 54), f"Shape X_eval={X_eval.shape} khác (116203, 54)"
    assert y_eval.shape == (116203,), f"Shape y_eval={y_eval.shape} khác (116203,)"
    assert len(eval_row_id) == 116203, f"len(eval_row_id)={len(eval_row_id)} khác 116203"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Vì sao không được tính trên toàn bộ dữ liệu hay trên eval:
      Để ngăn ngừa rò rỉ thông tin (data leakage) từ validation/eval sang tập huấn luyện.
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled.astype(np.float32)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)
    X_tr_std = apply_standardizer(X_tr, mean, std)
    X_val_std = apply_standardizer(X_val, mean, std)
    X_eval_std = apply_standardizer(X_eval, mean, std)

    # Đưa lên device
    X_tr_t = torch.tensor(X_tr_std, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)
    X_val_t = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)
    X_eval_t = torch.tensor(X_eval_std, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device)

    # In thông tin kiểm tra
    print(f"X_tr: {X_tr_t.shape} ({X_tr_t.dtype}), y_tr: {y_tr_t.shape}")
    print(f"X_val: {X_val_t.shape} ({X_val_t.dtype}), y_val: {y_val_t.shape}")
    print(f"X_eval: {X_eval_t.shape} ({X_eval_t.dtype}), y_eval: {y_eval_t.shape}")

    # Accuracy mốc đoán lớp đa số trên tập val
    counts = torch.bincount(y_val_t)
    majority_class = counts.argmax().item()
    majority_acc = (y_val_t == majority_class).float().mean().item()
    print(f"Lớp đa số trên val: {majority_class}, tỷ lệ = {counts[majority_class].item()/len(y_val_t):.4f}")
    print(f"Accuracy mốc 'đoán luôn lớp đa số' trên val = {majority_acc:.4f} (mốc thấp nhất cần vượt)")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
