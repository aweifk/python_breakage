# -*- coding: utf-8 -*-
"""
三通道 1D-DRSN 断刀检测模型训练脚本
- 输入: (N, 3, 500) 三通道时序，通道顺序 RMS / AMP / Power
- 窗口: 500 点，步长 200
- 输出: 正常 / 断刀 两类概率
- 交付产物: drsn_model.pth / drsn_norm.json / drsn_threshold.txt
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ======================== 常量 ========================
WINDOW_SIZE = 500
STEP = 200
NUM_CLASSES = 2
# 文档规定通道顺序: ch0=RMS, ch1=AMP, ch2=Power
CHANNEL_ORDER = ["rms", "amp", "power"]
EPS = 1e-8

# 增强参数
AUG_PER_POSITIVE = 6        # 每个正样本窗口的时间偏移增强份数
AUG_NOISE_RATIO = 0.02      # 噪声标准差相对各通道 std 的比例
AUG_SHIFT_RANGE = STEP      # 时间偏移范围（±STEP）

# 训练参数默认值
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.001


# ======================== 模型定义 ========================
class SoftThreshold(nn.Module):
    """自适应软阈值去噪模块。"""

    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        abs_x = torch.abs(x)
        A = self.gap(abs_x).squeeze(-1)   # (B, C)
        a = self.fc(A)                    # (B, C)
        T = (a * A).unsqueeze(-1)         # (B, C, 1)
        return torch.sign(x) * torch.relu(torch.abs(x) - T)


class RSBU(nn.Module):
    """残差收缩单元。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.soft_threshold = SoftThreshold(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.soft_threshold(out)
        out = out + identity
        return self.relu(out)


class DRSN1D_3Ch(nn.Module):
    """三通道一维深度残差收缩网络。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv0 = nn.Conv1d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU(inplace=True)

        self.drsn1 = RSBU(4, 16, stride=1)    # (B, 16, 500)
        self.drsn2 = RSBU(16, 32, stride=2)   # (B, 32, 250)
        self.drsn3 = RSBU(32, 64, stride=2)   # (B, 64, 125)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 3, 500)
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.drsn1(out)
        out = self.drsn2(out)
        out = self.drsn3(out)
        out = self.gap(out).squeeze(-1)   # (B, 64)
        out = self.dropout(out)
        return self.fc(out)               # (B, 2)


# ======================== 数据加载 ========================
def find_ae_columns(df):
    """按列名（不区分大小写）定位 amp/power/rms 三列。"""
    target = ["amp", "power", "rms"]
    found = {}
    for t in target:
        matched = [c for c in df.columns if c.lower() == t]
        if matched:
            found[t] = matched[0]
        else:
            raise ValueError(f"未找到列 '{t}'，请确保CSV包含 amp/power/rms 三列。")
    return found


def extract_file_windows(df):
    """对单个文件按 WINDOW_SIZE/STEP 滑窗。

    返回:
        X: (N, 3, WINDOW_SIZE) float32 原始窗口
        y: (N,) int64 标签（窗口内 label.max()==1 判为断刀）
        starts: (N,) int64 窗口起始下标
        sig: (3, L) float32 该文件三通道原始信号（供增强重切片）
        labels: (L,) int64 该文件逐点标签
    """
    col_map = find_ae_columns(df)
    sig = np.stack(
        [df[col_map[c]].to_numpy(dtype=np.float32) for c in CHANNEL_ORDER], axis=0
    )  # (3, L)
    labels = df["label"].to_numpy(dtype=np.int64)
    L = sig.shape[1]

    X, y, starts = [], [], []
    for i in range(0, L - WINDOW_SIZE + 1, STEP):
        X.append(sig[:, i : i + WINDOW_SIZE])
        y.append(1 if labels[i : i + WINDOW_SIZE].max() == 1 else 0)
        starts.append(i)

    X = np.stack(X).astype(np.float32) if X else np.zeros((0, 3, WINDOW_SIZE), dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    return X, y, starts, sig, labels


def create_windows_no_label(df, window_size=WINDOW_SIZE, step=STEP):
    """对无标签文件滑窗，返回 (X, starts)。供 GUI 预测使用。"""
    col_map = find_ae_columns(df)
    sig = np.stack(
        [df[col_map[c]].to_numpy(dtype=np.float32) for c in CHANNEL_ORDER], axis=0
    )
    L = sig.shape[1]
    X, starts = [], []
    for i in range(0, L - window_size + 1, step):
        X.append(sig[:, i : i + window_size])
        starts.append(i)
    X = np.stack(X).astype(np.float32) if X else np.zeros((0, 3, window_size), dtype=np.float32)
    return X, np.asarray(starts, dtype=np.int64)


def create_windows_with_label(df, window_size=WINDOW_SIZE, step=STEP):
    """对带标签文件滑窗，返回 (X, y, starts)。供 GUI 评估使用。"""
    col_map = find_ae_columns(df)
    sig = np.stack(
        [df[col_map[c]].to_numpy(dtype=np.float32) for c in CHANNEL_ORDER], axis=0
    )
    labels = df["label"].to_numpy(dtype=np.int64)
    L = sig.shape[1]
    X, y, starts = [], [], []
    for i in range(0, L - window_size + 1, step):
        X.append(sig[:, i : i + window_size])
        y.append(1 if labels[i : i + window_size].max() == 1 else 0)
        starts.append(i)
    X = np.stack(X).astype(np.float32) if X else np.zeros((0, 3, window_size), dtype=np.float32)
    return X, np.asarray(y, dtype=np.int64), np.asarray(starts, dtype=np.int64)


def load_all_windows(data_path):
    """读取文件夹内全部 CSV，合并窗口并保留文件级元数据。"""
    if os.path.isfile(data_path):
        csv_files = [data_path]
    elif os.path.isdir(data_path):
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".csv")]
    else:
        raise FileNotFoundError(f"路径不存在: {data_path}")
    if not csv_files:
        raise FileNotFoundError(f"文件夹 {data_path} 中没有 CSV 文件")

    all_X, all_y, all_starts, all_file = [], [], [], []
    file_sigs, file_labels = [], []

    for fname in sorted(csv_files):
        df = pd.read_csv(fname)
        if "label" not in df.columns:
            raise ValueError(f"{fname} 缺少 'label' 列")
        X, y, starts, sig, labels = extract_file_windows(df)
        if len(X) == 0:
            continue
        fid = len(file_sigs)
        file_sigs.append(sig)
        file_labels.append(labels)
        all_X.append(X)
        all_y.append(y)
        all_starts.append(starts)
        all_file.append(np.full(len(X), fid, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    starts = np.concatenate(all_starts, axis=0)
    file_ids = np.concatenate(all_file, axis=0)
    return X, y, starts, file_ids, file_sigs, file_labels


def compute_norm_stats(X):
    """逐通道标量 mean/std（训练集统计量）。X: (N, 3, L)。"""
    mean = X.mean(axis=(0, 2)).astype(np.float32)   # (3,)
    std = X.std(axis=(0, 2)).astype(np.float32)     # (3,)
    return mean, std


def normalize(X, mean, std):
    return (X - mean[None, :, None]) / (std[None, :, None] + EPS)


# ======================== 时间偏移增强 ========================
def augment_train_positives(
    train_idx,
    starts,
    file_ids,
    file_sigs,
    file_labels,
    aug_per=AUG_PER_POSITIVE,
    shift_range=AUG_SHIFT_RANGE,
    noise_ratio=AUG_NOISE_RATIO,
    channel_std=None,
    seed=None,
):
    """对训练集中正样本窗口做时间偏移 + 高斯噪声增强。

    时间偏移通过在原文件信号上重切片实现，偏移后仍须包含断刀点（label.max()==1）。
    """
    rng = np.random.default_rng(seed)
    if channel_std is None:
        channel_std = np.ones(3, dtype=np.float32)

    new_X, new_y = [], []
    for idx in train_idx:
        fid = int(file_ids[idx])
        start = int(starts[idx])
        sig = file_sigs[fid]
        labels = file_labels[fid]
        L = sig.shape[1]
        lo = max(0, start - shift_range)
        hi = min(start + shift_range, L - WINDOW_SIZE)

        for _ in range(aug_per):
            if lo > hi:
                break
            i2 = int(rng.integers(lo, hi + 1))
            window = sig[:, i2 : i2 + WINDOW_SIZE].copy()
            if labels[i2 : i2 + WINDOW_SIZE].max() != 1:
                continue  # 偏移后不再包含断刀点，丢弃
            noise = rng.normal(0.0, 1.0, size=window.shape).astype(np.float32)
            window += noise * (noise_ratio * channel_std)[:, None]
            new_X.append(window)
            new_y.append(1)

    return new_X, new_y


# ======================== 模型评估辅助 ========================
def predict_proba(model, X, device, batch_size=256):
    """返回 (N, 2) 概率，断刀概率在第 1 列。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 2), dtype=np.float32)


def search_best_threshold(y_true, break_proba):
    """在验证集上搜索使断刀 F1 最大的阈值（仿 XGBoost.py）。"""
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.1, 0.9, 0.01):
        pred = (break_proba >= th).astype(int)
        if np.any(pred):
            f1 = f1_score(y_true, pred, pos_label=1)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
    return float(best_th)


# ======================== 训练主函数 ========================
def train_drsn(
    data_path,
    save_dir="./",
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    seed=42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 1. 数据加载与窗口提取
    X, y, starts, file_ids, file_sigs, file_labels = load_all_windows(data_path)
    print(f"总窗口数: {len(X)}, 正样本(断刀): {int(y.sum())}, 比例: {y.mean():.4f}")

    # 2. 分层划分: 64 / 16 / 20
    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.2, random_state=seed, stratify=y[idx_trainval]
    )
    print(f"train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    # 3. 归一化统计量（仅用训练集）
    norm_mean, norm_std = compute_norm_stats(X[idx_train])

    # 4. 训练集正样本增强（时间偏移 + 噪声，仅作用于训练集）
    aug_X, aug_y = augment_train_positives(
        idx_train,
        starts,
        file_ids,
        file_sigs,
        file_labels,
        aug_per=AUG_PER_POSITIVE,
        shift_range=AUG_SHIFT_RANGE,
        noise_ratio=AUG_NOISE_RATIO,
        channel_std=norm_std,
        seed=seed,
    )
    if aug_X:
        X_train = np.concatenate([X[idx_train], np.stack(aug_X, axis=0)], axis=0)
        y_train = np.concatenate([y[idx_train], np.asarray(aug_y, dtype=np.int64)], axis=0)
    else:
        X_train = X[idx_train]
        y_train = y[idx_train]
    print(f"增强后训练集: {len(X_train)} (正样本 {int(y_train.sum())})")

    X_train = normalize(X_train, norm_mean, norm_std)
    X_val = normalize(X[idx_val], norm_mean, norm_std)
    X_test = normalize(X[idx_test], norm_mean, norm_std)
    y_val = y[idx_val]
    y_test = y[idx_test]

    # 5. 类别权重（按增强后训练集逆类别频率）
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = n_neg / max(n_pos, 1.0)
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    print(f"类别权重: [1.0, {pos_weight:.2f}]")

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 6. 训练
    model = DRSN1D_3Ch(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X_train)

        val_proba = predict_proba(model, X_val, device)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        val_f1 = f1_score(y_val, val_pred, pos_label=1) if np.any(val_pred) else 0.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_break_f1={val_f1:.4f}")

    print(f"最佳验证断刀 F1: {best_val_f1:.4f}")

    # 7. 载入最佳权重，搜索最佳阈值
    model.load_state_dict(best_state)
    val_proba = predict_proba(model, X_val, device)[:, 1]
    best_threshold = search_best_threshold(y_val, val_proba)
    print(f"最佳阈值: {best_threshold:.3f}")

    # 8. 测试集评估
    test_proba = predict_proba(model, X_test, device)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred, pos_label=1)
    f1 = f1_score(y_test, test_pred, pos_label=1)
    print("\n========== 测试集评估 ==========")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"断刀召回率 (Recall): {rec:.4f}")
    print(f"断刀 F1: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, test_pred, target_names=["正常", "断刀"], zero_division=0))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, test_pred))

    # 9. 保存产物
    os.makedirs(save_dir, exist_ok=True)
    model_file = os.path.join(save_dir, "drsn_model.pth")
    norm_file = os.path.join(save_dir, "drsn_norm.json")
    threshold_file = os.path.join(save_dir, "drsn_threshold.txt")

    torch.save(best_state, model_file)
    with open(norm_file, "w", encoding="utf-8") as f:
        json.dump({"mean": norm_mean.tolist(), "std": norm_std.tolist()}, f, ensure_ascii=False)
    with open(threshold_file, "w", encoding="utf-8") as f:
        f.write(str(best_threshold))

    print(f"\n模型已保存: {model_file}")
    print(f"归一化统计量已保存: {norm_file}")
    print(f"最佳阈值已保存: {threshold_file}")

    return model, best_threshold


def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    os.chdir(_script_dir())
    data_dir = os.path.normpath(os.path.join(_script_dir(), "..", "labeled_data"))
    train_drsn(data_dir, save_dir=_script_dir())
