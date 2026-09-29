# -*- coding: utf-8 -*-
"""
XGBoost 断刀检测模型训练脚本（增量版）
- 保留原有 15 维特征（含 decay_ratio/slope/peak_to_post_ratio/post_peak_mean）
- 新增 2 个峰后低位特征（解决"对刀后加工"误报）
- 特征维度：17 × 3 通道 = 51 维
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ======================== 特征定义 ========================
WINDOW_SIZE = 60
STEP = 20
LOW_THRESHOLD_RATIO = 0.30   # "低位"定义：低于峰值的30%（较宽松，避免漏判）

# 基础 11 维
STAT_NAMES_BASE = [
    'mean', 'var', 'std', 'kurtosis', 'skewness',
    'peak', 'peak_to_peak', 'form_factor', 'crest_factor',
    'impulse_factor', 'margin_factor'
]

# 原有 4 维趋势特征（保留，效果很好）
STAT_NAMES_TREND_OLD = [
    'decay_ratio',          # 前半段/后半段均值比
    'slope',                # 线性回归斜率
    'peak_to_post_ratio',   # 峰值/峰后均值
    'post_peak_mean',       # 峰值后均值
]

# 新增 2 维峰后特征（解决对刀后加工误报）
STAT_NAMES_TREND_NEW = [
    'post_peak_low_fraction',  # 峰后低值点占比（断刀时接近1）
    'post_peak_max_ratio',     # 峰后最大/峰值（断刀时很小）
]

# 完整特征列表（17 维）
STAT_NAMES = STAT_NAMES_BASE + STAT_NAMES_TREND_OLD + STAT_NAMES_TREND_NEW


def extract_time_domain_features(data):
    """
    时域特征提取：11 基础 + 4 原趋势 + 2 新峰后 = 17 维
    """
    data = np.array(data, dtype=float)
    n = len(data)
    if n == 0:
        return {s: 0 for s in STAT_NAMES}

    # ---------- 1. 基础 11 维 ----------
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    peak = np.max(np.abs(data))
    peak_to_peak = np.max(data) - np.min(data)
    rms = np.sqrt(np.mean(data ** 2))
    abs_mean = np.mean(np.abs(data))
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(data)))
    kurtosis = np.mean(((data - mean) / std) ** 4) if std > 1e-10 else 0
    skewness = np.mean(((data - mean) / std) ** 3) if std > 1e-10 else 0
    form_factor = rms / abs_mean if abs_mean > 1e-10 else 0
    crest_factor = peak / rms if rms > 1e-10 else 0
    impulse_factor = peak / abs_mean if abs_mean > 1e-10 else 0
    margin_factor = peak / sqrt_abs_mean if sqrt_abs_mean > 1e-10 else 0

    # ---------- 2. 原有 4 维趋势特征 ----------
    # (a) 前半段/后半段均值比
    half = n // 2
    first_half_mean = np.mean(data[:half]) if half > 0 else 0.0
    second_half_mean = np.mean(data[half:]) if (n - half) > 0 else 0.0
    decay_ratio = first_half_mean / (second_half_mean + 1e-6)

    # (b) 线性回归斜率
    if n > 1:
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, data, 1)[0]
    else:
        slope = 0.0

    # (c) 峰值位置与峰后均值
    peak_idx = int(np.argmax(np.abs(data)))
    if peak_idx < n - 1:
        post_peak_mean = np.mean(data[peak_idx + 1:])
    else:
        post_peak_mean = abs(data[peak_idx])

    # (d) 峰值 / 峰后均值
    peak_to_post_ratio = peak / (post_peak_mean + 1e-6)

    # ---------- 3. 新增 2 维峰后低位特征 ----------
    if (n - peak_idx - 1) >= 3:
        post = data[peak_idx + 1:]
        # 峰后低值点占比：低于"峰值×30%"的点占比
        # 断刀：峰后所有点都很低 → 接近 1
        # 对刀后加工：峰后仍有很多点较高 → 接近 0
        low_threshold = peak * LOW_THRESHOLD_RATIO
        post_peak_low_fraction = np.mean(post < low_threshold)
        # 峰后最大/峰值
        post_peak_max_ratio = np.max(np.abs(post)) / (peak + 1e-6)
    else:
        post_peak_low_fraction = 0.0
        post_peak_max_ratio = 1.0

    return {
        # 基础 11 维
        'mean': mean, 'var': var, 'std': std,
        'kurtosis': kurtosis, 'skewness': skewness,
        'peak': peak, 'peak_to_peak': peak_to_peak,
        'form_factor': form_factor, 'crest_factor': crest_factor,
        'impulse_factor': impulse_factor, 'margin_factor': margin_factor,
        # 原趋势 4 维
        'decay_ratio': decay_ratio,
        'slope': slope,
        'peak_to_post_ratio': peak_to_post_ratio,
        'post_peak_mean': post_peak_mean,
        # 新峰后 2 维
        'post_peak_low_fraction': post_peak_low_fraction,
        'post_peak_max_ratio': post_peak_max_ratio,
    }


def find_ae_columns(df):
    target = ['amp', 'power', 'rms']
    found = {}
    for t in target:
        matched = [c for c in df.columns if c.lower() == t]
        if matched:
            found[t] = matched[0]
        else:
            raise ValueError(f"未找到列 '{t}'，请确保CSV包含 amp/power/rms 三列。")
    return found


def create_windows_with_label(df, window_size=WINDOW_SIZE, step=STEP):
    col_map = find_ae_columns(df)
    X, y = [], []
    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        feature_vector = []
        for feat in ['amp', 'power', 'rms']:
            col = col_map[feat]
            stats = extract_time_domain_features(window[col].values)
            feature_vector.extend([stats[s] for s in STAT_NAMES])
        X.append(feature_vector)
        y.append(1 if window['label'].max() == 1 else 0)
    return np.array(X), np.array(y)


# ======================== 训练主函数 ========================
def train_xgboost(data_path, model_save_path='./', test_size=0.2, random_state=42):
    # 1. 读取数据
    if os.path.isfile(data_path):
        print(f"读取单个文件: {data_path}")
        df = pd.read_csv(data_path)
    elif os.path.isdir(data_path):
        print(f"读取文件夹: {data_path}")
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"文件夹 {data_path} 中没有 CSV 文件")
        dfs = []
        for fname in csv_files:
            print(f"  加载 {fname}")
            dfs.append(pd.read_csv(os.path.join(data_path, fname)))
        df = pd.concat(dfs, ignore_index=True)
        print(f"合并后总行数: {len(df)}")
    else:
        raise FileNotFoundError(f"路径不存在: {data_path}")

    if 'label' not in df.columns:
        raise ValueError("CSV 文件必须包含 'label' 列（0/1）")

    # 2. 特征提取
    X, y = create_windows_with_label(df, WINDOW_SIZE, STEP)
    print(f"总样本数: {len(X)}, 特征维度: {X.shape[1]}")
    print(f"正样本（断刀）比例: {np.mean(y):.3f}")

    # 3. 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练
    print("训练 XGBoost 模型（增量版，51维）...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train_scaled, y_train)

    # 6. 评估
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred)
    print("\n========== 测试集评估（阈值=0.5） ==========")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"召回率 (Recall):   {rec:.4f}")
    print(f"F1 分数:           {f1_value:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '断刀']))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 7. 最佳阈值
    best_threshold = 0.5
    best_f1 = 0.0
    for th in np.arange(0.1, 0.9, 0.01):
        pred = (y_pred_proba >= th).astype(int)
        f1_tmp = f1_score(y_test, pred)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_threshold = th
    print(f"\n最佳阈值: {best_threshold:.3f}, 对应 F1 = {best_f1:.4f}")

    y_pred_best = (y_pred_proba >= best_threshold).astype(int)
    print(f"使用最佳阈值后: 准确率={accuracy_score(y_test, y_pred_best):.4f}, "
          f"召回率={recall_score(y_test, y_pred_best):.4f}")

    # 8. 保存
    os.makedirs(model_save_path, exist_ok=True)
    model_file = os.path.join(model_save_path, 'xgboost_model.pkl')
    scaler_file = os.path.join(model_save_path, 'scaler.pkl')
    threshold_file = os.path.join(model_save_path, 'threshold.txt')

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    with open(threshold_file, 'w') as f:
        f.write(str(best_threshold))

    print(f"\n模型已保存: {model_file}")
    print(f"标准化器已保存: {scaler_file}")
    print(f"最佳阈值已保存: {threshold_file}")

    # 9. 特征重要性
    importance = model.feature_importances_
    feature_names = []
    for feat in ['amp', 'power', 'rms']:
        for stat in STAT_NAMES:
            feature_names.append(f"{feat}_{stat}")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop-15 重要特征:")
    for i in range(min(15, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i + 1:2d}. {feature_names[idx]}: {importance[idx]:.4f}")

    return model, scaler, best_threshold


if __name__ == '__main__':
    data_path = 'labeled_data'
    train_xgboost(data_path, model_save_path='./')