# -*- coding: utf-8 -*-
"""
=============================================================================
文件名称 : ae_rf_train.py
功能描述 : 用随机森林训练断刀检测分类器（不需要LLM/API/GPU）
           1) 加载 dataset/normal/ broken/ 所有CSV
           2) 提取滑窗特征（复用现有FeatureExtractor）
           3) 按文件划分 train/valid/test
           4) 训练 RandomForest
           5) 输出滑窗级 + 文件级混淆矩阵
           6) 保存模型 rf_break_tool.pkl

用法:
    python ae_rf_train.py
=============================================================================
"""
import os, sys, glob, random, json, logging, traceback
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from ae_llm_break_tool_detect import (
    CONFIG, AEDataLoader, FeatureExtractor, WindowFeature
)
from ae_batch_pipeline import load_one_csv, scan_dataset, split_files
from ae_window_config import WINDOW_SEC, STEP_SEC, MIN_EVENTS_PER_WINDOW, load_sliding_window_params

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
CFG = {
    "data_dir": "./dataset",
    "output_dir": "./output_rf",
    "window_sec": WINDOW_SEC,
    "step_sec": STEP_SEC,
    "min_events": MIN_EVENTS_PER_WINDOW,
    "seed": 42,
    # 模型参数
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RF-Train")

# 特征列（和WindowFeature字段对应）
FEATURE_COLS = [
    "tool_diameter",
    "amp_mean", "amp_std", "amp_p99", "amp_max",
    "power_mean", "power_std", "power_p99", "power_max",
    "rms_max", "asl_mean", "centroid_mean", "high_energy_ratio",
    "kurtosis", "crest_factor", "skewness", "impulse_factor",
    # 7维突变形态特征
    "peak_amp_ratio", "max_rise_slope", "pulse_half_width", "rise_time",
    "sharpness", "fall_slope", "baseline_diff",
    "power_drop_ratio",
    # 电流特征（多传感器融合）
    "spindle_current_mean", "spindle_current_std", "spindle_current_max",
    "x_current_mean", "x_current_std", "x_current_max",
    "y_current_mean", "y_current_std", "y_current_max",
    "z_current_mean", "z_current_std", "z_current_max",
]

# 追加：按直径正常基线的相对比值特征（跨直径可比，断刀时趋近0）
from ae_batch_pipeline import DIAL_RATIO_COLS as _DIAL_RATIO_COLS
FEATURE_COLS += list(_DIAL_RATIO_COLS)

# 追加：按直径空切/实切上下限的相对特征
from ae_batch_pipeline import DIA_THR_RATIO_COLS as _DIA_THR_RATIO_COLS
FEATURE_COLS += list(_DIA_THR_RATIO_COLS)


def feats_to_matrix(feats: list):
    """把WindowFeature列表转成(X, y, file_info)"""
    rows, labels, files = [], [], []
    for f in feats:
        row = [getattr(f, c, 0.0) or 0.0 for c in FEATURE_COLS]
        rows.append(row)
        labels.append(1 if f.status == "断刀" else 0)
        files.append(getattr(f, "source_file", "unknown"))
    return np.array(rows), np.array(labels), files


def main():
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # 扫描数据
    logger.info("扫描数据集...")
    files_by_label = scan_dataset(CFG["data_dir"])
    # 空切+基线数据都当正常类
    files_by_label["正常"] = (files_by_label["正常"]
                              + files_by_label.get("空切", [])
                              + files_by_label.get("基线", []))

    # 按文件划分
    splits = {"train": {"正常": [], "断刀": []},
              "valid": {"正常": [], "断刀": []},
              "test":  {"正常": [], "断刀": []}}
    for label, flist in files_by_label.items():
        if label in ("空切", "基线"):
            continue
        tr, va, te = split_files(flist, CFG["seed"])
        splits["train"][label] = tr
        splits["valid"][label] = va
        splits["test"][label]  = te

    wp = load_sliding_window_params()
    ex = FeatureExtractor(
        wp["window_sec"], wp["step_sec"], wp["min_events_per_window"])

    # 提取所有窗特征
    all_data = {"train": [], "valid": [], "test": []}
    file_feats = {"train": [], "valid": [], "test": []}  # 按文件存，用于文件级评估

    for sp in ["train", "valid", "test"]:
        for label in ["正常", "断刀"]:
            for path in splits[sp][label]:
                try:
                    df = load_one_csv(path)
                    feats = ex.extract(df)
                    # power_drop_ratio
                    for i, f in enumerate(feats):
                        if i == 0:
                            f.power_drop_ratio = 1.0
                        else:
                            p = feats[i-1].power_mean
                            f.power_drop_ratio = f.power_mean / p if p > 1e-6 else 1.0
                    # 打窗级标签（文件标签）
                    for f in feats:
                        f.source_file = os.path.basename(path)
                        f.gt_label = label
                        # 窗级标签：断刀文件=1，正常=0
                        # 但正常文件里的断刀窗也当1（模型学极端值）
                    all_data[sp].extend(feats)
                    file_feats[sp].append((os.path.basename(path), label, feats))
                except Exception as e:
                    logger.error(f"处理失败 {path}: {e}")

    # 构造X/y
    def build_xy(feats):
        X, y, files = feats_to_matrix(feats)
        # 标签：断刀文件=1，正常文件=0
        y2 = np.array([1 if f.gt_label == "断刀" else 0 for f in feats])
        return X, y2, files

    X_train, y_train, _ = build_xy(all_data["train"])
    X_valid, y_valid, _ = build_xy(all_data["valid"])
    X_test,  y_test,  _ = build_xy(all_data["test"])

    logger.info(f"训练集: {len(X_train)} 窗, 断刀={y_train.sum()}, 正常={(1-y_train).sum()}")
    logger.info(f"验证集: {len(X_valid)} 窗")
    logger.info(f"测试集: {len(X_test)} 窗")

    # 训练
    logger.info("训练 RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=CFG["n_estimators"],
        max_depth=CFG["max_depth"],
        min_samples_leaf=CFG["min_samples_leaf"],
        class_weight="balanced",  # 不平衡样本
        random_state=CFG["seed"],
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 滑窗级评估
    for name, X, y in [("train", X_train, y_train),
                        ("valid", X_valid, y_valid),
                        ("test",  X_test,  y_test)]:
        pred = clf.predict(X)
        acc = (pred == y).mean()
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        logger.info(f"  {name}: TP={tp} FN={fn} TN={tn} FP={fp} "
                    f"acc={acc:.3f} recall={recall:.3f} prec={prec:.3f}")

    # 文件级评估：一个文件里任意1个窗预测断刀 => 文件断刀
    logger.info("\n=== 文件级评估（RF预测）===")
    for sp in ["train", "valid", "test"]:
        tp = fn = tn = fp = 0
        for fname, label, feats in file_feats[sp]:
            Xf, _, _ = feats_to_matrix(feats)
            proba = clf.predict_proba(Xf)[:, 1]
            # 文件判断刀：任意窗概率>0.5
            pred_broken = (proba > 0.5).any()
            is_broken = (label == "断刀")
            if is_broken and pred_broken: tp += 1
            elif is_broken and not pred_broken: fn += 1
            elif not is_broken and not pred_broken: tn += 1
            else: fp += 1
        total = tp + fn + tn + fp
        acc = (tp + tn) / total if total else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        logger.info(f"  {sp}: TP={tp} FN={fn} TN={tn} FP={fp} "
                    f"acc={acc:.3f} recall={recall:.3f} prec={prec:.3f}")

    # 特征重要性
    logger.info("\n=== 特征重要性 Top10 ===")
    imp = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                 key=lambda x: -x[1])
    for name, v in imp[:10]:
        logger.info(f"  {name:25s} {v:.3f}")

    # 保存模型
    model_path = os.path.join(CFG["output_dir"], "rf_break_tool.pkl")
    joblib.dump({"model": clf, "features": FEATURE_COLS}, model_path)
    logger.info(f"\n模型已保存: {model_path}")
    logger.info(f"推理接口见 ae_rf_infer.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"异常: {e}")
        traceback.print_exc()
        sys.exit(1)
