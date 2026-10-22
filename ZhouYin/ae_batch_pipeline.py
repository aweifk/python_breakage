# -*- coding: utf-8 -*-
"""
=============================================================================
文件名称 : ae_batch_pipeline.py
功能描述 : 批量处理几十组AE CSV数据，一键完成：
           1) 目录扫描（normal/ broken/ 两个文件夹）
           2) 自动阈值标定（用所有正常样本）
           3) 滑窗特征提取 + 规则打标签
           4) train/valid/test 划分（按文件，不按行，避免泄漏）
           5) 生成 SFT 训练数据（sharegpt JSONL）
           6) 输出数据分布统计

目录结构（你本地这样放就行）：
    dataset/
    ├── normal/           # 所有正常切削CSV
    │   ├── sample_01.csv
    │   └── sample_02.csv
    ├── broken/           # 所有断刀/崩刃CSV
    │   ├── sample_01.csv
    │   └── sample_02.csv
    └── unknown/          # （可选）现场待标注

用法：
    改下方 DATA_DIR 指向你的数据集根目录，然后：
    python ae_batch_pipeline.py

作者    : PHM Lab
创建日期: 2026-09-09
=============================================================================
"""

import os
import sys
import json
import glob
import random
import logging
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# 复用主程序模块
from ae_llm_break_tool_detect import (
    CONFIG, AEDataLoader, FeatureExtractor,
    PromptBuilder, RuleDiagnoser, WindowFeature,
    ThresholdCalibrator,
)
from ae_window_config import (
    WINDOW_SEC,
    STEP_SEC,
    MIN_EVENTS_PER_WINDOW,
    load_sliding_window_params,
)

# =============================================================================
# 全局配置
# =============================================================================

BATCH_CFG = {
    # 你的数据集根目录（改成你本地路径）
    "data_dir": "./dataset",

    # 子文件夹名
    "normal_dir":  "normal",
    "broken_dir":  "broken",
    "aircut_dir":  "aircut",
    "baseline_dir": "baseline",

    # 滑窗参数（默认 ae_window_config；main 读 gui_config.json）
    "window_sec": WINDOW_SEC,
    "step_sec": STEP_SEC,
    "min_events_per_window": MIN_EVENTS_PER_WINDOW,

    # 划分比例（按文件分，不按行）
    "valid_ratio": 0.1,    # 从正常+断刀里各拿10%做验证
    "test_ratio": 0.1,    # 各拿10%做测试

    # 输出
    "output_dir": "./output_batch",

    "seed": 42,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Batch-Pipeline")


# =============================================================================
# 模块1：目录扫描
# =============================================================================

def parse_diameter_from_filename(fname: str) -> float:
    """从文件名解析刀具直径（仅当 CSV 列无有效数据时作为兜底）。

    微径刀具命名（D 后数字常省略小数点）：
      带小数点 : D1.5 / 0.8断刀 / D0.6实切  -> 直接取
      紧凑无小数点: D06->0.6  D08->0.8  D15->1.5  D1/D2/D4/D6/D8 -> 对应mm

    注意：文件名里的设备/采集哈希（如 ...5D1B08CF...）也含 "D1"，
    不能用裸正则取第一个 D<num>。优先锚定标签词相邻的直径 token。
    """
    import re

    def decode(tok: str) -> float:
        try:
            if '.' in tok:
                return float(tok)
            # 紧凑写法：首位为整数部分，其余位为小数（D06->0.6, D15->1.5, D1->1.0）
            if len(tok) >= 2 and tok[0].isdigit():
                return float(f"{tok[0]}.{tok[1:]}")
            return float(tok)
        except Exception:
            return 0.0

    labels = r'(?:实切|断刀|RealCut|ToolBreak|DryRun|AirCut|空切)'
    # 1) 标签词后紧跟 _D<num>：RealCut_D15_ / ToolBreak_D8_ / DryRun_D6_
    m = re.search(labels + r'[_ ]+[Dd](\d+\.?\d*)', fname, re.IGNORECASE)
    if m:
        return decode(m.group(1))
    # 2) 数字(可带D)紧跟标签词：1.5断刀 / 0.8断刀 / D6实切 / param_2断刀
    m = re.search(r'[Dd]?(\d+\.?\d*)[_ ]*' + labels, fname, re.IGNORECASE)
    if m:
        return decode(m.group(1))
    # 注意：不再做"任意 D<num>"兜底——文件名里的设备哈希(如 ...5D1B08CF...)
    # 也含 D1，裸 D 匹配会把刀号 T18 这类文件名误解析成直径1。
    # 文件名里没有紧贴标签词的直径标记时，判定为"文件名无直径信息"，交回 CSV 列。
    return 0.0


def scan_dataset(root: str) -> dict:
    """扫描 root 下 normal/broken/aircut/baseline 文件夹"""
    result = {"正常": [], "断刀": [], "空切": [], "基线": []}
    for label, subdir in [("正常", BATCH_CFG["normal_dir"]),
                          ("断刀", BATCH_CFG["broken_dir"]),
                          ("空切", BATCH_CFG.get("aircut_dir", "aircut")),
                          ("基线", BATCH_CFG.get("baseline_dir", "baseline"))]:
        d = os.path.join(root, subdir)
        if not os.path.isdir(d):
            logger.warning(f"目录不存在: {d}")
            continue
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        result[label] = files
        logger.info(f"  {label}: {len(files)} 个文件")
    return result


# =============================================================================
# 模块2：数据加载（兼容 spindle_speed 负值）
# =============================================================================

def load_one_csv(path: str) -> pd.DataFrame:
    """加载单个CSV，构造统一时间轴，兼容负值转速和缺列"""
    fname = os.path.basename(path)
    # 只读需要的列，大文件省内存
    use_cols = ["amp", "power", "rms", "asl", "peak_freq", "centroid_freq",
                "spindle_load_current", "feed_rate",
                "load_current_x", "load_current_y", "load_current_z",
                "time_sec", "time_nsec", "spindle_speed", "tool_diameter", "t", "program_line"]
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in use_cols)
    except ValueError:
        # 列名不完全匹配，退回全读
        df = pd.read_csv(path, encoding="utf-8-sig")

    # 核心列必须有
    must_have = ["amp", "power"]
    missing_must = [c for c in must_have if c not in df.columns]
    if missing_must:
        raise ValueError(
            f"[{fname}] 缺少核心列 {missing_must}。"
            f"实际列名: {list(df.columns)}")

    # 可选列缺失补0，并打印 warning 指明文件
    optional_cols = ["rms", "asl", "peak_freq", "centroid_freq",
                     "spindle_load_current", "feed_rate",
                     "load_current_x", "load_current_y", "load_current_z",
                     "time_sec", "time_nsec", "spindle_speed", "tool_diameter"]
    missing_opt = [c for c in optional_cols if c not in df.columns]
    if missing_opt:
        logger.warning(f"[{fname}] 缺列 {missing_opt}，自动补0")
    for c in missing_opt:
        df[c] = 0.0

    # ---- 刀具直径：训练 / 推理逻辑对齐 ----
    # CSV 存在有效(>0 且有限) tool_diameter 列 → 使用该列，每个窗口取中位数；
    # 列内 0 / NaN 等无效值先用整文件有效值中位数填补，避免空窗 / NaN 进入模型。
    # 全文件无任何有效数据 → 才从文件名解析直径，整文件广播。
    if "tool_diameter" in df.columns:
        s = pd.to_numeric(df["tool_diameter"], errors="coerce")
        valid = s[(s.notna()) & (s > 0)]
        if len(valid) > 0:
            file_dia = float(valid.median())
            # 保留列原值（>0），无效行用文件中位数兜底；窗口中位数在 extractor 内计算
            df["tool_diameter"] = s.where(s > 0, file_dia)
        else:
            dia = parse_diameter_from_filename(fname)
            df["tool_diameter"] = dia if dia > 0 else 0.0
            if dia > 0:
                logger.info(f"[{fname}] tool_diameter 列无有效数据，从文件名解析: D{dia}")
    else:
        dia = parse_diameter_from_filename(fname)
        df["tool_diameter"] = dia if dia > 0 else 0.0
        if dia > 0:
            logger.info(f"[{fname}] 无 tool_diameter 列，从文件名解析: D{dia}")

    # 构造时间轴
    t0 = df["time_sec"].iloc[0]
    df["t"] = (df["time_sec"] - t0).astype(float) \
              + df["time_nsec"].astype(float) / 1e9

    # spindle_speed 取绝对值
    if "spindle_speed" in df.columns:
        df["spindle_speed"] = df["spindle_speed"].abs()

    return df.sort_values("t").reset_index(drop=True)


# =============================================================================
# 模块3：按文件划分 train/valid/test
# =============================================================================

def split_files(file_list: list, seed: int = 42):
    """按文件名打乱后分 train/valid/test"""
    random.seed(seed)
    files = list(file_list)
    random.shuffle(files)
    n = len(files)
    n_valid = max(1, int(n * BATCH_CFG["valid_ratio"])) if n >= 5 else 0
    n_test  = max(1, int(n * BATCH_CFG["test_ratio"]))  if n >= 5 else 0

    valid = files[:n_valid]
    test  = files[n_valid:n_valid+n_test]
    train = files[n_valid+n_test:]
    return train, valid, test


# =============================================================================
# 主流程
# =============================================================================

def main():
    random.seed(BATCH_CFG["seed"])
    np.random.seed(BATCH_CFG["seed"])
    os.makedirs(BATCH_CFG["output_dir"], exist_ok=True)

    # ---- Step 1: 扫描数据 ----
    logger.info("=" * 60)
    logger.info("Step 1: 扫描数据集")
    logger.info("=" * 60)
    root = BATCH_CFG["data_dir"]
    if not os.path.isdir(root):
        logger.error(f"数据集目录不存在: {root}")
        logger.error("请修改 BATCH_CFG['data_dir'] 指向你的数据根目录")
        sys.exit(1)

    files_by_label = scan_dataset(root)

    # ---- Step 2: 划分 train/valid/test ----
    logger.info("=" * 60)
    logger.info("Step 2: 按文件划分 train/valid/test")
    logger.info("=" * 60)
    splits = {"train": {"正常": [], "断刀": []},
              "valid": {"正常": [], "断刀": []},
              "test":  {"正常": [], "断刀": []}}

    for label, flist in files_by_label.items():
        tr, va, te = split_files(flist, BATCH_CFG["seed"])
        splits["train"][label] = tr
        splits["valid"][label] = va
        splits["test"][label]  = te
        logger.info(f"  {label}: train={len(tr)}, valid={len(va)}, test={len(te)}")

    # ---- Step 3: 用 train 里的正常样本标定阈值 ----
    logger.info("=" * 60)
    logger.info("Step 3: 自动标定阈值（用train里的正常样本）")
    logger.info("=" * 60)
    wp = load_sliding_window_params()
    extractor = FeatureExtractor(
        wp["window_sec"], wp["step_sec"], wp["min_events_per_window"])

    calibrator = ThresholdCalibrator(CONFIG["calibration"])
    normal_feats_for_calib = []
    for path in splits["train"]["正常"]:
        df = load_one_csv(path)
        normal_feats_for_calib.extend(extractor.extract(df))

    calibrator.calibrate(normal_feats_for_calib)
    calibrator.save(os.path.join(BATCH_CFG["output_dir"],
                                 "threshold_calibration.json"))

    # ---- Step 4: 跑所有文件，提取特征 + 规则打标签 ----
    logger.info("=" * 60)
    logger.info("Step 4: 提取特征 + 规则打标签")
    logger.info("=" * 60)
    rule = RuleDiagnoser(calibrator)
    pb = PromptBuilder()

    all_feats = {"train": [], "valid": [], "test": []}
    file_stat = []

    for split_name in ["train", "valid", "test"]:
        for label in ["正常", "断刀"]:
            for path in splits[split_name][label]:
                try:
                    df = load_one_csv(path)
                    feats = extractor.extract(df)
                    n_evt = len(df)
                    dur = df["t"].iloc[-1] - df["t"].iloc[0] if len(df) > 1 else 0

                    n_break = n_warn = n_normal = 0

                    # 先算跨窗趋势：power_drop_ratio
                    for i, f in enumerate(feats):
                        if i == 0:
                            f.power_drop_ratio = 1.0
                        else:
                            prev = feats[i-1].power_mean
                            if prev > 1e-6:
                                f.power_drop_ratio = f.power_mean / prev
                            else:
                                f.power_drop_ratio = 1.0

                    for f in feats:
                        f = rule.diagnose(f)
                        f.source_file = os.path.basename(path)  # type: ignore
                        f.gt_label = label                       # type: ignore
                        all_feats[split_name].append(f)
                        if f.status == "断刀":  n_break += 1
                        elif f.status == "预警": n_warn += 1
                        else: n_normal += 1

                        # 文件级熔断：连续断刀窗 + 断刀后功率骤降
                        #  - max_run >= 10：大量连续异常，直接断刀
                        #  - max_run >= 5 且 末段/前段power < 0.5：连续异常+空切
                        #  - n_break >= 30：断刀窗总数多
                        #  - 连续预警窗 >=3 且 drop_ratio<0.7 => 预警（有下降趋势才关注）
                        max_run = run = 0
                        max_warn_run = warn_run = 0
                        for f in feats:
                            if f.status == "断刀":
                                run += 1; max_run = max(max_run, run)
                                warn_run = 0
                            elif f.status == "预警":
                                warn_run += 1
                                max_warn_run = max(max_warn_run, warn_run)
                                run = 0
                            else:
                                run = 0; warn_run = 0
                        break_ratio = n_break / max(len(feats), 1)
                        warn_ratio = n_warn / max(len(feats), 1)

                        # 前后段power比值（断刀后空切特征）
                        third = max(1, len(feats) // 3)
                        head_pm = np.mean([f.power_mean for f in feats[:third]])
                        tail_pm = np.mean([f.power_mean for f in feats[-third:]])
                        drop_ratio = tail_pm / head_pm if head_pm > 1e-6 else 1.0

                        if max_run >= 8 or n_break >= 25 or \
                           (max_run >= 5 and drop_ratio < 0.6):
                            conclusion = "断刀"
                        elif max_warn_run >= 3 and drop_ratio < 0.5:
                            conclusion = "预警"
                        else:
                            conclusion = "正常"

                    file_stat.append({
                        "file": os.path.basename(path),
                        "split": split_name,
                        "gt": label,
                        "events": n_evt,
                        "duration_s": round(dur, 1),
                        "windows": len(feats),
                        "rule_break": n_break,
                        "rule_warn":  n_warn,
                        "rule_normal": n_normal,
                        "break_ratio": round(break_ratio, 3),
                        "warn_ratio": round(warn_ratio, 3),
                        "max_consec_break": max_run,
                        "head_power": round(head_pm, 1),
                        "tail_power": round(tail_pm, 1),
                        "power_drop_ratio": round(drop_ratio, 3),
                        "file_conclusion": conclusion,
                    })
                except Exception as e:
                    logger.error(f"  处理失败 {path}: {e}")
                    traceback.print_exc()

    # ---- Step 5: 生成 SFT 数据 ----
    logger.info("=" * 60)
    logger.info("Step 5: 生成 SFT 训练数据")
    logger.info("=" * 60)

    def build_sft_example(f: WindowFeature) -> dict:
        user_msg = pb.build_user_prompt(f)
        asst_msg = json.dumps({
            "status": f.status,
            "confidence": round(f.confidence, 2),
            "reasons": f.reasons,
            "advice": f.advice,
        }, ensure_ascii=False)
        return {
            "messages": [
                {"role": "system", "content": pb.SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        }

    # 类别平衡：正常下采样到和断刀差不多
    for split_name in ["train", "valid", "test"]:
        feats = all_feats[split_name]
        by_status = defaultdict(list)
        for f in feats:
            by_status[f.status].append(f)

        n_break = len(by_status["断刀"])
        n_normal_target = int(n_break * 1.2) if n_break > 0 else len(by_status["正常"])
        if len(by_status["正常"]) > n_normal_target:
            random.shuffle(by_status["正常"])
            by_status["正常"] = by_status["正常"][:n_normal_target]

        balanced = by_status["正常"] + by_status["预警"] + by_status["断刀"]
        random.shuffle(balanced)

        out_path = os.path.join(BATCH_CFG["output_dir"],
                                f"sft_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fp:
            for f in balanced:
                fp.write(json.dumps(build_sft_example(f),
                                    ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(balanced)} 条 -> {out_path}")

    # ---- Step 6: 统计报告 ----
    logger.info("=" * 60)
    logger.info("Step 6: 输出统计")
    logger.info("=" * 60)

    stat = {
        "files": file_stat,
        "split_files": {
            s: {l: [os.path.basename(p) for p in splits[s][l]]
                for l in ["正常", "断刀"]}
            for s in ["train", "valid", "test"]
        },
    }
    with open(os.path.join(BATCH_CFG["output_dir"],
                           "batch_stat.json"), "w",
              encoding="utf-8") as fp:
        json.dump(stat, fp, ensure_ascii=False, indent=2)

    # 文件明细表
    pd.DataFrame(file_stat).to_csv(
        os.path.join(BATCH_CFG["output_dir"], "file_detail.csv"),
        index=False, encoding="utf-8-sig")

    # 打印汇总
    print("\n" + "=" * 60)
    print("文件处理汇总（规则诊断 vs 真实标签）")
    print("=" * 60)
    df_stat = pd.DataFrame(file_stat)
    print(df_stat.to_string(index=False))

    # 混淆矩阵
    print("\n=== 滑窗级混淆矩阵（规则诊断 vs 真实标签）===")
    for split_name in ["train", "valid", "test"]:
        feats = all_feats[split_name]
        tp = sum(1 for f in feats
                 if f.gt_label == "断刀" and f.status in ("断刀", "预警"))
        fn = sum(1 for f in feats
                 if f.gt_label == "断刀" and f.status == "正常")
        tn = sum(1 for f in feats
                 if f.gt_label == "正常" and f.status == "正常")
        fp = sum(1 for f in feats
                 if f.gt_label == "正常" and f.status in ("断刀", "预警"))
        total = len(feats)
        acc = (tp + tn) / total if total else 0
        print(f"  {split_name}: TP={tp} FN={fn} TN={tn} FP={fp} "
              f"准确率={acc:.3f}")

    # 文件级熔断混淆矩阵（实际工程部署看这个）
    print("\n=== 文件级熔断混淆矩阵（任意窗断刀 => 文件断刀）===")
    df_stat = pd.DataFrame(file_stat)
    for split_name in ["train", "valid", "test"]:
        sub = df_stat[df_stat["split"] == split_name]
        # 把"预警"也算检出
        tp = len(sub[(sub["gt"] == "断刀") &
                     (sub["file_conclusion"].isin(["断刀", "预警"]))])
        fn = len(sub[(sub["gt"] == "断刀") &
                     (sub["file_conclusion"] == "正常")])
        tn = len(sub[(sub["gt"] == "正常") &
                     (sub["file_conclusion"] == "正常")])
        fp = len(sub[(sub["gt"] == "正常") &
                     (sub["file_conclusion"].isin(["断刀", "预警"]))])
        total = len(sub)
        acc = (tp + tn) / total if total else 0
        # 召回率 = 断刀文件被检出的比例
        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        print(f"  {split_name}: TP={tp} FN={fn} TN={tn} FP={fp} "
              f"准确率={acc:.3f} 召回率={recall:.3f} 精确率={precision:.3f}")

    logger.info(f"\n全部完成，输出目录: {BATCH_CFG['output_dir']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"主流程异常: {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# 模块：按刀具直径的正常加工基线（相对比值特征）
# 不同直径正常切削的功率/电流量级差可达百倍，全局标准化会误报大直径正常件。
# 这里按直径 bin 学正常基线，检测时用"当前值/该直径正常基线"做相对量。
# =============================================================================

# 需要做相对化的绝对量级特征
DIAL_BASE_FEATS = ["power_mean", "amp_mean",
                   "spindle_current_mean", "spindle_current_max"]
# 新增的比值特征列名（追加到 FEATURE_COLS 末尾）
DIAL_RATIO_COLS = [c + "_dia_ratio" for c in DIAL_BASE_FEATS]


def build_dia_baseline(normal_feats):
    """从正常样本窗按刀具直径(1位小数bin)学各特征的中位数基线。
    返回 {dia: {feat: median}, "_global": {feat: median}} 供未知直径兜底。"""
    by_dia = {}
    global_vals = {c: [] for c in DIAL_BASE_FEATS}
    for ft in normal_feats:
        d = round(float(getattr(ft, "tool_diameter", 0) or 0), 1)
        by_dia.setdefault(d, {c: [] for c in DIAL_BASE_FEATS})
        for c in DIAL_BASE_FEATS:
            v = float(getattr(ft, c, 0.0) or 0.0)
            by_dia[d][c].append(v)
            global_vals[c].append(v)
    baseline = {}
    for d, dd in by_dia.items():
        baseline[d] = {c: (float(np.median(v)) if v else 1.0)
                       for c, v in dd.items()}
    baseline["_global"] = {c: (float(np.median(v)) if v else 1.0)
                          for c, v in global_vals.items()}
    return baseline


def apply_dia_baseline(feats, baseline):
    """给每个 WindowFeature 写入 *_dia_ratio 比值特征。
    断刀时功率/电流骤降 -> 比值趋近0；正常时比值≈1，跨直径可比。"""
    if not baseline:
        return
    g = baseline.get("_global", {})
    for ft in feats:
        d = round(float(getattr(ft, "tool_diameter", 0) or 0), 1)
        b = baseline.get(d, g)
        for c in DIAL_BASE_FEATS:
            base = float(b.get(c, 0) or 0)
            val = float(getattr(ft, c, 0.0) or 0.0)
            setattr(ft, c + "_dia_ratio",
                    val / base if base > 1e-6 else 1.0)


# 按直径空切/实切上下限的相对特征（阈值表 dia_thresholds.json）
DIA_THR_RATIO_COLS = ["power_over_normal_p95", "rms_over_normal_p95",
                      "power_over_aircut_p90"]
_DIA_THR_CACHE = None


def _load_dia_thr():
    global _DIA_THR_CACHE
    if _DIA_THR_CACHE is None:
        p = os.path.join(os.path.dirname(__file__), "dia_thresholds.json")
        try:
            import json
            _DIA_THR_CACHE = json.load(open(p, encoding="utf-8"))
        except Exception:
            _DIA_THR_CACHE = {}
    return _DIA_THR_CACHE


def apply_dia_thr_ratios(feats):
    """按每个窗口直径，用 normal/aircut 上下限算相对特征：
    power_over_normal_p95: 切削功率/该直径实切p95
    rms_over_normal_p95:   rms/该直径实切p95（断刀冲击>>1）
    power_over_aircut_p90: 功率/该直径空切p90（断刀后≈1或更低）"""
    table = _load_dia_thr()
    if not table:
        return
    keys = list(table.values())
    g_p95p = float(np.mean([d["normal_p95_power"] for d in keys]))
    g_p95r = float(np.mean([d["normal_p95_rms"] for d in keys]))
    g_ac = float(np.mean([d["aircut_p90_power"] for d in keys]))
    for ft in feats:
        d = round(float(getattr(ft, "tool_diameter", 0) or 0), 1)
        if d > 0:
            near = min(table.keys(), key=lambda k: abs(float(k) - d))
            b = table[near]
        else:
            b = {"normal_p95_power": g_p95p, "normal_p95_rms": g_p95r,
                 "aircut_p90_power": g_ac}
        p95p = float(b["normal_p95_power"]) or 1.0
        p95r = float(b["normal_p95_rms"]) or 1.0
        ac = float(b["aircut_p90_power"]) or 1.0
        pm = float(getattr(ft, "power_mean", 0.0) or 0.0)
        rm = float(getattr(ft, "rms_max", 0.0) or 0.0)
        setattr(ft, "power_over_normal_p95", pm / p95p)
        setattr(ft, "rms_over_normal_p95", rm / p95r)
        setattr(ft, "power_over_aircut_p90", pm / ac)
