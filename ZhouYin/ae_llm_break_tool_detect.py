# -*- coding: utf-8 -*-
"""
=============================================================================
文件名称 : ae_llm_break_tool_detect.py
功能描述 : 基于大语言模型"分支1：特征文本化路线"的微径刀具断刀检测
           方案实现（含 OpenAI/Qwen API 调用 + 可视化）。

流程：
    原始AE事件流 -> 滑窗特征提取 -> 结构化文本Prompt ->
    LLM诊断推理（OpenAI兼容API / 内置规则双模式）->
    诊断报告 + amp/power 时间曲线可视化。

LLM接入：
    兼容 OpenAI Chat Completions 接口，可对接：
      - 阿里云百炼 DashScope (qwen-plus / qwen-max)
      - DeepSeek / Moonshot / 本地 vLLM / Ollama
    通过环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 配置。

作者    : PHM Lab
创建日期: 2026-09-09
Python  : 3.10+
依赖    : pandas, numpy, matplotlib, openai
=============================================================================
"""

import os
import sys
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端，服务器也能跑
import matplotlib.pyplot as plt

from ae_window_config import (
    WINDOW_SEC,
    STEP_SEC,
    MIN_EVENTS_PER_WINDOW,
    load_sliding_window_params,
)

# =============================================================================
# 全局配置区
# =============================================================================

CONFIG = {
    # ---- 输入数据 ----
    "data_dir": "/home/user/.doubao/agent_mode/workspace/.sessions/"
                "38440756026883586/attachments",
    "files": {
        "D1.0_正常": "D1.0实切_Ae_Batch_20260827133115_5_param.csv",
        "D1.0_断刀": "D1.0断刀_Ae_Batch_20260827155235_6_param.csv",
        "D4_正常":   "D4实切_Ae_Batch_20260827131318_2_param.csv",
        "D4_断刀":   "D4断刀_Ae_Batch_20260827153039_3_param.csv",
    },

    # ---- 滑窗参数（默认见 ae_window_config；运行时 main 读 gui_config.json）----
    "window_sec": WINDOW_SEC,
    "step_sec": STEP_SEC,
    "min_events_per_window": MIN_EVENTS_PER_WINDOW,

    # ---- 机理阈值（仅作为兜底/物理硬限；实际由 ThresholdCalibrator 自动标定）----
    "threshold_fallback": {
        "power_max_warn":   200.0,
        "power_max_break":  1000.0,
        "amp_max_warn":     85.0,
        "amp_max_break":    95.0,
        "rms_max_warn":     1.0,
        "high_energy_ratio_warn": 0.02,
        "centroid_shift_warn": 50.0,
    },

    # ---- 自动阈值标定（3σ 原则）----
    # 用哪些文件作为"正常样本"来统计基线分布
    # 注意：窗内max本身就是极值统计量，天然接近 mean+2σ，
    #       所以预警线用3σ（正常样本不会触发），断刀线用6σ（强异常）
    "calibration": {
        "normal_files": ["D1.0_正常", "D4_正常"],
        "warn_sigma":   3.0,    # 预警线 = mean + 3σ
        "break_sigma":  6.0,    # 断刀线 = mean + 6σ
        # 物理硬限兜底（防止3σ算出来离谱）
        "hard_floor": {
            "power_max":   20.0,
            "amp_max":     65.0,
            "rms_max":     0.5,
        },
    },

    # ---- LLM API 配置（OpenAI兼容）----
    # 推荐用环境变量，不要把 key 硬编码进代码：
    #   export LLM_API_KEY="sk-xxxx"
    #   export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    #   export LLM_MODEL="qwen-plus"
    # 不设环境变量时自动回退到 RuleDiagnoser
    "llm": {
        "api_key":  os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model":    os.environ.get("LLM_MODEL", "qwen-plus"),
        "temperature": 0.1,
        "max_tokens": 512,
        "timeout": 30,
    },

    # ---- 输出 ----
    "output_dir": "/home/user/Doubao/chats/38440756026883586/bearing_llm/output",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AE-BreakTool")


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class WindowFeature:
    t_start: float = 0.0
    t_end:   float = 0.0
    n_events: int = 0
    tool_diameter: float = 0.0
    spindle_speed: float = 0.0
    feed_rate:     float = 0.0
    spindle_load:  float = 0.0
    amp_mean: float = 0.0
    amp_std:  float = 0.0
    amp_p99:  float = 0.0
    amp_max:  float = 0.0
    power_mean: float = 0.0
    power_std:  float = 0.0
    power_p99:  float = 0.0
    power_max:  float = 0.0
    rms_max: float = 0.0
    asl_mean: float = 0.0
    centroid_mean: float = 0.0
    high_energy_ratio: float = 0.0
    # ---- 新增时域冲击特征 ----
    kurtosis: float = 0.0       # 峭度（>3为冲击）
    crest_factor: float = 0.0   # 峰值因子 = amp_max / rms
    skewness: float = 0.0       # 偏度
    impulse_factor: float = 0.0 # 脉冲因子 = amp_max / mean(amp)
    # ---- 7维突变形态特征 ----
    peak_amp_ratio: float = 0.0     # 峰值倍率 = amp_max / amp_mean
    max_rise_slope: float = 0.0     # 最大上升斜率
    pulse_half_width: float = 0.0    # 脉冲半高宽（秒）
    rise_time: float = 0.0           # 上升时间（秒）
    sharpness: float = 0.0           # 波形尖锐度
    fall_slope: float = 0.0          # 回落斜率
    baseline_diff: float = 0.0       # 峰前后基线落差
    # ---- 趋势特征（断刀后骤降）----
    power_drop_ratio: float = 1.0  # 当前power_mean / 前窗power_mean，<0.3骤降
    # ---- 电流特征（多传感器融合）----
    spindle_current_mean: float = 0.0
    spindle_current_std:  float = 0.0
    spindle_current_max: float = 0.0
    x_current_mean: float = 0.0
    x_current_std:  float = 0.0
    x_current_max:  float = 0.0
    y_current_mean: float = 0.0
    y_current_std:  float = 0.0
    y_current_max:  float = 0.0
    z_current_mean: float = 0.0
    z_current_std:  float = 0.0
    z_current_max:  float = 0.0
    status:      str = "正常"
    confidence:  float = 0.0
    reasons: List[str] = field(default_factory=list)
    advice:      str = ""
    raw_llm:     str = ""   # LLM原始返回


# =============================================================================
# 模块1：数据加载
# =============================================================================

class AEDataLoader:
    REQUIRED_COLS = [
        "time_sec", "time_nsec", "amp", "power", "rms", "asl",
        "peak_freq", "centroid_freq",
        "spindle_speed", "feed_rate", "spindle_load_current",
    ]

    def load(self, path: str) -> pd.DataFrame:
        # 统一走 ae_batch_pipeline.load_one_csv：与训练/推理完全一致的
        # 缺列补0、时间轴、spindle_speed取绝对值、tool_diameter(列优先,每窗中位数,
        # 无效再回退文件名解析)。函数内延迟 import，避免与 ae_batch_pipeline 循环依赖。
        from ae_batch_pipeline import load_one_csv
        return load_one_csv(path)


# =============================================================================
# 模块2：滑窗特征提取
# =============================================================================

class FeatureExtractor:
    def __init__(self, window_sec, step_sec, min_events):
        self.w = window_sec
        self.s = step_sec
        self.min_evt = min_events

    def extract(self, df: pd.DataFrame) -> List[WindowFeature]:
        feats = []
        if len(df) == 0:
            return feats
        t_end_total = df["t"].iloc[-1]
        t = 0.0
        while t <= t_end_total:
            w = df[(df["t"] >= t) & (df["t"] < t + self.w)]
            if len(w) >= self.min_evt:
                feats.append(self._extract_one(w, t))
            t += self.s
        logger.info(f"  滑窗特征 {len(feats)} 个")
        return feats

    def _extract_one(self, w: pd.DataFrame, t_start: float) -> WindowFeature:
        f = WindowFeature(t_start=t_start, t_end=t_start + self.w,
                           n_events=len(w))

        # 列名容错：缺失列用0填充，避免KeyError
        def g(col, default=0.0):
            if col in w.columns:
                return float(w[col].median())
            return default

        # 工艺参数
        f.tool_diameter = g("tool_diameter")
        f.spindle_speed = g("spindle_speed")
        f.feed_rate     = g("feed_rate")
        f.spindle_load  = g("spindle_load_current")

        # 信号统计列（缺失列用0填充）
        def gm(col, stat="max"):
            if col not in w.columns:
                return 0.0
            s = w[col]
            if stat == "mean":   return float(s.mean())
            if stat == "std":    return float(s.std())
            if stat == "p99":    return float(s.quantile(0.99))
            if stat == "max":    return float(s.max())
            return 0.0

        f.amp_mean = gm("amp", "mean")
        f.amp_std  = gm("amp", "std")
        f.amp_p99  = gm("amp", "p99")
        f.amp_max  = gm("amp", "max")

        f.power_mean = gm("power", "mean")
        f.power_std  = gm("power", "std")
        f.power_p99  = gm("power", "p99")
        f.power_max  = gm("power", "max")

        f.rms_max  = gm("rms", "max")
        f.asl_mean = gm("asl", "mean")
        f.centroid_mean = gm("centroid_freq", "mean")

        # 高能量事件占比（power>100）
        if "power" in w.columns:
            f.high_energy_ratio = float((w["power"] > 100).mean())
        else:
            f.high_energy_ratio = 0.0

        # ---- 时域冲击特征 ----
        if "amp" in w.columns and len(w) > 10:
            amp_arr = w["amp"].values.astype(np.float64)
            amp_mu = amp_arr.mean()
            amp_sd = amp_arr.std()
            if amp_sd > 1e-9:
                # 峭度（超额）：高斯=0，冲击信号>>0
                f.kurtosis = float(
                    np.mean(((amp_arr - amp_mu)/amp_sd)**4) - 3)
                # 偏度
                f.skewness = float(
                    np.mean(((amp_arr - amp_mu)/amp_sd)**3))
            # 峰值因子 = max / rms
            amp_rms = float(np.sqrt(np.mean(amp_arr**2)))
            if amp_rms > 1e-9:
                f.crest_factor = float(amp_arr.max() / amp_rms)
            # 脉冲因子 = max / mean
            if amp_mu > 1e-9:
                f.impulse_factor = float(amp_arr.max() / amp_mu)

            # ---- 7维突变形态特征（基于amp时序）----
            if "t" in w.columns and "amp" in w.columns and len(w) > 10:
                t_arr = w["t"].values.astype(np.float64)
                a_arr = w["amp"].values.astype(np.float64)
                base = float(np.median(a_arr))
                peak_idx = int(np.argmax(a_arr))
                peak_val = a_arr[peak_idx]

                # 1. 峰值倍率
                if base > 1e-9:
                    f.peak_amp_ratio = float(peak_val / base)

                # 2. 最大上升斜率（peak前后）
                if peak_idx > 0 and peak_idx < len(a_arr)-1:
                    dt_rise = t_arr[peak_idx] - t_arr[peak_idx-1]
                    if dt_rise > 1e-9:
                        f.max_rise_slope = float(
                            (a_arr[peak_idx] - a_arr[peak_idx-1]) / dt_rise)

                # 3. 脉冲半高宽（peak两侧下降到一半的时间差）
                half = base + (peak_val - base) / 2
                left_idx = peak_idx
                for k in range(peak_idx, -1, -1):
                    if a_arr[k] < half:
                        left_idx = k
                        break
                right_idx = peak_idx
                for k in range(peak_idx, len(a_arr)):
                    if a_arr[k] < half:
                        right_idx = k
                        break
                if right_idx > left_idx:
                    f.pulse_half_width = float(
                        t_arr[right_idx] - t_arr[left_idx])

                # 4. 上升时间（从10%到90%峰值）
                lo = base + 0.1*(peak_val-base)
                hi = base + 0.9*(peak_val-base)
                t_lo = t_hi = peak_idx
                for k in range(peak_idx, -1, -1):
                    if a_arr[k] < lo:
                        t_lo = k; break
                for k in range(peak_idx, -1, -1):
                    if a_arr[k] < hi:
                        t_hi = k; break
                if t_arr[peak_idx] > t_arr[t_lo]:
                    f.rise_time = float(t_arr[peak_idx] - t_arr[t_lo])

                # 5. 波形尖锐度 = peak / 半高宽
                if f.pulse_half_width > 1e-6:
                    f.sharpness = float(peak_val / f.pulse_half_width)

                # 6. 回落斜率
                if peak_idx < len(a_arr)-1:
                    dt_fall = t_arr[min(peak_idx+2, len(a_arr)-1)] - t_arr[peak_idx]
                    if dt_fall > 1e-9:
                        f.fall_slope = float(
                            (peak_val - a_arr[min(peak_idx+2, len(a_arr)-1)]) / dt_fall)

                # 7. 峰前后基线落差
                pre_base = float(np.median(a_arr[:max(peak_idx,1)]))
                post_base = float(np.median(a_arr[min(peak_idx+1, len(a_arr)-1):]))
                f.baseline_diff = float(pre_base - post_base)

        # ---- 电流特征（多传感器融合）----
        def _cur_stats(col, mean_attr, std_attr, max_attr):
            if col in w.columns:
                s = w[col].astype(float)
                setattr(f, mean_attr, float(s.mean()))
                setattr(f, std_attr, float(s.std()))
                setattr(f, max_attr, float(s.abs().max()))
        _cur_stats("spindle_load_current", "spindle_current_mean",
                   "spindle_current_std", "spindle_current_max")
        _cur_stats("load_current_x", "x_current_mean",
                   "x_current_std", "x_current_max")
        _cur_stats("load_current_y", "y_current_mean",
                   "y_current_std", "y_current_max")
        _cur_stats("load_current_z", "z_current_mean",
                   "z_current_std", "z_current_max")

        return f


# =============================================================================
# 模块3：特征文本化
# =============================================================================

class PromptBuilder:
    SYSTEM_PROMPT = """你是一名资深数控刀具磨损与断刀诊断专家，熟悉声发射(AE)信号机理。
你的任务是根据给定的AE观测特征，按链式推理判断刀具状态。
严格按以下JSON格式输出，不要输出多余文字：
{"status": "正常/预警/断刀", "confidence": 0.0~1.0,
 "reasons": ["依据1","依据2"], "advice": "继续加工/降速观察/立即停机检查"}
"""

    KNOWLEDGE = """
【机理参考】
- 正常切削: power 10~35, amp 55~68dB, centroid ~320Hz
- 断刀冲击: power 飙升至正常30~数千倍, amp 接近100dB满量程,
  centroid下移, 高能量事件连续出现
- 干扰排除: 切屑撞击多为孤立尖峰, 颤振为周期性谐波无强冲击
"""

    def build_user_prompt(self, f: WindowFeature) -> str:
        return f"""观测时间窗: {f.t_start:.1f}~{f.t_end:.1f}s
工艺: 主轴{f.spindle_speed:.0f}rpm, 进给{f.feed_rate:.0f}, 负载{f.spindle_load:.2f}A
amp(dB): mean={f.amp_mean:.1f}, std={f.amp_std:.1f},
         P99={f.amp_p99:.1f}, max={f.amp_max:.1f}
power: mean={f.power_mean:.1f}, P99={f.power_p99:.1f},
       max={f.power_max:.1f}
rms_max={f.rms_max:.2f}, asl_mean={f.asl_mean:.1f}
centroid={f.centroid_mean:.0f}Hz, 高能量事件占比={f.high_energy_ratio*100:.1f}%
事件数={f.n_events}
时域特征: 峭度={f.kurtosis:.1f}(高斯=0,>5脉冲冲击),
         峰值因子={f.crest_factor:.1f}, 脉冲因子={f.impulse_factor:.1f}
趋势: power骤降比={f.power_drop_ratio:.2f}(<0.3表示切削能量骤降到空载)
{self.KNOWLEDGE}"""


# =============================================================================
# 模块3.5：阈值自动标定（3σ 原则）
# =============================================================================

class ThresholdCalibrator:
    """
    从正常样本统计各指标的 mean ± std，按刀具直径分组标定。
    不同直径刀具切削能量差几个数量级，必须分组。
    预警线 = mean + warn_sigma·std
    断刀线 = mean + break_sigma·std
    """

    TARGETS = ["amp_max", "power_max", "rms_max", "high_energy_ratio"]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fallback = CONFIG["threshold_fallback"]
        self.stats_by_dia: dict = {}     # {diameter: stats}
        self.th_by_dia: dict = {}        # {diameter: threshold_dict}
        self.global_stats: dict = {}
        self.global_threshold: dict = {}

    def _dia_key(self, dia: float) -> float:
        """刀具直径分桶key，保留1位小数"""
        return round(float(dia), 1)

    def calibrate(self, feats: List[WindowFeature]) -> dict:
        """按刀具直径分组标定，同时算全局默认阈值"""
        if not feats:
            logger.warning("正常样本为空，使用 fallback 阈值")
            self.global_threshold = dict(self.fallback)
            return self.global_threshold

        # 按直径分组
        groups = {}
        for f in feats:
            k = self._dia_key(f.tool_diameter)
            groups.setdefault(k, []).append(f)

        logger.info(f"按刀具直径分组标定，共 {len(groups)} 组: "
                    f"{sorted(groups.keys())}")

        for dia, group_feats in sorted(groups.items()):
            th, stats = self._calibrate_one_group(group_feats)
            self.th_by_dia[dia] = th
            self.stats_by_dia[dia] = stats

        # 全局默认（所有正常样本混在一起，作为未知直径的fallback）
        self.global_threshold, self.global_stats = \
            self._calibrate_one_group(feats)

        self._print_report()
        return self.global_threshold

    def _calibrate_one_group(self, feats: List[WindowFeature]):
        """对一组正常样本（同直径）算阈值
        用分位数而非 mean+nσ：大直径正常切削本身就有强冲击，
        分位数对偏态分布更鲁棒。
        预警线 = P95
        断刀线 = max(P99 * 1.3, 正常max * 1.05)
        """
        stats = {}
        th = {}
        n = len(feats)

        for metric in self.TARGETS:
            vals = np.array([getattr(f, metric) for f in feats],
                            dtype=np.float64)
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            p95 = float(np.percentile(vals, 95))
            p99 = float(np.percentile(vals, 99))
            vmax = float(np.max(vals))
            stats[metric] = {
                "n": n, "mean": mu, "std": sd,
                "min": float(np.min(vals)), "max": vmax,
                "p95": p95, "p99": p99,
            }

            # 分位数法
            warn_t = p95
            break_t = max(p99 * 1.3, vmax * 1.05)

            # 硬限兜底
            floor = self.cfg["hard_floor"].get(metric, 0.0)
            th[f"{metric}_warn"]  = round(max(warn_t,  floor), 4)
            th[f"{metric}_break"] = round(max(break_t, floor), 4)

        # centroid
        c_vals = np.array([f.centroid_mean for f in feats], dtype=np.float64)
        c_mu, c_sd = float(np.mean(c_vals)), float(np.std(c_vals, ddof=1))
        th["centroid_normal_low"] = c_mu - 3 * c_sd
        th["centroid_shift_warn"] = 3 * c_sd
        stats["centroid_mean"] = {
            "mean": c_mu, "std": c_sd, "normal_low": c_mu - 3 * c_sd}

        return th, stats

    def get_threshold(self, dia: float) -> dict:
        """根据刀具直径取对应阈值；没标定过用全局默认"""
        k = self._dia_key(dia)
        if k in self.th_by_dia:
            return self.th_by_dia[k]
        return self.global_threshold

    def get_baseline_power(self, dia: float) -> float:
        k = self._dia_key(dia)
        if k in self.stats_by_dia:
            return self.stats_by_dia[k]["power_max"]["mean"]
        return self.global_stats.get("power_max", {}).get("mean", 35.0)

    def _print_report(self):
        logger.info("=" * 70)
        logger.info(f"{'刀具直径':>10s} | {'power均值':>10s} | "
                    f"{'power断刀线':>12s} | {'amp断刀线':>10s} | 样本数")
        logger.info("-" * 70)
        for dia, th in sorted(self.th_by_dia.items()):
            s = self.stats_by_dia[dia]
            logger.info(
                f"{'D'+str(dia):>10s} | "
                f"{s['power_max']['mean']:>10.1f} | "
                f"{th['power_max_break']:>12.1f} | "
                f"{th['amp_max_break']:>10.1f} | "
                f"{s['power_max']['n']:>5d}")
        logger.info("=" * 70)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump({
                "by_diameter": {
                    str(k): {"stats": self.stats_by_dia[k],
                             "threshold": self.th_by_dia[k]}
                    for k in self.th_by_dia
                },
                "global": {"stats": self.global_stats,
                           "threshold": self.global_threshold},
            }, fp, ensure_ascii=False, indent=2)
        logger.info(f"标定结果已保存: {path}")


# =============================================================================
# 模块4a：规则诊断器（LLM不可用时的fallback）
# =============================================================================

class RuleDiagnoser:
    """基于自动标定阈值的规则诊断器（评分制），按刀具直径选阈值"""

    def __init__(self, calibrator: ThresholdCalibrator):
        self.cal = calibrator

    def diagnose(self, f: WindowFeature) -> WindowFeature:
        th = self.cal.get_threshold(f.tool_diameter)
        base_power = self.cal.get_baseline_power(f.tool_diameter)
        reasons = []
        score = 0.0

        # power_max：双阈值
        if f.power_max >= th["power_max_break"]:
            score += 1.0
            reasons.append(
                f"power_max={f.power_max:.0f}≥断刀线"
                f"{th['power_max_break']:.0f}，"
                f"为D{f.tool_diameter:.1f}正常基线{base_power:.0f}的"
                f"{f.power_max/max(base_power,1):.0f}倍")
        elif f.power_max >= th["power_max_warn"]:
            score += 0.4
            reasons.append(
                f"power_max={f.power_max:.0f}≥预警线"
                f"{th['power_max_warn']:.0f}")

        # amp_max：双阈值
        if f.amp_max >= th["amp_max_break"]:
            score += 0.6
            reasons.append(
                f"amp_max={f.amp_max:.1f}dB≥断刀线"
                f"{th['amp_max_break']:.1f}dB")
        elif f.amp_max >= th["amp_max_warn"]:
            score += 0.2
            reasons.append(
                f"amp_max={f.amp_max:.1f}dB≥预警线"
                f"{th['amp_max_warn']:.1f}dB")

        # rms_max
        if f.rms_max >= th.get("rms_max_break", th["rms_max_warn"]*1.5):
            score += 0.5
            reasons.append(f"rms_max={f.rms_max:.2f}≥断刀线")
        elif f.rms_max >= th["rms_max_warn"]:
            score += 0.3
            reasons.append(
                f"rms_max={f.rms_max:.2f}≥预警线"
                f"{th['rms_max_warn']:.2f}")

        # 高能量事件占比
        if f.high_energy_ratio >= th["high_energy_ratio_warn"]:
            score += 0.3
            reasons.append(
                f"高能量事件占比{f.high_energy_ratio*100:.1f}%"
                f"≥{th['high_energy_ratio_warn']*100:.1f}%")

        # centroid 低频下移
        c_low = th.get("centroid_normal_low", 300.0)
        if f.centroid_mean < c_low:
            score += 0.15
            reasons.append(
                f"centroid={f.centroid_mean:.0f}Hz<正常下限{c_low:.0f}Hz")

        # ---- 新增：时域冲击特征 ----
        # 峭度>5 说明有脉冲冲击（高斯分布超额峭度=0）
        if f.kurtosis > 5.0:
            score += 0.3
            reasons.append(f"峭度={f.kurtosis:.1f}>5，脉冲冲击显著")
        elif f.kurtosis > 3.0:
            score += 0.1
            reasons.append(f"峭度={f.kurtosis:.1f}偏高")

        # 峰值因子>5 说明尖峰突出
        if f.crest_factor > 5.0:
            score += 0.2
            reasons.append(f"峰值因子={f.crest_factor:.1f}>5")

        # 脉冲因子>4
        if f.impulse_factor > 4.0:
            score += 0.1
            reasons.append(f"脉冲因子={f.impulse_factor:.1f}")

        # ---- 新增：断刀后骤降模式 ----
        # power_drop_ratio<0.3 说明切削能量骤降到空载
        if f.power_drop_ratio < 0.3 and f.power_max > 50:
            score += 0.5
            reasons.append(
                f"power骤降至{f.power_drop_ratio*100:.0f}%（断刀后空载特征）")

        if score >= 1.0:
            f.status, f.confidence = "断刀", min(0.99, 0.6 + score*0.15)
            f.advice = "立即停机检查刀具"
        elif score >= 0.3:
            f.status, f.confidence = "预警", 0.5 + score*0.2
            f.advice = "降速观察"
        else:
            f.status, f.confidence = "正常", max(0.85, 1-score)
            f.advice = "继续加工"
            if not reasons:
                reasons.append("所有指标在正常范围内")

        f.reasons = reasons
        f.raw_llm = "[RuleDiagnoser auto-calibrated by diameter]"
        return f


# =============================================================================
# 模块4b：LLM API 诊断器（OpenAI兼容接口）
# =============================================================================

class LLMDiagnoser:
    """通过 OpenAI 兼容 Chat Completions 接口调用 Qwen/DeepSeek/本地vLLM。"""

    def __init__(self, llm_cfg: dict, prompt_builder: PromptBuilder,
                 calibrator: ThresholdCalibrator = None):
        self.cfg = llm_cfg
        self.pb = prompt_builder
        self.calibrator = calibrator
        self.client = None
        self.available = False

        if not llm_cfg.get("api_key"):
            logger.warning("LLM_API_KEY 未设置，将自动回退到 RuleDiagnoser")
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=llm_cfg["api_key"],
                base_url=llm_cfg["base_url"],
                timeout=llm_cfg["timeout"],
            )
            self.available = True
            logger.info(f"LLM 已连接: {llm_cfg['base_url']}, "
                        f"model={llm_cfg['model']}")
        except Exception as e:
            logger.warning(f"OpenAI client 初始化失败: {e}，回退规则模式")
            self.available = False

    def diagnose(self, f: WindowFeature) -> WindowFeature:
        if not self.available:
            return RuleDiagnoser(self.calibrator).diagnose(f)

        user_msg = self.pb.build_user_prompt(f)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg["model"],
                messages=[
                    {"role": "system", "content": self.pb.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=self.cfg["temperature"],
                max_tokens=self.cfg["max_tokens"],
            )
            text = resp.choices[0].message.content.strip()
            f.raw_llm = text
            return self._parse_llm_output(f, text)

        except Exception as e:
            logger.warning(f"LLM调用失败({e})，该窗回退规则诊断")
            return RuleDiagnoser(self.calibrator).diagnose(f)

    @staticmethod
    def _parse_llm_output(f: WindowFeature, text: str) -> WindowFeature:
        """从LLM返回中解析JSON；失败则用规则兜底"""
        try:
            # 兼容 ```json ... ``` 包裹
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            data = json.loads(clean)
            f.status = data.get("status", "正常")
            f.confidence = float(data.get("confidence", 0.5))
            f.reasons = list(data.get("reasons", []))
            f.advice = data.get("advice", "继续加工")
        except Exception:
            # JSON解析失败，按关键词粗判
            if "断刀" in text:
                f.status, f.confidence = "断刀", 0.7
            elif "预警" in text:
                f.status, f.confidence = "预警", 0.6
            else:
                f.status, f.confidence = "正常", 0.7
            f.reasons = [f"LLM原始输出: {text[:200]}"]
            f.advice = "人工复核"
        return f


# =============================================================================
# 模块5：可视化
# =============================================================================

class Visualizer:
    """画 amp/power 随时间曲线，断刀窗用红色背景标注"""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # 中文显示
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei",
                                          "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot(self, name: str, df: pd.DataFrame,
             feats: List[WindowFeature]) -> str:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 散点：每个AE事件
        axes[0].scatter(df["t"], df["amp"], s=2, c="steelblue",
                        alpha=0.5, label="amp (dB)")
        axes[0].set_ylabel("Amplitude (dB)")
        axes[0].set_title(f"AE Signal - {name}")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.3)

        axes[1].scatter(df["t"], df["power"], s=2, c="darkorange",
                        alpha=0.5, label="power")
        axes[1].set_ylabel("AE Power")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3, which="both")

        # 标注断刀/预警窗口
        for f in feats:
            if f.status == "断刀":
                color, alpha = "red", 0.15
            elif f.status == "预警":
                color, alpha = "orange", 0.15
            else:
                continue
            for ax in axes:
                ax.axvspan(f.t_start, f.t_end, color=color, alpha=alpha)

        # 在断刀窗顶部加标记
        break_t = [f.t_start for f in feats if f.status == "断刀"]
        if break_t:
            axes[0].axhline(100, color="red", ls="--", lw=0.8,
                            alpha=0.5, label="满量程参考")

        plt.tight_layout()
        out_path = os.path.join(self.out_dir, f"viz_{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"  可视化: {out_path}")
        return out_path


# =============================================================================
# 模块6：报告
# =============================================================================

class ReportGenerator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def to_markdown(self, name: str, feats: List[WindowFeature]) -> str:
        broken = [f for f in feats if f.status == "断刀"]
        warn   = [f for f in feats if f.status == "预警"]
        normal = [f for f in feats if f.status == "正常"]
        first_break_t = broken[0].t_start if broken else None

        lines = [
            f"# 断刀检测报告 - {name}",
            f"",
            f"- 生成: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- 滑窗: {len(feats)} | 正常: {len(normal)} | "
            f"预警: **{len(warn)}** | 断刀: **{len(broken)}**",
            f"- 首次断刀: "
            + (f"**{first_break_t:.1f}s**"
               if first_break_t is not None else "未检出"),
            f"",
            f"## 诊断结论",
        ]
        if broken:
            lines.append(f"> **判定: 已断刀** ({len(broken)}窗触发)")
        elif warn:
            lines.append("> **判定: 疑似崩刃，建议停机检查**")
        else:
            lines.append("> **判定: 切削正常**")

        if feats:
            worst = max(feats, key=lambda x: (
                1 if x.status == "断刀" else
                0.5 if x.status == "预警" else 0,
                x.power_max))
            lines += [
                f"",
                f"## 最严重窗口 (t={worst.t_start:.1f}s)",
                f"- power_max = **{worst.power_max:.1f}**",
                f"- amp_max   = **{worst.amp_max:.1f} dB**",
                f"- rms_max   = {worst.rms_max:.2f}",
                f"- confidence = {worst.confidence:.2f}",
                f"",
                f"### 推理依据",
            ]
            for r in worst.reasons:
                lines.append(f"- {r}")
            lines += [f"", f"### 建议: {worst.advice}"]

        md = "\n".join(lines)
        with open(os.path.join(self.out_dir, f"report_{name}.md"),
                  "w", encoding="utf-8") as fp:
            fp.write(md)

        # CSV
        pd.DataFrame([{
            "t_start": round(f.t_start, 2),
            "amp_max": round(f.amp_max, 2),
            "power_max": round(f.power_max, 2),
            "rms_max": round(f.rms_max, 3),
            "status": f.status,
            "confidence": round(f.confidence, 3),
        } for f in feats]).to_csv(
            os.path.join(self.out_dir, f"windows_{name}.csv"),
            index=False, encoding="utf-8-sig")
        return md


# =============================================================================
# 主流程
# =============================================================================

def run_single(name, path, loader, extractor, diagnoser, reporter, viz):
    logger.info(f"处理: {name}")
    try:
        df = loader.load(path)
        feats = extractor.extract(df)
        feats = [diagnoser.diagnose(f) for f in feats]
        reporter.to_markdown(name, feats)
        viz.plot(name, df, feats)
        n_b = sum(1 for f in feats if f.status == "断刀")
        n_w = sum(1 for f in feats if f.status == "预警")
        print(f"  => [{name}] 断刀={n_b} 预警={n_w} 总={len(feats)}")
    except Exception as e:
        logger.error(f"{name} 失败: {e}")
        traceback.print_exc()


def main():
    logger.info("=" * 60)
    logger.info("AE断刀检测 (3σ自动标定 + LLM API + 可视化) 启动")
    logger.info("=" * 60)

    loader = AEDataLoader()
    wp = load_sliding_window_params()
    CONFIG["window_sec"] = wp["window_sec"]
    CONFIG["step_sec"] = wp["step_sec"]
    CONFIG["min_events_per_window"] = wp["min_events_per_window"]
    extractor = FeatureExtractor(
        wp["window_sec"], wp["step_sec"], wp["min_events_per_window"])
    pb = PromptBuilder()

    # ---- Step 1: 用正常样本自动标定阈值 ----
    calibrator = ThresholdCalibrator(CONFIG["calibration"])
    normal_feats_all = []
    for name in CONFIG["calibration"]["normal_files"]:
        if name not in CONFIG["files"]:
            logger.warning(f"标定文件 {name} 不在 files 配置中，跳过")
            continue
        path = os.path.join(CONFIG["data_dir"], CONFIG["files"][name])
        df = loader.load(path)
        normal_feats_all.extend(extractor.extract(df))

    calibrated_th = calibrator.calibrate(normal_feats_all)
    calibrator.save(os.path.join(CONFIG["output_dir"],
                                 "threshold_calibration.json"))

    # ---- Step 2: 构造诊断器（优先LLM，无key自动回退规则）----
    diagnoser = LLMDiagnoser(CONFIG["llm"], pb, calibrator=calibrator)
    if not diagnoser.available:
        logger.info("使用 RuleDiagnoser（按直径自动标定阈值）模式")
        diagnoser = RuleDiagnoser(calibrator)

    reporter = ReportGenerator(CONFIG["output_dir"])
    viz = Visualizer(CONFIG["output_dir"])

    # ---- Step 3: 跑全部样本 ----
    for name, fn in CONFIG["files"].items():
        run_single(name, os.path.join(CONFIG["data_dir"], fn),
                   loader, extractor, diagnoser, reporter, viz)

    logger.info("完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"主流程异常: {e}")
        traceback.print_exc()
        sys.exit(1)
