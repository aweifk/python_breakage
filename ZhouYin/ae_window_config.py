# -*- coding: utf-8 -*-
"""滑动窗参数：代码默认值 + 从 gui_config.json 读取运行时配置。"""
from __future__ import annotations

import json
import os

_ZHOUYIN_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_SEC = 5.0
STEP_SEC = 2.5
MIN_EVENTS_PER_WINDOW = 20
MIN_EVT = MIN_EVENTS_PER_WINDOW

GUI_CONFIG_PATH = os.path.join(_ZHOUYIN_DIR, "gui_config.json")


def load_sliding_window_params(config_path: str | None = None) -> dict:
    """
    优先 gui_config.json 的 window_sec / step_sec / min_events_per_window；
    缺文件或缺键时使用本模块默认值。兼容旧键 min_events。
    """
    path = config_path or GUI_CONFIG_PATH
    out = {
        "window_sec": float(WINDOW_SEC),
        "step_sec": float(STEP_SEC),
        "min_events_per_window": int(MIN_EVENTS_PER_WINDOW),
    }
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        if "window_sec" in cfg:
            out["window_sec"] = float(cfg["window_sec"])
        if "step_sec" in cfg:
            out["step_sec"] = float(cfg["step_sec"])
        if "min_events_per_window" in cfg:
            out["min_events_per_window"] = int(cfg["min_events_per_window"])
        elif "min_events" in cfg:
            out["min_events_per_window"] = int(cfg["min_events"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return out


def apply_sliding_params_to_extractor(extractor, params: dict | None = None) -> None:
    """同步 FeatureExtractor 的 w / s / min_evt。"""
    p = params or load_sliding_window_params()
    extractor.w = float(p["window_sec"])
    extractor.s = float(p["step_sec"])
    extractor.min_evt = int(p["min_events_per_window"])
