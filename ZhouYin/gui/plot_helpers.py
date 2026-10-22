# -*- coding: utf-8 -*-
"""共用波形图：物理规则阈值线与报警窗标注（与实时监控一致）"""
from __future__ import annotations

import json
import os

import numpy as np

_ZHOUYIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLOT_DETAIL_KEYS = (
    "rule_broken_start",
    "both_break_idx",
    "rule_evidence",
    "rule_broken",
    "model_file_broken",
    "final_broken",
    "xgb_break_idx",
    "if_break_idx",
    "rule_drop_start",
    "model_alert_window",
    "window_times",
)


def dia_thresholds_path() -> str:
    return os.path.join(_ZHOUYIN_DIR, "dia_thresholds.json")


def detail_to_json(detail: dict | None, reason_lines: list | None = None) -> str | None:
    if not detail:
        return None
    payload = {k: detail[k] for k in PLOT_DETAIL_KEYS if k in detail}
    if reason_lines:
        payload["_reason_lines"] = list(reason_lines)
    return json.dumps(payload, ensure_ascii=False) if payload else None


def resolve_detection_result(detector, fpath: str | None, detail_json: str | None):
    """优先历史 detail_json；否则 predict_file。"""
    if detail_json:
        try:
            raw = json.loads(detail_json)
            if isinstance(raw, dict) and "detail" in raw:
                return raw
            detail = raw if isinstance(raw, dict) else {}
            reason_lines = detail.pop("_reason_lines", None)
            return {
                "detail": detail,
                "broken": detail.get("final_broken"),
                "reason_lines": reason_lines or [],
            }
        except Exception:
            pass
    if detector and fpath and os.path.isfile(fpath):
        return detector.predict_file(fpath)
    return None


def time_axis(df):
    if "t" in df.columns:
        return df["t"].values, "时间 (秒)"
    return np.arange(len(df)), "样本序号"


def plot_waveform_curves(axis_channels, df, title: str | None = None):
    """axis_channels: [(ax, channel_name), ...]"""
    t, xlabel = time_axis(df)
    for ax, ch in axis_channels:
        ax.clear()
        if ch in df.columns:
            ax.plot(t, df[ch].values, linewidth=0.5, color=_channel_color(ch))
        else:
            ax.plot(t, np.zeros(len(t)), linewidth=0.5, color="#999")
        ax.set_ylabel(ch)
        ax.grid(alpha=0.3)
    if axis_channels:
        axis_channels[-1][0].set_xlabel(xlabel)
    if title and axis_channels:
        axis_channels[0][0].set_title(title)


def _channel_color(ch: str) -> str:
    cl = ch.lower()
    if "power" in cl:
        return "#27ae60"
    if "rms" in cl:
        return "#2980b9"
    if "amp" in cl:
        return "#8e44ad"
    return "#555"


def _thr_lines(ax, ch: str, td: dict):
    cl = ch.lower()
    if "power" in cl:
        hi = td["power_shock_hi"]
        lo = td["power_drop_ref_lo"]
        ax.axhline(hi, color="red", ls="--", lw=1, alpha=0.75)
        ax.axhline(lo, color="orange", ls="--", lw=1, alpha=0.75)
        ax.text(
            0.01, hi, " " + td["power_shock_hi_label"],
            color="red", fontsize=7, va="bottom",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            0.01, lo, " " + td["power_drop_ref_lo_label"],
            color="darkorange", fontsize=7, va="top",
            transform=ax.get_yaxis_transform(),
        )
    elif "rms" in cl:
        hi = td["rms_shock_hi"]
        ax.axhline(hi, color="red", ls="--", lw=1, alpha=0.75)
        ax.text(
            0.01, hi, " " + td["rms_shock_hi_label"],
            color="red", fontsize=7, va="bottom",
            transform=ax.get_yaxis_transform(),
        )


def plot_detection_annotations(
    axis_channels,
    df,
    detector,
    result=None,
    show_legend_note: bool = True,
):
    """
    在已画曲线的轴上叠加阈值线与报警窗。
    返回 {"program_line": str, "break_t": float|None}
    """
    out = {"program_line": "-", "break_t": None}
    if not axis_channels:
        return out

    plot_axes = [ax for ax, _ in axis_channels]

    try:
        from ae_xgboost_isolationforest import compute_rule_display_thresholds
        with open(dia_thresholds_path(), encoding="utf-8") as f:
            d_thr = json.load(f)
        dia = float(df["tool_diameter"].median()) if "tool_diameter" in df.columns else 0
        if dia > 0 and detector:
            near = min(d_thr.keys(), key=lambda k: abs(float(k) - dia))
            td = compute_rule_display_thresholds(dia, d_thr[near], detector)
            for ax, ch in axis_channels:
                _thr_lines(ax, ch, td)
    except Exception:
        pass

    feats = detector.extractor.extract(df) if detector else []
    if not feats:
        return out

    detail = (result or {}).get("detail") or {}
    both_idx = list(detail.get("both_break_idx") or [])
    if not both_idx and detector:
        both_idx = detector.infer_dual_model_windows(feats).get("both_break_idx") or []

    rule_start = int(detail.get("rule_broken_start", -1))

    if rule_start >= 0:
        for i in range(rule_start, len(feats)):
            t0, t1 = feats[i].t_start, feats[i].t_end
            for ax in plot_axes:
                ax.axvspan(t0, t1, color="#dc2626", alpha=0.15, zorder=0)

    for i in both_idx:
        if i < 0 or i >= len(feats):
            continue
        t0, t1 = feats[i].t_start, feats[i].t_end
        for ax in plot_axes:
            ax.axvspan(t0, t1, color="#dc2626", alpha=0.22, zorder=1)
            ax.axvline(t0, color="#991b1b", lw=1.2, alpha=0.85, zorder=2)
            ax.axvline(t1, color="#991b1b", lw=1.2, alpha=0.85, zorder=2)

    legend_ax = axis_channels[0][0]
    if show_legend_note and (rule_start >= 0 or both_idx):
        legend_ax.text(
            0.01, 0.98,
            "浅红=物理规则段  深红框=双模型(XGB+IF)共同窗",
            transform=legend_ax.transAxes, fontsize=7, va="top",
            color="#64748b",
        )

    break_t = None
    if "power" in df.columns and "t" in df.columns:
        power_arr = df["power"].values
        t_arr = df["t"].values
        search_windows = both_idx if both_idx else (
            list(range(rule_start, len(feats))) if rule_start >= 0 else [])
        best_p = 0
        best_t = None
        for wi in search_windows:
            if wi < 0 or wi >= len(feats):
                continue
            t0, t1 = feats[wi].t_start, feats[wi].t_end
            mask = (t_arr >= t0) & (t_arr <= t1)
            if mask.any():
                idx = int(np.argmax(np.where(mask, power_arr, -np.inf)))
                if power_arr[idx] > best_p:
                    best_p = power_arr[idx]
                    best_t = float(t_arr[idx])
        if best_t is not None:
            break_t = best_t
            out["break_t"] = break_t
            legend_ax.text(
                break_t, legend_ax.get_ylim()[1] * 0.95, "断刀",
                color="#dc2626", fontsize=12, fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#dc2626"),
            )

    if "program_line" in df.columns and "t" in df.columns:
        gt_t = None
        if result is not None and result.get("detail", {}).get("rule_broken_start", -1) >= 0:
            rs = result["detail"]["rule_broken_start"]
            if rs < len(feats):
                gt_t = feats[rs].t_start
        elif break_t is not None:
            gt_t = break_t
        if gt_t is not None:
            mask = (df["t"] >= gt_t - 1.0) & (df["t"] <= gt_t + 1.0)
            if mask.any():
                program_line = str(df.loc[mask, "program_line"].iloc[0])
                out["program_line"] = program_line
                for ax in plot_axes:
                    ax.axvline(gt_t, color="#dc2626", ls=":", lw=1.5, alpha=0.7)
                try:
                    axis_channels[-1][0].text(
                        gt_t, axis_channels[-1][0].get_ylim()[1] * 0.9,
                        f"G{program_line}", color="#dc2626", fontsize=10,
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dc2626"),
                    )
                except Exception:
                    pass

    for ax, _ in axis_channels:
        ax.autoscale_view()

    return out
