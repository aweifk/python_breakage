# -*- coding: utf-8 -*-
"""
=============================================================================
文件名称 : ae_xgboost_isolationforest.py
功能描述 : XGBoost + IsolationForest 双模型断刀检测
          - XGBoost: 监督主检测
          - IsolationForest: 无监督双确认
          - 两个模型都判断刀才报警（降低误报）
用法:
    python ae_xgboost_isolationforest.py            # 批量测试
    python ae_xgboost_isolationforest.py 文件.csv   # 单文件检测
=============================================================================
"""
import os, sys, logging, traceback
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from ae_llm_break_tool_detect import FeatureExtractor
from ae_batch_pipeline import (
    load_one_csv, scan_dataset, build_dia_baseline, apply_dia_baseline,
    apply_dia_thr_ratios,
)
from ae_rf_train import FEATURE_COLS
from ae_window_config import (
    WINDOW_SEC,
    STEP_SEC,
    MIN_EVT,
    load_sliding_window_params,
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("XGB+IF")

XGB_MODEL = "output_xgb/xgb_break_tool.pkl"
IF_MODEL = "output_lgb/isolationforest.pkl"
THRESHOLD = 0.5       # XGBoost概率阈值（优先保召回：0914断刀全抓到，误报偏多）
CONSEC_WINDOWS = 2   # 连续N窗确认

# 按直径的空切/实切阈值表（由 normal+aircut 统计生成），推理时按直径取用
import json as _json
try:
    _DIA_THR = _json.load(open(os.path.join(os.path.dirname(__file__),
                              "dia_thresholds.json"), encoding="utf-8"))
except Exception:
    _DIA_THR = {}


def dia_thr_lookup(dia):
    """按直径取阈值，未知直径用全局均值。"""
    dia = float(dia or 0)
    if dia > 0 and _DIA_THR:
        nearest = min(_DIA_THR.keys(), key=lambda k: abs(float(k) - dia))
        return _DIA_THR[nearest]
    # 全局兜底
    keys = list(_DIA_THR.values())
    if keys:
        return {k: float(np.mean([d[k] for d in keys])) for k in keys[0]}
    return {"aircut_p90_power": 40.0, "aircut_p99_rms": 2.0,
            "normal_p10_power": 50.0, "normal_p95_power": 1000.0,
            "normal_p95_rms": 10.0}


def pcoef_for_dia(dia, detector):
    """分径 power 冲击系数，与 predict_file 一致。"""
    dia = float(dia or 0)
    if dia <= 1.0:
        return float(getattr(detector, "pcoef_small", 3.0))
    if dia <= 4.0:
        return float(getattr(detector, "pcoef_mid", 4.0))
    return float(getattr(detector, "pcoef_large", 2.5))


def compute_rule_display_thresholds(dia, thr_row, detector):
    """
    物理规则绘图/推理共用的阈值（含乘系数冲击线与 2×空切底噪）。
    thr_row: dia_thresholds.json 中某一档 dict。
    """
    dia = round(float(dia or 0), 1)
    thr_row = thr_row or {}
    rms_coef = float(getattr(detector, "rms_coef", 1.2))
    pcoef = pcoef_for_dia(dia, detector)
    p95p = float(thr_row.get("normal_p95_power", 1000.0))
    p95r = float(thr_row.get("normal_p95_rms", 10.0))
    aircut_p90 = float(thr_row.get("aircut_p90_power", 0))
    floor_thr = max(aircut_p90, 5.0)
    power_shock_hi = p95p * pcoef
    power_drop_ref_lo = 2.0 * floor_thr
    rms_shock_hi = p95r * rms_coef
    return {
        "dia_used": dia,
        "pcoef": pcoef,
        "rms_coef": rms_coef,
        "floor_thr": floor_thr,
        "rthr": rms_shock_hi,
        "power_rthr": power_shock_hi,
        "power_shock_hi": power_shock_hi,
        "power_drop_ref_lo": power_drop_ref_lo,
        "rms_shock_hi": rms_shock_hi,
        "normal_p95_power": p95p,
        "normal_p95_rms": p95r,
        "power_shock_hi_label": (
            f"冲击线=实切p95×p={p95p:.1f}×{pcoef:g}={power_shock_hi:.1f}"
        ),
        "power_drop_ref_lo_label": (
            f"掉底参考=2×F，F=max(aircut_p90,5)={floor_thr:.1f}→{power_drop_ref_lo:.1f}"
        ),
        "rms_shock_hi_label": (
            f"冲击线=实切p95_rms×k={p95r:.2f}×{rms_coef:g}={rms_shock_hi:.2f}"
        ),
    }


def _feat_index_at_t_start(feats, t0: float, eps: float = 1e-6) -> int:
    """第一个 t_start ≈ t0 的窗索引，不存在返回 -1。"""
    for j, f in enumerate(feats):
        if abs(float(f.t_start) - float(t0)) <= eps:
            return j
    return -1


def _model_both_confirmed_mask(feats, both_broken, drop_lo: float):
    """
    双模型同窗 i 的确认窗为 [t_start+W, t_start+2W]；
    通过 iff 该窗存在且 power_mean < drop_lo（2×空切底噪）。
    返回 (confirmed_bool_array, evidence_list)。
    """
    n = len(feats)
    confirmed = np.zeros(n, dtype=bool)
    evidence = []
    if n == 0:
        return confirmed, evidence
    w_sec = float(feats[0].t_end - feats[0].t_start)
    drop_lo = float(drop_lo)
    both_arr = np.asarray(both_broken, dtype=bool)
    if both_arr.size != n:
        both_arr = np.resize(both_arr, n)
    for i in range(n):
        if not both_arr[i]:
            continue
        fi = feats[i]
        t_confirm = float(fi.t_start) + w_sec
        j = _feat_index_at_t_start(feats, t_confirm)
        rec = {
            "both_window": int(i),
            "confirm_window": int(j),
            "both_t_start": float(fi.t_start),
            "both_t_end": float(fi.t_end),
            "confirm_t_start": t_confirm,
            "confirm_t_end": t_confirm + w_sec,
            "drop_lo": drop_lo,
            "passed": False,
        }
        if j < 0:
            rec["fail_reason"] = "无确认窗(该时段有效窗不足)"
        else:
            fj = feats[j]
            pm = float(fj.power_mean)
            rec["confirm_power_mean"] = pm
            rec["confirm_t_start"] = float(fj.t_start)
            rec["confirm_t_end"] = float(fj.t_end)
            if pm < drop_lo:
                confirmed[i] = True
                rec["passed"] = True
            else:
                rec["fail_reason"] = (
                    f"确认窗power_mean={pm:.1f}≥掉底线{drop_lo:.1f}(2×空切)"
                )
        evidence.append(rec)
    return confirmed, evidence


# 各直径正常加工 power 中位数（来自 dataset/normal）：用于空切门判定
_DIA_NORMAL_PMED = {
    0.6: 70.0, 0.8: 38.0, 1.0: 1321.0, 1.5: 158.0,
    2.0: 685.0, 4.0: 856.0, 6.0: 3353.0, 8.0: 12073.0,
}


def dia_normal_power_median(dia):
    """该直径正常加工 power 中位数，未知直径用全局中位。"""
    dia = float(dia or 0)
    if dia <= 0:
        return float(np.median(list(_DIA_NORMAL_PMED.values())))
    nearest = min(_DIA_NORMAL_PMED.keys(), key=lambda d: abs(d - dia))
    return _DIA_NORMAL_PMED[nearest]


# 各直径正常加工基线（来自 dataset/normal 统计）：
#   {dia: (p95_power, p95_rms)}
_DIA_NORMAL = {
    0.6: (818.0, 6.5),
    0.8: (162.0, 4.5),
    1.0: (3650.0, 23.3),
    1.5: (971.0, 7.9),
    2.0: (3900.0, 16.0),
    4.0: (6783.0, 22.4),
    6.0: (124360.0, 55.3),
    8.0: (36597.0, 34.9),
}
SHOCK_POW_BUFFER = 1.3   # power 超正常p95的倍数
SHOCK_RMS_BUFFER = 1.5   # rms 超正常p95的倍数


def _longest_true_run(mask):
    """布尔序列中最长连续 True 段的窗索引列表。"""
    best, cur = [], []
    for i, b in enumerate(mask):
        if b:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def _window_times_for(feats, indices):
    """为指定窗索引构建 t_start/t_end 字典。"""
    wt = {}
    for i in indices:
        if 0 <= i < len(feats):
            wt[int(i)] = {
                "t_start": round(float(feats[i].t_start), 2),
                "t_end": round(float(feats[i].t_end), 2),
            }
    return wt


def _fmt_win(idx, window_times):
    if idx is None or int(idx) < 0:
        return ""
    i = int(idx)
    t = window_times.get(i)
    if t:
        return f"窗{i} ({t['t_start']}~{t['t_end']}s)"
    return f"窗{i}"


def _fmt_win_list(idxs, window_times, max_show=8):
    if not idxs:
        return "无"
    idxs = [int(x) for x in idxs]
    parts = [_fmt_win(i, window_times) for i in idxs[:max_show]]
    if len(idxs) > max_show:
        parts.append(f"…共{len(idxs)}窗")
    return "、".join(parts)


def _spike_kind_at(s, rm, pmax, rthr, power_rthr):
    r_hit = float(rm[s]) > rthr
    p_hit = float(pmax[s]) > power_rthr
    if r_hit and p_hit:
        return "rms与power同时冲击"
    if r_hit:
        return "rms冲击"
    if p_hit:
        return "power冲击"
    return "冲击"


def _cmp_spike_line(label, value, thr, hit):
    op = ">" if hit else "≤"
    return f"{label}={value:.2f} {op} {label}_thr={thr:.2f}"


def _format_rule_evidence_lines(evidence, window_times=None):
    """物理规则报警时的量化原因（峰值、阈值、峰后统计、不等式）。"""
    if not evidence:
        return []
    wt = window_times or {}
    lines = []
    mode = evidence.get("mode", "")
    floor_thr = float(evidence.get("floor_thr", 0))
    rthr = float(evidence.get("rthr", 0))
    power_rthr = float(evidence.get("power_rthr", 0))
    rms_coef = evidence.get("rms_coef")
    pcoef = evidence.get("pcoef")
    dia = evidence.get("dia")

    def spike_detail(spike_win, rms_v, pwr_v, kind):
        if spike_win is None or int(spike_win) < 0:
            return
        sw = int(spike_win)
        r_hit = bool(evidence.get("rms_hit")) if sw == evidence.get("spike_window") else (
            float(rms_v) > rthr)
        p_hit = bool(evidence.get("power_hit")) if sw == evidence.get("spike_window") else (
            float(pwr_v) > power_rthr)
        coef_note = ""
        if rms_coef is not None and pcoef is not None:
            coef_note = f"（rms_thr=normal_p95_rms×{rms_coef}，power_thr=normal_p95_power×{pcoef}）"
        lines.append(
            f"【物理规则/尖峰】突变窗{_fmt_win(sw, wt)}，直径D{dia:g}；"
            f"{_cmp_spike_line('rms_max', float(rms_v), rthr, r_hit)}；"
            f"{_cmp_spike_line('power_max', float(pwr_v), power_rthr, p_hit)}"
            f"{coef_note} → {kind or _spike_kind_at(sw, np.array([rms_v]), np.array([pwr_v]), rthr, power_rthr)}"
        )

    if mode == "small_spike_drop":
        sw = evidence.get("spike_window", -1)
        spike_detail(sw, evidence.get("rms_max", 0), evidence.get("power_max", 0),
                     evidence.get("spike_kind"))
        after_count = evidence.get("after_count", 0)
        after_med = float(evidence.get("after_med", 0))
        lines.append(
            f"【物理规则/峰后】峰后有效滑动窗共 {after_count} 个（尖峰后所有≥{MIN_EVT}点/5s 的窗）；"
            f"峰后 power_mean 中位数 P_after={after_med:.1f}"
        )
        if evidence.get("start_break"):
            lim = float(evidence.get("limit_2f", floor_thr * 2))
            lines.append(
                f"【物理规则/掉底】文件开头断刀模式（峰前窗<3，不要求真切削）；"
                f"空切底噪 F={floor_thr:.1f}；"
                f"判定：P_after({after_med:.1f}) ≤ 2×F={lim:.1f} → 满足掉底报断刀"
            )
        else:
            before_med = float(evidence.get("before_med", 0))
            lim_half = before_med * 0.5
            lim_2f = floor_thr * 2.0
            lim_before = floor_thr * 0.3
            lines.append(
                f"【物理规则/掉底】尖峰前 P_before={before_med:.1f}（需 ≥ 0.3×F={lim_before:.1f}，已满足）；"
                f"空切底噪 F={floor_thr:.1f}；"
                f"判定：P_after({after_med:.1f}) ≤ 0.5×P_before={lim_half:.1f} 且 "
                f"P_after({after_med:.1f}) ≤ 2×F={lim_2f:.1f} → 满足掉底报断刀"
            )
        if evidence.get("no_spike_after_s_plus2"):
            lines.append(
                "【物理规则/约束】D<6：尖峰窗 s+2 之后无新的冲击尖峰（已满足）"
            )

    elif mode == "large_state_machine":
        drop_start = evidence.get("drop_start_window", -1)
        drop_n = evidence.get("drop_consec_windows", 0)
        after_med = float(evidence.get("after_med", 0))
        before_med = float(evidence.get("before_med", 0))
        lim_half = before_med * 0.5
        lim_2f = floor_thr * 2.0
        lim_before = floor_thr * 0.3
        lines.append(
            f"【物理规则/峰后】掉底段起点{_fmt_win(drop_start, wt)}；"
            f"连续 {drop_n} 窗 power_mean≤F={floor_thr:.1f}；"
            f"该段 power_mean 中位数 P_after={after_med:.1f}"
        )
        lines.append(
            f"【物理规则/掉底】切削段 P_before={before_med:.1f}；"
            f"判定：段内曾出现尖峰 且 P_before({before_med:.1f}) > 0.3×F={lim_before:.1f} 且 "
            f"P_after({after_med:.1f}) < 0.5×P_before={lim_half:.1f} 且 "
            f"P_after({after_med:.1f}) < 2×F={lim_2f:.1f} → 满足状态机报断刀"
        )
        sw = evidence.get("representative_spike_window", -1)
        if sw is not None and int(sw) >= 0:
            spike_detail(
                sw,
                evidence.get("rep_rms_max", 0),
                evidence.get("rep_power_max", 0),
                evidence.get("rep_spike_kind"),
            )
        spike_list = evidence.get("spike_windows_in_cut") or []
        if spike_list:
            lines.append(
                f"【物理规则/尖峰】本切削段内冲击窗：{_fmt_win_list(spike_list, wt, max_show=6)}"
            )

    return lines


def format_detection_reason(broken, detail, window_times=None):
    """根据 detail 生成可读判定原因（含窗位）。"""
    window_times = window_times or {}
    lines = []
    d = detail or {}
    wt = window_times

    def w(idx):
        return _fmt_win(idx, wt)

    model_ok = bool(d.get("model_file_broken"))
    rule_ok = bool(d.get("rule_broken"))
    air_gate = bool(d.get("air_cut_gate"))
    both_idx = d.get("both_break_idx") or []
    model_alert = int(d.get("model_alert_window", -1))
    rule_start = int(d.get("rule_broken_start", -1))
    rule_drop = int(d.get("rule_drop_start", -1))
    rule_kind = d.get("rule_spike_kind") or ""
    rule_mode = d.get("rule_alert_mode") or ""

    if broken:
        parts = []
        if model_ok and model_alert >= 0:
            parts.append(f"模型主报警{w(model_alert)}")
        if rule_ok and rule_start >= 0:
            parts.append(f"物理规则尖峰/掉底{w(rule_start)}")
        head = "判定为断刀"
        if parts:
            head += "：" + "；".join(parts)
        lines.append(f"【结论】{head}")

        if model_ok:
            prob = d.get("model_alert_prob")
            prob_s = f"，XGB概率={prob:.3f}" if prob is not None else ""
            lines.append(
                f"【模型通路】XGB>{THRESHOLD}共{d.get('xgb_broken_windows', 0)}窗，"
                f"IF异常共{d.get('if_broken_windows', 0)}窗，"
                f"双模型共同共{d.get('both_broken_windows', 0)}窗；"
                f"双模型共同窗位：{_fmt_win_list(both_idx, wt)}"
            )
            conf_idx = d.get("both_confirmed_break_idx") or []
            if conf_idx:
                lines.append(
                    f"【模型通路/掉底门控】通过确认窗的共同窗："
                    f"{_fmt_win_list(conf_idx, wt)}"
                )
            if model_alert >= 0:
                lines.append(f"【模型通路】主报警窗{w(model_alert)}{prob_s}")
                gate_ev = d.get("model_drop_gate_evidence") or []
                for ev in gate_ev:
                    if ev.get("both_window") == model_alert and ev.get("passed"):
                        cj = int(ev.get("confirm_window", -1))
                        lines.append(
                            f"【模型通路/掉底门控】主窗确认窗{w(cj)} "
                            f"power_mean={ev.get('confirm_power_mean', 0):.1f}"
                            f"<{ev.get('drop_lo', 0):.1f}(2×空切)"
                        )
                        break
            run = d.get("consec_alert_run") or []
            if run:
                lines.append(
                    f"【模型通路】连续异常确认段（任一模型异常，需≥{CONSEC_WINDOWS}窗）："
                    f"{_fmt_win_list(run, wt)}"
                )

        if rule_ok:
            if rule_mode == "large_state_machine":
                lines.append(
                    f"【物理规则/突变】大径(D≥6)状态机：切削段内曾出现冲击尖峰，"
                    f"随后连续低 power 掉底；掉底段起点{w(rule_start)}"
                )
            else:
                sk = f"，突变类型={rule_kind}" if rule_kind else ""
                lines.append(
                    f"【物理规则/突变】冲击尖峰窗{w(rule_start)}{sk}；"
                    f"掉底自尖峰后窗{w(rule_drop) if rule_drop >= 0 else '下一窗'}起（power中位数显著下降）"
                )
            spike_list = d.get("rule_spike_windows") or []
            if spike_list:
                lines.append(
                    f"【物理规则/突变】全文件冲击候选窗：{_fmt_win_list(spike_list, wt, max_show=6)}"
                )
            ev = d.get("rule_evidence") or {}
            lines.extend(_format_rule_evidence_lines(ev, wt))
    else:
        lines.append("【结论】判定为正常")
        if air_gate:
            msg = "【空切门】全文件无冲击尖峰且 power 均在空切底噪内，强制判正常"
            if d.get("model_file_broken"):
                msg += f"（模型曾满足报警条件，主窗{w(model_alert)}，已被空切门否决）"
            lines.append(msg)
        if not model_ok:
            sub = []
            if d.get("both_broken_windows", 0) == 0:
                sub.append("无双模型共同命中窗")
            elif d.get("both_confirmed_windows", 0) == 0:
                sub.append(
                    "有双模型共同窗但未通过确认窗掉底门控"
                    "（下一格窗 power_mean 须 < 2×空切底噪，或无确认窗）"
                )
            elif d.get("max_consecutive", 0) < CONSEC_WINDOWS:
                sub.append(
                    f"虽有通过掉底门控的共同窗但未满足连续异常≥{CONSEC_WINDOWS}窗"
                    f"（当前最大连续={d.get('max_consecutive', 0)}）"
                )
            xgb_n = d.get("xgb_broken_windows", 0)
            if_n = d.get("if_broken_windows", 0)
            if xgb_n == 0:
                sub.append("XGB未超过阈值")
            if if_n == 0:
                sub.append("IF未判异常")
            if sub:
                lines.append(f"【模型通路】未触发文件级报警：{'；'.join(sub)}")
            gate_ev = d.get("model_drop_gate_evidence") or []
            fails = [ev for ev in gate_ev if not ev.get("passed")][:3]
            for ev in fails:
                bi = int(ev.get("both_window", -1))
                reason = ev.get("fail_reason") or "未通过"
                pm = ev.get("confirm_power_mean")
                extra = f"，确认窗power_mean={pm:.1f}" if pm is not None else ""
                lines.append(
                    f"【模型通路/掉底门控】共同窗{w(bi)}未通过：{reason}{extra}"
                )
            if both_idx and not model_ok:
                lines.append(
                    f"【模型通路】双模型共同窗位（原始同窗命中）：{_fmt_win_list(both_idx, wt)}"
                )
        if not rule_ok:
            if d.get("total_windows", 0) < 6:
                lines.append("【物理规则】窗口数不足6，未执行完整物理规则")
            else:
                lines.append("【物理规则】未找到合法的「冲击尖峰 + 掉底」组合")

    summary = lines[0].replace("【结论】", "") if lines else "无判定原因"
    return lines, summary


def dia_shock_bounds(dia):
    """返回 (rms冲击阈值, power冲击阈值)，按最接近的直径查表。"""
    dia = float(dia or 0)
    if dia <= 0:
        # 未知直径：用全局中位
        rp = np.median([v[0] for v in _DIA_NORMAL.values()])
        rr = np.median([v[1] for v in _DIA_NORMAL.values()])
        return rr * SHOCK_RMS_BUFFER, rp * SHOCK_POW_BUFFER
    nearest = min(_DIA_NORMAL.keys(), key=lambda d: abs(d - dia))
    p95p, p95r = _DIA_NORMAL[nearest]
    return p95r * SHOCK_RMS_BUFFER, p95p * SHOCK_POW_BUFFER


class DualModelDetector:
    def __init__(self):
        self.segmented = False
        self.models = {}
        wp = load_sliding_window_params()
        self.extractor = FeatureExtractor(
            wp["window_sec"], wp["step_sec"], wp["min_events_per_window"])
        self.window_sec = wp["window_sec"]
        self.step_sec = wp["step_sec"]
        # 物理规则系数（可被GUI覆盖）
        self.rms_coef = 1.2
        self.pcoef_small = 3.0
        self.pcoef_mid = 4.0
        self.pcoef_large = 2.5

        # 优先加载分直径模型 S/M/L
        seg_ok = True
        for seg in ["S", "M", "L"]:
            xp = f"output_xgb/xgb_{seg}.pkl"
            ip = f"output_xgb/if_{seg}.pkl"
            if not (os.path.exists(xp) and os.path.exists(ip)):
                seg_ok = False
                break
        if seg_ok:
            for seg in ["S", "M", "L"]:
                xgb_data = joblib.load(f"output_xgb/xgb_{seg}.pkl")
                if_data = joblib.load(f"output_xgb/if_{seg}.pkl")
                self.models[seg] = {
                    "xgb": xgb_data["model"],
                    "features": xgb_data.get("features", FEATURE_COLS),
                    "if": if_data["model"],
                    "if_mu": if_data["mu"],
                    "if_sd": if_data["sd"],
                }
            self.segmented = True
            logger.info("分直径模型加载完成 (S/M/L)")
        else:
            # 回退到单模型
            xgb_data = joblib.load(XGB_MODEL)
            self.xgb = xgb_data["model"]
            self.features = xgb_data.get("features", FEATURE_COLS)
            self.dia_baseline = xgb_data.get("dia_baseline", None)
            if_data = joblib.load(IF_MODEL)
            self.if_model = if_data["model"]
            self.if_mu = if_data["mu"]
            self.if_sd = if_data["sd"]
            logger.info("单模型加载完成（分模型文件不全，回退）")

    def _route(self, dia):
        """按直径路由到 S/M/L"""
        if dia <= 1.5: return "S"
        elif dia <= 4.0: return "M"
        else: return "L"

    def _prepare_feats_for_infer(self, feats):
        """与 predict_file 一致的特征前处理（会原地修改 feats）。"""
        for i, f in enumerate(feats):
            if i == 0:
                f.power_drop_ratio = 1.0
            else:
                p = feats[i - 1].power_mean
                f.power_drop_ratio = f.power_mean / p if p > 1e-6 else 1.0
        apply_dia_baseline(feats, getattr(self, "dia_baseline", None))
        apply_dia_thr_ratios(feats)

    def _run_model_masks(self, feats):
        """XGB + IF 窗级掩码，与 predict_file 模型段一致。"""
        dia_model = round(float(getattr(feats[0], "tool_diameter", 0) or 0), 1)
        if self.segmented:
            seg = self._route(dia_model)
            m = self.models[seg]
            feats_list = m["features"]
            xgb_clf = m["xgb"]
            if_clf = m["if"]
            if_mu, if_sd = m["if_mu"], m["if_sd"]
        else:
            feats_list = self.features
            xgb_clf = self.xgb
            if_clf = self.if_model
            if_mu, if_sd = self.if_mu, self.if_sd
        X = np.array([[getattr(f, c, 0.0) or 0.0 for c in feats_list] for f in feats])
        xgb_prob = xgb_clf.predict_proba(X)[:, 1]
        xgb_broken_mask = xgb_prob > THRESHOLD
        X_norm = (X - if_mu) / if_sd
        if_pred = if_clf.predict(X_norm)
        if_broken_mask = if_pred == -1
        both_broken = xgb_broken_mask & if_broken_mask
        return xgb_prob, xgb_broken_mask, if_broken_mask, both_broken

    def infer_dual_model_windows(self, feats):
        """供 GUI 绘图：双模型共同命中窗索引（不跑物理规则）。"""
        if not feats:
            return {
                "both_break_idx": [], "xgb_break_idx": [], "if_break_idx": [],
                "model_alert_window": -1, "model_alert_prob": None,
            }
        self._prepare_feats_for_infer(feats)
        xgb_prob, xgb_m, if_m, both = self._run_model_masks(feats)
        both_idx = np.where(both)[0].tolist()
        alert_w = -1
        alert_p = None
        if both_idx:
            alert_w = int(max(both_idx, key=lambda i: float(xgb_prob[i])))
            alert_p = float(xgb_prob[alert_w])
        return {
            "both_break_idx": both_idx,
            "xgb_break_idx": np.where(xgb_m)[0].tolist(),
            "if_break_idx": np.where(if_m)[0].tolist(),
            "model_alert_window": alert_w,
            "model_alert_prob": alert_p,
            "xgb_prob": xgb_prob,
        }

    def predict_file(self, csv_path: str):
        """检测单个文件，返回：是否断刀、置信度、详情"""
        df = load_one_csv(csv_path)
        feats = self.extractor.extract(df)
        if not feats:
            msg = "无有效滑动窗口，无法检测"
            return {
                "broken": False, "confidence": 0, "reason": "无有效窗口",
                "reason_lines": [f"【结论】{msg}"], "reason_summary": msg,
            }

        self._prepare_feats_for_infer(feats)
        xgb_prob, xgb_broken_mask, if_broken_mask, both_broken = self._run_model_masks(feats)

        # 双模型都判断刀的窗
        either_broken = xgb_broken_mask | if_broken_mask

        # 连续窗判断（OR逻辑：任一模型判断刀就算）
        consec = 0
        max_consec = 0
        for b in either_broken:
            if b:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        consec_alert_run = _longest_true_run(either_broken)

        dia_model = round(float(getattr(feats[0], "tool_diameter", 0) or 0), 1)
        _thr_row = dia_thr_lookup(dia_model)
        _disp_thr = compute_rule_display_thresholds(dia_model, _thr_row, self)
        model_drop_lo = float(_disp_thr["power_drop_ref_lo"])

        both_confirmed, model_drop_gate_evidence = _model_both_confirmed_mask(
            feats, both_broken, model_drop_lo)

        # 判决逻辑：同窗双模型 + 确认窗掉底 + 连续2窗（任一模型 OR）
        model_file_broken = bool(
            both_confirmed.any() and max_consec >= CONSEC_WINDOWS)
        file_broken = model_file_broken

        both_break_idx = np.where(both_broken)[0].tolist()
        both_confirmed_break_idx = np.where(both_confirmed)[0].tolist()
        model_alert_window = -1
        model_alert_prob = None
        if both_confirmed_break_idx:
            model_alert_window = int(max(
                both_confirmed_break_idx, key=lambda i: float(xgb_prob[i])))
            model_alert_prob = float(xgb_prob[model_alert_window])

        # 物理规则（按直径从 normal/aircut 学的上下限）：
        #   冲击线 = 该直径实切 rms p95 × 1.5；掉底线 = 该直径空切 power p90。
        # 断刀 = rms冲击尖峰(超实切上限) 之后，后续功率相对该文件尖峰前切削功率显著下降。
        pm = np.array([f.power_mean for f in feats], dtype=float)
        pmax = np.array([getattr(f, "power_max", f.power_mean) for f in feats], dtype=float)
        rm = np.array([f.rms_max for f in feats], dtype=float)
        rule_broken = False
        rule_spike_idx = -1
        rule_drop_start = -1
        rule_spike_kind = ""
        rule_alert_mode = ""
        air_cut_gate = False
        spike = np.zeros(len(pm), dtype=bool)
        rule_spike_windows = []
        rule_evidence = {}
        if len(pm) >= 6:
            dia = round(float(getattr(feats[0], "tool_diameter", 0) or 0), 1)
            thr = dia_thr_lookup(dia)
            _rule_thr = compute_rule_display_thresholds(dia, thr, self)
            rthr = _rule_thr["rthr"]
            floor_thr = _rule_thr["floor_thr"]
            power_rthr = _rule_thr["power_rthr"]
            _pcoef = _rule_thr["pcoef"]
            spike = (rm > rthr) | (pmax > power_rthr)
            rule_spike_windows = np.where(spike)[0].tolist()
            # 该文件正常切削功率基线：所有 > 空切底噪 的窗 power 中位数
            cut_mask = pm > floor_thr
            cut_med = float(np.median(pm[cut_mask])) if cut_mask.any() else float(np.median(pm))
            drop_ratio = 0.3   # 尖峰后功率中位数降到尖峰前的30%以下 = 掉刀
            for s in np.where(spike)[0]:
                s = int(s)
                before = pm[max(0, s - 30):s]
                after = pm[s + 1:]
                if len(after) < 3:
                    continue
                after_med = float(np.median(after))
                start_break = len(before) < 3  # 断刀发生在文件开头
                if start_break:
                    # 文件开头断刀：不要求尖窗前有切削，只看after是否掉到空切水平
                    if after_med > floor_thr * 2.0:
                        continue
                else:
                    before_med = float(np.median(before))
                    # 尖峰前必须是真切削（比空切底噪高）
                    if before_med < floor_thr * 0.3:
                        continue
                    # 掉底：after 掉到 before 的一半以下，且接近空切水平
                    if after_med > before_med * 0.5:
                        continue
                    if after_med > floor_thr * 2.0:
                        continue
                # 允许尖峰跨2个窗，第3个窗起不能再有新尖峰（小直径严格，大直径放宽）
                if dia < 6.0 and spike[s + 2:].any():
                    continue
                rule_broken = True
                rule_spike_idx = s
                rule_drop_start = s + 1
                rule_spike_kind = _spike_kind_at(s, rm, pmax, rthr, power_rthr)
                rule_alert_mode = "small_spike_drop"
                _before_med_val = None if start_break else float(np.median(before))
                rule_evidence = {
                    "mode": "small_spike_drop",
                    "dia": dia,
                    "floor_thr": float(floor_thr),
                    "rthr": float(rthr),
                    "power_rthr": float(power_rthr),
                    "rms_coef": float(self.rms_coef),
                    "pcoef": float(_pcoef),
                    "normal_p95_rms": float(thr["normal_p95_rms"]),
                    "normal_p95_power": float(thr["normal_p95_power"]),
                    "spike_window": s,
                    "rms_max": float(rm[s]),
                    "power_max": float(pmax[s]),
                    "power_mean": float(pm[s]),
                    "rms_hit": bool(float(rm[s]) > rthr),
                    "power_hit": bool(float(pmax[s]) > power_rthr),
                    "spike_kind": rule_spike_kind,
                    "after_count": int(len(after)),
                    "after_med": float(after_med),
                    "before_med": _before_med_val,
                    "start_break": bool(start_break),
                    "limit_2f": float(floor_thr * 2.0),
                    "no_spike_after_s_plus2": bool(dia < 6.0 and not spike[s + 2:].any()),
                }
                break
            # 大直径(D>=6)实时状态机：按时间顺序检测
            # 切削段中出现冲击尖峰 -> 掉底持续consec窗空切 -> 报警；中途恢复切削则取消
            if not rule_broken and dia >= 6.0:
                consec = int(getattr(self, 'consec_need', 3))
                in_cut = False
                drop_count = 0
                cut_start = -1
                drop_vals = []
                cut_had_spike = False   # 当前切削段是否出现过冲击尖峰
                spike_windows_in_cut = []
                for i in range(len(pm)):
                    if pm[i] > floor_thr:
                        if not in_cut:
                            in_cut = True
                            cut_start = i
                            cut_had_spike = bool(spike[i])
                            spike_windows_in_cut = [i] if spike[i] else []
                        elif spike[i]:
                            cut_had_spike = True
                            spike_windows_in_cut.append(i)
                        drop_count = 0
                        drop_vals = []
                    else:
                        if in_cut:
                            drop_count += 1
                            drop_vals.append(pm[i])
                            if drop_count >= consec:
                                after_med = float(np.median(drop_vals))
                                before_med = float(np.median(pm[max(0, cut_start):i - drop_count + 1]))
                                if (cut_had_spike and
                                    before_med > floor_thr * 0.3 and
                                    after_med < before_med * 0.5 and
                                    after_med < floor_thr * 2.0):
                                    rule_broken = True
                                    rule_spike_idx = i - drop_count + 1
                                    rule_drop_start = rule_spike_idx
                                    rule_alert_mode = "large_state_machine"
                                    rule_spike_kind = "切削段内冲击+连续掉底"
                                    rep_sw = int(spike_windows_in_cut[-1]) if spike_windows_in_cut else -1
                                    rule_evidence = {
                                        "mode": "large_state_machine",
                                        "dia": dia,
                                        "floor_thr": float(floor_thr),
                                        "rthr": float(rthr),
                                        "power_rthr": float(power_rthr),
                                        "rms_coef": float(self.rms_coef),
                                        "pcoef": float(_pcoef),
                                        "drop_start_window": int(rule_spike_idx),
                                        "drop_consec_windows": int(drop_count),
                                        "after_med": float(after_med),
                                        "before_med": float(before_med),
                                        "cut_had_spike": bool(cut_had_spike),
                                        "spike_windows_in_cut": list(spike_windows_in_cut),
                                        "representative_spike_window": rep_sw,
                                    }
                                    if rep_sw >= 0:
                                        rule_evidence["rep_rms_max"] = float(rm[rep_sw])
                                        rule_evidence["rep_power_max"] = float(pmax[rep_sw])
                                        rule_evidence["rep_spike_kind"] = _spike_kind_at(
                                            rep_sw, rm, pmax, rthr, power_rthr)
                                        rule_evidence["rms_hit"] = bool(float(rm[rep_sw]) > rthr)
                                        rule_evidence["power_hit"] = bool(
                                            float(pmax[rep_sw]) > power_rthr)
                                    break
            # 空切门：无冲击尖峰、且power全在空切范围 -> 纯空载，不可能断刀
            if (not spike.any()) and (pm.max() <= floor_thr):
                air_cut_gate = True
                rule_broken = False
                file_broken = False
                rule_evidence = {}
        file_broken = file_broken or rule_broken

        confidence = float(
            max(xgb_prob[both_confirmed]) if both_confirmed.any()
            else (max(xgb_prob[both_broken]) if both_broken.any()
                  else xgb_prob.max()))

        idx_for_times = set(both_break_idx)
        idx_for_times.update(both_confirmed_break_idx)
        for ev in model_drop_gate_evidence:
            cw = ev.get("confirm_window", -1)
            if cw is not None and int(cw) >= 0:
                idx_for_times.add(int(cw))
        idx_for_times.update(consec_alert_run)
        if model_alert_window >= 0:
            idx_for_times.add(model_alert_window)
        if rule_spike_idx >= 0:
            idx_for_times.add(rule_spike_idx)
        if rule_drop_start >= 0:
            idx_for_times.add(rule_drop_start)
        for si in rule_spike_windows[:12]:
            idx_for_times.add(int(si))
        if rule_evidence:
            for key in ("spike_window", "drop_start_window", "representative_spike_window"):
                wi = rule_evidence.get(key, -1)
                if wi is not None and int(wi) >= 0:
                    idx_for_times.add(int(wi))
            for wi in rule_evidence.get("spike_windows_in_cut") or []:
                idx_for_times.add(int(wi))
        window_times = _window_times_for(feats, idx_for_times)

        detail = {
            "total_windows": len(feats),
            "xgb_broken_windows": int(xgb_broken_mask.sum()),
            "if_broken_windows": int(if_broken_mask.sum()),
            "both_broken_windows": int(both_broken.sum()),
            "max_consecutive": max_consec,
            "max_xgb_prob": float(xgb_prob.max()),
            "rule_broken_start": rule_spike_idx,
            "model_file_broken": model_file_broken,
            "rule_broken": rule_broken,
            "air_cut_gate": air_cut_gate,
            "final_broken": file_broken,
            "xgb_break_idx": np.where(xgb_broken_mask)[0].tolist(),
            "if_break_idx": np.where(if_broken_mask)[0].tolist(),
            "both_break_idx": both_break_idx,
            "both_confirmed_windows": int(both_confirmed.sum()),
            "both_confirmed_break_idx": both_confirmed_break_idx,
            "model_drop_gate_evidence": model_drop_gate_evidence,
            "model_drop_lo": model_drop_lo,
            "model_alert_window": model_alert_window,
            "model_alert_prob": model_alert_prob,
            "consec_alert_run": consec_alert_run,
            "consec_alert_start": consec_alert_run[0] if consec_alert_run else -1,
            "rule_drop_start": rule_drop_start,
            "rule_spike_kind": rule_spike_kind,
            "rule_alert_mode": rule_alert_mode,
            "rule_spike_windows": rule_spike_windows,
            "rule_evidence": rule_evidence,
            "window_times": window_times,
        }
        reason_lines, reason_summary = format_detection_reason(
            file_broken, detail, window_times)
        # 模型实际使用的刀具直径（每窗中位数，首窗即代表值）
        used_diameter = float(feats[0].tool_diameter) if feats else 0.0
        return {
            "broken": file_broken, "confidence": confidence, "detail": detail,
            "diameter": used_diameter,
            "reason_lines": reason_lines, "reason_summary": reason_summary,
        }


def batch_test(detector):
    """批量测试dataset目录"""
    files_by_label = scan_dataset("./dataset")
    test_files = (files_by_label["断刀"] + files_by_label["正常"]
                  + files_by_label.get("空切", []))

    tp = fn = tn = fp = 0
    print("="*80)
    print(f"{'文件':<50} {'真实':>6} {'预测':>6} {'XGB':>6} {'IF':>6} {'置信度':>6}")
    print("="*80)
    for path in sorted(test_files):
        name = os.path.basename(path)
        gt = "断刀" if "断刀" in name else "正常"
        try:
            r = detector.predict_file(path)
            pred = "断刀" if r["broken"] else "正常"
            d = r["detail"]
            if gt == "断刀" and pred == "断刀": tp += 1
            elif gt == "断刀" and pred == "正常": fn += 1
            elif gt == "正常" and pred == "正常": tn += 1
            else: fp += 1
            mark = " ✓" if gt == pred else " ✗"
            print(f"{name:<50} {gt:>6} {pred:>6} "
                  f"{d['xgb_broken_windows']:>6} {d['if_broken_windows']:>6} "
                  f"{r['confidence']:>6.2f}{mark}")
        except Exception as e:
            logger.error(f"跳过 {name}: {e}")

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    print("="*80)
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"准确率={acc:.3f} 召回率={rec:.3f} 精确率={prec:.3f}")


def main():
    detector = DualModelDetector()
    if len(sys.argv) > 1:
        # 单文件检测
        path = sys.argv[1]
        r = detector.predict_file(path)
        print(f"文件: {path}")
        print(f"判断: {'断刀' if r['broken'] else '正常'}")
        print(f"置信度: {r['confidence']:.3f}")
        print(f"详情: {r['detail']}")
    else:
        batch_test(detector)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        traceback.print_exc(); sys.exit(1)
