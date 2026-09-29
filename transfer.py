# -*- coding: utf-8 -*-
"""
XGBoost 断刀检测模型 -> C# 可部署 ONNX 转换脚本

交付产物:
  - xgboost_model_csharp.onnx   内置 StandardScaler，输入未标准化 51 维特征
  - xgboost_model_csharp_test_vectors.json  C# 端到端验证用黄金测试向量

公开契约:
  输入:  input            float32[batch, 51]  未标准化特征
  输出:  probabilities    float32[batch, 2]   [P(正常), P(断刀)]

C# 判定: probabilities[1] > threshold (默认 0.24，见 threshold.txt)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as rt
import onnxmltools
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.preprocessing import StandardScaler

from XGBoost import (
    STAT_NAMES,
    WINDOW_SIZE,
    STEP,
    extract_time_domain_features,
    find_ae_columns,
)

MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLD_PATH = "threshold.txt"
LABELED_DATA_DIR = "labeled_data"
ONNX_OUTPUT = "xgboost_model_csharp.onnx"
TEST_VECTORS_OUTPUT = "xgboost_model_csharp_test_vectors.json"
TARGET_OPSET = 14
FEATURE_VERSION = "AE-51-V1"
CHANNEL_ORDER = ["amp", "power", "rms"]
CLASS_NAMES = ["正常", "断刀"]
PROBA_TOLERANCE = 1e-4
MIN_VECTORS_PER_CLASS = 2
MAX_VECTORS_PER_CLASS = 5
INPUT_NAME = "input"
OUTPUT_NAME = "probabilities"
SCALER_SUB_OUT = "_embedded_scaler_sub"
SCALER_OUT = "_embedded_scaler_output"


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_threshold(path: str = THRESHOLD_PATH) -> float:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到阈值文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"阈值文件为空: {path}")
    threshold = float(text)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"阈值必须位于 [0,1]，实际为 {threshold}")
    return threshold


def load_model_and_scaler(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
) -> tuple[Any, StandardScaler, int]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"未找到标准化器: {scaler_path}")

    model = None
    load_errors: list[str] = []
    for loader_name, loader in (("joblib", joblib.load), ("pickle", pickle.load)):
        try:
            if loader_name == "pickle":
                with open(model_path, "rb") as f:
                    model = loader(f)
            else:
                model = loader(model_path)
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{loader_name}: {exc}")
    if model is None:
        raise RuntimeError("模型加载失败: " + "; ".join(load_errors))

    scaler = joblib.load(scaler_path)
    if not isinstance(scaler, StandardScaler):
        raise TypeError(f"scaler 类型应为 StandardScaler，实际为 {type(scaler).__name__}")

    if hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)
    elif hasattr(model, "get_booster"):
        n_features = int(model.get_booster().num_features())
    else:
        n_features = len(scaler.mean_)
    if n_features != len(scaler.mean_):
        raise ValueError(
            f"特征维度不一致: model={n_features}, scaler={len(scaler.mean_)}"
        )
    if n_features != 51 or len(scaler.scale_) != 51:
        raise ValueError(
            f"当前特征契约要求 51 维，实际 model={n_features}, "
            f"scaler.scale_={len(scaler.scale_)}"
        )
    return model, scaler, n_features


def convert_xgboost_to_onnx(model: Any, n_features: int) -> onnx.ModelProto:
    initial_types = [(INPUT_NAME, FloatTensorType([None, n_features]))]
    try:
        return onnxmltools.convert_xgboost(
            model,
            initial_types=initial_types,
            target_opset=TARGET_OPSET,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"onnxmltools.convert_xgboost 失败: {exc}，尝试 skl2onnx...")
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType

        return convert_sklearn(
            model,
            initial_types=[(INPUT_NAME, SklFloatTensorType([None, n_features]))],
            target_opset=TARGET_OPSET,
        )


def _resolve_probability_tensor(graph: onnx.GraphProto) -> str:
    for node in graph.node:
        if node.op_type == "ZipMap" and node.input:
            return node.input[0]
    for out in graph.output:
        name = out.name.lower()
        if "prob" in name:
            return out.name
    if len(graph.output) >= 2:
        return graph.output[1].name
    if len(graph.output) == 1:
        return graph.output[0].name
    raise RuntimeError("无法在 ONNX 图中定位概率张量输出")


def fuse_scaler_and_fix_outputs(
    model: onnx.ModelProto,
    scaler: StandardScaler,
) -> onnx.ModelProto:
    graph = model.graph
    if not graph.input:
        raise RuntimeError("ONNX 图缺少输入")
    raw_input_name = graph.input[0].name
    if raw_input_name != INPUT_NAME:
        graph.input[0].name = INPUT_NAME
        raw_input_name = INPUT_NAME

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale_safe = np.where(np.abs(scale) < 1e-12, 1.0, scale).astype(np.float32)
    graph.initializer.extend(
        [
            numpy_helper.from_array(mean, name="scaler_mean"),
            numpy_helper.from_array(scale_safe, name="scaler_scale"),
        ]
    )

    prob_source = _resolve_probability_tensor(graph)
    new_nodes: list[onnx.NodeProto] = []
    new_nodes.append(
        helper.make_node(
            "Sub",
            [INPUT_NAME, "scaler_mean"],
            [SCALER_SUB_OUT],
            name="EmbedStandardScaler_Sub",
        )
    )
    new_nodes.append(
        helper.make_node(
            "Div",
            [SCALER_SUB_OUT, "scaler_scale"],
            [SCALER_OUT],
            name="EmbedStandardScaler_Div",
        )
    )

    for node in graph.node:
        if node.op_type == "ZipMap":
            continue
        cloned = onnx.NodeProto()
        cloned.CopyFrom(node)
        for idx, inp in enumerate(cloned.input):
            if inp == raw_input_name:
                cloned.input[idx] = SCALER_OUT
        new_nodes.append(cloned)

    if prob_source != OUTPUT_NAME:
        new_nodes.append(
            helper.make_node(
                "Identity",
                [prob_source],
                [OUTPUT_NAME],
                name="ProbabilitiesOutput",
            )
        )

    del graph.node[:]
    graph.node.extend(new_nodes)

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                OUTPUT_NAME, TensorProto.FLOAT, [None, 2]
            )
        ]
    )

    if graph.input[0].type.tensor_type.shape.dim:
        graph.input[0].type.tensor_type.shape.dim[1].dim_value = len(mean)

    # XGBoost 图通常只有 ai.onnx.ml 域；Sub/Div 属于默认 ONNX 域。
    if not any(opset.domain == "" for opset in model.opset_import):
        model.opset_import.extend([helper.make_opsetid("", TARGET_OPSET)])

    onnx.checker.check_model(model)
    return model


def _attach_metadata(
    model: onnx.ModelProto,
    threshold: float,
    n_features: int,
) -> None:
    props = {p.key: p.value for p in model.metadata_props}
    updates = {
        "feature_version": FEATURE_VERSION,
        "n_features": str(n_features),
        "threshold": str(threshold),
        "break_probability_threshold": str(threshold),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "window_size": str(WINDOW_SIZE),
        "step": str(STEP),
        "channel_order": ",".join(CHANNEL_ORDER),
        "feature_order": ",".join(STAT_NAMES),
        "class_names": ",".join(CLASS_NAMES),
    }
    for key, value in updates.items():
        if key in props:
            for prop in model.metadata_props:
                if prop.key == key:
                    prop.value = value
                    break
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value


def validate_consistency(
    model: Any,
    scaler: StandardScaler,
    onnx_path: str,
    n_features: int,
    test_vectors: dict[str, Any] | None = None,
) -> None:
    rng = np.random.default_rng(42)
    raw = rng.normal(0.0, 100.0, size=(8, n_features)).astype(np.float32)

    scaled = scaler.transform(raw)
    expected = model.predict_proba(scaled).astype(np.float32)

    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {inp.name: raw for inp in sess.get_inputs()}
    if INPUT_NAME not in inputs and sess.get_inputs():
        inputs = {sess.get_inputs()[0].name: raw}
    outputs = sess.run([OUTPUT_NAME], inputs)
    actual = outputs[0].astype(np.float32)

    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"随机样本最大概率差: {max_diff:.6e}")
    if max_diff > PROBA_TOLERANCE:
        raise RuntimeError(
            f"ONNX 与 sklearn 不一致: max_diff={max_diff} > {PROBA_TOLERANCE}"
        )

    if test_vectors:
        for item in test_vectors.get("vectors", []):
            vec = np.asarray(item["input"], dtype=np.float32).reshape(1, -1)
            out = sess.run([OUTPUT_NAME], {INPUT_NAME: vec})[0][0]
            exp = np.asarray(item["expected_probabilities"], dtype=np.float32)
            diff = float(np.max(np.abs(out - exp)))
            if diff > PROBA_TOLERANCE:
                raise RuntimeError(
                    f"测试向量 {item.get('id')} 不一致: diff={diff}"
                )


def build_test_vectors(
    model: Any,
    scaler: StandardScaler,
    threshold: float,
    labeled_dir: str = LABELED_DATA_DIR,
) -> dict[str, Any]:
    if not os.path.isdir(labeled_dir):
        raise FileNotFoundError(f"未找到标注数据目录: {labeled_dir}")

    buckets: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for fname in sorted(os.listdir(labeled_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(labeled_dir, fname)
        df = pd.read_csv(path)
        if "label" not in df.columns:
            continue
        columns = find_ae_columns(df)
        for start in range(0, len(df) - WINDOW_SIZE + 1, STEP):
            window = df.iloc[start : start + WINDOW_SIZE]
            label = int(window["label"].max() == 1)
            if len(buckets[label]) >= MAX_VECTORS_PER_CLASS:
                continue
            raw_window: dict[str, list[float]] = {}
            features: list[float] = []
            for channel in CHANNEL_ORDER:
                values = window[columns[channel]].astype(float).to_numpy()
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"{fname} 第 {start} 个窗口的 {channel} 含 NaN/Infinity"
                    )
                raw_window[channel] = values.tolist()
                stats = extract_time_domain_features(values)
                features.extend(float(stats[name]) for name in STAT_NAMES)
            raw = np.asarray(features, dtype=np.float32)
            proba = model.predict_proba(scaler.transform(raw.reshape(1, -1)))[0]
            entry = {
                "id": f"{fname}__start_{start}",
                "source_file": fname,
                "window_start": int(start),
                "window_size": WINDOW_SIZE,
                "step": STEP,
                "label": label,
                "class_name": CLASS_NAMES[label],
                "raw_window": raw_window,
                "input": raw.astype(float).tolist(),
                "expected_probabilities": proba.astype(float).tolist(),
                "expected_break_alarm": bool(proba[1] > threshold),
            }
            buckets[label].append(entry)
        if all(len(buckets[label]) >= MAX_VECTORS_PER_CLASS for label in (0, 1)):
            break

    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        if len(buckets[label]) < MIN_VECTORS_PER_CLASS:
            raise RuntimeError(
                f"类别 {label} 测试向量不足 {MIN_VECTORS_PER_CLASS} 条 "
                f"(当前 {len(buckets[label])})"
            )
        selected.extend(buckets[label][:MAX_VECTORS_PER_CLASS])

    return {
        "feature_version": FEATURE_VERSION,
        "n_features": len(scaler.mean_),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "class_names": CLASS_NAMES,
        "channel_order": CHANNEL_ORDER,
        "stat_names": STAT_NAMES,
        "feature_names": [
            f"{channel}_{stat}"
            for channel in CHANNEL_ORDER
            for stat in STAT_NAMES
        ],
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "threshold": threshold,
        "proba_tolerance": PROBA_TOLERANCE,
        "decision_rule": "probabilities[1] > threshold",
        "vectors": selected,
    }


def save_onnx(model: onnx.ModelProto, path: str = ONNX_OUTPUT) -> None:
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def save_test_vectors(payload: dict[str, Any], path: str = TEST_VECTORS_OUTPUT) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    os.chdir(_script_dir())
    print("=" * 60)
    print("XGBoost -> C# ONNX 交付转换")
    print("=" * 60)

    threshold = load_threshold()
    model, scaler, n_features = load_model_and_scaler()
    print(f"模型已加载, n_features={n_features}, threshold={threshold}")

    print("转换 XGBoost 为 ONNX...")
    raw_onnx = convert_xgboost_to_onnx(model, n_features)
    print("嵌入 StandardScaler 并固定 probabilities 输出...")
    fused = fuse_scaler_and_fix_outputs(raw_onnx, scaler)
    _attach_metadata(fused, threshold, n_features)
    save_onnx(fused, ONNX_OUTPUT)
    size_kb = os.path.getsize(ONNX_OUTPUT) / 1024
    print(f"已保存: {ONNX_OUTPUT} ({size_kb:.1f} KB)")

    print("构建黄金测试向量...")
    vectors = build_test_vectors(model, scaler, threshold, LABELED_DATA_DIR)
    save_test_vectors(vectors, TEST_VECTORS_OUTPUT)
    print(f"已保存: {TEST_VECTORS_OUTPUT} ({len(vectors['vectors'])} 条)")

    print("验证 sklearn / ONNX 一致性...")
    validate_consistency(model, scaler, ONNX_OUTPUT, n_features, vectors)
    print("一致性验证通过")

    sess = rt.InferenceSession(ONNX_OUTPUT, providers=["CPUExecutionProvider"])
    sample = np.asarray(vectors["vectors"][0]["input"], dtype=np.float32).reshape(1, -1)
    for _ in range(5):
        sess.run([OUTPUT_NAME], {INPUT_NAME: sample})
    t0 = time.perf_counter()
    for _ in range(100):
        sess.run([OUTPUT_NAME], {INPUT_NAME: sample})
    avg_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"推理基准: {avg_ms:.3f} ms/次 (batch=1)")

    print("=" * 60)
    print("完成")
    print(f"  ONNX: {ONNX_OUTPUT}")
    print(f"  测试向量: {TEST_VECTORS_OUTPUT}")
    print(f"  输入: {INPUT_NAME} float32[batch,{n_features}] (未标准化)")
    print(f"  输出: {OUTPUT_NAME} float32[batch,2]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # noqa: BLE001
        print(f"错误: {err}", file=sys.stderr)
        raise