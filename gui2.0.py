# -*- coding: utf-8 -*-
"""
XGBoost 断刀检测 GUI（1.0.24版）
- 特征维度：17维 × 3通道 = 51维
- 后处理四层判据：峰值强度 / 低位阈值 / 整体占比 / 滑窗检查
- 界面简洁：只显示最终判定结果
- 波形图：极细半透明红虚线 + 顶部三角箭头
- 波形窗口支持鼠标拖动
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd
import joblib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QTextEdit, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QEvent
from PyQt6.QtGui import QFont, QColor, QBrush
import matplotlib
from sympy import false

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 日志记录重定向
# ============================================================
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug.log")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout

print(f"=== 日志记录开始，日志文件: {LOG_FILE} ===")
print(f"Python 版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# ============================================================
# 设置中文字体
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 特征提取（17维/通道，共51维）
# ============================================================
WINDOW_SIZE = 60
STEP = 20
LOW_THRESHOLD_RATIO = 0.30

# ---------- 后处理验证参数（四层判据） ----------
POST_LOOKAHEAD = 20000
POST_MIN_LEN = 500
POST_LOW_RATIO = 0.10
POST_LOW_FRACTION_THRESHOLD = 0.90
POST_SEGMENT_SIZE = 500
POST_SEGMENT_LOW_THRESHOLD = 0.70
PEAK_STRENGTH_RATIO = 5.0
BASELINE_EXCLUDE_PERCENTILE = 20
BASELINE_PERCENTILE = 50

STAT_NAMES_BASE = [
    'mean', 'var', 'std', 'kurtosis', 'skewness',
    'peak', 'peak_to_peak', 'form_factor', 'crest_factor',
    'impulse_factor', 'margin_factor'
]

STAT_NAMES_TREND_OLD = [
    'decay_ratio', 'slope', 'peak_to_post_ratio', 'post_peak_mean',
]

STAT_NAMES_TREND_NEW = [
    'post_peak_low_fraction', 'post_peak_max_ratio',
]

STAT_NAMES = STAT_NAMES_BASE + STAT_NAMES_TREND_OLD + STAT_NAMES_TREND_NEW


def extract_time_domain_features(data):
    data = np.array(data, dtype=float)
    n = len(data)
    if n == 0:
        return {s: 0 for s in STAT_NAMES}

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

    half = n // 2
    first_half_mean = np.mean(data[:half]) if half > 0 else 0.0
    second_half_mean = np.mean(data[half:]) if (n - half) > 0 else 0.0
    decay_ratio = first_half_mean / (second_half_mean + 1e-6)

    if n > 1:
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, data, 1)[0]
    else:
        slope = 0.0

    peak_idx = int(np.argmax(np.abs(data)))
    if peak_idx < n - 1:
        post_peak_mean = np.mean(data[peak_idx + 1:])
    else:
        post_peak_mean = abs(data[peak_idx])

    peak_to_post_ratio = peak / (post_peak_mean + 1e-6)

    if (n - peak_idx - 1) >= 3:
        post = data[peak_idx + 1:]
        low_threshold = peak * LOW_THRESHOLD_RATIO
        post_peak_low_fraction = np.mean(post < low_threshold)
        post_peak_max_ratio = np.max(np.abs(post)) / (peak + 1e-6)
    else:
        post_peak_low_fraction = 0.0
        post_peak_max_ratio = 1.0

    return {
        'mean': mean, 'var': var, 'std': std,
        'kurtosis': kurtosis, 'skewness': skewness,
        'peak': peak, 'peak_to_peak': peak_to_peak,
        'form_factor': form_factor, 'crest_factor': crest_factor,
        'impulse_factor': impulse_factor, 'margin_factor': margin_factor,
        'decay_ratio': decay_ratio,
        'slope': slope,
        'peak_to_post_ratio': peak_to_post_ratio,
        'post_peak_mean': post_peak_mean,
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


def create_windows_no_label(df, window_size=WINDOW_SIZE, step=STEP):
    col_map = find_ae_columns(df)
    X, indices = [], []
    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        feature_vector = []
        for feat in ['amp', 'power', 'rms']:
            col = col_map[feat]
            stats = extract_time_domain_features(window[col].values)
            feature_vector.extend([stats[s] for s in STAT_NAMES])
        X.append(feature_vector)
        indices.append(i)
    return np.array(X), np.array(indices)


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


# ============================================================
# 计算文件基线
# ============================================================
def compute_baseline(power_values):
    if len(power_values) == 0:
        return 0.0
    abs_values = np.abs(power_values)
    low_cut = np.percentile(abs_values, BASELINE_EXCLUDE_PERCENTILE)
    active_values = abs_values[abs_values > low_cut]
    if len(active_values) < 10:
        return float(np.median(abs_values))
    return float(np.median(active_values))


# ============================================================
# 后处理验证函数（四层强化）
# ============================================================
def validate_break_postprocess(power_values, peak_global_idx,
                                baseline,
                                lookahead=POST_LOOKAHEAD,
                                min_len=POST_MIN_LEN,
                                low_ratio=POST_LOW_RATIO,
                                low_fraction_threshold=POST_LOW_FRACTION_THRESHOLD,
                                segment_size=POST_SEGMENT_SIZE,
                                segment_low_threshold=POST_SEGMENT_LOW_THRESHOLD,
                                peak_strength_ratio=PEAK_STRENGTH_RATIO):
    """
    四层后处理判据：
    ① 峰值强度：峰值 ≥ 基线 × 5
    ② 峰后低位阈值：低于峰值 10%
    ③ 整体低位占比：≥90%
    ④ 滑窗检查：任意 500 点窗口内 ≥70% 低位
    """
    peak_value = np.abs(power_values[peak_global_idx])
    if peak_value < 1e-6:
        return False

    # 判据 ①
    if baseline > 1e-6 and peak_value < baseline * peak_strength_ratio:
        return False

    start = peak_global_idx + 1
    end = min(peak_global_idx + 1 + lookahead, len(power_values))
    post_values = power_values[start:end]

    if len(post_values) < min_len:
        return False

    low_threshold = peak_value * low_ratio
    low_mask = np.abs(post_values) < low_threshold

    # 判据 ②
    overall_low_fraction = np.mean(low_mask)
    if overall_low_fraction < low_fraction_threshold:
        return False

    # 判据 ③
    n_windows = len(post_values) // segment_size
    if n_windows == 0:
        return True

    for i in range(n_windows):
        seg_mask = low_mask[i * segment_size:(i + 1) * segment_size]
        seg_low_fraction = np.mean(seg_mask)
        if seg_low_fraction < segment_low_threshold:
            return False

    return True


# ============================================================
# 核心检测器类
# ============================================================
class BreakDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = 0.5
        self.is_trained = False
        self.expected_features = None

    def load_model(self, model_path="xgboost_model.pkl",
                   scaler_path="scaler.pkl",
                   threshold_path="threshold.txt"):
        try:
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                return False
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
            self.expected_features = self.model.n_features_in_
            self.is_trained = True
            print(f"✅ XGBoost 模型加载成功，期望特征数: {self.expected_features}")
            return True
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            traceback.print_exc()
            return False

    def predict_file(self, file_path, threshold=None, enable_postprocess=True):
        if not self.is_trained:
            raise RuntimeError("模型未加载")
        if threshold is None:
            threshold = self.threshold

        df = pd.read_csv(file_path)
        X, indices = create_windows_no_label(df, WINDOW_SIZE, STEP)
        if len(X) == 0:
            return False, -1, 0.0
        if X.shape[1] != self.expected_features:
            raise ValueError(f"特征维度不匹配: X有{X.shape[1]}维，模型期望{self.expected_features}维")

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        candidate_mask = proba > threshold
        if not np.any(candidate_mask):
            return False, -1, np.max(proba) if len(proba) > 0 else 0.0

        candidates = np.where(candidate_mask)[0]
        candidates = candidates[np.argsort(proba[candidates])[::-1]]

        if not enable_postprocess:
            max_idx = candidates[0]
            return True, indices[max_idx], float(proba[max_idx])

        col_map = find_ae_columns(df)
        power_values = df[col_map['power']].values
        baseline = compute_baseline(power_values)

        for c_idx in candidates:
            w_start = indices[c_idx]
            window_power = power_values[w_start:w_start + WINDOW_SIZE]
            peak_local = int(np.argmax(np.abs(window_power)))
            peak_global = w_start + peak_local

            if validate_break_postprocess(power_values, peak_global, baseline):
                return True, w_start, float(proba[c_idx])

        return False, -1, np.max(proba) if len(proba) > 0 else 0.0

    def evaluate_file(self, file_path, threshold=None, enable_postprocess=True):
        if not self.is_trained:
            raise RuntimeError("模型未加载")
        if threshold is None:
            threshold = self.threshold

        df = pd.read_csv(file_path)
        if 'label' not in df.columns:
            raise ValueError("文件缺少 'label' 列，无法评估")

        X, indices = create_windows_no_label(df, WINDOW_SIZE, STEP)
        if len(X) == 0:
            return [], [], [], []
        if X.shape[1] != self.expected_features:
            raise ValueError(f"特征维度不匹配: X有{X.shape[1]}维，模型期望{self.expected_features}维")

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        preds = (proba > threshold).astype(int)

        if enable_postprocess and np.any(preds):
            col_map = find_ae_columns(df)
            power_values = df[col_map['power']].values
            baseline = compute_baseline(power_values)

            new_preds = np.zeros_like(preds)
            candidate_idx = np.where(preds == 1)[0]
            for c_idx in candidate_idx:
                w_start = indices[c_idx]
                window_power = power_values[w_start:w_start + WINDOW_SIZE]
                peak_local = int(np.argmax(np.abs(window_power)))
                peak_global = w_start + peak_local
                if validate_break_postprocess(power_values, peak_global, baseline):
                    new_preds[c_idx] = 1
            preds = new_preds

        true_labels = []
        for i in range(0, len(df) - WINDOW_SIZE, STEP):
            window = df.iloc[i:i + WINDOW_SIZE]
            true_labels.append(1 if window['label'].max() == 1 else 0)

        min_len = min(len(true_labels), len(preds))
        return (true_labels[:min_len], preds[:min_len],
                proba[:min_len], indices[:min_len])

    def get_feature_importance(self):
        if self.model is None:
            return None
        return self.model.feature_importances_


# ============================================================
# 评估线程
# ============================================================
class EvaluateThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, detector, folder, threshold, enable_postprocess=True):
        super().__init__()
        self.detector = detector
        self.folder = folder
        self.threshold = threshold
        self.enable_postprocess = enable_postprocess

    def run(self):
        try:
            files = [f for f in os.listdir(self.folder) if f.endswith('.csv')]
            all_true = []
            all_pred = []
            file_results = []
            total = len(files)
            for i, fname in enumerate(files):
                self.log.emit(f"评估 {fname} ...")
                file_path = os.path.join(self.folder, fname)
                try:
                    true, pred, proba, indices = self.detector.evaluate_file(
                        file_path, self.threshold, self.enable_postprocess)
                    all_true.extend(true)
                    all_pred.extend(pred)
                    is_break = any(pred)
                    first_idx = indices[np.argmax(pred)] if is_break else -1
                    file_results.append({
                        'filename': fname,
                        'true_label': 1 if any(true) else 0,
                        'pred_label': 1 if is_break else 0,
                        'break_index': first_idx,
                        'confidence': np.max(proba) if len(proba) > 0 else 0.0
                    })
                except Exception as e:
                    self.log.emit(f"  处理 {fname} 失败: {e}")
                    traceback.print_exc()
                self.progress.emit(int((i+1)/total*100))

            if all_true and all_pred:
                cm = confusion_matrix(all_true, all_pred)
                report = classification_report(all_true, all_pred, target_names=['正常', '断刀'], output_dict=True)
                accuracy = report['accuracy']
                result = {
                    'cm': cm,
                    'report': report,
                    'accuracy': accuracy,
                    'file_results': file_results
                }
                self.finished.emit(result)
            else:
                self.finished.emit(None)
        except Exception as e:
            self.log.emit(f"❌ 评估失败: {e}")
            traceback.print_exc()
            self.finished.emit(None)


# ============================================================
# 预测线程
# ============================================================
class PredictAllThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    file_result = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, detector, folder, files, threshold, enable_postprocess=True):
        super().__init__()
        self.detector = detector
        self.folder = folder
        self.files = files
        self.threshold = threshold
        self.enable_postprocess = enable_postprocess

    def run(self):
        total = len(self.files)
        for i, fname in enumerate(self.files):
            self.log.emit(f"预测 {fname} ...")
            file_path = os.path.join(self.folder, fname)
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"文件不存在: {file_path}")
                df = pd.read_csv(file_path)
                cols_lower = [c.lower() for c in df.columns]
                required = ['amp', 'power', 'rms']
                missing = [r for r in required if r not in cols_lower]
                if missing:
                    raise ValueError(f"缺少列: {missing}")

                is_break, idx, conf = self.detector.predict_file(
                    file_path, self.threshold, self.enable_postprocess)

                result = {
                    'filename': fname,
                    'is_break': is_break,
                    'break_index': idx,
                    'confidence': conf
                }
                self.file_result.emit(result)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                traceback.print_exc()
                self.log.emit(f"❌ 预测失败 {fname}: {error_msg}")
                result = {
                    'filename': fname,
                    'error': error_msg,
                    'is_break': False,
                    'break_index': -1,
                    'confidence': 0.0
                }
                self.file_result.emit(result)
            self.progress.emit(int((i+1)/total*100))
        self.finished.emit()


# ============================================================
# 波形查看对话框（支持鼠标拖动窗口）
# ============================================================
class WaveformDialog(QWidget):
    def __init__(self, file_path, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"波形 - {result['filename']}")
        self.setMinimumSize(800, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        # 拖动状态
        self._drag_pos = None

        layout = QVBoxLayout(self)

        # 顶部信息栏（可拖动）
        if result.get('is_break', False) and result.get('break_index', -1) >= 0:
            moment_text = f"{result['break_index']} ~ {result['break_index'] + WINDOW_SIZE}"
        else:
            moment_text = "无"
        self.info_label = QLabel(
            f"📌 文件: {result['filename']}  |  断刀: {'是' if result.get('is_break', False) else '否'}  "
            f"|  时刻: {moment_text}  "
            f"|  置信度: {result.get('confidence', 0.0):.3f}    （按住此栏可拖动窗口）"
        )
        self.info_label.setFont(QFont("Microsoft YaHei", 10))
        self.info_label.setStyleSheet(
            "background: #f0f4f8; padding: 6px; border-bottom: 1px solid #cfd8dc;"
        )
        self.info_label.setCursor(Qt.CursorShape.SizeAllCursor)
        layout.addWidget(self.info_label)

        self.canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.ax = self.canvas.figure.subplots(3, 1, sharex=True)
        layout.addWidget(self.canvas)

        try:
            df = pd.read_csv(file_path)
            col_map = find_ae_columns(df)
            for i, feat in enumerate(['amp', 'power', 'rms']):
                col = col_map[feat]
                self.ax[i].plot(df[col].values, linewidth=1, zorder=2)
                self.ax[i].set_ylabel(feat)
                self.ax[i].grid(True, alpha=0.3)

            self.ax[-1].set_xlabel("样本索引")

            # 断刀时刻标记：红细虚线 + 顶部三角箭头
            if result.get('is_break', False) and result.get('break_index', -1) >= 0:
                break_start = result['break_index']
                break_end = break_start + WINDOW_SIZE
                for x in (break_start, break_end):
                    for ax in self.ax:
                        ax.axvline(x=x, color='red', linestyle='--',
                                   linewidth=0.8, alpha=0.7, zorder=1)
                    for ax in self.ax:
                        y_min, y_max = ax.get_ylim()
                        ax.plot([x], [y_max], marker='v', color='red',
                                markersize=7, clip_on=False, zorder=3)

            self.canvas.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.ax[0].text(0.5, 0.5, f"绘图错误: {str(e)}", ha='center', va='center')
            self.canvas.draw()
            traceback.print_exc()

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # 在 info 标签和 canvas 上安装事件过滤器
        self.info_label.installEventFilter(self)
        self.canvas.installEventFilter(self)

    # ---------- 窗口级拖动 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (event.globalPosition().toPoint()
                              - self.frameGeometry().topLeft())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

    # ---------- 事件过滤器：让子控件上的拖动也生效 ----------
    def eventFilter(self, obj, event):
        if obj is self.info_label or obj is self.canvas:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_pos = (event.globalPosition().toPoint()
                                      - self.frameGeometry().topLeft())
                    return False

            elif event.type() == QEvent.Type.MouseMove:
                if self._drag_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                    self.move(event.globalPosition().toPoint() - self._drag_pos)
                    return True

            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_pos = None

        return super().eventFilter(obj, event)


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔧 XGBoost 断刀检测系统（1.0.24版）")
        self.setMinimumSize(1200, 750)

        self.detector = BreakDetector()
        self.test_folder = None
        self.eval_folder = None
        self.results = []
        self.eval_results = None
        self.current_threshold = 0.5
        self.thread = None
        self.test_files = []
        self.total_files = 0

        self.init_ui()
        self.apply_style()
        self.load_saved_model()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.log_text.append("⏳ 等待线程结束...")
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        event.accept()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        title = QLabel("🔧 XGBoost 断刀检测系统（1.0.24版）")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #0d47a1; padding: 5px;")
        main_layout.addWidget(title)

        top_layout = QHBoxLayout()

        status_group = QGroupBox("模型状态")
        status_layout = QVBoxLayout()
        self.model_status = QLabel("模型未加载")
        self.model_status.setFont(QFont("Microsoft YaHei", 11))
        status_layout.addWidget(self.model_status)
        status_group.setLayout(status_layout)
        top_layout.addWidget(status_group)

        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()
        self.eval_btn = QPushButton("📊 评估模式 (带标签)")
        self.eval_btn.clicked.connect(self.load_eval_folder)
        self.eval_btn2 = QPushButton("📈 开始评估")
        self.eval_btn2.clicked.connect(self.start_evaluation)
        self.eval_btn2.setEnabled(False)
        self.test_btn = QPushButton("📁 预测模式 (无标签)")
        self.test_btn.clicked.connect(self.load_test_folder)
        self.predict_btn = QPushButton("🔮 开始预测")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        btn_layout.addWidget(self.eval_btn)
        btn_layout.addWidget(self.eval_btn2)
        btn_layout.addWidget(self.test_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_group.setLayout(btn_layout)
        top_layout.addWidget(btn_group)

        th_group = QGroupBox("阈值调节")
        th_layout = QVBoxLayout()
        th_layout.addWidget(QLabel("当前阈值:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        th_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.500")
        self.threshold_label.setFont(QFont("Consolas", 12))
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        th_layout.addWidget(self.threshold_label)
        th_group.setLayout(th_layout)
        top_layout.addWidget(th_group)

        main_layout.addLayout(top_layout)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("background: #fafafa; border: 1px solid #cfd8dc; border-radius: 4px;")
        main_layout.addWidget(self.log_text)

        tab_widget = QTabWidget()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(
            ["文件名", "断刀", "时刻", "置信度", "状态", "说明"]
        )
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.result_table.setColumnWidth(0, 450)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.cellDoubleClicked.connect(self.view_waveform)
        tab_widget.addTab(self.result_table, "预测/评估结果")

        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_widget)
        self.cm_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.cm_ax = self.cm_canvas.figure.add_subplot(111)
        metrics_layout.addWidget(self.cm_canvas)
        self.report_label = QLabel("分类报告：")
        self.report_label.setFont(QFont("Consolas", 10))
        self.report_label.setWordWrap(True)
        metrics_layout.addWidget(self.report_label)
        tab_widget.addTab(self.metrics_widget, "混淆矩阵 & 报告")

        self.imp_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.imp_ax = self.imp_canvas.figure.add_subplot(111)
        tab_widget.addTab(self.imp_canvas, "特征重要性")

        main_layout.addWidget(tab_widget)

        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("💾 导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        export_layout.addStretch()
        export_layout.addWidget(self.export_btn)
        main_layout.addLayout(export_layout)

        self.statusBar().showMessage("就绪")

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f7fa; }
            QGroupBox { border: 1px solid #b0bec5; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { border-radius: 4px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background: #e3f2fd; }
            QPushButton:pressed { background: #bbdefb; }
            QTableWidget { alternate-background-color: #fafafa; background: white; }
            QHeaderView::section { background: #eceff1; padding: 4px; border: 1px solid #cfd8dc; }
            QProgressBar { height: 16px; border-radius: 8px; background: #eceff1; }
            QProgressBar::chunk { background: #1976d2; border-radius: 8px; }
        """)

    def on_threshold_changed(self, value):
        self.current_threshold = value / 100.0
        self.threshold_label.setText(f"{self.current_threshold:.3f}")

    def load_saved_model(self):
        if self.detector.load_model(model_path="xgboost_model.pkl",
                                    scaler_path="scaler.pkl",
                                    threshold_path="threshold.txt"):
            self.model_status.setText(f"✅ XGBoost 模型已加载（{self.detector.expected_features}维）")
            self.model_status.setStyleSheet("color: #2e7d32;")
            self.eval_btn2.setEnabled(True)
            self.predict_btn.setEnabled(False)
            self.current_threshold = self.detector.threshold
            self.threshold_slider.setValue(int(self.current_threshold*100))
            self.threshold_label.setText(f"{self.current_threshold:.3f}")
            self.log_text.append(f"✅ 模型加载成功，特征维度: {self.detector.expected_features}")
            self.plot_importance()
        else:
            self.model_status.setText("❌ 模型加载失败，请检查文件")
            self.model_status.setStyleSheet("color: #c62828;")
            self.log_text.append("❌ 未找到模型文件")
            QMessageBox.warning(self, "警告", "请确保 xgboost_model.pkl, scaler.pkl, threshold.txt 存在。")

    def plot_importance(self):
        importances = self.detector.get_feature_importance()
        if importances is None:
            return
        self.imp_ax.clear()
        idx = np.argsort(importances)[::-1][:20]
        feature_names = []
        for feat in ['amp', 'power', 'rms']:
            for stat in STAT_NAMES:
                feature_names.append(f"{feat}_{stat}")
        names = np.array(feature_names)[idx][::-1]
        vals = importances[idx][::-1]
        self.imp_ax.barh(names, vals)
        self.imp_ax.set_xlabel("重要性")
        self.imp_ax.set_title("Top-20 特征 (XGBoost)")
        self.imp_canvas.figure.tight_layout()
        self.imp_canvas.draw()

    def _cleanup_thread(self):
        if self.thread is not None:
            try:
                for sig_name in ['log', 'progress', 'file_result', 'finished']:
                    try:
                        getattr(self.thread, sig_name).disconnect()
                    except Exception:
                        pass
                self.thread.deleteLater()
            except Exception:
                pass
            self.thread = None

    # ---------- 评估 ----------
    def load_eval_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择评估文件夹 (包含 label 列)", "./")
        if folder:
            self.eval_folder = folder
            self.log_text.append(f"📊 评估文件夹: {folder}")
            self.eval_btn2.setEnabled(self.detector.is_trained)

    def start_evaluation(self):
        if not self.detector.is_trained:
            QMessageBox.warning(self, "警告", "模型未加载")
            return
        if not hasattr(self, 'eval_folder') or not os.path.exists(self.eval_folder):
            QMessageBox.warning(self, "警告", "请先加载评估文件夹")
            return

        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            self.thread = None

        self.eval_btn2.setEnabled(False)
        self.eval_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.log_text.append("📊 开始评估...")
        self.progress.setValue(0)

        self.thread = EvaluateThread(self.detector, self.eval_folder,
                                     self.current_threshold, enable_postprocess=True)
        self.thread.log.connect(self.log_text.append)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_eval_finished)
        self.thread.start()

    def on_eval_finished(self, result):
        self.eval_btn2.setEnabled(True)
        self.eval_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.progress.setValue(100)
        if result is None:
            self.log_text.append("❌ 评估失败")
            QTimer.singleShot(500, self._cleanup_thread)
            return
        self.eval_results = result

        cm = result['cm']
        self.cm_ax.clear()
        self.cm_ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.cm_ax.set_title('混淆矩阵')
        self.cm_ax.set_xlabel('预测标签')
        self.cm_ax.set_ylabel('真实标签')
        self.cm_ax.set_xticks([0, 1])
        self.cm_ax.set_yticks([0, 1])
        self.cm_ax.set_xticklabels(['正常', '断刀'])
        self.cm_ax.set_yticklabels(['正常', '断刀'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.cm_ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > cm.max()/2 else "black")
        self.cm_canvas.figure.tight_layout()
        self.cm_canvas.draw()

        report = result['report']
        report_text = f"准确率: {result['accuracy']:.3f}\n"
        report_text += f"正常: precision={report['正常']['precision']:.3f}, recall={report['正常']['recall']:.3f}, f1={report['正常']['f1-score']:.3f}\n"
        report_text += f"断刀: precision={report['断刀']['precision']:.3f}, recall={report['断刀']['recall']:.3f}, f1={report['断刀']['f1-score']:.3f}"
        self.report_label.setText(report_text)

        self.result_table.setRowCount(0)
        for item in result['file_results']:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            fname_item = QTableWidgetItem(item['filename'])
            fname_item.setToolTip(item['filename'])
            self.result_table.setItem(row, 0, fname_item)
            is_break = item['pred_label'] == 1
            self.result_table.setItem(row, 1, QTableWidgetItem("是" if is_break else "否"))
            self.result_table.setItem(row, 2, QTableWidgetItem(str(item['break_index']) if is_break else "-"))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{item['confidence']:.3f}"))

            true_label = item['true_label']
            pred_label = item['pred_label']
            if true_label == pred_label:
                status = "✓ 正确"
                detail = "正确"
                color = QColor(200, 230, 201)
                text_color = QColor(46, 125, 50)
            else:
                if true_label == 1 and pred_label == 0:
                    status = "✗ 漏报"
                    detail = "漏报 (断刀未检出)"
                    color = QColor(255, 205, 210)
                    text_color = QColor(198, 40, 40)
                else:
                    status = "✗ 误报"
                    detail = "误报 (正常判为断刀)"
                    color = QColor(255, 224, 178)
                    text_color = QColor(255, 152, 0)
            status_item = QTableWidgetItem(status)
            status_item.setBackground(QBrush(color))
            status_item.setForeground(QBrush(text_color))
            self.result_table.setItem(row, 4, status_item)
            self.result_table.setItem(row, 5, QTableWidgetItem(detail))

        self.export_btn.setEnabled(True)

        file_results = result['file_results']
        total_files = len(file_results)
        false_pos = sum(1 for r in file_results if r['true_label']==0 and r['pred_label']==1)
        false_neg = sum(1 for r in file_results if r['true_label']==1 and r['pred_label']==0)
        self.log_text.append(f"✅ 评估完成! 总文件: {total_files}, 误报: {false_pos}, 漏报: {false_neg}")
        self.statusBar().showMessage(f"评估完成: 准确率 {result['accuracy']:.3f}, 误报 {false_pos}, 漏报 {false_neg}")
        QTimer.singleShot(500, self._cleanup_thread)

    # ---------- 预测 ----------
    def load_test_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择预测文件夹 (无 label)", "./")
        if folder:
            self.test_folder = folder
            self.test_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            self.log_text.append(f"📁 预测文件夹: {folder}, 共 {len(self.test_files)} 个CSV")
            self.results = []
            self.result_table.setRowCount(0)
            self.predict_btn.setEnabled(self.detector.is_trained and len(self.test_files) > 0)
            self.export_btn.setEnabled(False)

    def start_prediction(self):
        if not self.detector.is_trained:
            QMessageBox.warning(self, "警告", "模型未加载")
            return
        if not hasattr(self, 'test_folder') or not os.path.exists(self.test_folder):
            QMessageBox.warning(self, "警告", "请先加载预测文件夹")
            return

        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            self.thread = None

        self.predict_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.eval_btn2.setEnabled(False)
        self.log_text.append("🚀 开始预测...")
        self.progress.setValue(0)
        self.results = []
        self.result_table.setRowCount(0)

        self.thread = PredictAllThread(self.detector, self.test_folder, self.test_files,
                                       self.current_threshold, enable_postprocess=True)
        self.thread.log.connect(self.log_text.append)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.file_result.connect(self.on_single_predicted)
        self.thread.finished.connect(self.on_all_predicted)
        self.thread.start()

    def on_single_predicted(self, result):
        if 'error' in result:
            return

        self.results.append(result)
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)

        fname_item = QTableWidgetItem(result.get('filename', ''))
        fname_item.setToolTip(result.get('filename', ''))
        self.result_table.setItem(row, 0, fname_item)

        is_break = result.get('is_break', False)
        self.result_table.setItem(row, 1, QTableWidgetItem("是" if is_break else "否"))
        self.result_table.setItem(row, 2, QTableWidgetItem(
            str(result.get('break_index', -1)) if is_break else "-"))

        conf = result.get('confidence', 0.0)
        self.result_table.setItem(row, 3, QTableWidgetItem(f"{conf:.3f}"))

        status_item = QTableWidgetItem("断刀" if is_break else "正常")
        if is_break:
            status_item.setBackground(QBrush(QColor(255, 205, 210)))
            status_item.setForeground(QBrush(QColor(198, 40, 40)))
        else:
            status_item.setBackground(QBrush(QColor(200, 230, 201)))
            status_item.setForeground(QBrush(QColor(46, 125, 50)))
        self.result_table.setItem(row, 4, status_item)
        self.result_table.setItem(row, 5, QTableWidgetItem(""))

    def on_all_predicted(self):
        self.predict_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.eval_btn2.setEnabled(True)
        self.export_btn.setEnabled(True)

        break_count = sum(1 for r in self.results if r.get('is_break', False))
        self.log_text.append(f"✅ 预测完成! 断刀: {break_count}, 正常: {len(self.results)-break_count}")
        self.statusBar().showMessage(f"预测完成: 断刀 {break_count} 个")

        QTimer.singleShot(500, self._cleanup_thread)

    # ---------- 导出 ----------
    def export_results(self):
        if self.eval_results is not None:
            data = self.eval_results['file_results']
            df = pd.DataFrame(data)
            default_name = "evaluation_results.csv"
        elif hasattr(self, 'results') and self.results:
            data = self.results
            df = pd.DataFrame(data)
            default_name = "prediction_results.csv"
        else:
            QMessageBox.information(self, "提示", "没有结果可导出")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default_name, "CSV Files (*.csv)"
        )
        if file_path:
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "成功", f"结果已保存到:\n{file_path}")

    # ---------- 波形查看 ----------
    def view_waveform(self, row, col):
        if self.eval_results is not None:
            file_results = self.eval_results['file_results']
            if row < len(file_results):
                item = file_results[row]
                fname = item['filename']
                if hasattr(self, 'eval_folder'):
                    file_path = os.path.join(self.eval_folder, fname)
                    if os.path.exists(file_path):
                        result = {
                            'filename': fname,
                            'is_break': item['pred_label'] == 1,
                            'break_index': item['break_index'],
                            'confidence': item['confidence']
                        }
                        dialog = WaveformDialog(file_path, result, self)
                        dialog.show()
                    return
        if hasattr(self, 'results') and row < len(self.results):
            result = self.results[row]
            fname = result.get('filename')
            if fname and hasattr(self, 'test_folder'):
                file_path = os.path.join(self.test_folder, fname)
                if os.path.exists(file_path):
                    dialog = WaveformDialog(file_path, result, self)
                    dialog.show()


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())