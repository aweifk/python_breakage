# -*- coding: utf-8 -*-
"""
=============================================================================
文件名称 : ae_gui_qt.py
功能描述 : 工业级断刀检测GUI (PyQt5)
用法: python ae_gui_qt.py
=============================================================================
"""
import os, sys, csv, sqlite3, json
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QTableWidget, QTableWidgetItem,
    QTextEdit, QGroupBox, QStatusBar, QToolBar, QAction, QFileDialog,
    QMessageBox, QHeaderView, QSplitter, QFrame, QGridLayout, QLineEdit, QMenu,
    QProgressDialog, QCheckBox, QDateEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDate
from PyQt5.QtGui import QColor, QFont, QPalette
from gui.tab_history import TabHistory
from gui.tab_stats import TabStats

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(__file__))
from ae_xgboost_isolationforest import DualModelDetector
from ae_window_config import (
    WINDOW_SEC,
    STEP_SEC,
    MIN_EVENTS_PER_WINDOW,
    load_sliding_window_params,
    apply_sliding_params_to_extractor,
)
from gui.plot_helpers import (
    plot_detection_annotations,
    plot_waveform_curves,
    resolve_detection_result,
    detail_to_json,
)

HIST_DETAIL_JSON_ROLE = Qt.UserRole + 1


class _ResultCombo(QComboBox):
    """'实际结果'下拉。故意不设任何 stylesheet，避免 Qt 自绘 popup
    导致悬停项文字与背景同色；配色交给 Fusion 原生(悬停蓝底白字)。"""
    pass


class MainWindow(TabHistory, TabStats, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("微径刀具断刀检测系统 V1.0")
        self.resize(1400, 850)
        self.detector = None
        self.stats = {"total": 0, "broken": 0, "normal": 0}
        self._init_db()

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._load_config()
        self._load_model()
        self._load_history_from_db()
        self._refresh_stats()
        self._count_files()

    def _init_db(self):
        self.db = sqlite3.connect("detect_history.db")
        self.db.execute("""CREATE TABLE IF NOT EXISTS history
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             time TEXT, file TEXT, result TEXT,
             confidence REAL, diameter TEXT,
             gcode_line TEXT, actual_result TEXT DEFAULT '未知')""")
        # 兼容旧库
        try: self.db.execute("ALTER TABLE history ADD COLUMN actual_result TEXT DEFAULT '未知'")
        except: pass
        try: self.db.execute("ALTER TABLE history ADD COLUMN filepath TEXT DEFAULT ''")
        except: pass
        try: self.db.execute("ALTER TABLE history ADD COLUMN detail_json TEXT DEFAULT ''")
        except: pass
        self.db.commit()

    # ========== 菜单栏 ==========
    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("文件(&F)")
        act_open = QAction("打开CSV文件", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        m_file.addAction(act_open)

        act_batch = QAction("批量检测目录", self)
        act_batch.triggered.connect(self.open_dir)
        m_file.addAction(act_batch)
        m_file.addSeparator()
        act_exp = QAction("导出历史记录", self)
        act_exp.triggered.connect(self.export_log)
        m_file.addAction(act_exp)

        act_screenshot = QAction("保存曲线截图", self)
        act_screenshot.triggered.connect(self.save_screenshot)
        m_file.addAction(act_screenshot)

        act_report = QAction("导出检测报告(HTML)", self)
        act_report.triggered.connect(self.export_report)
        m_file.addAction(act_report)
        m_file.addSeparator()
        act_export_log = QAction("导出运行日志", self)
        act_export_log.triggered.connect(self.export_runlog)
        m_file.addAction(act_export_log)
        m_file.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_theme = mb.addMenu("主题(&T)")
        self._theme_actions = {}
        for tname in ["亮色(默认)", "暗色", "护眼绿", "深蓝"]:
            act = QAction(tname, self, checkable=True)
            act.triggered.connect(lambda checked, n=tname: self._apply_theme(n))
            m_theme.addAction(act)
            self._theme_actions[tname] = act
        m_help = mb.addMenu("帮助(&H)")
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    def _about(self):
        QMessageBox.about(self, "关于",
            "微径刀具断刀检测系统 V1.0")

    # ========== 工具栏 ==========
    def _build_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("打开文件", self.open_file)
        tb.addAction("批量检测", self.open_dir)
        tb.addSeparator()

        tb.addWidget(QLabel("  刀具直径: "))
        self.cb_dia = QComboBox()
        self.cb_dia.addItems(["自动", "D0.4", "D0.6", "D0.8", "D1.0",
                              "D1.5", "D2", "D4", "D6", "D8"])
        tb.addWidget(self.cb_dia)
        tb.addSeparator()

        tb.addWidget(QLabel("  上通道:"))
        self.cb_top = QComboBox()
        self.cb_top.addItems(["amp", "power", "rms", "asl", "spindle_load_current"])
        self.cb_top.currentTextChanged.connect(self._replot)
        tb.addWidget(self.cb_top)

        tb.addWidget(QLabel("  下通道:"))
        self.cb_bot = QComboBox()
        self.cb_bot.addItems(["power", "amp", "rms", "asl", "spindle_load_current"])
        self.cb_bot.currentTextChanged.connect(self._replot)
        tb.addWidget(self.cb_bot)
        tb.addSeparator()

        tb.addWidget(QLabel("  置信度阈值: "))
        self.spin_thr = QSpinBox()
        self.spin_thr.setRange(30, 95)
        self.spin_thr.setValue(70)
        self.spin_thr.setSuffix(" %")
        tb.addWidget(self.spin_thr)

        # 状态灯（右侧）
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Expanding, spacer.sizePolicy().Preferred)
        tb.addWidget(spacer)
        self.lbl_light = QLabel("● 待机")
        self.lbl_light.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.lbl_light.setStyleSheet("color: #95a5a6; padding-right: 10px;")
        tb.addWidget(self.lbl_light)

    # ========== 中央Tab ==========
    def _build_central(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_monitor(), "实时监控")
        self.tabs.addTab(self._build_history(), "历史记录")
        self.tabs.addTab(self._build_dataview(), "数据查看")
        self.tabs.addTab(self._build_compare(), "对比分析")
        self.tabs.addTab(self._build_stats(), "统计分析")
        self.tabs.addTab(self._build_settings(), "参数设置")
        self.tabs.addTab(self._build_feature(), "特征分析")
        self.tabs.addTab(self._build_train(), "模型训练")
        self.tabs.addTab(self._build_threshold(), "阈值分析")
        self.tabs.addTab(self._build_log(), "运行日志")
        # 快捷键
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        sc_open = QShortcut(QKeySequence("Ctrl+O"), self)
        sc_open.activated.connect(self.open_file)
        sc_batch = QShortcut(QKeySequence("Ctrl+B"), self)
        sc_batch.activated.connect(self.open_dir)

    def _build_monitor(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        # 左：曲线
        left = QVBoxLayout()
        self.fig = Figure(figsize=(9, 5), dpi=90)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        self.nav = NavToolbar(self.canvas, self)
        left.addWidget(self.nav)
        left.addWidget(self.canvas)
        # 绑定滚轮缩放
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # 右：状态面板
        right = QGroupBox("检测状态")
        rl = QVBoxLayout(right)
        self.lbl_result = QLabel("等待检测")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setFont(QFont("微软雅黑", 24, QFont.Bold))
        self.lbl_result.setStyleSheet("color: #7f8c8d; padding: 15px; font-size: 24pt; font-weight: bold;")
        rl.addWidget(self.lbl_result)

        self.info_table = QTableWidget(9, 2)
        self.info_table.setHorizontalHeaderLabels(["项目", "值"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.info_table.verticalHeader().setVisible(False)
        info_items = ["文件", "刀具直径", "总窗口", "XGB报警窗",
                      "IF报警窗", "双模型共同", "最大连续窗", "最大概率",
                      "断刀G代码行号"]
        for i, k in enumerate(info_items):
            self.info_table.setItem(i, 0, QTableWidgetItem(k))
            self.info_table.setItem(i, 1, QTableWidgetItem("-"))
        rl.addWidget(self.info_table)

        rl.addWidget(QLabel("判定原因"))
        self.txt_reason = QTextEdit()
        self.txt_reason.setReadOnly(True)
        self.txt_reason.setPlaceholderText("打开 CSV 检测后将显示详细判定原因（含报警窗位）")
        self.txt_reason.setMinimumHeight(180)
        self.txt_reason.setStyleSheet(
            "background:#f8fafc; border:1px solid #dcdde1; border-radius:4px; padding:6px;")
        rl.addWidget(self.txt_reason)

        self.lbl_stat = QLabel("总检测: 0  正常: 0  断刀: 0")
        self.lbl_stat.setAlignment(Qt.AlignCenter)
        self.lbl_stat.setStyleSheet("padding: 8px; background: #ecf0f1; border-radius: 4px;")
        rl.addWidget(self.lbl_stat)

        lay.addLayout(left, stretch=3)
        lay.addWidget(right, stretch=1)
        return w

    def _build_dataview(self):
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # 左侧：三个子图 amp/rms/power
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.dv_fig = Figure(figsize=(10, 8), dpi=90)
        self.dv_canvas = FigureCanvas(self.dv_fig)
        self.dv_ax_amp = self.dv_fig.add_subplot(311)
        self.dv_ax_rms = self.dv_fig.add_subplot(312, sharex=self.dv_ax_amp)
        self.dv_ax_pwr = self.dv_fig.add_subplot(313, sharex=self.dv_ax_rms)
        for ax in (self.dv_ax_amp, self.dv_ax_rms, self.dv_ax_pwr):
            ax.grid(True, alpha=0.3)
        self.dv_ax_amp.set_ylabel("amp")
        self.dv_ax_rms.set_ylabel("rms")
        self.dv_ax_pwr.set_ylabel("power")
        self.dv_ax_pwr.set_xlabel("时间 (秒)")
        self.dv_fig.tight_layout()
        ll.addWidget(self.dv_canvas, stretch=1)

        # 底部按钮
        btn_row = QHBoxLayout()
        b_import = QPushButton("导入多组CSV")
        b_import.setStyleSheet("background:#0d9488;color:#fff;padding:8px 20px;border-radius:4px;")
        b_import.clicked.connect(self._dv_import)
        b_clear = QPushButton("清除数据")
        b_clear.setStyleSheet("background:#0d9488;color:#fff;padding:8px 20px;border-radius:4px;")
        b_clear.clicked.connect(self._dv_clear)
        b_leg = QPushButton("显示/隐藏图例")
        b_leg.setStyleSheet("background:#0d9488;color:#fff;padding:8px 20px;border-radius:4px;")
        b_leg.clicked.connect(lambda: self._dv_toggle_legend())
        b_all = QPushButton("全选")
        b_all.setStyleSheet("background:#0d9488;color:#fff;padding:8px 20px;border-radius:4px;")
        b_all.clicked.connect(lambda: self._dv_check_all(True))
        b_none = QPushButton("取消全选")
        b_none.setStyleSheet("background:#0d9488;color:#fff;padding:8px 20px;border-radius:4px;")
        b_none.clicked.connect(lambda: self._dv_check_all(False))
        for b in (b_import, b_clear, b_leg, b_all, b_none):
            btn_row.addWidget(b)
        btn_row.addStretch()
        ll.addLayout(btn_row)

        # 右侧：文件列表
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        rl.addWidget(QLabel("数据类型"))
        self.dv_list = QVBoxLayout()
        self.dv_list.setSpacing(4)
        self.dv_list.setContentsMargins(4, 4, 4, 4)
        self.dv_list.setAlignment(Qt.AlignTop)
        wrap = QWidget()
        wrap.setLayout(self.dv_list)
        from PyQt5.QtWidgets import QScrollArea
        sc = QScrollArea()
        sc.setWidget(wrap)
        sc.setWidgetResizable(True)
        rl.addWidget(sc, stretch=1)
        right.setMinimumWidth(360)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.dv_data = []  # [(checkbox, name, df, fpath, detail_json)]
        self.dv_legend_visible = True
        return w

    def _hist_detail_json(self, row: int) -> str | None:
        item = self.hist_table.item(row, 0)
        if not item:
            return None
        v = item.data(HIST_DETAIL_JSON_ROLE)
        return v if v else None

    def _dv_on_cb_toggled(self, changed_cb):
        if changed_cb.isChecked():
            for cb, _, _, _, _ in self.dv_data:
                if cb is not changed_cb and cb.isChecked():
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
        self._dv_redraw()

    def _dv_import(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择CSV文件", "", "CSV (*.csv)")
        if not paths:
            return
        from ae_batch_pipeline import load_one_csv
        for p in paths:
            try:
                df = load_one_csv(p)
            except Exception as e:
                self._log(f"加载失败 {os.path.basename(p)}: {e}")
                continue
            name = os.path.basename(p)
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda st, c=cb: self._dv_on_cb_toggled(c))
            self.dv_list.addWidget(cb)
            self.dv_data.append((cb, name, df, p, None))
            self._dv_on_cb_toggled(cb)
        self._dv_redraw()

    def _dv_clear(self):
        for cb, _, _, _, _ in self.dv_data:
            cb.setParent(None)
        self.dv_data = []
        for ax in (self.dv_ax_amp, self.dv_ax_rms, self.dv_ax_pwr):
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.dv_ax_amp.set_ylabel("amp")
        self.dv_ax_rms.set_ylabel("rms")
        self.dv_ax_pwr.set_ylabel("power")
        self.dv_ax_pwr.set_xlabel("时间 (秒)")
        self.dv_canvas.draw()

    def _dv_check_all(self, checked):
        for cb, _, _, _, _ in self.dv_data:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._dv_redraw()

    def _dv_toggle_legend(self):
        self.dv_legend_visible = not self.dv_legend_visible
        self._dv_redraw()

    def _dv_redraw(self):
        axis_channels = [
            (self.dv_ax_amp, "amp"),
            (self.dv_ax_rms, "rms"),
            (self.dv_ax_pwr, "power"),
        ]
        active = None
        for entry in self.dv_data:
            if entry[0].isChecked():
                active = entry
                break
        if not active:
            for ax, _ in axis_channels:
                ax.clear()
                ax.grid(True, alpha=0.3)
            self.dv_ax_amp.set_ylabel("amp")
            self.dv_ax_rms.set_ylabel("rms")
            self.dv_ax_pwr.set_ylabel("power")
            self.dv_fig.tight_layout()
            self.dv_canvas.draw()
            return

        cb, name, df, fpath, detail_json = active
        plot_waveform_curves(axis_channels, df, title=name)
        det = getattr(self, "detector", None)
        if det and fpath:
            pred = resolve_detection_result(det, fpath, detail_json)
            plot_detection_annotations(axis_channels, df, det, pred)
        elif det:
            pred = resolve_detection_result(det, None, detail_json)
            if pred:
                plot_detection_annotations(axis_channels, df, det, pred)
            else:
                self._log("数据查看：无文件路径且无可用的历史 detail，仅显示曲线")
        else:
            self._log("数据查看：请先在参数设置加载模型以显示阈值与报警窗")
        self.dv_fig.tight_layout()
        self.dv_canvas.draw()

    def _build_compare(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        # 工具栏
        bar = QHBoxLayout()
        bar.addWidget(QLabel("直径:"))
        self.cb_cmp_dia = QComboBox()
        self.cb_cmp_dia.addItems(["D0.4", "D0.6", "D0.8", "D1", "D1.5", "D2", "D4", "D6", "D8"])
        bar.addWidget(self.cb_cmp_dia)
        b_load = QPushButton("加载对比")
        b_load.setStyleSheet("background:#0d9488;color:#fff;padding:6px 16px;border-radius:4px;")
        b_load.clicked.connect(self._cmp_load)
        bar.addWidget(b_load)
        b_clr = QPushButton("清除")
        b_clr.setStyleSheet("background:#e74c3c;color:#fff;padding:6px 16px;border-radius:4px;")
        b_clr.clicked.connect(self._cmp_clear)
        bar.addWidget(b_clr)
        bar.addStretch()
        lay.addLayout(bar)
        # 滚动区域
        from PyQt5.QtWidgets import QScrollArea
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._cmp_container = QWidget()
        self._cmp_container.setObjectName("cmpContainer")
        self._cmp_container.setStyleSheet("#cmpContainer { background: #fafafa; }")
        self._cmp_layout = QVBoxLayout(self._cmp_container)
        self._cmp_layout.setSpacing(4)
        sc.setWidget(self._cmp_container)
        sc.setMinimumSize(600, 400)
        lay.addWidget(sc, stretch=1)
        return w

    def _cmp_load(self):
        dia = self.cb_cmp_dia.currentText()
        try:
            cur = self.db.execute(
                "SELECT file, filepath, result FROM history WHERE diameter=? ORDER BY file",
                (dia,))
            rows = cur.fetchall()
        except:
            rows = []
        while self._cmp_layout.count():
            it = self._cmp_layout.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if not rows:
            self._cmp_layout.addWidget(QLabel(f"直径 {dia} 无历史记录"))
            return
        # 限制显示前20个，避免卡顿
        self._cmp_rows = rows[:20]
        if len(rows) > 20:
            self._cmp_layout.addWidget(QLabel(f"共{len(rows)}个文件，只显示前20个（滚动查看）"))
        from ae_batch_pipeline import load_one_csv
        ex = self._make_feature_extractor()
        prog = QProgressDialog("正在加载对比数据...", "取消", 0, len(self._cmp_rows), self)
        prog.setWindowTitle("加载中")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()
        row_widget = None
        for i, (fname, fpath, result) in enumerate(self._cmp_rows):
            if i % 4 == 0:
                row_widget = QWidget()
                row_widget.setLayout(QHBoxLayout())
                row_widget.layout().setContentsMargins(0, 0, 0, 0)
                self._cmp_layout.addWidget(row_widget)
            card = QWidget()
            card.setStyleSheet("border:1px solid #ddd;border-radius:4px;padding:4px;")
            cl = QVBoxLayout(card)
            import re as _re
            _m = _re.search(r'(\d{14})[_-](\w+?_[Dd]\d+[._])', fname)
            _short = _m.group(1) + "_" + _m.group(2) if _m else fname[:25]
            title = QLabel(_short)
            title.setStyleSheet(f"font-size:9px;font-weight:bold;color:{'#e74c3c' if result=='断刀' else '#27ae60'};")
            cl.addWidget(title)
            try:
                df = load_one_csv(fpath)
                feats = ex.extract(df)
                pm = [f.power_mean for f in feats]
                rm = [f.rms_max for f in feats]
                fig = Figure(figsize=(3.2, 2.2), dpi=72)
                ax1 = fig.add_subplot(211)
                ax1.plot(rm, color="#e74c3c" if result=="断刀" else "#27ae60", lw=0.6)
                ax1.set_ylabel("rms", fontsize=7)
                ax1.tick_params(labelsize=6)
                ax2 = fig.add_subplot(212)
                ax2.plot(pm, color="#2980b9", lw=0.6)
                ax2.set_ylabel("power", fontsize=7)
                ax2.tick_params(labelsize=6)
                fig.tight_layout()
                cl.addWidget(FigureCanvas(fig))
            except Exception as e:
                cl.addWidget(QLabel(f"加载失败: {e}"))
            row_widget.layout().addWidget(card)
            prog.setValue(i + 1)
            prog.setLabelText(f"正在加载 {i+1}/{len(self._cmp_rows)}")
            QApplication.processEvents()
            if prog.wasCanceled():
                break
        # 底部弹簧，确保滚动条出现
        self._cmp_layout.addStretch()
        prog.close()
        self._cmp_container.adjustSize()

    def _cmp_clear(self):
        while self._cmp_layout.count():
            it = self._cmp_layout.takeAt(0)
            if it.widget(): it.widget().deleteLater()

        """导出统计分析报告为HTML"""
        from datetime import datetime
        cur = self.db.execute("SELECT result, diameter FROM history")
        rows = cur.fetchall()
        total = len(rows)
        cur2 = self.db.execute(
            "SELECT result, actual_result FROM history WHERE actual_result IN ('断刀','正常')")
        labeled = cur2.fetchall()
        tp = sum(1 for m, a in labeled if m=="断刀" and a=="断刀")
        fp = sum(1 for m, a in labeled if m=="断刀" and a=="正常")
        fn = sum(1 for m, a in labeled if m=="正常" and a=="断刀")
        tn = sum(1 for m, a in labeled if m=="正常" and a=="正常")
        n_lab = len(labeled)
        acc = (tp+tn)/n_lab*100 if n_lab else 0
        prec = tp/(tp+fp)*100 if (tp+fp) else 0
        rec = tp/(tp+fn)*100 if (tp+fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        # 按直径
        from collections import Counter
        dia_total = Counter(r[1] for r in rows)
        dia_broken = Counter(r[1] for r in rows if r[0]=="断刀")
        cur_dia = self.db.execute(
            "SELECT diameter, result, actual_result FROM history WHERE actual_result IN ('断刀','正常')")
        dia_fn = Counter(); dia_fp = Counter()
        for d, m, a in cur_dia.fetchall():
            if m == "正常" and a == "断刀": dia_fn[d] += 1
            if m == "断刀" and a == "正常": dia_fp[d] += 1
        # 误报漏报清单
        cur3 = self.db.execute(
            "SELECT time,file,diameter,confidence,result,actual_result,filepath FROM history WHERE actual_result IN ('断刀','正常')")
        _all_lab = cur3.fetchall()
        fps = [r for r in _all_lab if r[4]=="断刀" and r[5]=="正常"]
        fns = [r for r in _all_lab if r[4]=="正常" and r[5]=="断刀"]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = f"""<html><head><meta charset="utf-8"><style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:30px;color:#333}}
h1{{color:#2c3e50;border-bottom:3px solid #0d9488;padding-bottom:10px}}
h2{{color:#0d9488;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th,td{{border:1px solid #ddd;padding:8px 12px;text-align:center}}
th{{background:#0d9488;color:#fff}}
tr:nth-child(even){{background:#f8f9fa}}
.metric{{display:inline-block;background:#f0f9f8;border-radius:8px;padding:15px 25px;margin:5px;text-align:center}}
.metric .val{{font-size:24px;font-weight:bold;color:#0d9488}}
.metric .lbl{{font-size:12px;color:#666}}
.fp{{color:#e67e22}} .fn{{color:#e74c3c}}
</style></head><body>
<h1>断刀检测系统 - 统计分析报告</h1>
<p>生成时间：{now}</p>
<h2>一、总体指标</h2>
<div>
<div class="metric"><div class="val">{total}</div><div class="lbl">总检测数</div></div>
<div class="metric"><div class="val">{n_lab}</div><div class="lbl">已标注</div></div>
<div class="metric"><div class="val">{acc:.1f}%</div><div class="lbl">准确率</div></div>
<div class="metric"><div class="val">{prec:.1f}%</div><div class="lbl">精确率</div></div>
<div class="metric"><div class="val">{rec:.1f}%</div><div class="lbl">召回率</div></div>
<div class="metric"><div class="val">{f1:.1f}%</div><div class="lbl">F1</div></div>
</div>
<h2>二、混淆矩阵</h2>
<table><tr><th></th><th>实际断刀</th><th>实际正常</th></tr>
<tr><th>模型报警</th><td style="background:#d5f5e3">TP={tp}</td><td class="fp" style="background:#f9e79f">FP={fp}</td></tr>
<tr><th>模型正常</th><td class="fn" style="background:#fadbd8">FN={fn}</td><td style="background:#d5f5e3">TN={tn}</td></tr></table>
<h2>三、按刀具直径统计</h2>
<table><tr><th>直径</th><th>总数</th><th>模型报断刀数</th><th>模型报正常数</th><th>漏报</th><th>误报</th><th>模型判定断刀率</th></tr>
"""
        for dia in sorted(dia_total.keys()):
            t = dia_total[dia]; b = dia_broken.get(dia,0)
            _fn = dia_fn.get(dia,0); _fp = dia_fp.get(dia,0)
            fn_style = ' style="color:#e74c3c;font-weight:bold"' if _fn else ''
            fp_style = ' style="color:#e67e22;font-weight:bold"' if _fp else ''
            html += f"<tr><td>{dia}</td><td>{t}</td><td>{b}</td><td>{t-b}</td><td{fn_style}>{_fn}</td><td{fp_style}>{_fp}</td><td>{b/t*100:.1f}%</td></tr>"
        html += "</table>"
        # 辅助：生成rms/power图(base64)
        import base64, io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        def _gen_chart(fpath):
            try:
                import pandas as pd, numpy as np
                df = pd.read_csv(fpath)
                rms_col = next((c for c in df.columns if "rms" in c.lower()), None)
                pw_col = next((c for c in df.columns if "power" in c.lower() or "功率" in c), None)
                if not rms_col: return ""
                fig, axes = plt.subplots(2, 1, figsize=(8, 3), dpi=80)
                axes[0].plot(df[rms_col].values, lw=0.5, color="#27ae60")
                axes[0].set_ylabel("rms", fontsize=8)
                axes[0].tick_params(labelsize=7)
                if pw_col:
                    axes[1].plot(df[pw_col].values, lw=0.5, color="#2980b9")
                axes[1].set_ylabel("power", fontsize=8)
                axes[1].tick_params(labelsize=7)
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                return base64.b64encode(buf.getvalue()).decode()
            except:
                return ""
        html += f"<h2>四、误报清单（FP={fp}）</h2>"
        if fps:
            html += "<table><tr><th>时间</th><th>文件名</th><th>直径</th><th>置信度</th></tr>"
            for r in fps:
                html += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]:.3f}</td></tr>"
            html += "</table>"
            for r in fps:
                html += f"<div style='margin:10px 0;padding:10px;border:1px solid #f9e79f;border-radius:4px'>"
                html += f"<b>{r[1]}</b> 直径:{r[2]} 置信度:{r[3]:.3f}<br>"
                img = _gen_chart(r[6])
                if img:
                    html += f"<img src='data:image/png;base64,{img}' style='width:100%;max-width:800px;margin-top:5px'>"
                html += "</div>"
        else:
            html += "<p>无误报</p>"
        html += f"<h2>五、漏报清单（FN={fn}）</h2>"
        if fns:
            html += "<table><tr><th>时间</th><th>文件名</th><th>直径</th><th>置信度</th></tr>"
            for r in fns:
                html += f"<tr class='fn'><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]:.3f}</td></tr>"
            html += "</table>"
            for r in fns:
                html += f"<div style='margin:10px 0;padding:10px;border:1px solid #fadbd8;border-radius:4px'>"
                html += f"<b style='color:#e74c3c'>{r[1]}</b> 直径:{r[2]} 置信度:{r[3]:.3f}<br>"
                img = _gen_chart(r[6])
                if img:
                    html += f"<img src='data:image/png;base64,{img}' style='width:100%;max-width:800px;margin-top:5px'>"
                html += "</div>"
        else:
            html += "<p>无漏报</p>"
        html += "</body></html>"
        path, _ = QFileDialog.getSaveFileName(self, "保存报告", f"断刀检测统计报告_{datetime.now().strftime('%Y%m%d_%H%M')}.html", "HTML (*.html)")
        if path:
            open(path, "w", encoding="utf-8").write(html)
            self._log(f"统计报告已导出: {path}")
            QMessageBox.information(self, "导出成功", f"报告已保存:\n{path}")

    def _build_settings(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        # 主题选择
        box_theme = QGroupBox("界面主题")
        gt = QHBoxLayout(box_theme)
        gt.addWidget(QLabel("选择主题:"))
        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["亮色(默认)", "暗色", "护眼绿", "深蓝"])
        try:
            import json as _j
            _t = _j.load(open("gui_config.json", encoding="utf-8")).get("theme", "亮色(默认)")
            self.cb_theme.setCurrentText(_t)
        except: pass
        self.cb_theme.currentTextChanged.connect(self._apply_theme)
        gt.addWidget(self.cb_theme)
        gt.addStretch()
        lay.addWidget(box_theme)
        box = QGroupBox("检测参数")
        g = QGridLayout(box)

        g.addWidget(QLabel("窗口大小(秒):"), 0, 0)
        self.spin_win = QDoubleSpinBox()
        self.spin_win.setRange(1, 20)
        self.spin_win.setValue(WINDOW_SEC)
        self.spin_win.setSingleStep(0.5)
        g.addWidget(self.spin_win, 0, 1)

        g.addWidget(QLabel("步长(秒):"), 1, 0)
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.5, 10)
        self.spin_step.setValue(STEP_SEC)
        self.spin_step.setSingleStep(0.5)
        g.addWidget(self.spin_step, 1, 1)

        g.addWidget(QLabel("最少事件/窗:"), 2, 0)
        self.spin_min_events = QSpinBox()
        self.spin_min_events.setRange(1, 5000)
        self.spin_min_events.setValue(MIN_EVENTS_PER_WINDOW)
        g.addWidget(self.spin_min_events, 2, 1)

        g.addWidget(QLabel("连续窗确认数:"), 3, 0)
        self.spin_consec = QSpinBox()
        self.spin_consec.setRange(1, 10)
        self.spin_consec.setValue(3)
        g.addWidget(self.spin_consec, 3, 1)

        g.addWidget(QLabel("置信度阈值(%):"), 4, 0)
        self.spin_conf = QSpinBox()
        self.spin_conf.setRange(30, 95)
        self.spin_conf.setValue(70)
        g.addWidget(self.spin_conf, 4, 1)

        # 物理规则冲击系数
        g.addWidget(QLabel("rms冲击系数:"), 5, 0)
        self.spin_rms_coef = QDoubleSpinBox()
        self.spin_rms_coef.setRange(1.0, 5.0)
        self.spin_rms_coef.setSingleStep(0.1)
        self.spin_rms_coef.setValue(1.2)
        g.addWidget(self.spin_rms_coef, 5, 1)

        g.addWidget(QLabel("D≤1 power系数:"), 6, 0)
        self.spin_pcoef_s = QDoubleSpinBox()
        self.spin_pcoef_s.setRange(1.5, 8.0)
        self.spin_pcoef_s.setSingleStep(0.1)
        self.spin_pcoef_s.setValue(3.0)
        g.addWidget(self.spin_pcoef_s, 6, 1)

        g.addWidget(QLabel("D1.5~4 power系数:"), 7, 0)
        self.spin_pcoef_m = QDoubleSpinBox()
        self.spin_pcoef_m.setRange(1.5, 8.0)
        self.spin_pcoef_m.setSingleStep(0.1)
        self.spin_pcoef_m.setValue(4.0)
        g.addWidget(self.spin_pcoef_m, 7, 1)

        g.addWidget(QLabel("D≥6 power系数:"), 8, 0)
        self.spin_pcoef_l = QDoubleSpinBox()
        self.spin_pcoef_l.setRange(1.5, 8.0)
        self.spin_pcoef_l.setSingleStep(0.1)
        self.spin_pcoef_l.setValue(2.5)
        g.addWidget(self.spin_pcoef_l, 8, 1)

        btn_save = QPushButton("保存并应用")
        btn_save.clicked.connect(self._save_settings)
        g.addWidget(btn_save, 9, 0, 1, 2)
        # 修改即自动保存生效
        self.spin_win.valueChanged.connect(self._auto_save_detect)
        self.spin_step.valueChanged.connect(self._auto_save_detect)
        self.spin_min_events.valueChanged.connect(self._auto_save_detect)
        self.spin_consec.valueChanged.connect(self._auto_save_detect)
        self.spin_conf.valueChanged.connect(self._auto_save_detect)
        self.spin_rms_coef.valueChanged.connect(self._auto_save_detect)
        self.spin_pcoef_s.valueChanged.connect(self._auto_save_detect)
        self.spin_pcoef_m.valueChanged.connect(self._auto_save_detect)
        self.spin_pcoef_l.valueChanged.connect(self._auto_save_detect)

        lay.addWidget(box)
        lay.addStretch()
        return w

    def _apply_theme(self, name):
        self._theme_name = name
        if hasattr(self, "_themes"):
            self._app.setStyleSheet(self._themes.get(name, self._themes["亮色(默认)"]))
        # 勾选菜单
        for tn, act in self._theme_actions.items():
            act.setChecked(tn == name)
        # 刷新下拉框
        if hasattr(self, "cb_theme"):
            self.cb_theme.blockSignals(True)
            self.cb_theme.setCurrentText(name)
            self.cb_theme.blockSignals(False)
        # 重新刷新历史表行颜色
        if hasattr(self, "hist_table"):
            self._refresh_history_colors()
        import json
        cfg = {}
        try: cfg = json.load(open("gui_config.json", encoding="utf-8"))
        except: pass
        cfg["theme"] = name
        json.dump(cfg, open("gui_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def _refresh_history_colors(self):
        """根据当前主题重新设置所有行颜色"""
        for i in range(self.hist_table.rowCount()):
            hid = self.hist_table.item(i, 0).data(Qt.UserRole) if self.hist_table.item(i,0) else None
            model_r = self.hist_table.item(i, 2).text() if self.hist_table.item(i,2) else ""
            actual = self.hist_table.cellWidget(i, 7)
            actual_txt = actual.currentText() if actual else ""
            bg = self._hist_row_color(model_r, actual_txt)
            for c in range(self.hist_table.columnCount()):
                it = self.hist_table.item(i, c)
                if it: it.setBackground(bg)

    def _sliding_params_from_ui(self):
        if hasattr(self, "spin_win"):
            return {
                "window_sec": float(self.spin_win.value()),
                "step_sec": float(self.spin_step.value()),
                "min_events_per_window": int(self.spin_min_events.value()),
            }
        return load_sliding_window_params()

    def _apply_sliding_to_detector(self, params=None):
        p = params or self._sliding_params_from_ui()
        if not self.detector:
            return
        self.detector.window_sec = p["window_sec"]
        self.detector.step_sec = p["step_sec"]
        apply_sliding_params_to_extractor(self.detector.extractor, p)

    def _make_feature_extractor(self, params=None):
        from ae_llm_break_tool_detect import FeatureExtractor
        p = params or self._sliding_params_from_ui()
        return FeatureExtractor(
            p["window_sec"], p["step_sec"], p["min_events_per_window"])

    def _save_settings(self):
        cfg = {}
        try:
            with open("gui_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
        except: pass
        win = self.spin_win.value()
        step = self.spin_step.value()
        min_ev = self.spin_min_events.value()
        consec = self.spin_consec.value()
        conf = self.spin_conf.value()
        cfg["window_sec"] = win
        cfg["step_sec"] = step
        cfg["min_events_per_window"] = min_ev
        cfg["consec"] = consec
        cfg["confidence"] = conf
        with open("gui_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        # 实际生效
        if self.detector:
            self.detector.consec_need = consec
            self.detector.conf_threshold = conf / 100.0
            self._apply_sliding_to_detector()
        self.spin_thr.setValue(conf)
        self._log(
            f"参数已生效: 窗口={win}s 步长={step}s 最少事件={min_ev} "
            f"连续={consec} 置信度={conf}% rms系数={self.spin_rms_coef.value()} "
            f"p系数={self.spin_pcoef_s.value()}/{self.spin_pcoef_m.value()}/{self.spin_pcoef_l.value()}")
        if getattr(self, "cur_path", None):
            self._replot()
        QMessageBox.information(self, "保存成功", "参数已保存并生效")

    def _auto_save_detect(self):
        """检测参数修改时静默保存并生效，不弹窗"""
        if not hasattr(self, 'spin_win'):
            return
        cfg = {}
        try:
            with open("gui_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
        except: pass
        win = self.spin_win.value()
        step = self.spin_step.value()
        min_ev = self.spin_min_events.value()
        consec = self.spin_consec.value()
        conf = self.spin_conf.value()
        rms_c = self.spin_rms_coef.value()
        pcs = self.spin_pcoef_s.value()
        pcm = self.spin_pcoef_m.value()
        pcl = self.spin_pcoef_l.value()
        cfg["window_sec"] = win
        cfg["step_sec"] = step
        cfg["min_events_per_window"] = min_ev
        cfg["consec"] = consec
        cfg["confidence"] = conf
        cfg["rms_coef"] = rms_c
        cfg["pcoef_small"] = pcs
        cfg["pcoef_mid"] = pcm
        cfg["pcoef_large"] = pcl
        with open("gui_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        if self.detector:
            self.detector.consec_need = consec
            self.detector.conf_threshold = conf / 100.0
            self.detector.rms_coef = rms_c
            self.detector.pcoef_small = pcs
            self.detector.pcoef_mid = pcm
            self.detector.pcoef_large = pcl
            self._apply_sliding_to_detector()
        if hasattr(self, 'spin_thr'):
            self.spin_thr.setValue(conf)
        if getattr(self, "cur_path", None):
            self._replot()

    def _auto_save_train(self):
        """训练参数修改时静默保存"""
        if not hasattr(self, 'spin_n_est'):
            return
        cfg = {}
        try:
            with open("gui_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
        except: pass
        cfg["xgb_n_est"] = self.spin_n_est.value()
        cfg["xgb_depth"] = self.spin_depth.value()
        cfg["xgb_lr"] = self.spin_lr.value()
        cfg["if_n_est"] = self.spin_if.value()
        with open("gui_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    def _build_feature(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.feat_table = QTableWidget(0, 2)
        self.feat_table.setHorizontalHeaderLabels(["特征名", "值"])
        self.feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.feat_table)
        return w

    # ========== 模型训练Tab ==========
    def _build_train(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # 目录选择
        dir_box = QGroupBox("数据目录")
        dl = QGridLayout(dir_box)

        self.le_normal = QLabel("./dataset/normal")
        self.le_broken = QLabel("./dataset/broken")
        self.le_aircut = QLabel("./dataset/aircut")
        self.lbl_normal_cnt = QLabel("")
        self.lbl_broken_cnt = QLabel("")
        self.lbl_aircut_cnt = QLabel("")

        for i, (label, var, cnt_lbl) in enumerate([
                ("正常切削目录:", self.le_normal, self.lbl_normal_cnt),
                ("断刀目录:", self.le_broken, self.lbl_broken_cnt),
                ("空切目录(可选):", self.le_aircut, self.lbl_aircut_cnt)]):
            dl.addWidget(QLabel(label), i, 0)
            var.setMaximumWidth(300)
            dl.addWidget(var, i, 1)
            btn = QPushButton("浏览")
            btn.setMaximumWidth(80)
            btn.clicked.connect(lambda checked, v=var: self._browse_dir(v))
            dl.addWidget(btn, i, 2)
            cnt_lbl.setStyleSheet("color: #2980b9; font-weight: bold;")
            dl.addWidget(cnt_lbl, i, 3)
            dl.setColumnStretch(4, 1)

        btn_refresh_cnt = QPushButton("刷新文件数")
        btn_refresh_cnt.setMaximumWidth(120)
        btn_refresh_cnt.clicked.connect(self._count_files)
        dl.addWidget(btn_refresh_cnt, 3, 0, 1, 2)

        # 参数
        par_box = QGroupBox("训练参数")
        pl = QGridLayout(par_box)
        pl.addWidget(QLabel("XGBoost树数:"), 0, 0)
        self.spin_n_est = QSpinBox(); self.spin_n_est.setRange(50, 1000); self.spin_n_est.setValue(200)
        self.spin_n_est.setMaximumWidth(120)
        pl.addWidget(self.spin_n_est, 0, 1)
        pl.addWidget(QLabel("最大深度:"), 0, 2)
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(2, 10); self.spin_depth.setValue(4)
        self.spin_depth.setMaximumWidth(120)
        pl.addWidget(self.spin_depth, 0, 3)
        pl.addWidget(QLabel("学习率:"), 1, 0)
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(0.01, 0.5); self.spin_lr.setValue(0.1); self.spin_lr.setSingleStep(0.05)
        self.spin_lr.setMaximumWidth(120)
        pl.addWidget(self.spin_lr, 1, 1)
        pl.addWidget(QLabel("IF树数:"), 1, 2)
        self.spin_if = QSpinBox(); self.spin_if.setRange(50, 500); self.spin_if.setValue(200)
        self.spin_if.setMaximumWidth(120)
        pl.addWidget(self.spin_if, 1, 3)
        # 训练参数修改即自动保存
        self.spin_n_est.valueChanged.connect(self._auto_save_train)
        self.spin_depth.valueChanged.connect(self._auto_save_train)
        self.spin_lr.valueChanged.connect(self._auto_save_train)
        self.spin_if.valueChanged.connect(self._auto_save_train)

        # 特征选择
        feat_box = QGroupBox("特征选择（训练用哪些传感器通道）")
        fl = QHBoxLayout(feat_box)
        self.chk_ae = QCheckBox("AE声发射(amp/power/频域/形态)")
        self.chk_ae.setChecked(True)
        self.chk_ae.setEnabled(False)  # 核心AE特征必选
        self.chk_spindle = QCheckBox("主轴负载电流")
        self.chk_spindle.setChecked(False)
        self.chk_xyz = QCheckBox("XYZ进给负载电流")
        self.chk_xyz.setChecked(False)
        self.chk_dia = QCheckBox("刀具直径+直径基线")
        self.chk_dia.setChecked(True)
        fl.addWidget(self.chk_ae); fl.addWidget(self.chk_spindle)
        fl.addWidget(self.chk_xyz); fl.addWidget(self.chk_dia)
        fl.addStretch()

        # 上半部分：数据目录(左6) + 训练参数(右4)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(dir_box)
        top.addWidget(par_box)
        top.setSizes([600, 400])
        lay.addWidget(top)
        lay.addWidget(feat_box)

        # 训练按钮
        btn_box = QHBoxLayout()
        self.chk_importance = QCheckBox("训练后计算并显示特征重要性")
        self.chk_importance.setChecked(True)
        btn_box.addWidget(self.chk_importance)
        self.chk_eval = QCheckBox("训练后验证训练集准确率")
        self.chk_eval.setChecked(True)
        btn_box.addWidget(self.chk_eval)
        btn_box.addStretch()
        self.btn_train = QPushButton("🚀 开始训练")
        self.btn_train.setStyleSheet("QPushButton { background: #27ae60; color: white; padding: 8px; font-weight: bold; }")
        self.btn_train.setMaximumWidth(200)
        self.btn_train.clicked.connect(self.start_train)
        btn_box.addWidget(self.btn_train)

        btn_rollback = QPushButton("↩ 回退旧模型")
        btn_rollback.setMaximumWidth(150)
        btn_rollback.clicked.connect(self._rollback_model)
        btn_box.addWidget(btn_rollback)
        btn_box.addStretch()
        lay.addLayout(btn_box)

        # 训练日志 | 特征重要性 | 训练结果
        bottom = QSplitter(Qt.Horizontal)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 9pt; }")
        bottom.addWidget(self.train_log)

        self.fig_imp = Figure(figsize=(5, 4), dpi=90)
        self.canvas_imp = FigureCanvas(self.fig_imp)
        bottom.addWidget(self.canvas_imp)

        # 训练结果表格
        result_box = QGroupBox("训练结果")
        rl = QVBoxLayout(result_box)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["数据集", "准确率", "精确率", "召回率"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rl.addWidget(self.result_table)
        bottom.addWidget(result_box)

        bottom.setSizes([400, 350, 300])
        lay.addWidget(bottom)

        return w

    def _count_files(self):
        import glob
        for d, lbl in [(self.le_normal.text(), self.lbl_normal_cnt),
                       (self.le_broken.text(), self.lbl_broken_cnt),
                       (self.le_aircut.text(), self.lbl_aircut_cnt)]:
            n = len(glob.glob(os.path.join(d, "*.csv")))
            lbl.setText(f"{n}个CSV")

    def _browse_dir(self, label):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d: label.setText(d)

    def _rollback_model(self):
        import glob, shutil
        baks = sorted(glob.glob("model_backups/xgb_*.pkl"))
        if not baks:
            QMessageBox.information(self, "提示", "没有备份的模型")
            return
        latest = baks[-1]
        ret = QMessageBox.question(self, "确认",
            f"回退到最近备份: {os.path.basename(latest)}？")
        if ret != QMessageBox.Yes: return
        shutil.copy2(latest, "output_xgb/xgb_break_tool.pkl")
        self._load_model()
        self._log(f"已回退到 {latest}")

    def _selected_features(self):
        """根据勾选框返回训练用的特征列列表"""
        from ae_rf_train import FEATURE_COLS
        spindle = [c for c in FEATURE_COLS if c.startswith("spindle_current")]
        axis = [c for c in FEATURE_COLS
                if c.startswith(("x_current", "y_current", "z_current"))]
        dia = ["tool_diameter"] + [c for c in FEATURE_COLS
                                    if c.endswith("_dia_ratio")]
        out = []
        for c in FEATURE_COLS:
            if c in spindle and not self.chk_spindle.isChecked(): continue
            if c in axis and not self.chk_xyz.isChecked(): continue
            if c in dia and not self.chk_dia.isChecked(): continue
            out.append(c)
        return out

    def start_train(self):
        # 备份旧模型
        import shutil
        if os.path.exists("output_xgb/xgb_break_tool.pkl"):
            bak_dir = "model_backups"
            os.makedirs(bak_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2("output_xgb/xgb_break_tool.pkl",
                         f"{bak_dir}/xgb_{ts}.pkl")
            self.train_log.append(f"旧模型已备份到 model_backups/xgb_{ts}.pkl")
        self.btn_train.setEnabled(False)
        self.train_log.append("开始训练...")
        wp = self._sliding_params_from_ui()
        self.worker = TrainWorker(
            normal_dir=self.le_normal.text(),
            broken_dir=self.le_broken.text(),
            aircut_dir=self.le_aircut.text(),
            n_est=self.spin_n_est.value(),
            depth=self.spin_depth.value(),
            lr=self.spin_lr.value(),
            if_n=self.spin_if.value(),
            use_features=self._selected_features(),
            window_sec=wp["window_sec"],
            step_sec=wp["step_sec"],
            min_events_per_window=wp["min_events_per_window"],
        )
        self.worker.log_signal.connect(lambda m: self.train_log.append(m))
        self.worker.done_signal.connect(self._train_done)
        self.worker.start()

    def _train_done(self, ok, msg):
        self.btn_train.setEnabled(True)
        self.train_log.append(msg)
        if ok:
            QMessageBox.information(self, "训练完成", "模型已保存并自动加载")
            self._load_model()
            if self.chk_importance.isChecked():
                self._plot_importance()
            else:
                self.train_log.append("已跳过特征重要性计算（未勾选）")
            if self.chk_eval.isChecked():
                self._fill_result_table()
            else:
                self.train_log.append("已跳过训练集准确率验证（未勾选）")
        else:
            QMessageBox.critical(self, "训练失败", msg)

    def _fill_result_table(self):
        """后台重跑训练集准确率，避免卡 UI。"""
        self._res_worker = _ResultEvalWorker(
            xgb_path="output_xgb/xgb_break_tool.pkl",
            normal_dir=self.le_normal.text(),
            broken_dir=self.le_broken.text(),
            extractor=self.detector.extractor,
        )
        self._res_worker.done.connect(self._on_result_done)
        self._res_worker.start()

    def _on_result_done(self, vals):
        if vals is None:
            self.train_log.append("结果计算失败")
            return
        self.result_table.setRowCount(1)
        for c, v in enumerate(vals):
            self.result_table.setItem(0, c, QTableWidgetItem(str(v)))

    def _plot_importance(self):
        try:
            import joblib
            data = joblib.load("output_xgb/xgb_break_tool.pkl")
            model = data["model"]
            feats = data["features"]
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]

            self.fig_imp.clear()
            ax = self.fig_imp.add_subplot(111)
            names = [feats[i] for i in idx]
            vals = [imp[i] for i in idx]
            ax.barh(range(len(idx)), vals, color="#2980b9")
            ax.set_yticks(range(len(idx)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_title("特征重要性Top15")
            self.fig_imp.tight_layout()
            self.canvas_imp.draw()
        except Exception as e:
            self.train_log.append(f"画特征重要性失败: {e}")

    def _build_threshold(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        title = QLabel("按刀具直径的空切/实切阈值上下限（来自 dataset/normal 与 dataset/aircut 统计）")
        title.setStyleSheet("font-weight:bold; font-size:12px;")
        lay.addWidget(title)

        cols = ["直径mm", "空切power_p90", "空切rms_p99",
                "实切power_p10", "实切power_p95", "实切rms_p95",
                "断刀pmax_p95", "断刀rms_p95"]
        self.thr_table = QTableWidget(0, len(cols))
        self.thr_table.setHorizontalHeaderLabels(cols)
        self.thr_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.thr_table)

        row = QHBoxLayout()
        self.thr_info = QLabel("")
        row.addWidget(self.thr_info)
        row.addStretch(1)
        btn = QPushButton("重新统计 normal/aircut/broken")
        btn.clicked.connect(self._rebuild_threshold)
        row.addWidget(btn)
        lay.addLayout(row)

        self._load_threshold_table()
        return w

    def _load_threshold_table(self):
        try:
            table = json.load(open(os.path.join(
                os.path.dirname(__file__), "dia_thresholds.json"), encoding="utf-8"))
        except Exception as e:
            self.thr_info.setText(f"读取阈值表失败: {e}")
            return
        self.thr_table.setRowCount(0)
        for k in sorted(table, key=lambda x: float(x)):
            v = table[k]
            r = self.thr_table.rowCount()
            self.thr_table.insertRow(r)
            vals = [k, v["aircut_p90_power"], v["aircut_p99_rms"],
                    v["normal_p10_power"], v["normal_p95_power"], v["normal_p95_rms"],
                    v.get("broken_p95_pmax", "-"), v.get("broken_p95_rms", "-")]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.thr_table.setItem(r, c, item)
        self.thr_info.setText(f"共 {len(table)} 个直径档；用于冲击线(实切rms_p95×1.5)与掉底线(空切power_p90)")

    def _rebuild_threshold(self):
        """异步重新统计 normal/aircut，带进度条，不卡界面。"""
        self._thr_worker = ThresholdWorker()
        self._thr_prog = QProgressDialog("正在统计阈值…", "取消", 0, 100, self)
        self._thr_prog.setWindowTitle("阈值统计")
        self._thr_prog.setWindowModality(Qt.WindowModal)
        self._thr_prog.setMinimumDuration(0)
        self._thr_prog.setValue(0)

        def on_progress(cur, tot, msg):
            if tot > 0:
                self._thr_prog.setMaximum(tot)
                self._thr_prog.setValue(cur)
            self._thr_prog.setLabelText(msg)

        def on_done(ok, msg):
            self._thr_prog.close()
            if ok:
                self._load_threshold_table()
                QMessageBox.information(self, "阈值分析", "阈值表已重新统计并保存。")
            else:
                QMessageBox.warning(self, "阈值分析", f"统计失败: {msg}")

        self._thr_worker.progress.connect(on_progress)
        self._thr_worker.done.connect(on_done)
        self._thr_worker.start()

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 10pt; }")
        lay.addWidget(self.log_text)
        self._log("系统启动，等待加载模型...")
        return w

    # ========== 状态栏 ==========
    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_left = QLabel("引擎未加载")
        sb.addWidget(self.sb_left)
        self.sb_time = QLabel("")
        sb.addPermanentWidget(self.sb_time)
        self._tick()

    def _tick(self):
        self.sb_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        QTimer.singleShot(1000, self._tick)

    def _log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{t}] {msg}")

    # ========== 模型 ==========
    def _load_model(self):
        try:
            self.detector = DualModelDetector()
            if hasattr(self, "spin_win"):
                self._apply_sliding_to_detector()
            import os.path
            mtime = os.path.getmtime("output_xgb/xgb_break_tool.pkl")
            mt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            _seg = getattr(self.detector, "segmented", False)
            self.sb_left.setText(f"检测引擎就绪 | 更新: {mt}")
            self._log("检测引擎加载完成")
        except Exception as e:
            self.sb_left.setText("引擎加载失败")
            self._log(f"模型加载失败: {e}")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    # ========== 状态灯 ==========
    def set_light(self, state):
        cmap = {"normal": ("#2ecc71", "● 正常"),
                "broken": ("#e74c3c", "● 断刀报警!"),
                "idle": ("#95a5a6", "● 待机"),
                "warn": ("#f39c12", "● 检测中...")}
        color, text = cmap.get(state, cmap["idle"])
        self.lbl_light.setText(text)
        self.lbl_light.setStyleSheet(f"color: {color}; padding-right: 10px; "
                                     f"font-weight: bold;")

    # ========== 文件操作 ==========
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开CSV", "", "CSV文件 (*.csv)")
        if path:
            self.detect_file(path)

    def open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if not d: return
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
        if not files:
            QMessageBox.information(self, "提示", "目录下没有CSV文件")
            return
        self._run_detect(files)

    def _run_detect(self, files):
        self._progress = QProgressDialog("正在检测...", "取消", 0, len(files), self)
        self._progress.setWindowTitle("检测中")
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.show()
        self._batch_records = []   # [(gt, pred)]，gt:1断刀/0正常/None未知
        self._batch_start = __import__("time").time()
        self.worker = DetectWorker(self.detector, files)
        def _prog(cur, tot):
            self._progress.setValue(cur)
            elapsed = __import__("time").time() - self._batch_start
            eta = (elapsed / max(cur, 1)) * (tot - cur)
            fname = os.path.basename(files[cur]) if cur < len(files) else ""
            mm, ss = int(eta)//60, int(eta)%60
            self._progress.setLabelText(f"正在检测 {cur}/{tot}\n当前: {fname[:40]}\n预计剩余: {mm}分{ss:02d}秒")
        self.worker.progress.connect(_prog)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished.connect(self._on_batch_done)
        self.worker.start()

    def _extract_gcode_line(self, fpath, r):
        """从CSV提取断刀时刻G代码行号"""
        try:
            from ae_batch_pipeline import load_one_csv
            df = load_one_csv(fpath)
            if "program_line" not in df.columns or "t" not in df.columns:
                return "-"
            ex = self._make_feature_extractor()
            feats = ex.extract(df)
            detail = r.get("detail", {})
            # 优先物理规则断刀起始窗
            rs = detail.get("rule_broken_start", -1)
            if rs < 0:
                # 模型判断：在XGB断刀窗里找power最大的窗
                idxs = detail.get("xgb_break_idx", [])
                if idxs:
                    valid = [i for i in idxs if i < len(feats)]
                    if valid:
                        rs = max(valid, key=lambda i: getattr(feats[i], "power_max", 0))
            if rs >= 0 and rs < len(feats):
                gt_t = feats[rs].t_start
                mask = (df["t"] >= gt_t - 1.0) & (df["t"] <= gt_t + 1.0)
                if mask.any():
                    return str(int(df.loc[mask, "program_line"].iloc[0]))
                # 兜底：取时间最近的行
                import numpy as _np
                tidx = int(_np.argmin(_np.abs(df["t"].values - gt_t)))
                return str(int(df.iloc[tidx]["program_line"]))
            # 没有断刀窗时取最后一行
            return str(int(df.iloc[-1]["program_line"]))
        except:
            return "-"

    def _on_file_done(self, fpath, r):
        name = os.path.basename(fpath)
        if isinstance(r, dict) and "error" in r:
            self._log(f"跳过 {name}: {r['error']}")
            return
        gcode = self._extract_gcode_line(fpath, r) if r.get("broken") else "-"
        self._add_history(name, r, gcode, fpath)
        if r["broken"]: self.stats["broken"] += 1
        else: self.stats["normal"] += 1
        gt = self._parse_gt(name, r.get("diameter"))
        pred = 1 if r["broken"] else 0
        if gt is not None:
            self._batch_records.append((gt, pred))

    def _on_batch_done(self, broken, total):
        self._progress.close()
        self.stats["total"] = self.stats["broken"] + self.stats["normal"]
        self._update_stat()

        # ---- 批量混淆矩阵与指标（仅统计文件名带已知标签的文件）----
        recs = getattr(self, "_batch_records", [])
        labeled = [(g, p) for g, p in recs if g is not None]
        n_unknown = len(recs) - len(labeled)
        tp = sum(1 for g, p in labeled if g == 1 and p == 1)
        fn = sum(1 for g, p in labeled if g == 1 and p == 0)
        tn = sum(1 for g, p in labeled if g == 0 and p == 0)
        fp = sum(1 for g, p in labeled if g == 0 and p == 1)
        n = len(labeled)
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        self._log(f"批量完成: 断刀{broken}个, 正常{total-broken}个 | "
                  f"TP={tp} FN={fn} TN={tn} FP={fp} "
                  f"acc={acc:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                  f"(已知标签{n}个, 未知{n_unknown}个未计入)")

        detail = (f"共检测 {total} 个文件，其中已知标签 {n} 个"
                  + (f"（{n_unknown} 个未知标签未计入）" if n_unknown else "")
                  + f"\n\n混淆矩阵：TP={tp}  FN={fn}  TN={tn}  FP={fp}"
                  + f"\n准确率 = {acc:.1%}"
                  + f"\n精确率 = {prec:.1%}"
                  + f"\n召回率 = {rec:.1%}"
                  + f"\nF1     = {f1:.1%}")
        QMessageBox.information(self, "批量完成", detail)
        self._refresh_stats()

    def _on_scroll(self, event):
        # 滚轮缩放当前axes
        ax = event.inaxes
        if not ax: return
        factor = 1.2 if event.button == "up" else 1/1.2
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        x0, x1 = xlim; y0, y1 = ylim
        cx = event.xdata; cy = event.ydata
        if cx is None: return
        ax.set_xlim([cx - (cx-x0)*factor, cx + (x1-cx)*factor])
        ax.set_ylim([cy - (cy-y0)*factor, cy + (y1-cy)*factor])
        self.canvas.draw_idle()

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存截图", "", "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"截图已保存: {path}")

    def export_report(self):
        if not self.cur_path:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", "", "HTML (*.html)")
        if not path: return
        name = os.path.basename(self.cur_path)
        rows = []
        for i in range(self.info_table.rowCount()):
            k = self.info_table.item(i, 0).text()
            v = self.info_table.item(i, 1).text()
            rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html = f"""<html><head><meta charset="utf-8"><title>检测报告</title>
        <style>body{{font-family:Microsoft YaHei;padding:30px}}
        table{{border-collapse:collapse;margin:20px 0}}td,th{{border:1px solid #ccc;padding:8px 16px}}
        h1{{color:#2980b9}}</style></head><body>
        <h1>断刀检测报告</h1>
        <p>时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <table>{''.join(rows)}</table></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        self._log(f"报告已导出: {path}")

    def export_runlog(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "", "Text (*.txt)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.run_log.toPlainText())
        self._log(f"日志已导出: {path}")

    def closeEvent(self, event):
        try:
            cfg = {}
            try:
                with open("gui_config.json", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
            cfg.update({
                "w": self.width(),
                "h": self.height(),
                "normal_dir": self.le_normal.text(),
                "broken_dir": self.le_broken.text(),
                "aircut_dir": self.le_aircut.text(),
            })
            with open("gui_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        event.accept()

    def _load_config(self):
        try:
            with open("gui_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            self.resize(cfg.get("w", 1400), cfg.get("h", 850))
            if hasattr(self, "spin_win"):
                self.spin_win.setValue(cfg.get("window_sec", WINDOW_SEC))
                self.spin_step.setValue(cfg.get("step_sec", STEP_SEC))
                me = cfg.get("min_events_per_window", cfg.get("min_events", MIN_EVENTS_PER_WINDOW))
                self.spin_min_events.setValue(int(me))
                self.spin_consec.setValue(cfg.get("consec", 3))
                self.spin_conf.setValue(cfg.get("confidence", 70))
                self.spin_rms_coef.setValue(cfg.get("rms_coef", 1.2))
                self.spin_pcoef_s.setValue(cfg.get("pcoef_small", 3.0))
                self.spin_pcoef_m.setValue(cfg.get("pcoef_mid", 4.0))
                self.spin_pcoef_l.setValue(cfg.get("pcoef_large", 2.5))
            if hasattr(self, "spin_n_est"):
                self.spin_n_est.setValue(cfg.get("xgb_n_est", 200))
                self.spin_depth.setValue(cfg.get("xgb_depth", 4))
                self.spin_lr.setValue(cfg.get("xgb_lr", 0.1))
                self.spin_if.setValue(cfg.get("if_n_est", 200))
        except: pass

    def export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出", "", "CSV (*.csv)")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["时间", "文件名", "判断", "置信度"])
            for i in range(self.hist_table.rowCount()):
                w.writerow([self.hist_table.item(i, c).text() for c in range(5)])
        self._log(f"导出历史记录: {path}")

    def _add_history(self, name, r, gcode="-", filepath=""):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if r["broken"] is None:
            tag = "检测失败"
        elif r["broken"]:
            tag = "断刀"
        else:
            tag = "正常"
        dia = self._get_dia(name)
        # 优先用模型实际使用的直径(来自CSV列/每窗中位数)，避免从文件名误解析
        if isinstance(r, dict) and r.get("diameter"):
            dia = f"D{r['diameter']:g}"
        # 真实标签由文件名自动解析：用文件名解析的直径(而非CSV列，避免写错)判断>6排除
        from ae_batch_pipeline import parse_diameter_from_filename
        _fn_dia = parse_diameter_from_filename(name)
        gt = self._parse_gt(name, _fn_dia)
        if gt == 1:
            actual = "断刀"
        elif gt == 0:
            actual = "正常"
        else:
            actual = None
        actual_disp = actual if actual else "未知"
        dj = detail_to_json(
            r.get("detail") if isinstance(r, dict) else None,
            r.get("reason_lines") if isinstance(r, dict) else None,
        ) or ""
        self.db.execute(
            "INSERT INTO history(time,file,result,confidence,diameter,gcode_line,actual_result,filepath,detail_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (now, name, tag, r["confidence"], dia, gcode, actual, filepath, dj))
        self.db.commit()
        hid = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
        # 加到表格第一行
        self.hist_table.insertRow(0)
        vals = [now, name, tag, f"{r['confidence']:.3f}", dia, gcode, filepath, actual_disp]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            _bg = self._hist_row_color(tag, actual_disp)
            item.setBackground(_bg)
            self.hist_table.setItem(0, c, item)
        self.hist_table.item(0, 0).setData(Qt.UserRole, hid)
        if dj:
            self.hist_table.item(0, 0).setData(HIST_DETAIL_JSON_ROLE, dj)
        # 最后一列加下拉（未知=不参与统计）
        combo = _ResultCombo()
        combo.addItems(["未知", "断刀", "正常"])
        combo.setCurrentText(actual_disp)
        combo.currentTextChanged.connect(self._on_actual_changed)
        self.hist_table.setCellWidget(0, 7, combo)
        # 报警声音
        if r["broken"]:
            QApplication.beep()

    def _load_history_from_db(self):
        cur = self.db.execute(
            "SELECT id,time,file,result,confidence,diameter,gcode_line,actual_result,filepath,detail_json "
            "FROM history ORDER BY id DESC LIMIT 500")
        for row in cur.fetchall():
            hid, *rest = row
            detail_json = rest[-1] if len(rest) > 8 else ""
            rest = rest[:8]
            # rest顺序: time,file,result,confidence,diameter,gcode,actual_result,filepath
            # 表格顺序: time,file,result,confidence,diameter,gcode,filepath,actual_result
            # 需要交换 actual_result 和 filepath
            rest_list = list(rest)
            r = rest_list[:6] + [rest_list[7], rest_list[6]]  # 后两列交换
            if r[7] is None: r[7] = "未知"
            self.hist_table.insertRow(0)
            for c, v in enumerate(r):
                if c == 3:  # 置信度列
                    try:
                        v = f"{float(v):.3f}"
                    except:
                        pass
                item = QTableWidgetItem(str(v))
                _bg = self._hist_row_color(row[3], str(r[7]))
                item.setBackground(_bg)
                self.hist_table.setItem(0, c, item)
            self.hist_table.item(0, 0).setData(Qt.UserRole, hid)
            if detail_json:
                self.hist_table.item(0, 0).setData(HIST_DETAIL_JSON_ROLE, detail_json)
            combo = _ResultCombo()
            combo.addItems(["未知", "断刀", "正常"])
            combo.setCurrentText(str(r[7]))
            combo.currentTextChanged.connect(self._on_actual_changed)
            self.hist_table.setCellWidget(0, 7, combo)

    def _on_actual_changed(self, val):
        """下拉框改实际结果时，遍历表格找到所属行（筛选后也正确）"""
        combo = self.sender()
        row = -1
        for r in range(self.hist_table.rowCount()):
            if self.hist_table.cellWidget(r, 7) is combo:
                row = r
                break
        if row < 0:
            return
        self._update_actual(row, val)

        rows = sorted(set(i.row() for i in self.hist_table.selectedItems()), reverse=True)
        if not rows:
            QMessageBox.warning(self, "提示", "请先选中要删除的行")
            return
        ret = QMessageBox.question(self, "确认", f"删除选中的 {len(rows)} 条记录？")
        if ret != QMessageBox.Yes: return
        for row in rows:
            hid_item = self.hist_table.item(row, 0)
            if hid_item:
                hid = hid_item.data(Qt.UserRole)
                if hid:
                    self.db.execute("DELETE FROM history WHERE id=?", (hid,))
            self.hist_table.removeRow(row)
        self.db.commit()
        self._log(f"删除了 {len(rows)} 条记录")

    def _update_actual(self, row, val):
        file_item = self.hist_table.item(row, 1)
        if not file_item: return
        fname = file_item.text()
        db_val = None if val == "未知" else val
        self.db.execute("UPDATE history SET actual_result=? WHERE file=?", (db_val, fname))
        self.db.commit()
        self._log(f"标注 {fname} → {val}")
        # 更新行背景色
        model_result = self.hist_table.item(row, 2).text() if self.hist_table.item(row, 2) else ""
        bg = self._hist_row_color(model_result, val)
        for c in range(self.hist_table.columnCount()):
            it = self.hist_table.item(row, c)
            if it: it.setBackground(bg)
        # 刷新统计页
        try:
            self._refresh_stats()
        except:
            pass

    def _update_stat(self):
        self.lbl_stat.setText(
            f"总检测: {self.stats['total']}  "
            f"正常: {self.stats['normal']}  断刀: {self.stats['broken']}")

    # ========== 检测 ==========
    def detect_file(self, path):
        if not self.detector: return
        self.set_light("warn")
        if hasattr(self, "txt_reason"):
            self.txt_reason.setPlainText("检测中，请稍候…")
        # 圆形进度条（busy 状态，无取消按钮）
        self._single_progress = QProgressDialog("正在检测，请稍候...", None, 0, 0, self)
        self._single_progress.setWindowTitle("检测中")
        self._single_progress.setWindowModality(Qt.WindowModal)
        self._single_progress.setMinimumDuration(0)
        self._single_progress.setCancelButton(None)
        self._single_progress.show()
        QApplication.processEvents()
        # 启动异步检测
        self._single_worker = SingleDetectWorker(self.detector, path)
        self._single_worker.done.connect(self._on_single_done)
        self._single_worker.start()

    def _on_single_done(self, path, r, error):
        self._single_progress.close()
        if error:
            self._log(f"检测失败: {error}")
            QMessageBox.critical(self, "错误", error)
            self.set_light("idle")
            return
        name = os.path.basename(path)
        d = r.get("detail") or {}
        dia_disp = self._get_dia(name)
        if r.get("diameter"):
            dia_disp = f"D{r['diameter']:g}"
        info_vals = [
            name[:30], dia_disp,
            str(d.get("total_windows", "-")),
            str(d.get("xgb_broken_windows", "-")),
            str(d.get("if_broken_windows", "-")),
            str(d.get("both_broken_windows", "-")),
            str(d.get("max_consecutive", "-")),
            f"{d.get('max_xgb_prob', 0):.3f}" if d else "-",
        ]
        for i, v in enumerate(info_vals):
            self.info_table.setItem(i, 1, QTableWidgetItem(v))
        reason_lines = r.get("reason_lines")
        if not reason_lines:
            fallback = r.get("reason") or r.get("reason_summary") or "暂无判定原因"
            reason_lines = [fallback]
        self.txt_reason.setPlainText("\n".join(reason_lines))
        if r["broken"]:
            self.lbl_result.setText("⚠ 断刀")
            self.lbl_result.setStyleSheet("color: #dc2626; padding: 15px; font-size: 24pt; font-weight: bold;")
            self.set_light("broken")
            self._log(f"[报警] {name} 置信度={r['confidence']:.3f} {r.get('reason_summary', '')}")
        else:
            self.lbl_result.setText("✓ 正常")
            self.lbl_result.setStyleSheet("color: #16a34a; padding: 15px; font-size: 24pt; font-weight: bold;")
            self.set_light("normal")
            self._log(f"[正常] {name} 置信度={r['confidence']:.3f} {r.get('reason_summary', '')}")
        self.stats["total"] += 1
        if r["broken"]: self.stats["broken"] += 1
        else: self.stats["normal"] += 1
        self._update_stat()
        self._last_plot_result = r
        program_line = self._plot_file(path, r)
        self.info_table.setItem(8, 1, QTableWidgetItem(str(program_line)))
        self._add_history(name, r, str(program_line), self.cur_path)
        self._show_features(path)

    def _get_dia(self, name):
        # 复用与训练/推理一致的解析：D15->1.5, D06->0.6, D8->8
        from ae_batch_pipeline import parse_diameter_from_filename
        dia = parse_diameter_from_filename(name)
        if dia and dia > 0:
            return f"D{dia:g}"
        return "-"

    @staticmethod
    def _parse_gt(name, dia=None):
        """从文件名自动解析真实标签(GT)：
           含 实切 / RealCut    -> 0 (正常)
           含 断刀 / ToolBreak  -> 1 (断刀)，但若刀具直径 >6mm 则判定为
                                   无真断刀实验，返回 None (不参与指标统计)
           其他(含空切/未命名)   -> None
        """
        n = name.lower()
        is_cut = ("实切" in name) or ("realcut" in n)
        is_break = ("断刀" in name) or ("toolbreak" in n)
        if is_cut:
            return 0
        if is_break:
            # 大于8mm刀具做不出断刀实验，此类 ToolBreak 文件名不代表真断刀
            if dia is not None and dia > 8.0:
                return None
            return 1
        return None

    @staticmethod
    def _style_actual_combo(combo):
        """直接给'实际结果'下拉弹出视图设样式表：
        未悬停白底深字；悬停/选中蓝底白字，保证任何状态下文字都可见。"""
        view = combo.view()
        view.setStyleSheet(
            "QListView{background:white;color:#2c3e50;border:1px solid #dcdde1;}"
            "QListView::item{background:white;color:#2c3e50;padding:4px 10px;min-height:22px;}"
            "QListView::item:hover,QListView::item:selected{"
            "background:#3498db;color:white;}"
        )

    def _replot(self):
        if hasattr(self, "cur_path") and self.cur_path:
            try:
                last_r = getattr(self, "_last_plot_result", None)
                self._plot_file(self.cur_path, last_r)
            except: pass

    def _plot_file(self, path, result=None):
        from ae_batch_pipeline import load_one_csv
        df = load_one_csv(path)
        self.cur_path = path
        t = df["t"].values if "t" in df.columns else np.arange(len(df))

        def get_col(name):
            if name in df.columns:
                return df[name].values
            return np.zeros(len(df))

        ch_top = self.cb_top.currentText()
        ch_bot = self.cb_bot.currentText()

        self.ax1.clear()
        self.ax1.plot(t, get_col(ch_top), linewidth=0.5, color="#2980b9")
        self.ax1.set_ylabel(ch_top)
        self.ax1.set_title(os.path.basename(path))
        self.ax1.grid(alpha=0.3)

        self.ax2.clear()
        self.ax2.plot(t, get_col(ch_bot), linewidth=0.5, color="#27ae60")
        self.ax2.set_ylabel(ch_bot)
        self.ax2.set_xlabel("时间 (秒)")
        self.ax2.grid(alpha=0.3)

        ann = plot_detection_annotations(
            [(self.ax1, ch_top), (self.ax2, ch_bot)],
            df, self.detector, result,
        )
        program_line = ann.get("program_line", "-")
        self.canvas.draw()
        # 重置工具栏视图历史，将当前自适应视图设为 home（reset 按钮可恢复）
        try:
            self.nav._views.clear()
            self.nav._positions.clear()
            self.nav.push_current()
        except Exception:
            pass
        return program_line

    def _show_features(self, path):
        from ae_batch_pipeline import load_one_csv
        self.feat_table.setRowCount(0)
        df = load_one_csv(path)
        feats = self.detector.extractor.extract(df)
        if not feats: return
        f = feats[len(feats)//2]
        for c in self.detector.features:
            v = getattr(f, c, 0)
            try: v = f"{float(v):.4f}"
            except: v = str(v)
            row = self.feat_table.rowCount()
            self.feat_table.insertRow(row)
            self.feat_table.setItem(row, 0, QTableWidgetItem(c))
            self.feat_table.setItem(row, 1, QTableWidgetItem(v))


class _ResultEvalWorker(QThread):
    done = pyqtSignal(object)   # list[str] or None

    def __init__(self, xgb_path, normal_dir, broken_dir, extractor):
        super().__init__()
        self.xgb_path = xgb_path
        self.normal_dir = normal_dir
        self.broken_dir = broken_dir
        self.extractor = extractor

    def run(self):
        try:
            import joblib, glob, os
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            from ae_batch_pipeline import load_one_csv
            data = joblib.load(self.xgb_path)
            model, feats = data["model"], data["features"]
            Xs, ys = [], []
            for d, label in [(self.normal_dir, 0), (self.broken_dir, 1)]:
                for f in glob.glob(os.path.join(d, "*.csv")):
                    try:
                        df = load_one_csv(f)
                        for win in self.extractor.extract(df):
                            Xs.append([getattr(win, c, 0.0) or 0.0 for c in feats])
                            ys.append(label)
                    except Exception:
                        pass
            X = np.array(Xs); y = np.array(ys)
            pred = model.predict(X)
            self.done.emit(["训练集", f"{accuracy_score(y, pred)*100:.1f}%",
                            f"{precision_score(y, pred)*100:.1f}%",
                            f"{recall_score(y, pred)*100:.1f}%"])
        except Exception:
            self.done.emit(None)


class ThresholdWorker(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, message
    done = pyqtSignal(bool, str)           # ok, message

    def run(self):
        try:
            import glob as _glob
            import numpy as _np
            from collections import defaultdict
            from ae_batch_pipeline import load_one_csv
            from ae_window_config import load_sliding_window_params
            wp = load_sliding_window_params()
            from ae_llm_break_tool_detect import FeatureExtractor
            ex = FeatureExtractor(
                wp["window_sec"], wp["step_sec"], wp["min_events_per_window"])
            root = os.path.join(os.path.dirname(__file__), "dataset")
            files = []
            for sub in ("normal", "aircut", "broken"):
                files += _glob.glob(os.path.join(root, sub, "*.csv"))
            total = len(files)
            self.progress.emit(0, total, "开始统计阈值…")

            def _collect(flist):
                # 文件等权：先每个文件取分位，再对 N 个文件平均
                d = defaultdict(lambda: {"p90": [], "p99": [], "p10": [], "p95": [], "r95": [], "pmax95": []})
                for i, f in enumerate(flist, 1):
                    try:
                        df = load_one_csv(f)
                        # 过滤直径异常行：只保留占多数的主直径数据（避免前几行写错直径污染统计）
                        dia = float(df["tool_diameter"].median())
                        if dia > 0:
                            df = df[df["tool_diameter"].round(1) == round(dia, 1)]
                        fe = ex.extract(df)
                    except Exception:
                        continue
                    if dia <= 0:
                        self.progress.emit(i, total, os.path.basename(f) + " (跳过)")
                        continue
                    pm = [w.power_mean for w in fe]
                    pmax = [getattr(w, "power_max", w.power_mean) for w in fe]
                    rm = [w.rms_max for w in fe]
                    if not pm:
                        self.progress.emit(i, total, os.path.basename(f) + " (空)")
                        continue
                    k = round(dia, 1)
                    d[k]["p90"].append(float(_np.percentile(pm, 90)))
                    d[k]["p99"].append(float(_np.percentile(rm, 99)))
                    d[k]["p10"].append(float(_np.percentile(pm, 10)))
                    d[k]["p95"].append(float(_np.percentile(pm, 95)))
                    d[k]["r95"].append(float(_np.percentile(rm, 95)))
                    d[k]["pmax95"].append(float(_np.percentile(pmax, 95)))
                    self.progress.emit(i, total, os.path.basename(f))
                return d

            def _in_sub(f, sub):
                return os.path.basename(os.path.dirname(f)) == sub
            normal = _collect([f for f in files if _in_sub(f, "normal")])
            aircut = _collect([f for f in files if _in_sub(f, "aircut")])
            broken = _collect([f for f in files if _in_sub(f, "broken")])
            table = {}
            for k in sorted(set(list(normal) + list(aircut) + list(broken))):
                ac = aircut.get(k, {}) or {}
                no = normal.get(k, {}) or {}
                br = broken.get(k, {}) or {}
                def _avg(x): return round(float(_np.mean(x)), 1) if x else 0
                def _avgr(x): return round(float(_np.mean(x)), 2) if x else 0
                table[str(k)] = {
                    "aircut_p90_power": _avg(ac.get("p90", [])),
                    "aircut_p99_rms": _avgr(ac.get("p99", [])),
                    "normal_p10_power": _avg(no.get("p10", [])),
                    "normal_p95_power": _avg(no.get("p95", [])),
                    "normal_p95_rms": _avgr(no.get("r95", [])),
                    "broken_p95_pmax": _avg(br.get("pmax95", [])),
                    "broken_p95_rms": _avgr(br.get("r95", [])),
                }
            json.dump(table, open(os.path.join(os.path.dirname(__file__),
                      "dia_thresholds.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            self.done.emit(True, f"完成，共 {len(table)} 个直径档")
        except Exception as e:
            self.done.emit(False, str(e))


class SingleDetectWorker(QThread):
    """单文件异步检测"""
    done = pyqtSignal(str, object, str)  # path, result, error
    def __init__(self, detector, path):
        super().__init__()
        self.detector = detector
        self.path = path
    def run(self):
        try:
            r = self.detector.predict_file(self.path)
            self.done.emit(self.path, r, "")
        except Exception as e:
            self.done.emit(self.path, None, str(e))


class DetectWorker(QThread):
    progress = pyqtSignal(int, int)       # current, total
    file_done = pyqtSignal(str, object)    # filename, result
    finished = pyqtSignal(int, int)       # broken_count, total

    def __init__(self, detector, files):
        super().__init__()
        self.detector = detector
        self.files = files

    def run(self):
        broken = 0
        for i, f in enumerate(self.files):
            try:
                if "空切" in os.path.basename(f) or "aircut" in f.lower():
                    r = {"broken": False, "confidence": 0.0, "note": "空切"}
                else:
                    r = self.detector.predict_file(f)
                self.file_done.emit(f, r)
                if r["broken"]: broken += 1
            except Exception as e:
                self.file_done.emit(f, {"broken": None, "confidence": 0, "error": str(e)})
            self.progress.emit(i+1, len(self.files))
        self.finished.emit(broken, len(self.files))


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, normal_dir, broken_dir, aircut_dir, n_est, depth, lr, if_n,
                 use_features=None, window_sec=None, step_sec=None,
                 min_events_per_window=None):
        super().__init__()
        self.normal_dir = normal_dir
        self.broken_dir = broken_dir
        self.aircut_dir = aircut_dir
        self.n_est = n_est
        self.depth = depth
        self.lr = lr
        self.if_n = if_n
        self.use_features = use_features  # 训练用特征列；None=全部
        wp = load_sliding_window_params()
        self.window_sec = window_sec if window_sec is not None else wp["window_sec"]
        self.step_sec = step_sec if step_sec is not None else wp["step_sec"]
        self.min_events_per_window = (
            min_events_per_window if min_events_per_window is not None
            else wp["min_events_per_window"])

    def run(self):
        try:
            import joblib
            from ae_llm_break_tool_detect import FeatureExtractor
            from ae_batch_pipeline import load_one_csv
            from ae_rf_train import FEATURE_COLS
            from xgboost import XGBClassifier
            from sklearn.ensemble import IsolationForest
            import numpy as np

            COLS = self.use_features or FEATURE_COLS
            self.log_signal.emit(f"启用特征 {len(COLS)}/{len(FEATURE_COLS)} 维")
            self.log_signal.emit(
                f"滑窗: {self.window_sec}s / 步长 {self.step_sec}s / "
                f"最少事件 {self.min_events_per_window}")

            ex = FeatureExtractor(
                self.window_sec, self.step_sec, self.min_events_per_window)
            from ae_batch_pipeline import (
                build_dia_baseline, apply_dia_baseline)

            def load_file_feats(path):
                """加载单个CSV，返回WindowFeature列表（含power_drop_ratio修正）"""
                df = load_one_csv(path)
                feats = ex.extract(df)
                for i, ft in enumerate(feats):
                    if i == 0: ft.power_drop_ratio = 1.0
                    else:
                        p = feats[i-1].power_mean
                        ft.power_drop_ratio = ft.power_mean/p if p>1e-6 else 1.0
                return feats

            def collect(d):
                """加载目录，返回 WindowFeature 列表（未打标签）"""
                feats_all = []
                if not os.path.isdir(d): return feats_all
                for f in sorted(os.listdir(d)):
                    if not f.endswith(".csv"): continue
                    try:
                        self.log_signal.emit(f"  加载 {f}")
                        feats_all.extend(load_file_feats(os.path.join(d, f)))
                    except Exception as e:
                        self.log_signal.emit(f"  跳过 {f}: {e}")
                return feats_all

            def segment_broken(feats):
                """窗级分段打标：
                - 尾部连续空切底噪(power<50)段 = 断刀(1)
                - 之前的正常切削段 = 正常(0)（断刀前是正常加工）
                - 断刀尖峰：floor起点往前2窗也算断刀
                返回 [(feat,label)]；无切削过程(纯空切)返回空"""
                pm = [ft.power_mean for ft in feats]
                n = len(pm)
                j = n
                while j > 0 and pm[j-1] < 50:   # 尾部连续 floor
                    j -= 1
                if j == 0:
                    return [], "纯空切无断刀过程,丢弃"
                if not any(v > 200 for v in pm[:j]):
                    return [], "断刀段前无切削,丢弃"
                k = max(0, j - 2)               # 含断刀瞬间尖峰
                out = [(ft, 1 if i >= k else 0) for i, ft in enumerate(feats)]
                return out, f"前{j}窗正常+尖峰{k}~{j},后{n-k}窗断刀"

            self.log_signal.emit("=== 加载正常数据 ===")
            normal_feats = collect(self.normal_dir)
            if os.path.isdir(self.aircut_dir):
                self.log_signal.emit("=== 加载空切数据 ===")
                normal_feats += collect(self.aircut_dir)
            self.log_signal.emit("=== 加载断刀数据(窗级分段) ===")
            broken_rows = []
            n_break_keep = n_break_drop = 0
            if os.path.isdir(self.broken_dir):
                for f in sorted(os.listdir(self.broken_dir)):
                    if not f.endswith(".csv"): continue
                    try:
                        self.log_signal.emit(f"  分析 {f}")
                        feats = load_file_feats(os.path.join(self.broken_dir, f))
                        seg, note = segment_broken(feats)
                        if not seg:
                            n_break_drop += 1
                            self.log_signal.emit(f"    {note}")
                            continue
                        n_break_keep += 1
                        self.log_signal.emit(f"    {note}")
                        broken_rows.extend(seg)
                    except Exception as e:
                        self.log_signal.emit(f"    跳过 {f}: {e}")
            self.log_signal.emit(f"断刀文件: 保留{n_break_keep} 丢弃{n_break_drop}")

            # 按正常数据学"各直径正常基线"，套用到全部窗
            self.log_signal.emit("=== 学习各直径正常基线 ===")
            baseline = build_dia_baseline(normal_feats)
            all_feats = normal_feats + [ft for ft, _ in broken_rows]
            apply_dia_baseline(all_feats, baseline)

            def to_rows(feats, label):
                rows = [[getattr(ft, c, 0.0) or 0.0 for c in COLS]
                        for ft in feats]
                ys = [label] * len(feats)
                return rows, ys

            r1, y1 = to_rows(normal_feats, 0)
            bfeats = [ft for ft, _ in broken_rows]
            bylb = [lb for _, lb in broken_rows]
            r3 = [[getattr(ft, c, 0.0) or 0.0 for c in COLS] for ft in bfeats]
            X = np.array(r1 + r3); y = np.array(list(y1) + bylb)
            self.log_signal.emit(f"总样本: {len(X)} (正常={int((y==0).sum())} 断刀={int((y==1).sum())})")

            # XGBoost
            self.log_signal.emit("=== 训练XGBoost ===")
            xgb = XGBClassifier(n_estimators=self.n_est, max_depth=self.depth,
                                learning_rate=self.lr, subsample=0.8,
                                colsample_bytree=0.8, scale_pos_weight=2,
                                random_state=42, eval_metric="logloss", verbosity=0)
            xgb.fit(X, y)
            os.makedirs("output_xgb", exist_ok=True)
            joblib.dump({"model": xgb, "features": COLS,
                         "dia_baseline": baseline},
                        "output_xgb/xgb_break_tool.pkl")
            self.log_signal.emit("XGBoost已保存 output_xgb/xgb_break_tool.pkl")

            # IsolationForest（只用正常）
            self.log_signal.emit("=== 训练IsolationForest ===")
            X_norm = X[y == 0]
            mu, sd = X_norm.mean(0), X_norm.std(0) + 1e-6
            ifm = IsolationForest(n_estimators=self.if_n, contamination=0.1,
                                   random_state=42)
            ifm.fit((X_norm - mu) / sd)
            os.makedirs("output_lgb", exist_ok=True)
            joblib.dump({"model": ifm, "mu": mu, "sd": sd, "features": COLS},
                        "output_lgb/isolationforest.pkl")
            self.log_signal.emit("IsolationForest已保存 output_lgb/isolationforest.pkl")

            self.done_signal.emit(True, "✅ 训练完成")
        except Exception as e:
            import traceback
            self.done_signal.emit(False, f"训练失败: {e}\n{traceback.format_exc()}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    THEMES = {
        "亮色(默认)": """
        QMainWindow { background: #f1f5f9; }
        QMenuBar { background:#f1f5f9; }
        QToolBar { background:#f1f5f9; border:none; spacing:5px; }
        QWidget { font-family: "Microsoft YaHei","微软雅黑",sans-serif; font-size: 9pt; color: #1e293b; }
        QGroupBox { font-weight:bold; color:#334155; border:1px solid #e2e8f0; border-radius:10px; margin-top:14px; padding-top:14px; background:#fff; }
        QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 8px; color:#2563eb; }
        QPushButton { background:#2563eb; color:#fff; border:none; padding:7px 18px; border-radius:6px; font-weight:500; }
        QPushButton:hover { background:#1d4ed8; } QPushButton:pressed { background:#1e40af; } QPushButton:disabled { background:#94a3b8; }
        QSpinBox,QDoubleSpinBox,QLineEdit,QComboBox { padding:5px 10px; border:1px solid #cbd5e1; border-radius:6px; background:#fff; selection-background-color:#bfdbfe; }
        QSpinBox:focus,QLineEdit:focus,QComboBox:focus { border:1px solid #2563eb; }
        QTabWidget::pane { border:1px solid #e2e8f0; border-radius:8px; background:#fff; }
        QTabBar::tab { padding:9px 22px; margin-right:3px; background:#e2e8f0; color:#475569; border:1px solid #e2e8f0; border-bottom:none; border-top-left-radius:8px; border-top-right-radius:8px; }
        QTabBar::tab:hover { background:#cbd5e1; color:#1e293b; }
        QTabBar::tab:selected { background:#fff; color:#2563eb; font-weight:bold; border-bottom:2px solid #2563eb; }
        QHeaderView::section { background:#1e293b; color:#f1f5f9; padding:8px; border:none; }
        QTableWidget { background:#fff; border:1px solid #e2e8f0; gridline-color:#f1f5f9; alternate-background-color:#f8fafc; }
        QTableWidget::item:selected { background:#dbeafe; color:#1e40af; }
        QTextEdit { border:1px solid #e2e8f0; border-radius:6px; background:#fff; }
        QProgressBar { border:1px solid #e2e8f0; border-radius:6px; text-align:center; background:#f1f5f9; }
        QProgressBar::chunk { background:#2563eb; border-radius:5px; }
        QScrollBar:vertical { background:#f1f5f9; width:10px; } QScrollBar::handle:vertical { background:#cbd5e1; min-height:30px; }
        QScrollBar:horizontal { background:#f1f5f9; height:10px; } QScrollBar::handle:horizontal { background:#cbd5e1; min-width:30px; }
        """,
        "暗色": """
        QMainWindow { background: #1e1e2e; }
        QMenuBar { background:#181825; color:#cdd6f4; }
        QMenuBar::item:selected { background:#313244; }
        QMenu { background:#181825; color:#cdd6f4; border:1px solid #313244; }
        QMenu::item:selected { background:#45475a; }
        QToolBar { background:#181825; border:none; spacing:5px; }
        QWidget { font-family:"Microsoft YaHei","微软雅黑",sans-serif; font-size:9pt; color:#cdd6f4; }
        QGroupBox { font-weight:bold; color:#cdd6f4; border:1px solid #313244; border-radius:10px; margin-top:14px; padding-top:14px; background:#181825; }
        QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 8px; color:#89b4fa; }
        QPushButton { background:#89b4fa; color:#1e1e2e; border:none; padding:7px 18px; border-radius:6px; font-weight:500; }
        QPushButton:hover { background:#b4befe; } QPushButton:pressed { background:#74c7ec; } QPushButton:disabled { background:#45475a; }
        QSpinBox,QDoubleSpinBox,QLineEdit,QComboBox { padding:5px 10px; border:1px solid #45475a; border-radius:6px; background:#313244; color:#cdd6f4; }
        QTabWidget::pane { border:1px solid #313244; background:#181825; }
        QTabBar::tab { padding:9px 22px; margin-right:3px; background:#313244; color:#a6adc8; border:none; border-top-left-radius:8px; border-top-right-radius:8px; }
        QTabBar::tab:hover { background:#45475a; color:#cdd6f4; }
        QTabBar::tab:selected { background:#181825; color:#89b4fa; font-weight:bold; border-bottom:2px solid #89b4fa; }
        QHeaderView::section { background:#313244; color:#cdd6f4; padding:8px; border:none; }
        QTableWidget { background:#181825; border:1px solid #313244; gridline-color:#313244; alternate-background-color:#1e1e2e; color:#cdd6f4; }
        QTableWidget::item:selected { background:#45475a; color:#89b4fa; }
        QTextEdit { border:1px solid #313244; border-radius:6px; background:#181825; color:#cdd6f4; }
        QProgressBar { border:1px solid #313244; border-radius:6px; text-align:center; background:#313244; color:#cdd6f4; }
        QProgressBar::chunk { background:#89b4fa; border-radius:5px; }
        QScrollBar:vertical { background:#181825; width:10px; } QScrollBar::handle:vertical { background:#45475a; min-height:30px; }
        QScrollBar:horizontal { background:#181825; height:10px; } QScrollBar::handle:horizontal { background:#45475a; min-width:30px; }
        """,
        "护眼绿": """
        QMainWindow { background: #e8f5e9; }
        QMenuBar { background:#e8f5e9; }
        QToolBar { background:#e8f5e9; border:none; spacing:5px; }
        QWidget { font-family:"Microsoft YaHei","微软雅黑",sans-serif; font-size:9pt; color:#1b5e20; }
        QGroupBox { font-weight:bold; color:#2e7d32; border:1px solid #c8e6c9; border-radius:10px; margin-top:14px; padding-top:14px; background:#f1f8e9; }
        QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 8px; color:#388e3c; }
        QPushButton { background:#43a047; color:#fff; border:none; padding:7px 18px; border-radius:6px; }
        QPushButton:hover { background:#66bb6a; } QPushButton:pressed { background:#388e3c; } QPushButton:disabled { background:#a5d6a7; }
        QSpinBox,QDoubleSpinBox,QLineEdit,QComboBox { padding:5px 10px; border:1px solid #a5d6a7; border-radius:6px; background:#fff; }
        QTabWidget::pane { border:1px solid #c8e6c9; background:#f1f8e9; }
        QTabBar::tab { padding:9px 22px; margin-right:3px; background:#c8e6c9; color:#2e7d32; border:none; border-top-left-radius:8px; border-top-right-radius:8px; }
        QTabBar::tab:hover { background:#a5d6a7; }
        QTabBar::tab:selected { background:#f1f8e9; color:#388e3c; font-weight:bold; border-bottom:2px solid #43a047; }
        QHeaderView::section { background:#388e3c; color:#fff; padding:8px; border:none; }
        QTableWidget { background:#fff; border:1px solid #c8e6c9; gridline-color:#e8f5e9; alternate-background-color:#f1f8e9; }
        QTableWidget::item:selected { background:#c8e6c9; color:#1b5e20; }
        QTextEdit { border:1px solid #c8e6c9; border-radius:6px; background:#fff; }
        QProgressBar { border:1px solid #c8e6c9; border-radius:6px; text-align:center; background:#e8f5e9; }
        QProgressBar::chunk { background:#43a047; border-radius:5px; }
        QScrollBar:vertical { background:#e8f5e9; width:10px; } QScrollBar::handle:vertical { background:#a5d6a7; min-height:30px; }
        QScrollBar:horizontal { background:#e8f5e9; height:10px; } QScrollBar::handle:horizontal { background:#a5d6a7; min-width:30px; }
        """,
        "深蓝": """
        QMainWindow { background: #0f172a; }
        QMenuBar { background:#1e293b; color:#e2e8f0; }
        QMenuBar::item:selected { background:#334155; }
        QMenu { background:#1e293b; color:#e2e8f0; border:1px solid #334155; }
        QMenu::item:selected { background:#0c4a6e; }
        QToolBar { background:#1e293b; border:none; spacing:5px; }
        QWidget { font-family:"Microsoft YaHei","微软雅黑",sans-serif; font-size:9pt; color:#e2e8f0; }
        QGroupBox { font-weight:bold; color:#e2e8f0; border:1px solid #1e3a5f; border-radius:10px; margin-top:14px; padding-top:14px; background:#1e293b; }
        QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 8px; color:#38bdf8; }
        QPushButton { background:#0284c7; color:#fff; border:none; padding:7px 18px; border-radius:6px; }
        QPushButton:hover { background:#0ea5e9; } QPushButton:pressed { background:#0369a1; } QPushButton:disabled { background:#475569; }
        QSpinBox,QDoubleSpinBox,QLineEdit,QComboBox { padding:5px 10px; border:1px solid #334155; border-radius:6px; background:#0f172a; color:#e2e8f0; }
        QTabWidget::pane { border:1px solid #1e3a5f; background:#1e293b; }
        QTabBar::tab { padding:9px 22px; margin-right:3px; background:#334155; color:#94a3b8; border:none; border-top-left-radius:8px; border-top-right-radius:8px; }
        QTabBar::tab:hover { background:#475569; color:#e2e8f0; }
        QTabBar::tab:selected { background:#1e293b; color:#38bdf8; font-weight:bold; border-bottom:2px solid #0ea5e9; }
        QHeaderView::section { background:#0c4a6e; color:#fff; padding:8px; border:none; }
        QTableWidget { background:#1e293b; border:1px solid #334155; gridline-color:#334155; alternate-background-color:#0f172a; color:#e2e8f0; }
        QTableWidget::item:selected { background:#0c4a6e; color:#fff; }
        QTextEdit { border:1px solid #334155; border-radius:6px; background:#0f172a; color:#e2e8f0; }
        QProgressBar { border:1px solid #334155; border-radius:6px; text-align:center; background:#1e293b; }
        QProgressBar::chunk { background:#0ea5e9; border-radius:5px; }
        QScrollBar:vertical { background:#1e293b; width:10px; } QScrollBar::handle:vertical { background:#475569; min-height:30px; }
        QScrollBar:horizontal { background:#1e293b; height:10px; } QScrollBar::handle:horizontal { background:#475569; min-width:30px; }
        """,
    }
    import json as _json
    _cfg = {}
    try: _cfg = _json.load(open("gui_config.json", encoding="utf-8"))
    except: pass
    _theme = _cfg.get("theme", "亮色(默认)")
    app.setStyleSheet(THEMES.get(_theme, THEMES["亮色(默认)"]))
    w = MainWindow()
    w._themes = THEMES
    w._app = app
    w._theme_name = _theme
    # 初始化时按主题刷新行颜色
    w._refresh_history_colors()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
