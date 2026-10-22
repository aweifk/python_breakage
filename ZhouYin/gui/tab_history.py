# -*- coding: utf-8 -*-
"""tab_history"""
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar


class TabHistory:
    def _build_history(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # 搜索栏
        bar = QHBoxLayout()
        bar.addWidget(QLabel("文件名:"))
        self.le_search_name = QLineEdit()
        self.le_search_name.setPlaceholderText("输入关键词")
        self.le_search_name.textChanged.connect(self._filter_history)
        bar.addWidget(self.le_search_name)

        bar.addWidget(QLabel("直径:"))
        self.cb_search_dia = QComboBox()
        self.cb_search_dia.addItems(["全部", "D0.4", "D0.6", "D0.8", "D1.0", "D1.5", "D2", "D4", "D6", "D8"])
        self.cb_search_dia.currentTextChanged.connect(self._filter_history)
        bar.addWidget(self.cb_search_dia)

        bar.addWidget(QLabel("结果:"))
        self.cb_search_result = QComboBox()
        self.cb_search_result.addItems(["全部", "断刀", "正常"])
        self.cb_search_result.currentTextChanged.connect(self._filter_history)
        bar.addWidget(self.cb_search_result)
        self.cb_err = QComboBox()
        self.cb_err.addItems(["全部", "漏检", "误报", "正确"])
        self.cb_err.currentTextChanged.connect(self._filter_history)
        bar.addWidget(self.cb_err)

        bar.addWidget(QLabel("从:"))
        from PyQt5.QtCore import QDate
        self.dt_from = QDateEdit(QDate.currentDate().addDays(-7))
        self.dt_from.setCalendarPopup(True)
        self.dt_from.setDisplayFormat("yyyy-MM-dd")
        self.dt_from.dateChanged.connect(self._filter_history)
        bar.addWidget(self.dt_from)
        bar.addWidget(QLabel("到:"))
        self.dt_to = QDateEdit(QDate.currentDate())
        self.dt_to.setCalendarPopup(True)
        self.dt_to.setDisplayFormat("yyyy-MM-dd")
        self.dt_to.dateChanged.connect(self._filter_history)
        bar.addWidget(self.dt_to)

        btn_sel_all = QPushButton("全选")
        btn_sel_all.clicked.connect(lambda: self._hist_select_all(True))
        bar.addWidget(btn_sel_all)
        btn_sel_none = QPushButton("取消全选")
        btn_sel_none.clicked.connect(lambda: self._hist_select_all(False))
        bar.addWidget(btn_sel_none)
        btn_export = QPushButton("导出分析报告")
        btn_export.clicked.connect(self._export_selected_report)
        bar.addWidget(btn_export)
        btn_del = QPushButton("删除选中")
        btn_del.clicked.connect(self._delete_history)
        bar.addWidget(btn_del)
        lay.addLayout(bar)

        self.hist_table = QTableWidget(0, 8)
        self.hist_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.hist_table.setHorizontalHeaderLabels(["时间", "文件名", "模型判断", "置信度", "直径", "G代码", "路径", "实际结果"])
        header = self.hist_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        self.hist_table.setColumnWidth(0, 160)
        self.hist_table.setColumnWidth(2, 80)
        self.hist_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.hist_table.setColumnWidth(4, 70)
        self.hist_table.setColumnWidth(5, 80)
        self.hist_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.hist_table.customContextMenuRequested.connect(self._hist_context_menu)
        lay.addWidget(self.hist_table)
        return w


    def _hist_select_all(self, sel):
        """全选/取消全选当前显示的行"""
        self.hist_table.clearSelection()
        if sel:
            for r in range(self.hist_table.rowCount()):
                for c in range(self.hist_table.columnCount()):
                    item = self.hist_table.item(r, c)
                    if item:
                        item.setSelected(True)


    def _export_selected_report(self):
        """导出选中文件的分析报告（HTML，含rms/power轨迹图）"""
        rows = sorted(set(i.row() for i in self.hist_table.selectedItems()))
        if not rows:
            QMessageBox.warning(self, "提示", "请先选择要导出的记录")
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ae_batch_pipeline import load_one_csv
        from gui.plot_helpers import (
            plot_detection_annotations,
            plot_waveform_curves,
            resolve_detection_result,
        )

        # 收集选中文件名
        files = []
        for r in rows:
            fname = self.hist_table.item(r, 1).text()
            pred = self.hist_table.item(r, 2).text()
            # 第7列是实际结果下拉框
            w7 = self.hist_table.cellWidget(r, 7)
            actual = w7.currentText() if w7 else (self.hist_table.item(r, 7).text() if self.hist_table.item(r, 7) else "")
            files.append((fname, pred, actual))

        out_path, _ = QFileDialog.getSaveFileName(self, "导出分析报告", "分析报告.html", "HTML (*.html)")
        if not out_path:
            return

        html_parts = [f"<html><head><meta charset='utf-8'><style>body{{font-family:Microsoft YaHei;margin:20px}}h2{{color:#2563eb}}.file{{margin-bottom:40px;border:1px solid #ddd;padding:15px;border-radius:8px}}</style></head><body>"]
        html_parts.append(f"<h1>断刀检测分析报告</h1><p>共 {len(files)} 个文件</p>")

        for idx, (fname, model_pred, actual) in enumerate(files):
            # 从表格取路径列（第6列）
            row_idx = rows[idx]
            path_item = self.hist_table.item(row_idx, 6)
            fpath = path_item.text() if path_item else ""
            if not fpath or not os.path.exists(fpath):
                html_parts.append(f"<div class='file'><h2>{idx+1}. {fname}</h2><p style='color:red'>文件路径不存在: {fpath}</p></div>")
                continue

            try:
                df = load_one_csv(fpath)
                detail_json = self._hist_detail_json(row_idx)
                det = getattr(self, "detector", None)
                infer = resolve_detection_result(det, fpath, detail_json)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
                axis_channels = [(ax1, "rms"), (ax2, "power")]
                plot_waveform_curves(axis_channels, df, title=fname)
                if det:
                    plot_detection_annotations(axis_channels, df, det, infer)
                fig.tight_layout()

                reason_lines = (infer or {}).get("reason_lines") or []
                if not reason_lines and infer and infer.get("detail"):
                    try:
                        from ae_xgboost_isolationforest import format_detection_reason
                        d = infer["detail"]
                        reason_lines = format_detection_reason(
                            bool(infer.get("broken")),
                            d,
                            d.get("window_times") or {},
                        )
                    except Exception:
                        pass

                import base64
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()

                mark_color = "#dc2626" if model_pred == "断刀" else "#16a34a"

                analysis = ""
                if reason_lines:
                    items = "".join(f"<li>{line}</li>" for line in reason_lines)
                    src = "历史入库 detail" if detail_json else "当前模型重新推理"
                    analysis = (
                        f"<p style='color:#64748b;font-size:12px'>判定原因来源：{src}</p>"
                        f"<ul>{items}</ul>"
                    )

                html_parts.append(
                    f"<div class='file'><h2>{idx+1}. {fname}</h2>"
                    f"<p>模型判断: <b style='color:{mark_color}'>{model_pred}</b> | 实际结果: {actual}</p>"
                    f"{analysis}"
                    f"<img src='data:image/png;base64,{img_b64}' style='width:100%'/>"
                    f"</div>")
            except Exception as e:
                html_parts.append(f"<div class='file'><h2>{idx+1}. {fname}</h2><p style='color:red'>加载失败: {e}</p></div>")

        html_parts.append("</body></html>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))
        self._log(f"分析报告已导出: {out_path}（{len(files)}个文件）")
        QMessageBox.information(self, "导出成功", f"已导出 {len(files)} 个文件: {out_path}")


    def _hist_context_menu(self, pos):
        item = self.hist_table.itemAt(pos)
        if not item: return
        row = item.row()
        fname = self.hist_table.item(row, 1).text()
        menu = QMenu()
        act_view = menu.addAction("📊 数据查看")
        act_export = menu.addAction("📄 导出单文件报告(HTML)")
        menu.addSeparator()
        act_broken = menu.addAction("→ 加入断刀训练集")
        act_normal = menu.addAction("→ 加入正常训练集")
        act_aircut = menu.addAction("→ 加入空切训练集")
        act = menu.exec_(self.hist_table.viewport().mapToGlobal(pos))
        if act == act_view:
            self._hist_load_to_dataview()
        elif act == act_export:
            self._export_single_report(row)
        elif act == act_broken:
            self._copy_to_train_dir(row, "broken")
        elif act == act_normal:
            self._copy_to_train_dir(row, "normal")
        elif act == act_aircut:
            self._copy_to_train_dir(row, "aircut")


    def _export_single_report(self, row):
        fname = self.hist_table.item(row, 1).text()
        model_r = self.hist_table.item(row, 2).text()
        conf = self.hist_table.item(row, 3).text()
        dia = self.hist_table.item(row, 4).text()
        gcode = self.hist_table.item(row, 5).text()
        fpath_item = self.hist_table.item(row, 6)
        fpath = fpath_item.text() if fpath_item else ""
        actual_w = self.hist_table.cellWidget(row, 7)
        actual = actual_w.currentText() if actual_w else ""
        import base64, io
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ae_batch_pipeline import load_one_csv
        from gui.plot_helpers import (
            plot_detection_annotations,
            plot_waveform_curves,
            resolve_detection_result,
        )
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        img_html = ""
        reason_html = ""
        reason_note = ""
        try:
            df = load_one_csv(fpath)
            detail_json = self._hist_detail_json(row)
            det = getattr(self, "detector", None)
            pred = resolve_detection_result(det, fpath, detail_json)
            fig, axes = plt.subplots(3, 1, figsize=(12, 6), dpi=80)
            axis_channels = [(axes[0], "rms"), (axes[1], "power"), (axes[2], "amp")]
            plot_waveform_curves(axis_channels, df, title=fname)
            if det:
                plot_detection_annotations(axis_channels, df, det, pred)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
            img_html = f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' style='width:100%'>"
            reason_lines = (pred or {}).get("reason_lines") or []
            if detail_json and reason_lines:
                reason_note = (
                    "<p style='color:#64748b;font-size:12px'>"
                    "判断原因来自历史入库时的 detail。</p>"
                )
            elif det and pred:
                reason_note = (
                    "<p style='color:#64748b;font-size:12px'>"
                    "判断原因按当前已加载模型重新推理生成。</p>"
                )
            if not reason_lines and det and fpath and os.path.isfile(fpath):
                try:
                    from ae_xgboost_isolationforest import format_detection_reason
                    if pred and pred.get("detail"):
                        reason_lines = format_detection_reason(
                            bool(pred.get("broken")),
                            pred.get("detail") or {},
                            (pred.get("detail") or {}).get("window_times") or {},
                        )
                except Exception as ex:
                    reason_lines = [f"推理失败，无法生成详细原因：{ex}"]
            if not reason_lines:
                if model_r == "断刀":
                    reason_lines = [
                        f"模型判定断刀（置信度{conf}），G代码行号 {gcode}",
                    ]
                else:
                    reason_lines = [
                        f"模型判定正常（置信度{conf}），未触发报警",
                    ]
            items = "".join(f"<li>{line}</li>" for line in reason_lines)
            reason_html = f"{reason_note}<ul>{items}</ul>"
        except Exception as e:
            img_html = f"<p style='color:red'>图表加载失败: {e}</p>"
            reason_html = f"<p style='color:red'>原因生成失败: {e}</p>"
        html = f"""<html><head><meta charset="utf-8"><style>
body{{font-family:"Microsoft YaHei";padding:30px;background:#f8fafc;color:#1e293b}}
h1{{color:#0d9488}} table{{border-collapse:collapse;margin:15px 0}}
th,td{{border:1px solid #ddd;padding:8px 14px}} th{{background:#0d9488;color:#fff}}
</style></head><body>
<h1>单文件检测分析报告</h1>
<table><tr><th>文件名</th><td>{fname}</td></tr>
<tr><th>检测结果</th><td style="color:{'#e74c3c' if model_r=='断刀' else '#27ae60'};font-weight:bold">{model_r}</td></tr>
<tr><th>置信度</th><td>{conf}</td></tr>
<tr><th>刀具直径</th><td>{dia}</td></tr>
<tr><th>G代码行号</th><td>{gcode}</td></tr>
<tr><th>实际结果</th><td>{actual}</td></tr>
<tr><th>文件路径</th><td>{fpath}</td></tr></table>
<h2>判断原因</h2>{reason_html}
<h2>信号波形（含阈值线）</h2>{img_html}
</body></html>"""
        out, _ = QFileDialog.getSaveFileName(self, "保存报告", f"单文件报告_{fname.replace('.csv','')}.html", "HTML (*.html)")
        if out:
            open(out, "w", encoding="utf-8").write(html)
            self._log(f"单文件报告已导出: {out}")
            QMessageBox.information(self, "导出成功", out)


    def _hist_load_to_dataview(self):
        """把选中行对应的CSV加载到数据查看页"""
        rows = sorted(set(i.row() for i in self.hist_table.selectedItems()))
        if not rows:
            rows = [item.row()]
        from ae_batch_pipeline import load_one_csv
        loaded = 0
        for r in rows:
            fname = self.hist_table.item(r, 1).text()
            fpath_item = self.hist_table.item(r, 6)
            fpath = fpath_item.text() if fpath_item else ""
            if not fpath or not os.path.exists(fpath):
                continue
            try:
                df = load_one_csv(fpath)
                detail_json = self._hist_detail_json(r)
                cb = QCheckBox(fname)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda st, c=cb: self._dv_on_cb_toggled(c))
                self.dv_list.addWidget(cb)
                self.dv_data.append((cb, fname, df, fpath, detail_json))
                self._dv_on_cb_toggled(cb)
                loaded += 1
            except Exception as e:
                self._log(f"加载 {fname} 失败: {e}")
        if loaded:
            self._dv_redraw()
            self.tabs.setCurrentIndex(2)  # 数据查看页
            self._log(f"已加载 {loaded} 个文件到数据查看")
        else:
            QMessageBox.information(self, "提示", "选中行没有有效的CSV路径")


    def _copy_to_train_dir(self, row, subdir):
        """把指定行对应的CSV从路径列复制到 dataset/{subdir}/ 目录"""
        import shutil
        fname = self.hist_table.item(row, 1).text() if self.hist_table.item(row, 1) else ""
        fp_item = self.hist_table.item(row, 6)
        src = fp_item.text() if fp_item else ""
        if not src or not os.path.exists(src):
            # 兜底：在dataset各子目录按文件名找
            for root in ["dataset/normal", "dataset/broken", "dataset/aircut"]:
                p = os.path.join(root, fname)
                if os.path.exists(p):
                    src = p; break
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, "找不到", f"文件 {fname}\n路径为空或不存在")
            return
        dst_dir = os.path.join("dataset", subdir)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fname)
        if os.path.abspath(src) == os.path.abspath(dst):
            self._log(f"{fname} 已在 dataset/{subdir}/，跳过")
            return
        shutil.copy2(src, dst)
        self._log(f"已复制 {fname} → dataset/{subdir}/")


    def _filter_history(self):
        name = self.le_search_name.text().lower()
        dia = self.cb_search_dia.currentText()
        result = self.cb_search_result.currentText()
        err = self.cb_err.currentText()
        for i in range(self.hist_table.rowCount()):
            row_name = self.hist_table.item(i, 1).text().lower() if self.hist_table.item(i, 1) else ""
            row_dia = self.hist_table.item(i, 4).text() if self.hist_table.item(i, 4) else ""
            row_result = self.hist_table.item(i, 2).text() if self.hist_table.item(i, 2) else ""
            # 第7列是下拉框组件，用cellWidget取值
            w7 = self.hist_table.cellWidget(i, 7)
            row_actual = w7.currentText() if w7 else (self.hist_table.item(i, 7).text() if self.hist_table.item(i, 7) else "")
            show = True
            if name and name not in row_name: show = False
            if dia != "全部":
                try:
                    _dn = float(dia.replace("D",""))
                    _rn = float(row_dia.replace("D",""))
                    if abs(_dn - _rn) > 1e-6: show = False
                except:
                    if dia not in row_dia: show = False
            if result != "全部" and result != row_result: show = False
            if err == "漏检" and not (row_actual == "断刀" and row_result == "正常"): show = False
            if err == "误报" and not (row_actual == "正常" and row_result == "断刀"): show = False
            if err == "正确" and not ((row_actual == "断刀" and row_result == "断刀") or
                                      (row_actual == "正常" and row_result == "正常")): show = False
            row_name2 = self.hist_table.item(i, 1).text() if self.hist_table.item(i, 1) else ""
            try:
                import re as _re
                m = _re.search(r'(20\d{12})', row_name2)
                if m:
                    from datetime import datetime as _dt
                    rt = _dt.strptime(m.group(1), "%Y%m%d%H%M%S")
                    if not (self.dt_from.date().toPyDate() <= rt.date() <= self.dt_to.date().toPyDate()):
                        show = False
            except: pass
            self.hist_table.setRowHidden(i, not show)


    def _hist_row_color(self, model_result, actual):
        """漏检红色，误报黄色，正确绿色，未知浅色。暗色主题用深色版"""
        dark = getattr(self, "_theme_name", "") in ("暗色", "深蓝")
        if actual in ("未知", "None", "") or actual is None:
            return QColor("#45475a") if dark else (QColor("#f4f6f7") if model_result == "断刀" else QColor("#eafaf1"))
        if model_result != actual:
            if model_result == "正常" and actual == "断刀":
                return QColor("#7f2d2d") if dark else QColor("#f1948a")  # 漏检
            else:
                return QColor("#7a6520") if dark else QColor("#f9e79f")  # 误报
        return QColor("#1e4d2b") if dark else QColor("#d5f5e3")  # 正确


    def _delete_history(self):
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


