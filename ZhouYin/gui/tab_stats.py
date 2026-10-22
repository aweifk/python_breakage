# -*- coding: utf-8 -*-
"""tab_stats"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar


class TabStats:
    def _build_stats(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # 刷新按钮
        bar = QHBoxLayout()
        btn_refresh = QPushButton("刷新统计")
        btn_refresh.clicked.connect(self._refresh_stats)
        bar.addWidget(btn_refresh)
        btn_export = QPushButton("导出报告")
        btn_export.setStyleSheet("background:#0d9488;color:#fff;padding:6px 16px;border-radius:4px;")
        btn_export.clicked.connect(self._export_stats_report)
        bar.addWidget(btn_export)
        bar.addStretch()
        lay.addLayout(bar)

        # 总指标：左文字 + 右三个环形图
        box1 = QGroupBox("总体指标")
        g1 = QHBoxLayout(box1)
        self.lbl_stat_total = QLabel("总检测数: -")
        self.lbl_stat_total.setStyleSheet("font-size: 12pt; padding: 10px;")
        g1.addWidget(self.lbl_stat_total)

        self.fig_metrics = Figure(figsize=(6, 2.5), dpi=80)
        self.canvas_metrics = FigureCanvas(self.fig_metrics)
        g1.addWidget(self.canvas_metrics, stretch=1)
        lay.addWidget(box1)

        # 按直径统计表 + 右侧饼图
        middle = QSplitter(Qt.Horizontal)
        box2 = QGroupBox("按刀具直径统计")
        g2 = QVBoxLayout(box2)
        self.stats_table = QTableWidget(0, 7)
        self.stats_table.setHorizontalHeaderLabels(["直径", "总数", "模型报断刀数", "模型报正常数", "漏报", "误报", "模型判定断刀率"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        g2.addWidget(self.stats_table)
        middle.addWidget(box2)

        box3 = QGroupBox("混淆矩阵分布")
        g3 = QVBoxLayout(box3)
        self.fig_pie = Figure(figsize=(3.5, 3.5), dpi=80)
        self.canvas_pie = FigureCanvas(self.fig_pie)
        g3.addWidget(self.canvas_pie)
        middle.addWidget(box3)
        middle.setSizes([800, 500])
        lay.addWidget(middle)

        return w


    def _export_stats_report(self):
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


    def _refresh_stats(self):
        try:
            cur = self.db.execute("SELECT result, diameter FROM history")
            rows = cur.fetchall()
            total = len(rows)
            broken = sum(1 for r in rows if r[0] == "断刀")
            normal = total - broken
            self.lbl_stat_total.setText(f"总检测数: {total}")

            # 按直径分组
            from collections import Counter, defaultdict
            dia_total = Counter(r[1] for r in rows)
            dia_broken = Counter(r[1] for r in rows if r[0] == "断刀")
            # 按直径统计漏报误报
            cur_dia = self.db.execute(
                "SELECT diameter, result, actual_result FROM history WHERE actual_result IN ('断刀','正常')")
            dia_fn = Counter(); dia_fp = Counter()
            for d, m, a in cur_dia.fetchall():
                if m == "正常" and a == "断刀": dia_fn[d] += 1
                if m == "断刀" and a == "正常": dia_fp[d] += 1
            self.stats_table.setRowCount(len(dia_total))
            for i, dia in enumerate(sorted(dia_total.keys())):
                t = dia_total[dia]
                b = dia_broken.get(dia, 0)
                n = t - b
                fn = dia_fn.get(dia, 0)
                fp = dia_fp.get(dia, 0)
                vals = [dia, str(t), str(b), str(n), str(fn), str(fp), f"{b/t*100:.1f}%"]
                for c, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    _dark = getattr(self, "_theme_name", "") in ("暗色", "深蓝")
                    if c == 4 and fn > 0: item.setBackground(QColor("#7f2d2d") if _dark else QColor("#fadbd8"))
                    if c == 5 and fp > 0: item.setBackground(QColor("#7a6520") if _dark else QColor("#f9e79f"))
                    self.stats_table.setItem(i, c, item)

            # 混淆矩阵（只算已标注的）
            cur2 = self.db.execute(
                "SELECT result, actual_result FROM history WHERE actual_result IN ('断刀','正常')")
            labeled = cur2.fetchall()
            if labeled:
                tp = sum(1 for m, a in labeled if m == "断刀" and a == "断刀")
                fp = sum(1 for m, a in labeled if m == "断刀" and a == "正常")
                fn = sum(1 for m, a in labeled if m == "正常" and a == "断刀")
                tn = sum(1 for m, a in labeled if m == "正常" and a == "正常")
                acc = (tp + tn) / len(labeled) * 100
                prec = tp / (tp + fp) * 100 if (tp + fp) else None
                rec = tp / (tp + fn) * 100 if (tp + fn) else None
                f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
                self.lbl_stat_total.setText(
                    f"总检测数: {total}   已标注: {len(labeled)}   "
                    f"准确率 {acc:.1f}%  精确率 {prec if prec is None else f'{prec:.1f}%'}  "
                    f"召回率 {rec if rec is None else f'{rec:.1f}%'}  "
                    f"F1 {f1 if f1 is None else f'{f1:.1f}%'}")

                # 画环形图
                self.fig_metrics.clear()
                metrics = [("准确率", acc, "#27ae60")]
                if prec is not None: metrics.append(("精确率", prec, "#2980b9"))
                if rec is not None: metrics.append(("召回率", rec, "#8e44ad"))
                if f1 is not None: metrics.append(("F1", f1, "#e67e22"))
                for i, (name, val, color) in enumerate(metrics):
                    ax = self.fig_metrics.add_subplot(1, len(metrics), i+1)
                    ax.pie([val, 100-val], colors=[color, "#ecf0f1"],
                           wedgeprops=dict(width=0.3), startangle=90)
                    ax.text(0, 0, f"{val:.1f}%", ha="center", va="center",
                            fontsize=12, fontweight="bold")
                    ax.set_title(name)
                self.fig_metrics.tight_layout()
                self.canvas_metrics.draw()

                self._log(f"混淆矩阵 TP={tp} FP={fp} FN={fn} TN={tn} 已标注={len(labeled)}")
                # 画饼图
                self.fig_pie.clear()
                ax = self.fig_pie.add_subplot(111)
                labels = [f"TP真断刀({tp})", f"FP误报({fp})", f"FN漏报({fn})", f"TN真正常({tn})"]
                sizes = [tp, fp, fn, tn]
                colors = ["#27ae60", "#e74c3c", "#f39c12", "#3498db"]
                sizes = [s for s in sizes if s > 0]
                labels2 = [l for l, s in zip(labels, [tp, fp, fn, tn]) if s > 0]
                colors2 = [c for c, s in zip(colors, [tp, fp, fn, tn]) if s > 0]
                if sizes:
                    wedges, texts, autotexts = ax.pie(
                        sizes, colors=colors2, autopct="%1.1f%%",
                        startangle=90, textprops={"fontsize": 8},
                        pctdistance=0.75)
                    ax.legend(wedges, labels2, loc="lower center", fontsize=8,
                              bbox_to_anchor=(0.5, -0.12), ncol=2)
                    ax.axis("equal")
                self.fig_pie.tight_layout()
                self.canvas_pie.draw()
        except Exception as e:
            self._log(f"统计失败: {e}")


