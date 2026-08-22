#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经元高维点阵图 —— 整合意识可视化窗口

功能：
- 将 6 维认知相空间中的专家/虚拟专家神经元用 PCA 投影到 2D 点阵
- 神经元之间用连线表示认知耦合关系
- 实时显示推理阶段与信息传递（线段上流动的粒子 + 消息气泡）

运行方式（独立进程）：
    python neuron_map.py --port 50000

事件协议（UDP JSON）：
    init      {"type":"init","all_vectors":[[..6d]..],
               "nodes":{"vectors":[..6d..],"labels":[..],"kinds":[..]},
               "edges":[[i,j,w],...]}
    phase     {"type":"phase","text":"...","level":N}
    signal    {"type":"signal","from":i,"to":j,"text":"..."}
    highlight {"type":"highlight","nodes":[i,..]}
    exit      {"type":"exit"}
"""

import sys
import json
import socket
import argparse

import numpy as np

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                           QRadialGradient)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel, QFrame)

# ── 配色（深色主题 · 现代非AI审美） ──
BG_COLOR = QColor(13, 15, 24)
PANEL_COLOR = QColor(22, 26, 40)
GRID_COLOR = QColor(38, 44, 66)
CLOUD_COLOR = QColor(90, 100, 140, 60)
REAL_COLOR = QColor(255, 205, 66)     # 真实专家：金色
REP_COLOR = QColor(86, 196, 255)      # 神经元代表：青色
EDGE_COLOR = QColor(110, 130, 190, 90)
PARTICLE_COLOR = QColor(255, 138, 76) # 信息粒子：橙色
TEXT_COLOR = QColor(220, 226, 245)
DIM_COLOR = QColor(140, 150, 180)

LEVEL_COLORS = {
    0: QColor(150, 160, 180),
    1: QColor(86, 196, 255),
    2: QColor(110, 240, 170),
    3: QColor(255, 205, 66),
    4: QColor(255, 110, 180),
}

LEVEL_NAMES = {
    0: "L0 直接综合",
    1: "L1 交叉耦合",
    2: "L2 序参量涌现",
    3: "L3 自组织临界",
    4: "L4 量子叠加",
}


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    """6 维向量 PCA 投影到 2D（中心化 → 协方差 → 特征分解）"""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 2))
    mean = vectors.mean(axis=0)
    X = vectors - mean
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    proj = X @ eigvecs[:, idx]
    span = proj.max(axis=0) - proj.min(axis=0)
    span[span < 1e-9] = 1.0
    proj = 2.0 * (proj - proj.min(axis=0)) / span - 1.0
    return proj


class NeuronCanvas(QWidget):
    """神经元点阵画布：绘制云点、节点、连线与信息粒子"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(760, 560)
        self.setAutoFillBackground(True)

        # 数据
        self.all_pts = []        # 背景云点 (归一化坐标)
        self.nodes = []          # [{x, y, r, color, label, kind}]
        self.edges = []          # [(i, j, w)]
        self.particles = []      # [{x1,y1,x2,y2,t,text,color}]
        self.highlight = {}      # node_idx -> expiry_ms
        self.phase_text = ""
        self.level = -1

        # 动画定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    # ── 数据装载 ──
    def load_init(self, payload: dict):
        all_vectors = payload.get("all_vectors", [])
        nodes = payload.get("nodes", {})
        edges = payload.get("edges", [])

        # 背景云点
        self.all_pts = []
        if all_vectors:
            proj = _pca_2d(np.array(all_vectors))
            self.all_pts = [(float(x), float(y)) for x, y in proj]

        # 节点（代表神经元/真实专家）
        self.nodes = []
        node_vectors = nodes.get("vectors", [])
        labels = nodes.get("labels", [])
        kinds = nodes.get("kinds", [])
        if node_vectors:
            nproj = _pca_2d(np.array(node_vectors))
            for idx, (x, y) in enumerate(nproj):
                kind = kinds[idx] if idx < len(kinds) else "rep"
                color = REAL_COLOR if kind == "real" else REP_COLOR
                r = 8.0 if kind == "real" else 5.5
                self.nodes.append({
                    "x": float(x), "y": float(y), "r": r,
                    "color": color,
                    "label": labels[idx] if idx < len(labels) else f"N{idx}",
                    "kind": kind,
                })

        # 边
        self.edges = [(int(a), int(b), float(w)) for a, b, w in edges]

        self.particles.clear()
        self.highlight.clear()
        self.update()

    def set_phase(self, text: str, level: int = -1):
        self.phase_text = text
        self.level = level
        self.update()

    def add_signal(self, from_i: int, to_j: int, text: str):
        """在线段上添加一个信息传递粒子"""
        # 找到端点坐标
        p1 = self._node_pos(from_i)
        p2 = self._node_pos(to_j)
        if p1 is None or p2 is None:
            # 尝试从边查找
            for a, b, _ in self.edges:
                if a == from_i and b == to_j:
                    p1 = self._node_pos(a)
                    p2 = self._node_pos(b)
                    break
            if p1 is None or p2 is None:
                return
        color = PARTICLE_COLOR
        if from_i == to_j:
            color = QColor(255, 205, 66)
        self.particles.append({
            "x1": p1[0], "y1": p1[1],
            "x2": p2[0], "y2": p2[1],
            "t": 0.0, "text": text, "color": color,
        })
        # 限制粒子数量，防止堆积
        if len(self.particles) > 40:
            self.particles = self.particles[-40:]
        self.update()

    def highlight_nodes(self, indices: list, duration_ms: int = 1600):
        now = self._now()
        for idx in indices:
            self.highlight[idx] = now + duration_ms
        self.update()

    # ── 辅助 ──
    def _now(self):
        import time
        return time.time() * 1000

    def _node_pos(self, idx: int):
        if 0 <= idx < len(self.nodes):
            return (self.nodes[idx]["x"], self.nodes[idx]["y"])
        return None

    def _tick(self):
        moved = False
        now = self._now()
        for p in self.particles:
            p["t"] += 0.018
            if p["t"] > 1.0:
                p["t"] = 1.0
            moved = True
        # 清理过期粒子
        self.particles = [p for p in self.particles if p["t"] < 1.0]
        # 清理高亮
        expired = [k for k, v in self.highlight.items() if v < now]
        for k in expired:
            del self.highlight[k]
        if moved or expired:
            self.update()

    # ── 绘制 ──
    def _map(self, x, y):
        """归一化 [-1,1] → 画布坐标"""
        w = self.width() - 120
        h = self.height() - 120
        cx = 60 + w / 2
        cy = 60 + h / 2
        return QPointF(cx + x * w / 2, cy + y * h / 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BG_COLOR)

        # 背景网格
        painter.setPen(QPen(GRID_COLOR, 1))
        step = 60
        for gx in range(0, self.width(), step):
            painter.drawLine(gx, 0, gx, self.height())
        for gy in range(0, self.height(), step):
            painter.drawLine(0, gy, self.width(), gy)

        # 背景云点
        if self.all_pts:
            painter.setPen(Qt.NoPen)
            for x, y in self.all_pts:
                pt = self._map(x, y)
                painter.setBrush(CLOUD_COLOR)
                painter.drawEllipse(pt, 1.6, 1.6)

        # 边（先画线）
        painter.setPen(QPen(EDGE_COLOR, 1))
        for a, b, w in self.edges:
            if a >= len(self.nodes) or b >= len(self.nodes):
                continue
            p1 = self._map(self.nodes[a]["x"], self.nodes[a]["y"])
            p2 = self._map(self.nodes[b]["x"], self.nodes[b]["y"])
            painter.drawLine(p1, p2)

        # 节点（后画点）
        for idx, node in enumerate(self.nodes):
            pt = self._map(node["x"], node["y"])
            is_hl = idx in self.highlight
            # 高亮光晕
            if is_hl:
                glow = QRadialGradient(pt, 26)
                c = QColor(node["color"])
                c.setAlpha(120)
                glow.setColorAt(0, c)
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(glow)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(pt, 26, 26)

            r = node["r"] * (1.4 if is_hl else 1.0)
            painter.setBrush(QBrush(node["color"]))
            painter.setPen(QPen(QColor(0, 0, 0, 80), 1))
            painter.drawEllipse(pt, r, r)

            # 标签（仅真实专家 + 代表）
            if node["kind"] in ("real",) or is_hl:
                painter.setPen(TEXT_COLOR)
                font = painter.font()
                font.setPointSize(7)
                painter.setFont(font)
                painter.drawText(
                    QPointF(pt.x() + r + 3, pt.y() + 3),
                    node["label"][:12]
                )

        # 信息粒子（最上层）
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for p in self.particles:
            t = p["t"]
            x = p["x1"] + (p["x2"] - p["x1"]) * t
            y = p["y1"] + (p["y2"] - p["y1"]) * t
            pt = self._map(x, y)
            # 粒子本体（光晕）
            glow = QRadialGradient(pt, 16)
            c = p["color"]
            c2 = QColor(c)
            c2.setAlpha(70)
            glow.setColorAt(0, c)
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(glow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(pt, 16, 16)
            painter.setBrush(QBrush(p["color"]))
            painter.drawEllipse(pt, 3.4, 3.4)

            # 消息气泡
            if p["text"]:
                label = p["text"][:18]
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label) + 12
                th = fm.height() + 6
                bx = pt.x() + 10
                by = pt.y() - th - 4
                # 限制在画布内
                bx = min(bx, self.width() - tw - 6)
                by = max(by, 4)
                bubble = QColor(24, 30, 48, 230)
                painter.setPen(QPen(QColor(c), 1))
                painter.setBrush(bubble)
                painter.drawRoundedRect(
                    int(bx), int(by), int(tw), int(th), 6, 6
                )
                painter.setPen(TEXT_COLOR)
                painter.drawText(
                    int(bx) + 6, int(by) + th - 5, label
                )

        # 底部相位文字
        if self.phase_text:
            painter.setPen(DIM_COLOR)
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                QPointF(70, self.height() - 22), self.phase_text
            )

        # 顶部层级徽章
        if self.level >= 0:
            lc = LEVEL_COLORS.get(self.level, DIM_COLOR)
            badge = LEVEL_NAMES.get(self.level, f"L{self.level}")
            painter.setPen(QPen(lc, 1))
            painter.setBrush(QColor(lc.red(), lc.green(), lc.blue(), 40))
            fm = painter.fontMetrics()
            bw = fm.horizontalAdvance(badge) + 20
            painter.drawRoundedRect(
                self.width() - bw - 30, 18, int(bw), 28, 14, 14
            )
            painter.setPen(lc)
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                int(self.width() - bw - 30 + 10),
                18 + 19,
                badge
            )

        painter.end()


class NeuronMapWindow(QMainWindow):
    """神经元点阵图主窗口"""

    def __init__(self, port: int):
        super().__init__()
        self.setWindowTitle("神经元高维点阵图 · 整合意识")
        self.resize(900, 680)
        self.setStyleSheet("background-color: #0d0f18; color: #dce2f5;")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 顶部信息条
        top = QHBoxLayout()
        title = QLabel("🧠 神经元高维点阵图")
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #86c4ff; background: #161a28;"
            "padding: 8px 14px; border-radius: 8px; border: 1px solid #2a3050;"
        )
        legend = QLabel(
            "● 真实专家    ● 神经元代表    ➜ 信息传递"
        )
        legend.setStyleSheet("color: #8c96b4; font-size: 12px; padding: 8px;")
        top.addWidget(title)
        top.addStretch()
        top.addWidget(legend)
        layout.addLayout(top)

        # 画布
        self.canvas = NeuronCanvas()
        layout.addWidget(self.canvas, 1)

        # 状态栏
        status_frame = QFrame()
        status_frame.setStyleSheet(
            "background: #161a28; border: 1px solid #2a3050; border-radius: 8px;"
        )
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 6, 12, 6)
        self.status_label = QLabel("等待推理事件...")
        self.status_label.setStyleSheet("color: #8c96b4; font-size: 12px;")
        self.phase_label = QLabel("")
        self.phase_label.setStyleSheet("color: #86c4ff; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.phase_label)
        layout.addWidget(status_frame)

        self.setCentralWidget(central)

        # UDP 监听
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.setblocking(False)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._drain)
        self._poll.start(30)

        self.show()
        self.raise_()

    # ── UDP 事件解析 ──
    def _drain(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(65536)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                evt = json.loads(data.decode("utf-8"))
                self._handle(evt)
            except Exception:
                continue

    def _handle(self, evt: dict):
        etype = evt.get("type", "")
        if etype == "init":
            self.canvas.load_init(evt)
            self.status_label.setText(
                f"神经元已装载: {len(self.canvas.nodes)} 节点 · "
                f"{len(self.canvas.all_pts)} 相空间点 · {len(self.canvas.edges)} 连接"
            )
        elif etype == "phase":
            self.canvas.set_phase(evt.get("text", ""), evt.get("level", -1))
            self.phase_label.setText(f"推理中: {evt.get('text', '')}")
        elif etype == "signal":
            self.canvas.add_signal(evt.get("from", 0), evt.get("to", 0),
                                   evt.get("text", ""))
        elif etype == "highlight":
            self.canvas.highlight_nodes(evt.get("nodes", []))
        elif etype == "status":
            self.status_label.setText(evt.get("text", ""))
        elif etype == "exit":
            QTimer.singleShot(100, self.close)

    def closeEvent(self, event):
        try:
            self._sock.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="神经元高维点阵图")
    parser.add_argument("--port", type=int, default=52000, help="UDP 监听端口")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = NeuronMapWindow(args.port)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
