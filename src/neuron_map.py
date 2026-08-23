#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经元高维点阵图 —— 整合意识可视化窗口（3D 投影版）

功能：
- 将 6 维认知相空间中的专家/虚拟专家神经元用 PCA 投影到 3D
- 鼠标拖拽旋转视角，滚轮缩放
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
    cognitive_center  {"type":"cognitive_center","vector":[6d],"all_vectors":[[..6d]..]}
    exit      {"type":"exit"}
"""

import sys
import json
import socket
import argparse
import math

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


def _pca_3d(vectors: np.ndarray) -> np.ndarray:
    """6 维向量 PCA 投影到 3D"""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[0] < 2:
        return np.zeros((vectors.shape[0], 3))
    mean = vectors.mean(axis=0)
    X = vectors - mean
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:3]
    proj = X @ eigvecs[:, idx]
    # 归一化到 [-1, 1]
    for d in range(3):
        span = proj[:, d].max() - proj[:, d].min()
        if span > 1e-9:
            proj[:, d] = 2.0 * (proj[:, d] - proj[:, d].min()) / span - 1.0
        else:
            proj[:, d] = 0.0
    return proj


class OrbitCamera:
    """3D 轨道摄像机：旋转 + 缩放"""

    def __init__(self):
        self.theta = 0.8          # 水平旋转角（弧度）
        self.phi = 0.4            # 垂直旋转角（弧度）
        self.zoom = 1.0           # 缩放
        self._last_pos = None     # 上次鼠标位置

    def rotate(self, dx: float, dy: float):
        """鼠标拖拽旋转"""
        self.theta += dx * 0.008
        self.phi += dy * 0.008
        self.phi = max(-math.pi / 2.1, min(math.pi / 2.1, self.phi))

    def zoom_in(self, factor: float = 1.1):
        self.zoom = max(0.2, min(5.0, self.zoom * factor))

    def zoom_out(self, factor: float = 1.1):
        self.zoom = max(0.2, min(5.0, self.zoom / factor))

    def project(self, x: float, y: float, z: float):
        """3D 点 → 2D 屏幕坐标（带透视）"""
        # 旋转矩阵：绕 Y 轴
        ct, st = math.cos(self.theta), math.sin(self.theta)
        # 先绕 Y 轴旋转
        rx = x * ct + z * st
        rz = -x * st + z * ct
        # 再绕 X 轴旋转
        cp, sp = math.cos(self.phi), math.sin(self.phi)
        ry = y * cp - rz * sp
        rz2 = y * sp + rz * cp

        # 透视投影
        perspective = 3.0
        scale = self.zoom * perspective / (perspective + rz2)
        return rx * scale, ry * scale, rz2

    def reset(self):
        self.theta = 0.8
        self.phi = 0.4
        self.zoom = 1.0


class NeuronCanvas(QWidget):
    """神经元点阵画布：3D 投影 + 鼠标交互"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(760, 560)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

        # 数据
        self.all_pts = []        # 背景云点 (3D 坐标)
        self.nodes = []          # [{x, y, z, r, color, label, kind}]
        self.edges = []          # [(i, j, w)]
        self.cloud_edges = []    # 云点之间的连接 [(i, j)]
        self.particles = []      # [{x1,y1,z1, x2,y2,z2, t, text, color}]
        self.highlight = {}      # node_idx -> expiry_ms
        self.phase_text = ""
        self.level = -1

        # 摄像机
        self.cam = OrbitCamera()

        # Item 19: 认知重心演化轨迹
        self.cognitive_trajectory = []  # [(x, y, z, round), ...]

        # ── 拓扑过渡动画 ──
        # 每次收到 init 事件后，从旧状态平滑过渡到新状态
        self._morph_t = 1.0       # 0.0 → 1.0，1.0 = 过渡完成
        self._morph_old = None    # {nodes, edges, all_pts} 旧状态快照

        # ── 持续信息流动（合成/推理阶段） ──
        self._active_phase = False     # 是否处于推理/输出阶段
        self._signal_buffer = []        # 信号缓冲区（持续重放）
        self._signal_idx = 0            # 当前播放的信号索引

        # 动画定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    # ── 平滑过渡数据装载 ──
    def load_init(self, payload: dict):
        """从旧状态平滑过渡到新状态"""
        all_vectors = payload.get("all_vectors", [])
        nodes = payload.get("nodes", {})
        edges = payload.get("edges", [])

        # 保存旧状态快照
        self._morph_old = {
            "all_pts": list(self.all_pts),
            "nodes": [dict(n) for n in self.nodes],
            "edges": list(self.edges),
        }

        # 计算新状态
        new_pts = []
        if all_vectors:
            proj = _pca_3d(np.array(all_vectors))
            new_pts = [(float(x), float(y), float(z)) for x, y, z in proj]

        new_nodes = []
        node_vectors = nodes.get("vectors", [])
        labels = nodes.get("labels", [])
        kinds = nodes.get("kinds", [])
        if node_vectors:
            nproj = _pca_3d(np.array(node_vectors))
            for idx, (x, y, z) in enumerate(nproj):
                kind = kinds[idx] if idx < len(kinds) else "rep"
                color = REAL_COLOR if kind == "real" else REP_COLOR
                r = 8.0 if kind == "real" else 5.5
                new_nodes.append({
                    "x": float(x), "y": float(y), "z": float(z), "r": r,
                    "color": color,
                    "label": labels[idx] if idx < len(labels) else f"N{idx}",
                    "kind": kind,
                })

        new_edges = [(int(a), int(b), float(w)) for a, b, w in edges]

        # 计算云点之间的连接（灰色小点之间的网络，k近邻法）
        new_cloud_edges = []
        if len(new_pts) >= 2:
            pts = np.array(new_pts, dtype=np.float64)
            seen = set()
            for i in range(len(pts)):
                dists = np.sum((pts - pts[i]) ** 2, axis=1)
                dists[i] = np.inf
                nearest = np.argsort(dists)[:2]  # 最近2个邻居
                for j in nearest:
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        new_cloud_edges.append((i, j))

        # 如果当前没有旧状态（首次加载），直接设置
        if not self.nodes:
            self.all_pts = new_pts
            self.nodes = new_nodes
            self.edges = new_edges
            self.cloud_edges = new_cloud_edges
            self._morph_t = 1.0
            self._morph_old = None
        else:
            # 保存新目标，开始过渡
            self._morph_target = {
                "all_pts": new_pts,
                "nodes": new_nodes,
                "edges": new_edges,
                "cloud_edges": new_cloud_edges,
            }
            self._morph_t = 0.0

        self.particles.clear()
        self.highlight.clear()
        self.update()

    def update_cognitive_center(self, payload: dict):
        """
        更新虚拟专家云点和认知重心（神经认知反馈闭环的可视化）。

        来自 emergence.py 合成的 cognitive_center 事件：
        {
            "type": "cognitive_center",
            "vector": [6d] P_final 认知重心向量,
            "all_vectors": [[..6d]..] 更新后的虚拟专家云点
        }
        """
        vector = payload.get("vector", None)
        all_vectors = payload.get("all_vectors", [])
        if vector:
            # 将认知重心向量作为"发光节点"显示
            vec3d = _pca_3d(np.array([vector]))[0]
            cx, cy, cz = float(vec3d[0]), float(vec3d[1]), float(vec3d[2])
            # Item 19: 记录认知重心轨迹
            round_num = payload.get("round", len(self.cognitive_trajectory))
            self.cognitive_trajectory.append((cx, cy, cz, round_num))
            # 保留最近 50 个轨迹点
            if len(self.cognitive_trajectory) > 50:
                self.cognitive_trajectory = self.cognitive_trajectory[-50:]
            # 认知重心节点（高亮金色）
            center_node = {
                "x": cx, "y": cy, "z": cz, "r": 12.0,
                "color": (255, 215, 0),  # 金色
                "label": "认知重心", "kind": "center",
            }
            # 替换或追加认知重心节点
            for i, n in enumerate(self.nodes):
                if n.get("kind") == "center":
                    self.nodes[i] = center_node
                    break
            else:
                self.nodes.append(center_node)

        if all_vectors:
            # 更新云点
            proj = _pca_3d(np.array(all_vectors))
            new_pts = [(float(x), float(y), float(z)) for x, y, z in proj]
            # 保留旧云点做平滑过渡
            old_pts = list(self.all_pts)
            self.all_pts = new_pts
            # 更新云点之间的连接
            if len(new_pts) >= 2:
                pts = np.array(new_pts, dtype=np.float64)
                seen = set()
                new_cloud_edges = []
                for i in range(len(pts)):
                    dists = np.sum((pts - pts[i]) ** 2, axis=1)
                    dists[i] = np.inf
                    nearest = np.argsort(dists)[:2]
                    for j in nearest:
                        key = (min(i, j), max(i, j))
                        if key not in seen:
                            seen.add(key)
                            new_cloud_edges.append((i, j))
                self.cloud_edges = new_cloud_edges

        self.update()

    # ── 获取当前过渡插值数据 ──
    def _get_morph_data(self):
        """返回当前过渡帧的数据（插值后的状态）"""
        if self._morph_t >= 1.0 or self._morph_target is None:
            return  # 过渡完成或无过渡，直接使用当前数据

        t = self._morph_t  # 0~1
        # 缓动函数：ease-out cubic，让动画更自然
        ease = 1.0 - (1.0 - t) ** 3

        old = self._morph_old
        target = self._morph_target

        # 云点插值（逐点一一对应）
        old_pts = old["all_pts"]
        target_pts = target["all_pts"]
        if old_pts and target_pts:
            n = min(len(old_pts), len(target_pts))
            self.all_pts = []
            for i in range(n):
                ox, oy, oz = old_pts[i]
                tx, ty, tz = target_pts[i]
                self.all_pts.append((
                    ox + (tx - ox) * ease,
                    oy + (ty - oy) * ease,
                    oz + (tz - oz) * ease,
                ))
            # 补齐多余的点（直接用旧/新值）
            if len(old_pts) > n:
                self.all_pts.extend(old_pts[n:])
            if len(target_pts) > n:
                self.all_pts.extend(target_pts[n:])
        elif target_pts:
            self.all_pts = list(target_pts)
        else:
            self.all_pts = []

        # 节点插值（按索引一一对应）
        old_nodes = old["nodes"]
        target_nodes = target["nodes"]
        old_n = len(old_nodes)
        target_n = len(target_nodes)
        n = min(old_n, target_n)
        self.nodes = []
        for i in range(n):
            on, tn = old_nodes[i], target_nodes[i]
            self.nodes.append({
                "x": on["x"] + (tn["x"] - on["x"]) * ease,
                "y": on["y"] + (tn["y"] - on["y"]) * ease,
                "z": on["z"] + (tn["z"] - on["z"]) * ease,
                "r": on["r"] + (tn["r"] - on["r"]) * ease,
                "color": tn["color"],  # 用目标颜色
                "label": tn["label"],
                "kind": tn["kind"],
            })
        # 旧节点淡出（多余索引）
        for i in range(n, old_n):
            c = QColor(old_nodes[i]["color"])
            c.setAlpha(int(255 * (1.0 - ease)))
            node = dict(old_nodes[i])
            node["color"] = c
            node["r"] *= (1.0 - ease * 0.5)
            self.nodes.append(node)
        # 新节点淡入（多余索引）
        for i in range(n, target_n):
            c = QColor(target_nodes[i]["color"])
            c.setAlpha(int(255 * ease))
            node = dict(target_nodes[i])
            node["color"] = c
            node["r"] *= (0.2 + ease * 0.8)
            self.nodes.append(node)

        # 边：过渡期间混合新旧
        old_edges = old["edges"]
        target_edges = target["edges"]
        self.edges = []
        # 旧边淡出
        for a, b, w in old_edges:
            if not any(a == ta and b == tb for ta, tb, _ in target_edges):
                self.edges.append((a, b, w * (1.0 - ease)))
        # 新边淡入
        for a, b, w in target_edges:
            self.edges.append((a, b, w * ease))

        # 云点连接：过渡期间直接用目标值（点数不变，索引一一对应）
        self.cloud_edges = target.get("cloud_edges", [])

    def set_phase(self, text: str, level: int = -1):
        self.phase_text = text
        self.level = level
        self._active_phase = bool(text)
        if not self._active_phase:
            # 推理结束，清空残留粒子
            self.particles.clear()
            self._signal_buffer = []
            self._signal_idx = 0
        self.update()

    def add_signal(self, from_i: int, to_j: int, text: str):
        """在线段上添加一个信息传递粒子，并自动创建回复信号"""
        p1 = self._node_pos(from_i)
        p2 = self._node_pos(to_j)
        if p1 is None or p2 is None:
            for a, b, _ in self.edges:
                if a == from_i and b == to_j:
                    p1 = self._node_pos(a)
                    p2 = self._node_pos(b)
                    break
            if p1 is None or p2 is None:
                return
        # 原始信号
        color = PARTICLE_COLOR
        if from_i == to_j:
            color = QColor(255, 205, 66)
        self.particles.append({
            "x1": p1[0], "y1": p1[1], "z1": p1[2],
            "x2": p2[0], "y2": p2[1], "z2": p2[2],
            "t": 0.0, "text": text, "color": color,
            "speed": 0.018,
        })
        # 回复信号（本地生成，不调 LLM）：目标 → 来源
        if from_i != to_j:
            _reply_text = f"↩ 收到"
            _rp1 = self._node_pos(to_j)
            _rp2 = self._node_pos(from_i)
            _reply_color = QColor(color)
            _reply_color.setAlpha(160)
            self.particles.append({
                "x1": _rp1[0], "y1": _rp1[1], "z1": _rp1[2],
                "x2": _rp2[0], "y2": _rp2[1], "z2": _rp2[2],
                "t": 0.0, "text": _reply_text, "color": _reply_color,
                "speed": 0.018,
                "_delay": 0.4,  # 延迟帧，等原始信号走一段后再出现
            })
        # 限制粒子总数
        if len(self.particles) > 60:
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
            n = self.nodes[idx]
            return (n["x"], n["y"], n["z"])
        return None

    def _tick(self):
        moved = False
        now = self._now()

        # 拓扑过渡动画推进
        if self._morph_t < 1.0 and hasattr(self, '_morph_target'):
            self._morph_t = min(1.0, self._morph_t + 0.025)  # ~1秒
            self._get_morph_data()
            moved = True
            if self._morph_t >= 1.0:
                self._morph_old = None
                self._morph_target = None

        # ── 信息粒子持续流动（合成/推理阶段） ──
        if self._active_phase and self.edges:
            # 从信号缓冲区逐条发射信号，播完即止（不循环）
            if self._signal_buffer and self._signal_idx < len(self._signal_buffer):
                sig = self._signal_buffer[self._signal_idx]
                self._signal_idx += 1
                self.add_signal(
                    sig.get("from", 0), sig.get("to", 0),
                    sig.get("text", "")
                )
            # 限制粒子总数
            if len(self.particles) > 120:
                self.particles = self.particles[-80:]

        # 推进粒子（带延迟的回复粒子在延迟前不移动）
        for p in self.particles:
            _delay = p.get("_delay", 0.0)
            if _delay > 0:
                p["_delay"] = max(0.0, _delay - 0.03)
                continue
            speed = p.get("speed", 0.018)
            p["t"] += speed
            if p["t"] > 1.0:
                p["t"] = 1.0
            moved = True
        self.particles = [p for p in self.particles if p["t"] < 1.0]
        expired = [k for k, v in self.highlight.items() if v < now]
        for k in expired:
            del self.highlight[k]
        if moved or expired:
            self.update()

    # ── 3D → 2D 映射 ──
    def _map(self, x, y, z):
        """3D 坐标 → 画布 QPointF"""
        sx, sy, _ = self.cam.project(x, y, z)
        w = self.width() - 120
        h = self.height() - 120
        cx = 60 + w / 2
        cy = 60 + h / 2
        return QPointF(cx + sx * w / 2, cy + sy * h / 2)

    def _map_depth(self, x, y, z):
        """返回投影后的深度值（用于排序）"""
        _, _, rz = self.cam.project(x, y, z)
        return rz

    # ── 鼠标交互 ──
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.cam._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.cam._last_pos = None
            self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if self.cam._last_pos is not None:
            pos = event.position()
            dx = pos.x() - self.cam._last_pos.x()
            dy = pos.y() - self.cam._last_pos.y()
            self.cam.rotate(dx, dy)
            self.cam._last_pos = pos
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.cam.zoom_in()
        else:
            self.cam.zoom_out()
        self.update()

    # ── 绘制 ──
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

        # 背景云点之间的连接线（灰色点网络）
        if self.cloud_edges and self.all_pts:
            painter.setPen(QPen(QColor(55, 65, 95, 35), 0.5))
            for i, j in self.cloud_edges:
                if i < len(self.all_pts) and j < len(self.all_pts):
                    x1, y1, z1 = self.all_pts[i]
                    x2, y2, z2 = self.all_pts[j]
                    p1 = self._map(x1, y1, z1)
                    p2 = self._map(x2, y2, z2)
                    painter.drawLine(p1, p2)

        # 背景云点（按深度排序，先画远的）
        if self.all_pts:
            cloud_depth = [(self._map_depth(x, y, z), x, y, z)
                           for x, y, z in self.all_pts]
            cloud_depth.sort(key=lambda t: t[0], reverse=True)
            painter.setPen(Qt.NoPen)
            for _, x, y, z in cloud_depth:
                pt = self._map(x, y, z)
                painter.setBrush(CLOUD_COLOR)
                painter.drawEllipse(pt, 1.6, 1.6)

        # 按深度排序节点和边
        node_depths = [
            (self._map_depth(n["x"], n["y"], n["z"]), i)
            for i, n in enumerate(self.nodes)
        ]
        node_depths.sort(key=lambda t: t[0], reverse=True)
        depth_order = [i for _, i in node_depths]

        # 边（投影到 2D 后绘制）
        edge_alpha = max(30, min(90, int(90 * self.cam.zoom)))
        edge_color = QColor(EDGE_COLOR)
        edge_color.setAlpha(edge_alpha)
        painter.setPen(QPen(edge_color, 1))

        for a, b, w in self.edges:
            if a >= len(self.nodes) or b >= len(self.nodes):
                continue
            na, nb = self.nodes[a], self.nodes[b]
            p1 = self._map(na["x"], na["y"], na["z"])
            p2 = self._map(nb["x"], nb["y"], nb["z"])
            # 根据权重控制线条透明度
            line_alpha = max(40, min(180, int(80 + w * 100)))
            line_color = QColor(edge_color)
            line_color.setAlpha(line_alpha)
            pen_width = max(0.5, w * 2.5)
            painter.setPen(QPen(line_color, pen_width))
            painter.drawLine(p1, p2)

        # 节点（按深度从远到近绘制）
        for idx in depth_order:
            node = self.nodes[idx]
            pt = self._map(node["x"], node["y"], node["z"])
            depth = node_depths[node_depths.index((self._map_depth(
                node["x"], node["y"], node["z"]), idx))][0]
            is_hl = idx in self.highlight

            # 深度缩放：远处的节点小一点
            depth_scale = max(0.5, 1.0 - depth * 0.15)

            # 高亮光晕
            if is_hl:
                glow = QRadialGradient(pt, 26 * depth_scale)
                c = QColor(node["color"])
                c.setAlpha(120)
                glow.setColorAt(0, c)
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(glow)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(pt, 26 * depth_scale, 26 * depth_scale)

            r = node["r"] * depth_scale * (1.4 if is_hl else 1.0)
            painter.setBrush(QBrush(node["color"]))
            painter.setPen(QPen(QColor(0, 0, 0, 80), 1))
            painter.drawEllipse(pt, r, r)

            # 标签
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

        # Item 19: 认知重心演化轨迹（渐变线）
        if len(self.cognitive_trajectory) >= 2:
            for i in range(1, len(self.cognitive_trajectory)):
                x1, y1, z1, _ = self.cognitive_trajectory[i - 1]
                x2, y2, z2, _ = self.cognitive_trajectory[i]
                p1 = self._map(x1, y1, z1)
                p2 = self._map(x2, y2, z2)
                # 渐变色：从暗到亮
                alpha = int(40 + 140 * i / len(self.cognitive_trajectory))
                width = 1.0 + 2.0 * i / len(self.cognitive_trajectory)
                trail_color = QColor(255, 215, 0, alpha)
                painter.setPen(QPen(trail_color, width))
                painter.drawLine(p1, p2)
        for p in self.particles:
            t = p["t"]
            x = p["x1"] + (p["x2"] - p["x1"]) * t
            y = p["y1"] + (p["y2"] - p["y1"]) * t
            z = p["z1"] + (p["z2"] - p["z1"]) * t
            pt = self._map(x, y, z)
            # 粒子光晕
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

        # 左下角状态与操作提示
        painter.setPen(DIM_COLOR)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        if self._active_phase:
            painter.setPen(LEVEL_COLORS.get(self.level, PARTICLE_COLOR) if self.level >= 0 else PARTICLE_COLOR)
            painter.drawText(QPointF(70, self.height() - 6),
                             "● 信息流动中 · 拖拽旋转 · 滚轮缩放")
        else:
            painter.setPen(DIM_COLOR)
            painter.drawText(QPointF(70, self.height() - 6),
                             "拖拽旋转 · 滚轮缩放")

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
        title = QLabel("🧠 神经元高维点阵图 (3D)")
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #86c4ff; background: #161a28;"
            "padding: 8px 14px; border-radius: 8px; border: 1px solid #2a3050;"
        )
        # 节点/边/云点 统计信息栏
        self.info_label = QLabel("节点: 0  ·  连接: 0  ·  云点: 0")
        self.info_label.setStyleSheet(
            "font-size: 12px; color: #dce2f5; background: #161a28;"
            "padding: 8px 14px; border-radius: 8px; border: 1px solid #2a3050;"
        )
        legend = QLabel(
            "● 真实专家    ● 神经元代表    ● 相空间云点    ➜ 信息传递"
        )
        legend.setStyleSheet("color: #8c96b4; font-size: 12px; padding: 8px;")
        top.addWidget(title)
        top.addSpacing(10)
        top.addWidget(self.info_label)
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
            # 顶部信息栏统计
            self.info_label.setText(
                f"节点: {len(self.canvas.nodes)}  ·  连接: {len(self.canvas.edges)}  ·  "
                f"云点: {len(self.canvas.all_pts)}"
            )
            self.status_label.setText("神经元已装载")
        elif etype == "phase":
            self.canvas.set_phase(evt.get("text", ""), evt.get("level", -1))
            self.phase_label.setText(f"推理中: {evt.get('text', '')}")
        elif etype == "signal":
            self.canvas.add_signal(evt.get("from", 0), evt.get("to", 0),
                                   evt.get("text", ""))
        elif etype == "signal_buffer":
            self.canvas._signal_buffer = evt.get("signals", [])
            self.canvas._signal_idx = 0
            self.status_label.setText(
                f"信号缓冲区已加载: {len(self.canvas._signal_buffer)} 条"
            )
        elif etype == "highlight":
            self.canvas.highlight_nodes(evt.get("nodes", []))
        elif etype == "status":
            self.status_label.setText(evt.get("text", ""))
        elif etype == "exit":
            QTimer.singleShot(100, self.close)
        elif etype == "cognitive_center":
            self.canvas.update_cognitive_center(evt)
            self.status_label.setText("认知重心已更新")
        elif etype == "emergence_trajectory":
            traj = evt.get("trajectory", [])
            if traj:
                self.status_label.setText(
                    f"涌现轨迹: {len(traj)} 轮 | 当前 L{evt.get('current_level', '?')}"
                )
        elif etype == "thinking":
            text = evt.get("text", "思考中…")
            self.status_label.setText(f"⏳ {text}")
            # 清除高亮，显示等待状态
            self.canvas.highlighted_nodes = set()

    def closeEvent(self, event):
        try:
            self._sock.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="神经元高维点阵图 (3D)")
    parser.add_argument("--port", type=int, default=52000, help="UDP 监听端口")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = NeuronMapWindow(args.port)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()