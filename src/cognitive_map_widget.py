"""
思维可视化认知地图 —— 力导向图实时呈现精华池观点生态

核心功能：
- 力导向布局：节点=精华，边=关系（引用/深化/反驳/澄清）
- 动态演化：每轮结束后自动更新，显示"思想生态"的生长
- 交互：点击节点展开详情，拖拽平移，滚轮缩放
"""

import math
import random
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                            QFontMetrics)
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem,
                                QGraphicsEllipseItem, QGraphicsLineItem,
                                QGraphicsTextItem, QWidget, QVBoxLayout,
                                QHBoxLayout, QLabel, QPushButton,
                                QDialog, QToolTip, QSizePolicy)


# ── 颜色方案 ──
NODE_COLORS = {
    "new":       QColor(66, 133, 244),    # 蓝色 - 新观点
    "refine":    QColor(52, 168, 83),     # 绿色 - 深化
    "challenge": QColor(234, 67, 53),     # 红色 - 反驳
    "clarify":   QColor(251, 188, 4),     # 黄色 - 澄清
    "meta":      QColor(154, 71, 215),    # 紫色 - 元讨论
    "default":   QColor(158, 158, 158),   # 灰色 - 默认
}

EDGE_COLORS = {
    "refine":    QColor(52, 168, 83, 120),
    "challenge": QColor(234, 67, 53, 100),
    "clarify":   QColor(251, 188, 4, 100),
    "cite":      QColor(66, 133, 244, 80),
    "default":   QColor(200, 200, 200, 60),
}

PLAYER_COLORS = [
    QColor(66, 133, 244),
    QColor(234, 67, 53),
    QColor(52, 168, 83),
    QColor(251, 188, 4),
    QColor(154, 71, 215),
    QColor(255, 112, 67),
    QColor(0, 188, 212),
    QColor(233, 30, 99),
]


class ForceDirectedLayout:
    """力导向布局算法"""

    def __init__(self, width: float = 800, height: float = 600,
                 repulsion: float = 3000, attraction: float = 0.01,
                 damping: float = 0.85, iterations: int = 100):
        self.width = width
        self.height = height
        self.repulsion = repulsion
        self.attraction = attraction
        self.damping = damping
        self.iterations = iterations
        self.nodes: Dict[int, Dict] = {}  # id -> {x, y, vx, vy, radius}
        self.edges: List[Tuple[int, int]] = []  # [(from_id, to_id)]

    def set_nodes(self, node_ids: List[int], radii: List[float]):
        """设置节点，随机初始化位置"""
        random.seed(42)
        self.nodes = {}
        for nid, radius in zip(node_ids, radii):
            margin = max(radius * 2, 30)
            self.nodes[nid] = {
                "x": random.uniform(margin, self.width - margin),
                "y": random.uniform(margin, self.height - margin),
                "vx": 0.0,
                "vy": 0.0,
                "radius": max(radius, 8),
            }

    def set_edges(self, edges: List[Tuple[int, int]]):
        self.edges = edges

    def run(self) -> None:
        """运行力导向布局模拟"""
        if len(self.nodes) < 2:
            return

        items = list(self.nodes.items())
        for _ in range(self.iterations):
            forces = {nid: {"fx": 0.0, "fy": 0.0} for nid in self.nodes}

            # 斥力：所有节点之间
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    nid1, n1 = items[i]
                    nid2, n2 = items[j]
                    dx = n1["x"] - n2["x"]
                    dy = n1["y"] - n2["y"]
                    dist = math.hypot(dx, dy) + 1
                    min_dist = n1["radius"] + n2["radius"] + 10
                    if dist < min_dist:
                        dist = min_dist
                    force = self.repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[nid1]["fx"] += fx
                    forces[nid1]["fy"] += fy
                    forces[nid2]["fx"] -= fx
                    forces[nid2]["fy"] -= fy

            # 引力：沿边连接
            for from_id, to_id in self.edges:
                if from_id in self.nodes and to_id in self.nodes:
                    n1 = self.nodes[from_id]
                    n2 = self.nodes[to_id]
                    dx = n2["x"] - n1["x"]
                    dy = n2["y"] - n1["y"]
                    dist = math.hypot(dx, dy) + 1
                    force = self.attraction * dist
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[from_id]["fx"] += fx
                    forces[from_id]["fy"] += fy
                    forces[to_id]["fx"] -= fx
                    forces[to_id]["fy"] -= fy

            # 中心引力
            cx, cy = self.width / 2, self.height / 2
            for nid, n in self.nodes.items():
                dx = cx - n["x"]
                dy = cy - n["y"]
                dist = math.hypot(dx, dy) + 1
                center_force = 0.001
                forces[nid]["fx"] += center_force * dx
                forces[nid]["fy"] += center_force * dy

            # 更新位置
            for nid, n in self.nodes.items():
                n["vx"] = (n["vx"] + forces[nid]["fx"]) * self.damping
                n["vy"] = (n["vy"] + forces[nid]["fy"]) * self.damping
                n["x"] += n["vx"]
                n["y"] += n["vy"]
                # 边界约束
                margin = n["radius"] + 5
                n["x"] = max(margin, min(self.width - margin, n["x"]))
                n["y"] = max(margin, min(self.height - margin, n["y"]))

    def get_position(self, node_id: int) -> Tuple[float, float]:
        """获取节点位置"""
        if node_id in self.nodes:
            return self.nodes[node_id]["x"], self.nodes[node_id]["y"]
        return 0, 0


class NodeItem(QGraphicsEllipseItem):
    """表示精华节点的图形项"""

    def __init__(self, node_id: int, x: float, y: float, radius: float,
                 color: QColor, label: str, tooltip: str, parent=None):
        rect = QRectF(-radius, -radius, radius * 2, radius * 2)
        super().__init__(rect, parent)
        self.node_id = node_id
        self._radius = radius
        self._color = color
        self._label_text = label
        self._tooltip_text = tooltip

        self.setBrush(QBrush(color))
        pen = QPen(QColor(255, 255, 255, 80), 1.5)
        self.setPen(pen)
        self.setZValue(10)
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # 标签文字
        self._label = QGraphicsTextItem(self)
        self._label.setPlainText(label)
        font = QFont("Microsoft YaHei", 8)
        self._label.setFont(font)
        self._label.setDefaultTextColor(QColor(255, 255, 255, 200))
        self._label.setPos(-radius, -radius - 18)

    def hoverEnterEvent(self, event):
        self.setScale(1.3)
        QToolTip.showText(event.screenPos(), self._tooltip_text, self)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setSelected(True)
            QToolTip.showText(event.screenPos(), self._tooltip_text, self)
        super().mousePressEvent(event)


class EdgeItem(QGraphicsLineItem):
    """表示节点间关系的图形项"""

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 color: QColor, width: float = 1.5, parent=None):
        super().__init__(x1, y1, x2, y2, parent)
        pen = QPen(color, width)
        pen.setStyle(Qt.DashLine if color.alpha() < 80 else Qt.SolidLine)
        self.setPen(pen)
        self.setZValue(0)


class CognitiveMapWidget(QGraphicsView):
    """认知地图主控件"""

    nodeClicked = Signal(int)  # 发出被点击的节点ID

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 30)))
        self.setFrameShape(QWidget.NoFrame)

        self._node_items: Dict[int, NodeItem] = {}
        self._edge_items: List[EdgeItem] = []
        self._layout = ForceDirectedLayout()
        self._running = False

        # 图例
        self._legend_items = []

    def build_map(self, essences: List[Dict],
                  player_names: List[str] = None) -> None:
        """
        从精华数据构建认知地图。

        essences: [{
            "id": int, "content": str, "contributor": str,
            "score": float, "tags": list, "type": str,
            "parent_id": int or None, "source_round": int,
            "challenged_by": list, "cited_by": list,
        }]
        """
        self._scene.clear()
        self._node_items.clear()
        self._edge_items = []

        if not essences:
            text = self._scene.addText("暂无精华数据", QFont("Microsoft YaHei", 14))
            text.setDefaultTextColor(QColor(150, 150, 150))
            text.setPos(200, 250)
            return

        # 构建节点
        player_color_map = {}
        if player_names:
            for i, name in enumerate(player_names):
                player_color_map[name] = PLAYER_COLORS[i % len(PLAYER_COLORS)]

        node_ids = [e["id"] for e in essences]
        scores = [min(e.get("score", 1), 5) for e in essences]
        base_radius = 12
        radii = [base_radius + s * 4 for s in scores]

        # 布局
        scene_w, scene_h = max(600, len(essences) * 60), 600
        self._layout = ForceDirectedLayout(scene_w, scene_h)
        self._layout.set_nodes(node_ids, radii)

        # 构建边
        edges = []
        edge_types = {}  # (from, to) -> type
        for e in essences:
            pid = e.get("parent_id")
            if pid is not None and pid in node_ids:
                edges.append((pid, e["id"]))
                edge_types[(pid, e["id"])] = e.get("type", "refine")
            # 反驳关系
            for challenger_id in e.get("challenged_by", []):
                if isinstance(challenger_id, int) and challenger_id in node_ids:
                    edges.append((challenger_id, e["id"]))
                    edge_types[(challenger_id, e["id"])] = "challenge"
            # 引用关系
            for cited_id in e.get("cited_by", []):
                if isinstance(cited_id, int) and cited_id in node_ids:
                    edges.append((e["id"], cited_id))
                    edge_types[(e["id"], cited_id)] = "cite"

        self._layout.set_edges(edges)
        self._layout.run()

        # 绘制边
        for from_id, to_id in edges:
            x1, y1 = self._layout.get_position(from_id)
            x2, y2 = self._layout.get_position(to_id)
            etype = edge_types.get((from_id, to_id), "default")
            color = EDGE_COLORS.get(etype, EDGE_COLORS["default"])
            edge_item = EdgeItem(x1, y1, x2, y2, color)
            self._scene.addItem(edge_item)
            self._edge_items.append(edge_item)

        # 绘制节点
        for e in essences:
            nid = e["id"]
            contributor = e.get("contributor", "?")
            score = e.get("score", 1)
            content = e.get("content", "")[:80]
            etype = e.get("type", "default")
            source_round = e.get("source_round", 0)

            # 节点颜色
            if etype in NODE_COLORS:
                color = QColor(NODE_COLORS[etype])
            elif contributor in player_color_map:
                color = QColor(player_color_map[contributor])
            else:
                color = QColor(NODE_COLORS["default"])

            x, y = self._layout.get_position(nid)
            radius = base_radius + min(score, 5) * 4
            # 高评分节点不透明，低评分半透明
            alpha = int(120 + 135 * min(score / 5.0, 1.0))
            color.setAlpha(alpha)

            label = contributor[:6]
            tooltip = (
                f"#{nid} | {contributor} | 评分: {score:.1f} | 第{source_round}轮\n"
                f"{content}"
            )
            node = NodeItem(nid, x, y, radius, color, label, tooltip)
            node.nodeClicked.connect(self.nodeClicked)
            self._scene.addItem(node)
            self._node_items[nid] = node

        # 添加图例
        self._add_legend()

        # 适应视图
        self.fitInView(self._scene.sceneRect().adjusted(-50, -50, 50, 50),
                       Qt.KeepAspectRatio)

    def _add_legend(self) -> None:
        """添加图例"""
        y = 20
        for etype, color in NODE_COLORS.items():
            label = etype.capitalize()
            item = self._scene.addEllipse(20, y, 12, 12, QPen(Qt.NoPen), QBrush(color))
            text = self._scene.addText(label, QFont("Microsoft YaHei", 9))
            text.setDefaultTextColor(QColor(200, 200, 200))
            text.setPos(38, y - 3)
            y += 20

    def update_from_essence_pool(self, essence_pool,
                                 player_names: List[str] = None) -> None:
        """从 EssencePool 对象更新地图"""
        items = essence_pool.items if hasattr(essence_pool, 'items') else []
        essences = []
        for e in items:
            essences.append({
                "id": e.id,
                "content": e.content,
                "contributor": e.contributor,
                "score": e.score,
                "tags": list(e.tags) if hasattr(e, 'tags') else [],
                "type": "new" if not e.parent_id else "refine" if e.parent_id else "default",
                "parent_id": e.parent_id,
                "source_round": e.source_round,
                "challenged_by": list(e.challenged_by) if hasattr(e, 'challenged_by') else [],
                "cited_by": list(e.cited_by) if hasattr(e, 'cited_by') else [],
            })
        self.build_map(essences, player_names)

    def wheelEvent(self, event):
        """滚轮缩放"""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)


class CognitiveMapDialog(QDialog):
    """认知地图对话框"""

    def __init__(self, essence_pool, player_names: List[str] = None,
                 title: str = "🧠 思维可视化 · 认知地图", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 650)
        self.setStyleSheet("""
            QDialog {
                background-color: #12121e;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_label = QLabel(f"    {title}")
        title_label.setStyleSheet("color: white; font-size: 16px; padding: 10px;")
        layout.addWidget(title_label)

        # 地图
        self.map_widget = CognitiveMapWidget(self)
        self.map_widget.update_from_essence_pool(essence_pool, player_names)
        layout.addWidget(self.map_widget)

        # 更新按钮
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 5, 10, 10)

        self.btn_refresh = QPushButton("🔄 刷新")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #333; color: white; padding: 6px 16px;
                border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #555; }
        """)
        self.btn_refresh.clicked.connect(
            lambda: self.map_widget.update_from_essence_pool(essence_pool, player_names)
        )
        btn_layout.addWidget(self.btn_refresh)

        btn_layout.addStretch()

        self.btn_close = QPushButton("关闭")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #555; color: white; padding: 6px 16px;
                border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #777; }
        """)
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)


def show_cognitive_map_dialog(essence_pool, player_names: List[str] = None,
                               parent=None) -> None:
    """便捷函数：显示认知地图对话框"""
    dialog = CognitiveMapDialog(essence_pool, player_names, parent=parent)
    dialog.exec_()


# ── CLI 模式文本版认知地图 ──

def text_cognitive_map(essence_pool) -> str:
    """
    生成文字版认知地图（用于CLI模式）。
    显示节点和边的树状结构。
    """
    items = essence_pool.items if hasattr(essence_pool, 'items') else []
    if not items:
        return "（精华池为空，无法生成认知地图）"

    lines = []
    lines.append("🧠 认知地图 · 观点关系网")
    lines.append("=" * 50)

    # 按轮次分组
    by_round = defaultdict(list)
    for e in items:
        by_round[e.source_round].append(e)

    # 为每个精华显示其关系
    for rnd in sorted(by_round.keys()):
        round_essences = by_round[rnd]
        for e in round_essences:
            score_bar = "★" * int(min(e.score, 5)) + "☆" * (5 - int(min(e.score, 5)))
            etype = "新观点" if not e.parent_id else "深化" if e.parent_id else "?"
            tag_str = ",".join(e.tags) if hasattr(e, 'tags') and e.tags else ""

            # 节点
            lines.append(f"  #{e.id} [{score_bar}] {e.contributor}: {e.content[:50]}...")
            if tag_str:
                lines.append(f"    标签: {tag_str}")

            # 关系边
            if e.parent_id:
                parent = next((x for x in items if x.id == e.parent_id), None)
                if parent:
                    lines.append(f"    └─ 深化自 #{e.parent_id} ({parent.contributor})")
            if e.challenged_by:
                for cid in e.challenged_by:
                    challenger = next((x for x in items if x.id == cid), None)
                    if challenger:
                        lines.append(f"    ⚔️ 反驳 #{cid} ({challenger.contributor})")
                    else:
                        lines.append(f"    ⚔️ 反驳关系 #{cid}")
            if e.cited_by:
                lines.append(f"    📎 被 {len(e.cited_by)} 条后续精华引用")

        if by_round.get(rnd):
            lines.append(f"  ─ 第{rnd}轮结束 ─")

    lines.append("=" * 50)
    lines.append(f"总计: {len(items)} 条精华, "
                 f"{sum(1 for e in items if e.parent_id)} 条关系边")
    return "\n".join(lines)