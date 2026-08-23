"""
反事实推演沙盘 —— "如果……会怎样？" 假设分析工具

核心功能：
- 加载检查点数据，允许用户修改关键参数
- 模拟"如果某条精华被移除/增强/弱化"会怎样
- 对比原始路径与反事实路径的差异
- 生成"假设分析报告"
"""

import json
import os
import copy
import datetime
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTextEdit, QWidget, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QScrollArea,
)


# ── 反事实引擎 ──

class CounterfactualEngine:
    """
    反事实推演引擎。

    Item 20: 深度集成时间维度耦合记忆，
    支持"如果历史不同会怎样"的跨轮次推演。

    加载一个检查点，应用"what-if"操作，生成模拟结果。
    """

    def __init__(self, data: Dict):
        self.original_data = data
        self.essence_pool_data = data.get("essence_pool", {})
        self.original_items: List[Dict] = copy.deepcopy(
            self.essence_pool_data.get("items", [])
        )
        self.modified_items: List[Dict] = copy.deepcopy(self.original_items)
        self.player_names = data.get("game_record", {}).get("player_names", [])
        self.problem = data.get("problem", data.get("game_record", {}).get("problem", ""))
        self.mode = data.get("discussion_mode", "physical")
        self.rounds = data.get("game_record", {}).get("rounds", [])

        # Item 20: 加载时间维度耦合记忆
        self.temporal_memory_data = data.get("temporal_memory", None)
        self.modified_temporal = copy.deepcopy(self.temporal_memory_data)

        # 操作历史
        self.operations: List[Dict] = []
        self._next_id = max((e.get("id", 0) for e in self.original_items), default=0) + 1

    # ── 修改操作 ──

    def boost_essence(self, item_id: int, amount: float = 3.0) -> bool:
        """增强某条精华的评分（模拟"如果这个观点被更多人认可"）"""
        item = self._find_modified(item_id)
        if not item:
            return False
        item["score"] = item.get("score", 0) + amount
        item["tags"] = list(set(item.get("tags", []) + ["反事实增强"]))
        self.operations.append({
            "type": "boost",
            "item_id": item_id,
            "amount": amount,
            "description": f"增强 #{item_id} 评分 +{amount}",
        })
        return True

    def suppress_essence(self, item_id: int, amount: float = 3.0) -> bool:
        """削弱某条精华的评分（模拟"如果这个观点被反驳"）"""
        item = self._find_modified(item_id)
        if not item:
            return False
        item["score"] = item.get("score", 0) - amount
        item["tags"] = list(set(item.get("tags", []) + ["反事实削弱"]))
        self.operations.append({
            "type": "suppress",
            "item_id": item_id,
            "amount": amount,
            "description": f"削弱 #{item_id} 评分 -{amount}",
        })
        return True

    def remove_essence(self, item_id: int) -> bool:
        """移除一条精华（模拟"如果这个观点从未被提出"）"""
        before = len(self.modified_items)
        self.modified_items = [e for e in self.modified_items if e.get("id") != item_id]
        if len(self.modified_items) < before:
            content = next((e.get("content", "")[:30] for e in self.original_items
                            if e.get("id") == item_id), "?")
            self.operations.append({
                "type": "remove",
                "item_id": item_id,
                "description": f"移除 #{item_id}: \"{content}...\"",
            })
            return True
        return False

    def add_hypothetical(self, content: str, contributor: str = "反事实推演",
                         score: float = 5.0, tags: List[str] = None) -> int:
        """添加一条假设性精华（模拟"如果某人提出了这个观点"）"""
        item = {
            "id": self._next_id,
            "content": content,
            "contributor": contributor,
            "source_round": self.rounds[-1].get("round_id", 1) if self.rounds else 1,
            "round": self.rounds[-1].get("round_id", 1) if self.rounds else 1,
            "score": score,
            "parent_id": None,
            "tags": tags or ["反事实假设"],
            "cited_by": [],
            "refined_by": [],
            "challenged_by": [],
            "approve_by": [],
            "reject_by": [],
            "abstain_by": [],
            "vote_reasons": [],
            "clarifications": [],
        }
        self._next_id += 1
        self.modified_items.append(item)
        self.operations.append({
            "type": "add",
            "item_id": item["id"],
            "description": f"添加假设: \"{content[:40]}...\"",
        })
        return item["id"]

    def change_mode(self, new_mode: str) -> None:
        """切换讨论模式"""
        self.mode = new_mode
        self.operations.append({
            "type": "mode_change",
            "description": f"切换模式: {new_mode}",
        })

    def reset(self) -> None:
        """重置所有修改"""
        self.modified_items = copy.deepcopy(self.original_items)
        self.modified_temporal = copy.deepcopy(self.temporal_memory_data)
        self.operations = []
        self.mode = self.original_data.get("discussion_mode", "physical")

    # ── Item 20: 时间记忆操作 ──

    def boost_temporal_coupling(self, factor: float = 1.5) -> bool:
        """
        "如果历史耦合更强会怎样" —— 增强所有历史耦合强度。
        factor: 耦合强度放大倍数（默认 1.5x）
        """
        if self.modified_temporal is None:
            return False
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if not coupling:
            return False
        matrix = np.array(coupling) * factor
        self.modified_temporal["cumulative_coupling"] = np.clip(matrix, -1.0, 1.0).tolist()
        self.operations.append({
            "type": "temporal_boost",
            "factor": factor,
            "description": f"增强时间耦合 ×{factor}（模拟更强历史协同效应）",
        })
        return True

    def weaken_temporal_coupling(self, factor: float = 0.5) -> bool:
        """
        "如果历史耦合更弱会怎样" —— 削弱所有历史耦合强度。
        factor: 耦合强度衰减倍数（默认 0.5x）
        """
        if self.modified_temporal is None:
            return False
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if not coupling:
            return False
        matrix = np.array(coupling) * factor
        self.modified_temporal["cumulative_coupling"] = matrix.tolist()
        self.operations.append({
            "type": "temporal_weaken",
            "factor": factor,
            "description": f"削弱时间耦合 ×{factor}（模拟更弱历史协同效应）",
        })
        return True

    def reset_temporal_memory(self) -> bool:
        """
        "如果历史从未发生过会怎样" —— 清空时间记忆。
        """
        if self.modified_temporal is None:
            return False
        # 清零耦合矩阵但保留结构
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if coupling:
            self.modified_temporal["cumulative_coupling"] = np.zeros_like(np.array(coupling)).tolist()
        self.modified_temporal["round"] = 0
        self.modified_temporal["topology_history"] = []
        self.modified_temporal["emergence_level_history"] = []
        self.operations.append({
            "type": "temporal_reset",
            "description": "清空时间记忆（模拟历史归零的假设场景）",
        })
        return True

    def get_temporal_stats(self) -> Dict:
        """获取时间记忆统计信息"""
        if self.modified_temporal is None:
            return {"available": False}
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if not coupling:
            return {"available": False, "reason": "无耦合数据"}
        matrix = np.array(coupling)
        active = np.sum(np.abs(matrix) > 0.05)
        total = matrix.shape[0] * (matrix.shape[1] - 1) if matrix.shape[0] > 0 else 1
        density = active / total if total > 0 else 0
        hist = self.modified_temporal.get("emergence_level_history", [])
        level_counts = {}
        for entry in hist:
            lv = entry.get("level", -1)
            level_counts[lv] = level_counts.get(lv, 0) + 1
        return {
            "available": True,
            "shape": list(matrix.shape),
            "active_connections": int(active),
            "connection_density": round(density, 4),
            "rounds_recorded": self.modified_temporal.get("round", 0),
            "topology_entries": len(self.modified_temporal.get("topology_history", [])),
            "emergence_level_distribution": level_counts,
        }

    # ── 分析 & 比较 ──

    def get_original_ranking(self) -> List[Dict]:
        """原始排名"""
        return sorted(self.original_items, key=lambda e: e.get("score", 0), reverse=True)

    def get_modified_ranking(self) -> List[Dict]:
        """修改后排名"""
        return sorted(self.modified_items, key=lambda e: e.get("score", 0), reverse=True)

    def compare_rankings(self) -> Dict:
        """
        比较原始和修改后的排名变化。

        返回：
          {
            "gained": [{"id", "content", "old_rank", "new_rank", "old_score", "new_score", "delta"}],
            "lost": [...],
            "unchanged": [...],
            "new_entries": [...],
            "removed": [...],
          }
        """
        original = self.get_original_ranking()
        modified = self.get_modified_ranking()

        orig_map = {e["id"]: {"rank": i, "score": e.get("score", 0)}
                     for i, e in enumerate(original)}
        mod_map = {e["id"]: {"rank": i, "score": e.get("score", 0)}
                    for i, e in enumerate(modified)}

        gained = []
        lost = []
        unchanged = []
        new_entries = []
        removed = []

        for e in modified:
            eid = e["id"]
            if eid in orig_map:
                old_rank = orig_map[eid]["rank"]
                new_rank = mod_map[eid]["rank"]
                delta = old_rank - new_rank  # 正数 = 排名上升
                if delta > 0:
                    gained.append({
                        "id": eid,
                        "content": e.get("content", "")[:50],
                        "old_rank": old_rank + 1,
                        "new_rank": new_rank + 1,
                        "old_score": orig_map[eid]["score"],
                        "new_score": mod_map[eid]["score"],
                        "delta": delta,
                    })
                elif delta < 0:
                    lost.append({
                        "id": eid,
                        "content": e.get("content", "")[:50],
                        "old_rank": old_rank + 1,
                        "new_rank": new_rank + 1,
                        "old_score": orig_map[eid]["score"],
                        "new_score": mod_map[eid]["score"],
                        "delta": delta,
                    })
                else:
                    unchanged.append(eid)
            else:
                new_entries.append({
                    "id": eid,
                    "content": e.get("content", "")[:50],
                    "score": e.get("score", 0),
                })

        for e in original:
            eid = e["id"]
            if eid not in mod_map:
                removed.append({
                    "id": eid,
                    "content": e.get("content", "")[:50],
                    "old_score": e.get("score", 0),
                })

        gained.sort(key=lambda x: x["delta"], reverse=True)
        lost.sort(key=lambda x: x["delta"])

        return {
            "gained": gained,
            "lost": lost,
            "unchanged": unchanged,
            "new_entries": new_entries,
            "removed": removed,
        }

    def get_statistics(self) -> Dict:
        """获取原始和修改后的统计对比"""
        def calc(items):
            if not items:
                return {"count": 0, "avg_score": 0, "max_score": 0, "min_score": 0, "std_dev": 0}
            scores = [e.get("score", 0) for e in items]
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            return {
                "count": len(items),
                "avg_score": round(avg, 2),
                "max_score": max(scores),
                "min_score": min(scores),
                "std_dev": round(variance ** 0.5, 2),
            }

        return {
            "original": calc(self.original_items),
            "modified": calc(self.modified_items),
        }

    def generate_counterfactual_synthesis(self) -> str:
        """
        基于修改后的精华池，生成"假设性综合方案"文本。
        模拟如果修改生效，最终方案会是什么样。
        """
        stats = self.get_statistics()
        comparison = self.compare_rankings()
        top_original = self.get_original_ranking()[:5]
        top_modified = self.get_modified_ranking()[:5]

        lines = []
        lines.append("=" * 60)
        lines.append("🔮 反事实推演报告")
        lines.append("=" * 60)
        lines.append(f"   问题: {self.problem[:60]}")
        lines.append(f"   模式: {self.mode}")
        lines.append(f"   操作数: {len(self.operations)}")
        lines.append("")

        # 操作记录
        if self.operations:
            lines.append("─" * 40)
            lines.append("📝 推演操作")
            lines.append("─" * 40)
            for op in self.operations:
                lines.append(f"  · {op.get('description', '')}")
            lines.append("")

        # Item 20: 时间记忆信息
        tstats = self.get_temporal_stats()
        if tstats.get("available"):
            lines.append("─" * 40)
            lines.append("⏳ 时间耦合记忆")
            lines.append("─" * 40)
            lines.append(f"  记录轮次: {tstats['rounds_recorded']}")
            lines.append(f"  活跃连接: {tstats['active_connections']} / {tstats['shape'][0]}×{tstats['shape'][1]}")
            lines.append(f"  连接密度: {tstats['connection_density']:.2%}")
            if tstats.get("emergence_level_distribution"):
                dist = tstats["emergence_level_distribution"]
                dist_str = ", ".join(f"L{k}: {v}次" for k, v in sorted(dist.items()))
                lines.append(f"  涌现层级分布: {dist_str}")
            # 有 temporal 操作时添加演化分析
            temporal_ops = [op for op in self.operations if "temporal" in op["type"]]
            if temporal_ops:
                lines.append("  📌 可推演: 耦合强度变化将影响跨轮次认知协同效应")
                if tstats["connection_density"] > 0.3:
                    lines.append("  🔗 当前耦合密度较高，历史协同效应显著")
                elif tstats["connection_density"] < 0.1:
                    lines.append("  🔗 当前耦合密度较低，历史协同效应较弱")
            lines.append("")

        # 统计对比
        lines.append("─" * 40)
        lines.append("📊 统计对比")
        lines.append("─" * 40)
        o = stats["original"]
        m = stats["modified"]
        lines.append(f"  精华数量: {o['count']} → {m['count']}")
        lines.append(f"  平均评分: {o['avg_score']} → {m['avg_score']} "
                      f"({'↑' if m['avg_score'] > o['avg_score'] else '↓'}{abs(m['avg_score'] - o['avg_score']):.1f})")
        lines.append(f"  最高评分: {o['max_score']} → {m['max_score']}")
        lines.append(f"  评分标准差: {o['std_dev']} → {m['std_dev']}")
        lines.append("")

        # 排名变化
        lines.append("─" * 40)
        lines.append("📈 排名变化")
        lines.append("─" * 40)

        if comparison["gained"]:
            lines.append("  🔼 排名上升:")
            for e in comparison["gained"][:5]:
                lines.append(f"    #{e['id']} \"{e['content']}\"  "
                              f"#{e['old_rank']} → #{e['new_rank']} "
                              f"({e['old_score']:.1f} → {e['new_score']:.1f})")
        if comparison["lost"]:
            lines.append("  🔽 排名下降:")
            for e in comparison["lost"][:5]:
                lines.append(f"    #{e['id']} \"{e['content']}\"  "
                              f"#{e['old_rank']} → #{e['new_rank']} "
                              f"({e['old_score']:.1f} → {e['new_score']:.1f})")
        if comparison["new_entries"]:
            lines.append("  ✨ 新增条目:")
            for e in comparison["new_entries"]:
                lines.append(f"    #{e['id']} \"{e['content']}\"  (评分 {e['score']:.1f})")
        if comparison["removed"]:
            lines.append("  🗑️ 已移除:")
            for e in comparison["removed"]:
                lines.append(f"    #{e['id']} \"{e['content']}\"  (原评分 {e['old_score']:.1f})")
        lines.append("")

        # 反事实方案
        lines.append("=" * 60)
        lines.append("🔮 反事实综合方案（假设推演）")
        lines.append("=" * 60)

        if top_modified:
            lines.append(f"\n  在\"{self.problem[:40]}...\"的讨论中，")
            if comparison["gained"]:
                g = comparison["gained"][0]
                lines.append(f"  如果 {g['content'][:30]} 获得更多认可（排名 #{g['old_rank']}→#{g['new_rank']}），")
                lines.append(f"  讨论的焦点将从原有共识转向更强调这一方向。")
            lines.append("")
            lines.append("  💡 核心观点（按影响力排序）:")
            for i, e in enumerate(top_modified, 1):
                tags = ", ".join(e.get("tags", []))[:30]
                lines.append(f"    {i}. {e.get('content', '')[:60]}")
                lines.append(f"       (评分:{e.get('score', 0):.1f} | {e.get('contributor', '?')} | {tags})")
                lines.append("")

        delta = m["avg_score"] - o["avg_score"]
        if delta > 0.5:
            lines.append("  📌 结论: 反事实推演表明，该修改将提升整体讨论质量，")
            lines.append(f"     平均精华评分提高 {delta:.2f}，共识更容易形成。")
        elif delta < -0.5:
            lines.append("  📌 结论: 反事实推演表明，该修改将降低讨论质量，")
            lines.append(f"     平均精华评分下降 {abs(delta):.2f}，可能引发更多分歧。")
        else:
            lines.append("  📌 结论: 反事实推演表明，该修改对整体讨论影响有限，")
            lines.append("     核心论点结构基本保持不变。")

        lines.append("")
        lines.append("=" * 60)
        lines.append("🔚 推演结束")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _find_modified(self, item_id: int) -> Optional[Dict]:
        for e in self.modified_items:
            if e.get("id") == item_id:
                return e
        return None

    def get_essence_by_id(self, item_id: int) -> Optional[Dict]:
        for e in self.original_items:
            if e.get("id") == item_id:
                return e
        return None


# ── 工具函数 ──

def load_counterfactual_from_checkpoint(path: str) -> Optional[CounterfactualEngine]:
    """从检查点文件加载反事实引擎"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CounterfactualEngine(data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ 加载检查点失败: {e}")
        return None


def text_counterfactual_summary(engine: CounterfactualEngine) -> str:
    """生成文本版反事实推演摘要"""
    return engine.generate_counterfactual_synthesis()


# ── GUI 反事实推演对话框 ──

class CounterfactualDialog(QDialog):
    """反事实推演沙盘对话框"""

    def __init__(self, engine: CounterfactualEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._init_ui()
        self._refresh_essence_list()
        self._update_report()

    def _init_ui(self):
        self.setWindowTitle("🔮 反事实推演沙盘")
        self.setMinimumSize(1000, 700)
        self.resize(1100, 750)

        self.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #d4d4d4;
            }
            QLabel {
                color: #c89b3c;
            }
            QGroupBox {
                color: #c89b3c;
                font-weight: bold;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QTextEdit {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 8px;
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #c89b3c;
                color: #0d0d0d;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #dbb052;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #666666;
            }
            QPushButton.danger {
                background-color: #c0392b;
            }
            QPushButton.danger:hover {
                background-color: #e74c3c;
            }
            QPushButton.secondary {
                background-color: #2a2a2a;
                color: #d4d4d4;
            }
            QPushButton.secondary:hover {
                background-color: #3a3a3a;
            }
            QListWidget {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px 6px;
                border-bottom: 1px solid #1e1e1e;
            }
            QListWidget::item:selected {
                background-color: #c89b3c;
                color: #0d0d0d;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QTabWidget::pane {
                border: 1px solid #2a2a2a;
                background-color: #0d0d0d;
            }
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #888888;
                padding: 6px 16px;
                border: 1px solid #2a2a2a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0d0d0d;
                color: #c89b3c;
            }
            QTableWidget {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                gridline-color: #1e1e1e;
                font-size: 12px;
            }
            QTableWidget::item:selected {
                background-color: #c89b3c;
                color: #0d0d0d;
            }
            QHeaderView::section {
                background-color: #1e1e1e;
                color: #c89b3c;
                padding: 4px;
                border: 1px solid #2a2a2a;
                font-weight: bold;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel(f"🔮 反事实推演沙盘 · {self.engine.problem[:50]}")
        title_font = QFont("Microsoft YaHei", 14, QFont.Weight.Bold)
        title.setFont(title_font)
        main_layout.addWidget(title)

        info = QLabel(f"📊 精华池: {len(self.engine.original_items)} 条  |  "
                       f"👥 {len(self.engine.player_names)} 位专家  |  "
                       f"当前操作: {len(self.engine.operations)} 次")
        info.setStyleSheet("color: #888888; font-size: 12px;")
        main_layout.addWidget(info)

        # 标签页
        tabs = QTabWidget()
        tabs.setFont(QFont("Microsoft YaHei", 10))

        # ── Tab 1: 操作面板 ──
        op_tab = QWidget()
        op_layout = QHBoxLayout(op_tab)
        op_layout.setSpacing(12)

        # 左侧：精华列表
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_layout.addWidget(QLabel("📋 精华列表（点击选中后操作）"))
        self.essence_list = QListWidget()
        self.essence_list.currentRowChanged.connect(self._on_essence_selected)
        left_layout.addWidget(self.essence_list, 1)

        # 选中精华信息
        self.selected_info = QLabel("未选中精华")
        self.selected_info.setStyleSheet("color: #555555; font-size: 11px; padding: 4px;")
        self.selected_info.setWordWrap(True)
        left_layout.addWidget(self.selected_info)

        op_layout.addWidget(left_panel)

        # 右侧：操作按钮
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 增强/削弱
        score_group = QGroupBox("评分调整")
        score_layout = QVBoxLayout(score_group)

        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("调整幅度:"))
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.5, 20.0)
        self.amount_spin.setValue(3.0)
        self.amount_spin.setSingleStep(0.5)
        amount_layout.addWidget(self.amount_spin)
        amount_layout.addStretch()
        score_layout.addLayout(amount_layout)

        btn_row = QHBoxLayout()
        self.btn_boost = QPushButton("⬆ 增强评分")
        self.btn_boost.clicked.connect(self._boost_selected)
        btn_row.addWidget(self.btn_boost)

        self.btn_suppress = QPushButton("⬇ 削弱评分")
        self.btn_suppress.clicked.connect(self._suppress_selected)
        btn_row.addWidget(self.btn_suppress)
        score_layout.addLayout(btn_row)

        self.btn_remove = QPushButton("🗑️ 移除精华", clicked=self._remove_selected)
        self.btn_remove.setStyleSheet("background-color: #c0392b; color: white;")
        score_layout.addWidget(self.btn_remove)
        right_layout.addWidget(score_group)

        # 添加假设观点
        add_group = QGroupBox("添加假设观点")
        add_layout = QVBoxLayout(add_group)

        add_layout.addWidget(QLabel("假设内容:"))
        self.hypo_input = QTextEdit()
        self.hypo_input.setMaximumHeight(60)
        self.hypo_input.setPlaceholderText("例如：如果专家们更关注数据隐私方面...")
        add_layout.addWidget(self.hypo_input)

        score_row = QHBoxLayout()
        score_row.addWidget(QLabel("初始评分:"))
        self.hypo_score = QDoubleSpinBox()
        self.hypo_score.setRange(1.0, 10.0)
        self.hypo_score.setValue(5.0)
        score_row.addWidget(self.hypo_score)
        score_row.addStretch()
        add_layout.addLayout(score_row)

        self.btn_add_hypo = QPushButton("➕ 添加假设观点")
        self.btn_add_hypo.clicked.connect(self._add_hypothetical)
        add_layout.addWidget(self.btn_add_hypo)
        right_layout.addWidget(add_group)

        # 模式切换
        mode_group = QGroupBox("讨论模式切换")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["physical", "mathematical", "balance", "converge", "explore"])
        self.mode_combo.setCurrentText(self.engine.mode)
        mode_layout.addWidget(self.mode_combo)
        self.btn_apply_mode = QPushButton("应用模式")
        self.btn_apply_mode.clicked.connect(self._apply_mode)
        mode_layout.addWidget(self.btn_apply_mode)
        mode_layout.addStretch()
        right_layout.addWidget(mode_group)

        # Item 20: 时间记忆操作
        if self.engine.temporal_memory_data is not None:
            temporal_group = QGroupBox("⏳ 时间耦合记忆（反事实）")
            temporal_layout = QVBoxLayout(temporal_group)

            temporal_info = QLabel('调整历史耦合强度，模拟"如果历史不同"的跨轮次推演')
            temporal_info.setStyleSheet("color: #888888; font-size: 11px;")
            temporal_info.setWordWrap(True)
            temporal_layout.addWidget(temporal_info)

            temporal_btn_row = QHBoxLayout()
            self.btn_boost_temporal = QPushButton("⬆ 增强历史耦合")
            self.btn_boost_temporal.clicked.connect(self._boost_temporal)
            temporal_btn_row.addWidget(self.btn_boost_temporal)

            self.btn_weaken_temporal = QPushButton("⬇ 削弱历史耦合")
            self.btn_weaken_temporal.clicked.connect(self._weaken_temporal)
            temporal_btn_row.addWidget(self.btn_weaken_temporal)
            temporal_layout.addLayout(temporal_btn_row)

            self.btn_reset_temporal = QPushButton("🗑️ 清空时间记忆（历史归零）")
            self.btn_reset_temporal.setStyleSheet("background-color: #c0392b; color: white;")
            self.btn_reset_temporal.clicked.connect(self._reset_temporal)
            temporal_layout.addWidget(self.btn_reset_temporal)

            right_layout.addWidget(temporal_group)

        # 重置
        self.btn_reset = QPushButton("🔄 重置所有操作")
        self.btn_reset.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        self.btn_reset.clicked.connect(self._reset_all)
        right_layout.addWidget(self.btn_reset)

        right_layout.addStretch()

        op_layout.addWidget(right_panel)
        tabs.addTab(op_tab, "🔧 操作")

        # ── Tab 2: 推演报告 ──
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        report_layout.setContentsMargins(8, 8, 8, 8)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        report_layout.addWidget(self.report_text, 1)

        report_btn_row = QHBoxLayout()
        self.btn_refresh_report = QPushButton("🔄 刷新报告")
        self.btn_refresh_report.clicked.connect(self._update_report)
        report_btn_row.addWidget(self.btn_refresh_report)

        self.btn_export = QPushButton("💾 导出报告")
        self.btn_export.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        self.btn_export.clicked.connect(self._export_report)
        report_btn_row.addWidget(self.btn_export)
        report_btn_row.addStretch()
        report_layout.addLayout(report_btn_row)

        tabs.addTab(report_tab, "📄 推演报告")

        # ── Tab 3: 对比表 ──
        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        compare_layout.setContentsMargins(8, 8, 8, 8)

        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(6)
        self.compare_table.setHorizontalHeaderLabels(
            ["ID", "内容", "原评分", "新评分", "原排名", "新排名"]
        )
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.compare_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        compare_layout.addWidget(self.compare_table, 1)

        self.btn_refresh_compare = QPushButton("🔄 刷新对比表")
        self.btn_refresh_compare.clicked.connect(self._refresh_compare_table)
        compare_layout.addWidget(self.btn_refresh_compare)

        tabs.addTab(compare_tab, "📊 对比表")

        # ── Tab 4: 统计 ──
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(8, 8, 8, 8)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text, 1)

        self.btn_refresh_stats = QPushButton("🔄 刷新统计")
        self.btn_refresh_stats.clicked.connect(self._refresh_stats)
        stats_layout.addWidget(self.btn_refresh_stats)

        tabs.addTab(stats_tab, "📈 统计")

        main_layout.addWidget(tabs, 1)

        # 底部关闭按钮
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        bottom_row.addWidget(self.btn_close)
        main_layout.addLayout(bottom_row)

    def _refresh_essence_list(self):
        """刷新精华列表"""
        self.essence_list.clear()
        items = self.engine.get_modified_ranking()
        for e in items:
            eid = e.get("id", 0)
            content = e.get("content", "")[:45]
            score = e.get("score", 0)
            tags = e.get("tags", [])
            tag_str = ""
            if "反事实增强" in tags:
                tag_str = " ⬆"
            elif "反事实削弱" in tags:
                tag_str = " ⬇"
            elif "反事实假设" in tags:
                tag_str = " ✨"

            # 检查是否与原始不同
            orig = self.engine.get_essence_by_id(eid)
            delta = ""
            if orig and orig.get("score", 0) != score:
                d = score - orig.get("score", 0)
                delta = f" ({'+' if d > 0 else ''}{d:.1f})"

            item = QListWidgetItem(f"#{eid} [{score:.1f}{delta}]{tag_str} {content}")
            if "反事实增强" in tags:
                item.setForeground(QColor(52, 168, 83))
            elif "反事实削弱" in tags:
                item.setForeground(QColor(234, 67, 53))
            elif "反事实假设" in tags:
                item.setForeground(QColor(66, 133, 244))
            self.essence_list.addItem(item)

    def _on_essence_selected(self, row: int):
        """精华选中回调"""
        items = self.engine.get_modified_ranking()
        if 0 <= row < len(items):
            e = items[row]
            tags = ", ".join(e.get("tags", []))
            self.selected_info.setText(
                f"#{e['id']} | {e.get('contributor', '?')} | 评分: {e.get('score', 0):.1f}\n"
                f"标签: {tags}\n"
                f"{e.get('content', '')[:80]}"
            )
        else:
            self.selected_info.setText("未选中精华")

    def _get_selected_id(self) -> Optional[int]:
        row = self.essence_list.currentRow()
        if row < 0:
            return None
        items = self.engine.get_modified_ranking()
        if row < len(items):
            return items[row].get("id")
        return None

    def _boost_selected(self):
        eid = self._get_selected_id()
        if eid is None:
            return
        amount = self.amount_spin.value()
        self.engine.boost_essence(eid, amount)
        self._refresh_essence_list()
        self._update_report()

    def _suppress_selected(self):
        eid = self._get_selected_id()
        if eid is None:
            return
        amount = self.amount_spin.value()
        self.engine.suppress_essence(eid, amount)
        self._refresh_essence_list()
        self._update_report()

    def _remove_selected(self):
        eid = self._get_selected_id()
        if eid is None:
            return
        reply = QMessageBox.question(
            self, "确认移除", f"确定要移除精华 #{eid} 吗？\n此操作将影响排名计算。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.remove_essence(eid)
            self._refresh_essence_list()
            self._update_report()

    def _add_hypothetical(self):
        content = self.hypo_input.toPlainText().strip()
        if not content:
            return
        score = self.hypo_score.value()
        self.engine.add_hypothetical(content, score=score)
        self.hypo_input.clear()
        self._refresh_essence_list()
        self._update_report()

    def _apply_mode(self):
        mode = self.mode_combo.currentText()
        self.engine.change_mode(mode)
        self._update_report()

    def _reset_all(self):
        reply = QMessageBox.question(
            self, "确认重置", "确定要重置所有操作吗？\n所有修改将被撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.reset()
            self.mode_combo.setCurrentText(self.engine.mode)
            self._refresh_essence_list()
            self._update_report()

    # ── Item 20: 时间记忆操作回调 ──

    def _boost_temporal(self):
        self.engine.boost_temporal_coupling(factor=1.5)
        self._update_report()

    def _weaken_temporal(self):
        self.engine.weaken_temporal_coupling(factor=0.5)
        self._update_report()

    def _reset_temporal(self):
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空时间记忆吗？\n这将模拟历史归零的假设场景。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.reset_temporal_memory()
            self._update_report()

    def _update_report(self):
        report = self.engine.generate_counterfactual_synthesis()
        self.report_text.setText(report)
        self._refresh_compare_table()
        self._refresh_stats()

    def _refresh_compare_table(self):
        comparison = self.engine.compare_rankings()

        # 收集所有需要显示的行
        rows = []
        for e in comparison["gained"]:
            rows.append((e["id"], e["content"], e["old_score"], e["new_score"],
                         e["old_rank"], e["new_rank"], "🟢"))
        for e in comparison["lost"]:
            rows.append((e["id"], e["content"], e["old_score"], e["new_score"],
                         e["old_rank"], e["new_rank"], "🔴"))
        for e in comparison["new_entries"]:
            rows.append((e["id"], e["content"], "-", e["score"], "-", "-", "✨"))
        for e in comparison["removed"]:
            rows.append((e["id"], e["content"], e["old_score"], "-",
                         "-", "-", "🗑️"))

        # 未变化的
        unchanged_ids = comparison["unchanged"]
        orig_ranking = self.engine.get_original_ranking()
        for i, e in enumerate(orig_ranking):
            if e["id"] in unchanged_ids:
                rows.append((e["id"], e.get("content", "")[:50], e.get("score", 0),
                             e.get("score", 0), i + 1, i + 1, "⚪"))

        self.compare_table.setRowCount(len(rows))
        for row_idx, (eid, content, old_s, new_s, old_r, new_r, icon) in enumerate(rows):
            self.compare_table.setItem(row_idx, 0, QTableWidgetItem(str(eid)))
            self.compare_table.setItem(row_idx, 1, QTableWidgetItem(f"{icon} {content}"))
            self.compare_table.setItem(row_idx, 2, QTableWidgetItem(str(old_s)))
            self.compare_table.setItem(row_idx, 3, QTableWidgetItem(str(new_s)))
            self.compare_table.setItem(row_idx, 4, QTableWidgetItem(str(old_r)))
            self.compare_table.setItem(row_idx, 5, QTableWidgetItem(str(new_r)))

        self.compare_table.resizeColumnsToContents()

    def _refresh_stats(self):
        stats = self.engine.get_statistics()
        o = stats["original"]
        m = stats["modified"]
        lines = []
        lines.append("=" * 50)
        lines.append("📊 统计对比")
        lines.append("=" * 50)
        lines.append(f"{'指标':<20} {'原始':<12} {'修改后':<12}")
        lines.append("-" * 50)
        lines.append(f"{'精华数量':<20} {o['count']:<12} {m['count']:<12}")
        lines.append(f"{'平均评分':<20} {o['avg_score']:<12} {m['avg_score']:<12}")
        lines.append(f"{'最高评分':<20} {o['max_score']:<12} {m['max_score']:<12}")
        lines.append(f"{'最低评分':<20} {o['min_score']:<12} {m['min_score']:<12}")
        lines.append(f"{'评分标准差':<20} {o['std_dev']:<12} {m['std_dev']:<12}")
        lines.append("")
        lines.append(f"操作次数: {len(self.engine.operations)}")
        if self.engine.operations:
            lines.append("")
            lines.append("操作记录:")
            for op in self.engine.operations:
                lines.append(f"  · {op.get('description', '')}")
        # Item 20: 时间记忆统计
        tstats = self.engine.get_temporal_stats()
        if tstats.get("available"):
            lines.append("")
            lines.append("─" * 50)
            lines.append("⏳ 时间耦合记忆统计")
            lines.append("─" * 50)
            lines.append(f"记录轮次: {tstats['rounds_recorded']}")
            lines.append(f"活跃连接: {tstats['active_connections']} / {tstats['shape'][0]}×{tstats['shape'][1]}")
            lines.append(f"连接密度: {tstats['connection_density']:.2%}")
            if tstats.get("emergence_level_distribution"):
                dist = tstats["emergence_level_distribution"]
                for k, v in sorted(dist.items()):
                    lines.append(f"  L{k} 出现次数: {v}")
        self.stats_text.setText("\n".join(lines))

    def _export_report(self):
        report = self.engine.generate_counterfactual_synthesis()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"counterfactual_{timestamp}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出推演报告", default_name,
            "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*.*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            QMessageBox.information(self, "导出成功", f"报告已导出到:\n{path}")


def open_counterfactual_dialog(checkpoint_path: str, parent=None) -> bool:
    """打开反事实推演对话框（便利函数）"""
    engine = load_counterfactual_from_checkpoint(checkpoint_path)
    if not engine:
        return False
    dialog = CounterfactualDialog(engine, parent)
    dialog.exec_()
    return True