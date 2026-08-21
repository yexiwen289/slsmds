"""
讨论回放时间机器 —— 逐轮回顾讨论全过程

核心功能：
- 加载断点/报告文件，提取每轮快照
- 逐轮前进/后退播放，展示发言、精华演化、状态变化
- GUI对话框 + CLI文本模式双支持
"""

import json
import os
import datetime
from typing import List, Dict, Optional, Tuple
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTextEdit, QWidget, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy,
)


# ── 回放引擎 ──

class DiscussionReplay:
    """
    讨论回放引擎：加载检查点数据，按轮次索引回放。

    可加载结构：
    - 完整检查点 JSON（含 essence_pool, players, game_record 等）
    - 游戏报告 JSON（只含 rounds 和最终方案）
    """

    def __init__(self, data: Dict):
        self.data = data
        self.rounds: List[Dict] = []
        self.essence_snapshots: List[Dict] = []
        self.player_names: List[str] = []
        self.problem: str = ""
        self._parse()

    def _parse(self):
        """解析加载的数据，构建轮次列表"""
        # 游戏记录中的 rounds
        game_record = self.data.get("game_record", {})
        rounds_data = game_record.get("rounds", [])
        essence_pool_data = self.data.get("essence_pool", {})
        self.player_names = game_record.get("player_names", [])
        self.problem = self.data.get("problem", game_record.get("problem", ""))

        # 构建每轮快照
        for i, rd in enumerate(rounds_data):
            # 计算到本轮为止的精华池状态
            round_essences = [e for e in essence_pool_data.get("items", [])
                              if e.get("source_round", 0) <= rd.get("round_id", 0)]
            self.essence_snapshots.append(round_essences)
            self.rounds.append(rd)

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    def get_round(self, round_idx: int) -> Optional[Dict]:
        """获取指定轮次的数据（0-indexed）"""
        if 0 <= round_idx < len(self.rounds):
            return self.rounds[round_idx]
        return None

    def get_essences_at_round(self, round_idx: int) -> List[Dict]:
        """获取到指定轮次为止的精华列表"""
        if 0 <= round_idx < len(self.essence_snapshots):
            return self.essence_snapshots[round_idx]
        return []

    def get_round_summary(self, round_idx: int) -> str:
        """生成单轮文字摘要"""
        rd = self.get_round(round_idx)
        if not rd:
            return "（无数据）"

        round_id = rd.get("round_id", round_idx + 1)
        players = rd.get("round_players", [])
        speeches = rd.get("speech_history", [])
        essences = rd.get("essences_added", [])
        pool_after = rd.get("pool_state_after", "")

        lines = []
        lines.append(f"═ 第 {round_id} 轮 ═")
        lines.append(f"👥 发言: {', '.join(players) if players else '无'}")
        lines.append("")

        if speeches:
            for s in speeches:
                name = s.get("player_name", "?")
                text = s.get("speech", "")[:80]
                insight = s.get("key_insight", "")
                action = s.get("action", "new")
                action_icon = {"new": "💡", "refine": "🔧", "challenge": "⚔️"}
                icon = action_icon.get(action, "💬")
                lines.append(f"  {icon} {name}: \"{text}\"")
                if insight:
                    lines.append(f"     → {insight}")
        else:
            lines.append("  （无发言记录）")

        if essences:
            lines.append("")
            lines.append(f"  📌 本轮提炼精华 ({len(essences)} 条):")
            for e in essences:
                content = e.get("content", "")[:60]
                etype = e.get("type", "论点")
                lines.append(f"    · [{etype}] {content}")

        if pool_after:
            lines.append("")
            lines.append(f"  📊 精华池: {pool_after[:80]}...")

        return "\n".join(lines)

    def get_full_timeline(self) -> str:
        """生成完整时间线文本"""
        parts = [f"🧠 讨论回放 · {self.problem[:40]}",
                 f"👥 参与者: {', '.join(self.player_names)}",
                 f"🔄 总轮次: {self.total_rounds}",
                 "=" * 50]
        for i in range(self.total_rounds):
            parts.append("")
            parts.append(self.get_round_summary(i))
            if i < self.total_rounds - 1:
                parts.append("")
                parts.append("─" * 40)
        return "\n".join(parts)

    def get_evolution_data(self) -> List[Dict]:
        """
        获取精华演化数据（每轮精华数量、评分变化等，用于可视化）
        """
        data = []
        for i in range(self.total_rounds):
            essences = self.get_essences_at_round(i)
            total = len(essences)
            avg_score = sum(e.get("score", 0) for e in essences) / max(total, 1)
            new_count = len([e for e in essences
                             if e.get("source_round", 0) == (self.rounds[i].get("round_id", i + 1) if self.rounds else 0)])
            data.append({
                "round": i + 1,
                "total_essences": total,
                "avg_score": round(avg_score, 2),
                "new_essences": new_count,
            })
        return data


# ── 工具函数 ──

def load_replay_from_file(path: str) -> Optional[DiscussionReplay]:
    """从文件加载回放数据"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DiscussionReplay(data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ 加载回放文件失败: {e}")
        return None


def text_replay(replay: DiscussionReplay, start_round: int = 0,
                end_round: int = -1) -> str:
    """生成文本版回放摘要"""
    if end_round < 0 or end_round >= replay.total_rounds:
        end_round = replay.total_rounds - 1
    if start_round < 0:
        start_round = 0

    parts = []
    for i in range(start_round, end_round + 1):
        parts.append(replay.get_round_summary(i))
        if i < end_round:
            parts.append("")
            parts.append("─" * 40)
    return "\n".join(parts)


# ── GUI 回放对话框 ──

class ReplayDialog(QDialog):
    """讨论回放时间机器对话框"""

    def __init__(self, replay: DiscussionReplay, parent=None):
        super().__init__(parent)
        self.replay = replay
        self._current_round = 0
        self._auto_playing = False
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_step)
        self._init_ui()
        self._show_round(0)

    def _init_ui(self):
        self.setWindowTitle("⏳ 讨论回放时间机器")
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)

        # 应用全局暗色风格
        self.setStyleSheet("""
            QDialog {
                background-color: #0d0d0d;
                color: #d4d4d4;
            }
            QLabel {
                color: #c89b3c;
                font-size: 13px;
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
            QSlider::groove:horizontal {
                height: 6px;
                background: #2a2a2a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #c89b3c;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #c89b3c;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #111111;
                color: #d4d4d4;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #1e1e1e;
            }
            QListWidget::item:selected {
                background-color: #c89b3c;
                color: #0d0d0d;
            }
            QListWidget::item:hover {
                background-color: #1e1e1e;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # 标题区
        title = QLabel(f"⏳ 讨论回放 · {self.replay.problem[:50]}")
        title_font = QFont("Microsoft YaHei", 14, QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        info = QLabel(f"👥 {len(self.replay.player_names)} 位专家  |  "
                       f"🔄 {self.replay.total_rounds} 轮讨论")
        info.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(info)

        # 主分割区
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # 左侧：轮次列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_layout.addWidget(QLabel("📋 轮次导航"))
        self.round_list = QListWidget()
        self.round_list.setMaximumWidth(200)
        for i in range(self.replay.total_rounds):
            rd = self.replay.get_round(i)
            r_id = rd.get("round_id", i + 1) if rd else i + 1
            speeches = len(rd.get("speech_history", [])) if rd else 0
            essences = len(rd.get("essences_added", [])) if rd else 0
            item = QListWidgetItem(f"第 {r_id} 轮  💬{speeches} 📌{essences}")
            self.round_list.addItem(item)
        self.round_list.currentRowChanged.connect(self._on_round_selected)
        left_layout.addWidget(self.round_list)

        # 进化数据摘要
        self.evo_label = QLabel("")
        self.evo_label.setStyleSheet("color: #555555; font-size: 11px; padding: 4px;")
        self.evo_label.setWordWrap(True)
        left_layout.addWidget(self.evo_label)

        splitter.addWidget(left_widget)

        # 右侧：回放内容
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 轮次进度标签
        self.round_label = QLabel("第 0 轮 / 共 0 轮")
        self.round_label.setStyleSheet("color: #c89b3c; font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.round_label)

        self.replay_text = QTextEdit()
        self.replay_text.setReadOnly(True)
        right_layout.addWidget(self.replay_text, 1)

        # 控制栏
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.btn_first = QPushButton("⏮ 开始")
        self.btn_first.setToolTip("跳转到第一轮")
        self.btn_first.clicked.connect(lambda: self._show_round(0))
        controls.addWidget(self.btn_first)

        self.btn_prev = QPushButton("◀ 上一轮")
        self.btn_prev.setToolTip("回退一轮")
        self.btn_prev.clicked.connect(self._prev_round)
        controls.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ 自动播放")
        self.btn_play.setToolTip("自动逐轮播放")
        self.btn_play.clicked.connect(self._toggle_auto_play)
        controls.addWidget(self.btn_play)

        self.btn_next = QPushButton("下一轮 ▶")
        self.btn_next.setToolTip("前进一轮")
        self.btn_next.clicked.connect(self._next_round)
        controls.addWidget(self.btn_next)

        self.btn_last = QPushButton("结束 ⏭")
        self.btn_last.setToolTip("跳转到最后一轮")
        self.btn_last.clicked.connect(lambda: self._show_round(self.replay.total_rounds - 1))
        controls.addWidget(self.btn_last)

        controls.addStretch()

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        controls.addWidget(self.btn_close)

        right_layout.addLayout(controls)

        # 进度滑块
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(8)
        self.round_slider = QSlider(Qt.Orientation.Horizontal)
        self.round_slider.setMinimum(0)
        self.round_slider.setMaximum(max(0, self.replay.total_rounds - 1))
        self.round_slider.setValue(0)
        self.round_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(QLabel("进度:"))
        slider_layout.addWidget(self.round_slider, 1)

        self.slider_info = QLabel("0%")
        self.slider_info.setStyleSheet("color: #888888; font-size: 11px; min-width: 40px;")
        slider_layout.addWidget(self.slider_info)

        right_layout.addLayout(slider_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter, 1)

    def _show_round(self, idx: int):
        """显示指定轮次"""
        if not self.replay or self.replay.total_rounds == 0:
            return
        idx = max(0, min(idx, self.replay.total_rounds - 1))
        self._current_round = idx

        # 更新标签
        total = self.replay.total_rounds
        self.round_label.setText(f"第 {idx + 1} 轮 / 共 {total} 轮")

        # 更新内容
        summary = self.replay.get_round_summary(idx)
        self.replay_text.setText(summary)
        self.replay_text.moveCursor(0)  # scroll to top

        # 更新滑块
        self.round_slider.blockSignals(True)
        self.round_slider.setValue(idx)
        self.round_slider.blockSignals(False)

        pct = int((idx + 1) / total * 100) if total > 0 else 0
        self.slider_info.setText(f"{pct}%")

        # 更新列表选中
        self.round_list.blockSignals(True)
        self.round_list.setCurrentRow(idx)
        self.round_list.blockSignals(False)

        # 更新按钮状态
        self.btn_prev.setEnabled(idx > 0)
        self.btn_first.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < total - 1)
        self.btn_last.setEnabled(idx < total - 1)

        # 更新进化摘要
        evo = self.replay.get_evolution_data()
        if idx < len(evo):
            e = evo[idx]
            self.evo_label.setText(
                f"📊 精华: {e['total_essences']} 条  |  "
                f"本轮新增: {e['new_essences']}  |  "
                f"均分: {e['avg_score']:.1f}"
            )

    def _next_round(self):
        self._show_round(self._current_round + 1)

    def _prev_round(self):
        self._show_round(self._current_round - 1)

    def _on_round_selected(self, row: int):
        if row >= 0:
            self._show_round(row)

    def _on_slider_changed(self, val: int):
        self._show_round(val)

    def _toggle_auto_play(self):
        self._auto_playing = not self._auto_playing
        if self._auto_playing:
            self.btn_play.setText("⏸ 暂停")
            self._auto_timer.start(2000)  # 每2秒前进一轮
        else:
            self.btn_play.setText("▶ 自动播放")
            self._auto_timer.stop()

    def _auto_step(self):
        if self._current_round >= self.replay.total_rounds - 1:
            self._auto_playing = False
            self.btn_play.setText("▶ 自动播放")
            self._auto_timer.stop()
            return
        self._next_round()

    def closeEvent(self, event):
        self._auto_timer.stop()
        super().closeEvent(event)


def open_replay_dialog(checkpoint_path: str, parent=None) -> bool:
    """打开回放对话框（便利函数）"""
    replay = load_replay_from_file(checkpoint_path)
    if not replay:
        return False
    dialog = ReplayDialog(replay, parent)
    dialog.exec_()
    return True