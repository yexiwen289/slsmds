"""
相空间神经元编辑器 —— 手动微调、创建、删除虚拟专家神经元的相空间位置

在设计理念上，反事实推演沙盘提供了全局性的"如果改变参数会怎样"的假设分析，
而本编辑器提供了微观层面的直接操控——你可以像雕塑家一样，直接移动每个
虚拟专家神经元在6维认知相空间中的位置，观察它们如何影响整体认知结构。

交互方式：纯 TUI，无需 GUI，可直接在终端中运行。

功能：
- 分页列出所有虚拟专家（含相空间向量和发言文本）
- 查看/编辑单个神经元的 6 维向量
- 创建新神经元（自定义向量 + 文本）
- 删除神经元
- 批量操作（平移、缩放、重置）
- 搜索/过滤（按文本、维度范围）
- 保存/恢复编辑状态
"""

import os
import sys
import copy
import math
import random
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict

import numpy as np

# ── 终端样式 ──

def C(s, code):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s

C_YELLOW  = lambda s: C(s, "93")
C_GREEN   = lambda s: C(s, "92")
C_RED     = lambda s: C(s, "91")
C_CYAN    = lambda s: C(s, "96")
C_MAGENTA = lambda s: C(s, "95")
C_DIM     = lambda s: C(s, "2")
C_BOLD    = lambda s: C(s, "1")
C_WHITE   = lambda s: C(s, "97")
C_BLUE    = lambda s: C(s, "94")

# ── 工具函数 ──

def _clear():
    os.system("cls" if os.name == "nt" else "clear")

def _header(text: str, width: int = 70):
    """绘制带边框的标题"""
    print(f"\n{C_BOLD('╔' + '═' * (width - 2) + '╗')}")
    left = (width - 2 - len(text)) // 2
    right = width - 2 - len(text) - left
    print(f"{C_BOLD('║')}{' ' * left}{C_YELLOW(text)}{' ' * right}{C_BOLD('║')}")
    print(f"{C_BOLD('╚' + '═' * (width - 2) + '╝')}\n")

def _box(text: str, width: int = 70):
    print(f"┏{'━' * (width - 2)}┓")
    print(f"┃{text:^{width - 2}}┃")
    print(f"┗{'━' * (width - 2)}┛")

def _box_single(text: str, width: int = 70):
    print(f"┃ {text:<{width - 3}}┃")

def _box_end(width: int = 70):
    print(f"┗{'━' * (width - 2)}┛")

def _input(prompt: str = "  ▸ ") -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "q"

def _pause():
    _input(f"\n  {C_DIM('按回车键继续...')}")

def _confirm(prompt: str = "确定? (y/n): ") -> bool:
    return _input(prompt).lower() in ("y", "yes", "是")

# ── 维度名称 ──

DIM_NAMES = ["逻辑一致性", "新颖性", "认知深度", "分歧度", "具体程度", "情感强度"]
DIM_KEYS = ["coherence", "novelty", "depth", "divergence", "specificity", "emotional"]


# ═══════════════════════════════════════════════════════════════
# 神经元编辑器核心
# ═══════════════════════════════════════════════════════════════

class NeuronEditor:
    """
    相空间神经元编辑器。

    工作流程：
    1. 接收 VirtualExpertGenerator 的虚拟讨论列表（或类似结构）
    2. 提供 TUI 界面让用户交互
    3. 返回修改后的虚拟讨论列表
    """

    def __init__(self, virtual_discussions: List[Dict],
                 real_discussions: List[Dict] = None,
                 custom_vectors: List[np.ndarray] = None):
        """
        Args:
            virtual_discussions: 虚拟专家的讨论列表，每项含 "speech", "key_insight", 等
            real_discussions: 真实专家列表（用于参考，不修改）
            custom_vectors: 可选，预计算的向量列表（与 virtual_discussions 一一对应）
        """
        self.virtual = virtual_discussions
        self.real = real_discussions or []
        self._custom_vectors = custom_vectors or []
        self._original_state = copy.deepcopy(virtual_discussions)
        self._modified = False
        self._sort_order = "index"  # index | coherence | novelty | depth | divergence | specificity | emotional
        self._search_query = ""
        self._page = 0
        self._page_size = 20

    def _get_vector(self, idx: int) -> np.ndarray:
        """获取指定虚拟专家的向量"""
        if idx < len(self._custom_vectors):
            cv = self._custom_vectors[idx]
            if isinstance(cv, np.ndarray) and cv.shape == (6,):
                return cv
        if idx < len(self.virtual):
            # 从发言文本重建向量
            from .emergence import OpinionPhaseVector
            d = self.virtual[idx]
            text = d.get("speech", "")
            ov = OpinionPhaseVector(text, d.get("player_name", ""))
            return ov.vector
        return np.zeros(6)

    def _rebuild_all_vectors(self):
        """重建所有自定义向量"""
        self._custom_vectors = []
        for i in range(len(self.virtual)):
            self._custom_vectors.append(self._get_vector(i))

    def _get_vectors_matrix(self) -> np.ndarray:
        """获取所有虚拟专家的向量矩阵 (N, 6)"""
        self._rebuild_all_vectors()
        return np.array(self._custom_vectors)

    def _set_vector(self, idx: int, new_vec: np.ndarray):
        """设置指定虚拟专家的向量"""
        if idx < len(self.virtual):
            new_vec = np.clip(new_vec, 0.0, 1.0)
            self._custom_vectors[idx] = new_vec
            # 同时更新 speech 文本（从向量重建）
            self._rebuild_speech(idx)
            self._modified = True

    def _rebuild_speech(self, idx: int):
        """从向量重建发言文本"""
        if idx >= len(self.virtual):
            return
        vec = self._custom_vectors[idx] if idx < len(self._custom_vectors) else np.zeros(6)
        # 基于向量值生成语义标签
        dim_labels = []
        for i, name in enumerate(DIM_NAMES):
            if vec[i] > 0.7:
                dim_labels.append(f"高{name}")
            elif vec[i] < 0.3:
                dim_labels.append(f"低{name}")
        # 生成文本
        texts = []
        coherence = vec[0]
        novelty = vec[1]
        depth = vec[2]
        divergence = vec[3]
        specificity = vec[4]
        emotional = vec[5]

        # 开场白
        if divergence > 0.6:
            texts.append(f"从另一个角度看，{self._random_phrase('divergence', coherence)}")
        elif novelty > 0.6:
            texts.append(f"本质上，{self._random_phrase('novelty', coherence)}")
        elif depth > 0.5:
            texts.append(f"从更深层次来理解，{self._random_phrase('depth', coherence)}")
        else:
            texts.append(f"从我的专业角度，{self._random_phrase('base', coherence)}")

        # 主体
        if depth > 0.5:
            texts.append(f"这涉及到{self._random_phrase('depth_detail', depth)}")
            if depth > 0.7:
                texts.append(f"进一步说，{self._random_phrase('depth_detail', depth)}")

        if novelty > 0.5:
            texts.append(f"一个被忽视的关键是{self._random_phrase('novelty_detail', novelty)}")

        if divergence > 0.5:
            texts.append(f"但需要注意，{self._random_phrase('divergence_detail', divergence)}")

        if specificity > 0.4:
            texts.append(f"例如，{self._random_phrase('specificity_detail', specificity)}")

        # 结尾
        if emotional > 0.6:
            texts.append(f"这至关重要——{self._random_phrase('emotional', emotional)}")
        elif coherence > 0.5:
            texts.append(f"因此，{self._random_phrase('conclusion', coherence)}")

        speech = " ".join(texts)
        self.virtual[idx]["speech"] = speech
        if "key_insight" in self.virtual[idx]:
            insight = self._random_phrase('insight', max(vec))
            self.virtual[idx]["key_insight"] = insight

    def _random_phrase(self, category: str, intensity: float) -> str:
        """生成随机短语"""
        phrases = {
            "base": [
                "这个问题需要从系统层面来理解",
                "我们需要考虑多个维度的影响",
                "这个问题的核心在于认知方式的转变",
                "答案隐藏在问题的结构之中",
                "我们需要重新审视基本假设",
            ],
            "divergence": [
                "如果用不同的框架来审视，会发现",
                "从对立的角度来看，情况恰恰相反",
                "这个观点虽然看似合理，但忽略了",
                "我们需要质疑这个前提",
                "一个相反的视角可能揭示",
            ],
            "novelty": [
                "一个全新的范式正在浮现",
                "涌现出的新模式表明",
                "突破性的见解在于",
                "被主流忽视的关键因素其实是",
                "这个问题的本质比我们想象的更复杂",
            ],
            "depth": [
                "表层现象之下隐藏着更深层的结构",
                "我们需要穿透表象，触及本质",
                "递归的思考揭示了一个悖论",
                "系统论告诉我们，局部最优不等于全局最优",
                "因果链的末端指向一个根本性的矛盾",
            ],
            "depth_detail": [
                "系统的自指涉性导致认知的边界条件发生变化",
                "反馈回路的非线性特征使得预测变得困难",
                "涌现的性质无法从组成部分的简单叠加中推导",
                "认知的层次结构决定了理解的深度",
                "维度的坍塌往往伴随着新的维度的诞生",
            ],
            "novelty_detail": [
                "认知框架的转变本身就是一种涌现现象",
                "自组织临界性揭示了系统如何在不稳定中创造新秩序",
                "相变点附近的涨落蕴含着系统的全部可能性",
                "量子叠加态在认知领域的类比暗示了多重真相的共存",
                "边界的模糊性恰恰是创新的源泉",
            ],
            "divergence_detail": [
                "过度强调一致性会抑制系统的多样性指数",
                "共识的代价往往是失去最有价值的异见",
                "系统的稳健性恰恰来自于内部的张力",
                "对立的力量在更高的层次上达到统一",
                "混沌的边缘是秩序与无序的辩证统一",
            ],
            "specificity_detail": [
                "在具体实践中，我们可以观察到模式识别中的偏差",
                "数据表明，认知多样性每增加10%，决策质量提升约7%",
                "案例分析揭示了框架效应如何影响判断",
                "实验结果表明，群体智慧需要独立性来维持",
                "在实际应用中，维度的选择会显著影响分析结果",
            ],
            "emotional": [
                "这关乎我们理解世界的方式，意义重大",
                "我们不能再忽视这个根本性的问题",
                "这不仅是理论问题，更是实践中的紧迫挑战",
                "我们需要勇气去面对这个认知的边界",
                "这个领域的探索将改变我们对智能的理解",
            ],
            "conclusion": [
                "我们需要在多个维度之间寻找平衡",
                "真正的理解来自于对复杂性的接受",
                "答案不是一个点，而是一个相空间中的区域",
                "持续的自我质疑是认知进化的动力",
                "在不确定性中寻找确定性本身就是一种悖论",
        ],
        "insight": [
            "认知多样性是系统智慧的核心指标",
            "相空间中的拓扑结构决定了认知的涌现层级",
            "自指涉性使得系统具有自我超越的能力",
            "非线性耦合是复杂系统演化的根本动力",
            "边界的定义同时决定了系统的可能性和局限性",
            "涌现不是叠加，而是维度的跃迁",
            "混沌边缘是创造力的生态位",
            "系统的约束条件同时也是其自由度的来源",
        ],
        }
        pool = phrases.get(category, phrases["base"])
        # 根据强度选择不同的短语
        intensity_idx = min(len(pool) - 1, int(intensity * len(pool)))
        return pool[intensity_idx % len(pool)]

    # ── 排序 ──

    def _get_sorted_indices(self) -> List[int]:
        """获取排序后的索引"""
        if self._sort_order == "index":
            return list(range(len(self.virtual)))
        dim_idx = DIM_KEYS.index(self._sort_order) if self._sort_order in DIM_KEYS else -1
        if dim_idx < 0:
            return list(range(len(self.virtual)))
        self._rebuild_all_vectors()
        vecs = np.array(self._custom_vectors)
        # 按该维度降序排列
        order = np.argsort(-vecs[:, dim_idx])
        return order.tolist()

    def _get_filtered_indices(self) -> List[int]:
        """获取过滤后的索引"""
        indices = self._get_sorted_indices()
        if not self._search_query:
            return indices
        query = self._search_query.lower()
        filtered = []
        for i in indices:
            d = self.virtual[i]
            text = d.get("speech", "").lower()
            insight = d.get("key_insight", "").lower()
            if query in text or query in insight:
                filtered.append(i)
        return filtered

    # ── 主菜单 ──

    def run(self) -> bool:
        """
        运行编辑器主循环。

        Returns:
            True 如果用户修改了数据
            False 如果用户未做任何修改
        """
        self._rebuild_all_vectors()
        while True:
            n_total = len(self.virtual)
            n_real = len(self.real)
            sorted_indices = self._get_filtered_indices()
            n_shown = len(sorted_indices)
            total_pages = max(1, (n_shown + self._page_size - 1) // self._page_size)
            self._page = min(self._page, total_pages - 1)
            start = self._page * self._page_size
            end = min(start + self._page_size, n_shown)
            page_indices = sorted_indices[start:end] if sorted_indices else []

            # 计算统计
            if n_shown > 0:
                self._rebuild_all_vectors()
                vecs = np.array([self._custom_vectors[i] for i in page_indices])
                dim_avgs = np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(6)
                dim_stds = np.std(vecs, axis=0) if len(vecs) > 0 else np.zeros(6)
                total_avg = np.mean(self._get_vectors_matrix(), axis=0)
            else:
                dim_avgs = np.zeros(6)
                dim_stds = np.zeros(6)
                total_avg = np.zeros(6)

            _clear()
            _header(f"🧠 神经元编辑器  —  {n_total} 个虚拟专家")
            _box(f"过滤: {n_shown}/{n_total}  |  第 {self._page + 1}/{total_pages} 页  |  排序: {self._sort_order}")
            _box_single(f"相空间维度均值: {'  '.join(f'{DIM_NAMES[i][:2]}={total_avg[i]:.2f}' for i in range(6))}")
            _box_end()
            print()

            # 列表头
            hdr = f"{C_DIM(' #')}  {C_DIM('逻辑一')} {C_DIM('新颖性')} {C_DIM('认知深')} {C_DIM('分歧度')} {C_DIM('具体程')} {C_DIM('情感强')}  {C_DIM('发言摘要')}"
            print(f"  {hdr}")
            print(f"  {C_DIM('─' * 68)}")

            # 当前页列表
            for row_idx, vidx in enumerate(page_indices):
                d = self.virtual[vidx]
                vec = self._custom_vectors[vidx] if vidx < len(self._custom_vectors) else np.zeros(6)
                speech = d.get("speech", "")[:35]
                num = start + row_idx + 1
                # 高亮极端值
                vals = []
                for v in vec:
                    if v > 0.8:
                        vals.append(C_GREEN(f"{v:.2f}"))
                    elif v < 0.2:
                        vals.append(C_RED(f"{v:.2f}"))
                    else:
                        vals.append(f"{v:.2f}")
                print(f"  {C_DIM(f'{num:>3}')}  {' '.join(vals)}  {speech}")

            print(f"  {C_DIM('─' * 68)}")

            # 维度统计
            if n_shown > 0:
                print(f"  {C_DIM('均值')}  {' '.join(f'{dim_avgs[i]:.2f}' for i in range(6))}  {C_DIM(f'当前页 {len(page_indices)} 个')}")
                print(f"  {C_DIM('标准差')} {' '.join(f'{dim_stds[i]:.2f}' for i in range(6))}")

            print()
            _box("操作")
            _box_single("[1-{n}] 编辑神经元  [c] 新建  [d] 删除  [s] 排序  [/] 搜索")
            _box_single("[b] 批量操作  [r] 重置  [p] 上一页  [n] 下一页  [q] 退出")
            _box_single(C_DIM(f"当前: 排序={self._sort_order} | 搜索='{self._search_query}'"), width=70)
            _box_end()

            cmd = _input()
            if cmd == "q":
                if self._modified:
                    if _confirm("有未保存的修改，确认退出? (y/n): "):
                        break
                else:
                    break
            elif cmd == "p":
                self._page = max(0, self._page - 1)
            elif cmd == "n":
                if page_indices and self._page < total_pages - 1:
                    self._page += 1
            elif cmd == "/":
                q = _input("搜索关键词: ")
                if q:
                    self._search_query = q
                    self._page = 0
            elif cmd == "s":
                self._sort_menu()
            elif cmd == "d":
                self._delete_menu()
            elif cmd == "b":
                self._batch_menu()
            elif cmd == "r":
                self._reset_menu()
            elif cmd == "c":
                self._create_menu()
            elif cmd.isdigit() or (cmd.startswith("-") and cmd[1:].isdigit()):
                idx = int(cmd) - 1
                if 0 <= idx < n_shown:
                    actual_idx = page_indices[idx]
                    self._edit_menu(actual_idx)
            # 每页浏览
            if page_indices and cmd == "":
                pass

        return self._modified

    # ── 排序菜单 ──

    def _sort_menu(self):
        _clear()
        _header("排序方式")
        print("  [1] 索引顺序")
        for i, name in enumerate(DIM_NAMES, 2):
            print(f"  [{i}] {name}（降序）")
        print(f"  [q] 返回")
        cmd = _input()
        options = ["index"] + DIM_KEYS
        if cmd == "q":
            return
        try:
            idx = int(cmd) - 1
            if 0 <= idx < len(options):
                self._sort_order = options[idx]
                self._page = 0
        except ValueError:
            pass

    # ── 编辑菜单 ──

    def _edit_menu(self, idx: int):
        """编辑单个神经元"""
        d = self.virtual[idx]
        vec = self._custom_vectors[idx] if idx < len(self._custom_vectors) else np.zeros(6)
        speech = d.get("speech", "")
        insight = d.get("key_insight", "")

        while True:
            _clear()
            _header(f"🧬 神经元 #{idx} 编辑")
            _box(f"发言文本")
            _box_single(f"{speech[:70]}")
            if len(speech) > 70:
                _box_single(f"{speech[70:140]}")
            if len(speech) > 140:
                _box_single(f"{speech[140:210]}")
            _box_single(f"关键洞察: {insight[:50]}")
            _box_end()
            print()
            print(f"  {C_BOLD('相空间向量 (6 维，范围 0.0 - 1.0)')}")
            print(f"  {'─' * 60}")
            for i, name in enumerate(DIM_NAMES):
                val = vec[i]
                bar_len = int(val * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(f"  [{i+1}] {name:<8}  {val:.3f}  {bar}")
            print(f"  {'─' * 60}")
            print()
            _box("操作")
            _box_single("[1-6] 修改维度值  [t] 编辑文本  [r] 随机化  [q] 返回")
            _box_end()

            cmd = _input()
            if cmd == "q":
                break
            elif cmd == "t":
                print(f"  当前文本: {speech}")
                new_speech = _input("新文本（留空不变）: ")
                if new_speech:
                    self.virtual[idx]["speech"] = new_speech
                    # 重建向量
                    from .emergence import OpinionPhaseVector
                    ov = OpinionPhaseVector(new_speech, d.get("player_name", ""))
                    self._custom_vectors[idx] = ov.vector
                    vec = ov.vector
                    self._modified = True
                    speech = new_speech
            elif cmd == "r":
                if _confirm("随机化此神经元的向量? (y/n): "):
                    new_vec = np.random.uniform(0.1, 0.9, 6)
                    self._set_vector(idx, new_vec)
                    vec = new_vec
                    speech = self.virtual[idx].get("speech", "")
            elif cmd in [str(i) for i in range(1, 7)]:
                dim = int(cmd) - 1
                val_str = _input(f"  {DIM_NAMES[dim]} 新值 (0.0-1.0, 留空不变): ")
                if val_str:
                    try:
                        val = float(val_str)
                        val = max(0.0, min(1.0, val))
                        new_vec = vec.copy()
                        new_vec[dim] = val
                        self._set_vector(idx, new_vec)
                        vec = new_vec
                        speech = self.virtual[idx].get("speech", "")
                    except ValueError:
                        print(f"  {C_RED('无效数值')}")
                        _pause()

    # ── 创建菜单 ──

    def _create_menu(self):
        """创建新神经元"""
        _clear()
        _header("➕ 创建新神经元")

        print("  请设置相空间向量值:")
        vec = np.zeros(6)
        for i, name in enumerate(DIM_NAMES):
            val_str = _input(f"  {name} (0.0-1.0, 默认 0.5): ")
            if val_str:
                try:
                    vec[i] = max(0.0, min(1.0, float(val_str)))
                except ValueError:
                    vec[i] = 0.5
            else:
                vec[i] = 0.5

        # 可选文本
        print()
        custom_text = _input("自定义发言文本（留空自动生成）: ")

        new_disc = {
            "speech": custom_text if custom_text else "",
            "key_insight": "用户创建",
            "player_name": "自定义",
            "source_round": 0,
            "round": 0,
            "score": 5.0,
            "tags": ["用户创建"],
            "cited_by": [],
            "refined_by": [],
            "challenged_by": [],
            "approve_by": [],
            "reject_by": [],
            "abstain_by": [],
            "vote_reasons": [],
            "clarifications": [],
        }
        self.virtual.append(new_disc)
        self._custom_vectors.append(vec)
        if not custom_text:
            self._rebuild_speech(len(self.virtual) - 1)
        self._modified = True
        print(f"\n  {C_GREEN('✓ 神经元已创建')}")
        _pause()

    # ── 删除菜单 ──

    def _delete_menu(self):
        """删除神经元"""
        _clear()
        _header("🗑️ 删除神经元")

        sorted_indices = self._get_filtered_indices()
        n_shown = len(sorted_indices)

        if n_shown == 0:
            print("  没有可删除的神经元。")
            _pause()
            return

        print("  选择要删除的神经元:")
        for i in range(min(10, n_shown)):
            idx = sorted_indices[i]
            d = self.virtual[idx]
            speech = d.get("speech", "")[:50]
            print(f"  [{i+1}] #{idx} {speech}")

        print(f"  [a] 删除全部 {n_shown} 个（当前过滤结果）")
        print(f"  [q] 返回")

        cmd = _input()
        if cmd == "q":
            return
        if cmd == "a":
            if _confirm(f"确定删除全部 {n_shown} 个神经元? (y/n): "):
                ids_to_remove = sorted(set(sorted_indices), reverse=True)
                for idx in ids_to_remove:
                    if idx < len(self.virtual):
                        self.virtual.pop(idx)
                        if idx < len(self._custom_vectors):
                            self._custom_vectors.pop(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已删除 {len(ids_to_remove)} 个神经元')}")
                _pause()
            return
        try:
            choice = int(cmd) - 1
            if 0 <= choice < min(10, n_shown):
                idx = sorted_indices[choice]
                if _confirm(f"确定删除神经元 #{idx}? (y/n): "):
                    self.virtual.pop(idx)
                    if idx < len(self._custom_vectors):
                        self._custom_vectors.pop(idx)
                    self._modified = True
                    print(f"  {C_GREEN('✓ 已删除')}")
                    _pause()
        except ValueError:
            pass

    # ── 批量操作菜单 ──

    def _batch_menu(self):
        """批量操作菜单"""
        _clear()
        _header("⚡ 批量操作")

        sorted_indices = self._get_filtered_indices()
        n_shown = len(sorted_indices)

        if n_shown == 0:
            print("  没有可操作的神经元。")
            _pause()
            return

        print(f"  当前过滤结果: {n_shown} 个神经元")
        print()
        print("  [1] 平移所有向量（每个维度加固定值）")
        print("  [2] 缩放所有向量（乘以系数）")
        print("  [3] 翻转维度（1 - 当前值）")
        print("  [4] 增加多样性（加随机噪声）")
        print("  [5] 向中心收缩（向 0.5 靠拢）")
        print("  [6] 向边缘扩散（远离 0.5）")
        print("  [7] 重置为原始状态")
        print("  [q] 返回")

        cmd = _input()
        if cmd == "q":
            return

        if cmd == "1":
            delta = _input("偏移量 (如 0.1 表示每个维度+0.1, 支持负数): ")
            try:
                d = float(delta)
                self._rebuild_all_vectors()
                matrix = np.array(self._custom_vectors)
                matrix = np.clip(matrix + d, 0.0, 1.0)
                for i, idx in enumerate(sorted_indices):
                    self._custom_vectors[idx] = matrix[i]
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已平移 {n_shown} 个神经元')}")
            except ValueError:
                print(f"  {C_RED('无效数值')}")

        elif cmd == "2":
            factor = _input("缩放系数 (如 0.8 表示缩小20%, 1.5 表示放大50%): ")
            try:
                f = float(factor)
                self._rebuild_all_vectors()
                matrix = np.array(self._custom_vectors)
                matrix = np.clip((matrix - 0.5) * f + 0.5, 0.0, 1.0)
                for i, idx in enumerate(sorted_indices):
                    self._custom_vectors[idx] = matrix[i]
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已缩放 {n_shown} 个神经元')}")
            except ValueError:
                print(f"  {C_RED('无效数值')}")

        elif cmd == "3":
            if _confirm("翻转所有维度? (1->0, 0.8->0.2): "):
                for idx in sorted_indices:
                    vec = self._custom_vectors[idx]
                    self._custom_vectors[idx] = 1.0 - vec
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已翻转 {n_shown} 个神经元')}")

        elif cmd == "4":
            noise_str = _input("噪声幅度 (0.0-0.5, 默认 0.2): ")
            try:
                noise = float(noise_str) if noise_str else 0.2
                noise = max(0.0, min(0.5, noise))
                for idx in sorted_indices:
                    vec = self._custom_vectors[idx]
                    self._custom_vectors[idx] = np.clip(
                        vec + np.random.uniform(-noise, noise, 6), 0.0, 1.0)
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已增加 {n_shown} 个神经元的多样性')}")
            except ValueError:
                print(f"  {C_RED('无效数值')}")

        elif cmd == "5":
            strength_str = _input("收缩强度 (0.0-1.0, 默认 0.3): ")
            try:
                strength = float(strength_str) if strength_str else 0.3
                strength = max(0.0, min(1.0, strength))
                for idx in sorted_indices:
                    vec = self._custom_vectors[idx]
                    self._custom_vectors[idx] = vec + (0.5 - vec) * strength
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已收缩 {n_shown} 个神经元')}")
            except ValueError:
                print(f"  {C_RED('无效数值')}")

        elif cmd == "6":
            strength_str = _input("扩散强度 (0.0-1.0, 默认 0.3): ")
            try:
                strength = float(strength_str) if strength_str else 0.3
                strength = max(0.0, min(1.0, strength))
                for idx in sorted_indices:
                    vec = self._custom_vectors[idx]
                    direction = np.where(vec > 0.5, 1.0 - vec, vec)
                    direction = np.clip(direction * 2, 0.0, 1.0)  # 归一化
                    self._custom_vectors[idx] = vec + direction * strength
                    self._custom_vectors[idx] = np.clip(self._custom_vectors[idx], 0.0, 1.0)
                    self._rebuild_speech(idx)
                self._modified = True
                print(f"  {C_GREEN(f'✓ 已扩散 {n_shown} 个神经元')}")
            except ValueError:
                print(f"  {C_RED('无效数值')}")

        elif cmd == "7":
            if _confirm("重置为原始状态? (y/n): "):
                self.virtual = copy.deepcopy(self._original_state)
                self._rebuild_all_vectors()
                self._modified = True
                print(f"  {C_GREEN('✓ 已重置')}")
        else:
            print(f"  {C_RED('无效选项')}")

        if self._modified:
            _pause()

    # ── 重置菜单 ──

    def _reset_menu(self):
        """重置到原始状态"""
        _clear()
        _header("🔄 重置")
        print("  [1] 重置当前过滤结果到原始值")
        print("  [2] 重置所有到原始值")
        print("  [q] 返回")
        cmd = _input()
        if cmd == "1":
            sorted_indices = self._get_filtered_indices()
            if _confirm(f"重置 {len(sorted_indices)} 个神经元? (y/n): "):
                for idx in sorted_indices:
                    if idx < len(self._original_state):
                        self.virtual[idx] = copy.deepcopy(self._original_state[idx])
                        self._rebuild_all_vectors()
                self._modified = True
        elif cmd == "2":
            if _confirm("重置所有? (y/n): "):
                self.virtual = copy.deepcopy(self._original_state)
                self._rebuild_all_vectors()
                self._modified = True

    # ── 统计信息 ──

    def get_statistics(self) -> Dict:
        """获取编辑统计信息"""
        self._rebuild_all_vectors()
        if not self._custom_vectors:
            return {"n_virtual": 0}
        matrix = np.array(self._custom_vectors)
        return {
            "n_virtual": len(self.virtual),
            "n_real": len(self.real),
            "modified": self._modified,
            "mean_vector": np.mean(matrix, axis=0).tolist(),
            "std_vector": np.std(matrix, axis=0).tolist(),
            "min_vector": np.min(matrix, axis=0).tolist(),
            "max_vector": np.max(matrix, axis=0).tolist(),
            "diversity_index": float(np.std(matrix)),
            "coverage": float(np.mean(np.max(matrix, axis=0) - np.min(matrix, axis=0))),
        }


# ═══════════════════════════════════════════════════════════════
# 便利函数 —— 直接集成到游戏流程中
# ═══════════════════════════════════════════════════════════════

def offer_neuron_editing(virtual_discussions: List[Dict],
                         real_discussions: List[Dict] = None,
                         prompt: str = "是否在合成前编辑虚拟专家神经元?",
                         auto_skip_threshold: int = 0) -> bool:
    """
    在虚拟专家生成后，提供神经元编辑选项。

    Args:
        virtual_discussions: 虚拟专家讨论列表
        real_discussions: 真实专家讨论列表（用于参考）
        prompt: 提示文本
        auto_skip_threshold: 如果虚拟专家数量小于此值，自动跳过

    Returns:
        True 如果用户进行了修改
    """
    if len(virtual_discussions) < auto_skip_threshold:
        return False

    print(f"\n  {C_DIM('─' * 50)}")
    print(f"  {C_YELLOW('🧠 相空间神经元编辑器')}")
    print(f"  {C_DIM(prompt)}")
    print(f"  {C_DIM('虚拟专家: ')}{len(virtual_discussions)} 个  |  "
          f"{C_DIM('真实专家: ')}{len(real_discussions) if real_discussions else 0} 个")
    print(f"  {C_DIM('你可以手动微调、创建、删除神经元，然后继续合成。')}")
    print(f"  {C_DIM('─' * 50)}")

    choice = _input("  进入编辑器? (y/n): ").lower()
    if choice not in ("y", "yes", "是"):
        return False

    editor = NeuronEditor(virtual_discussions, real_discussions)
    return editor.run()


def edit_virtual_discussions(generator, real_discussions: List[Dict] = None) -> bool:
    """
    从 VirtualExpertGenerator 直接编辑虚拟讨论。

    Args:
        generator: VirtualExpertGenerator 实例
        real_discussions: 真实专家列表

    Returns:
        True 如果进行了修改
    """
    virtual = generator.virtual_discussions
    if not virtual:
        return False
    return offer_neuron_editing(virtual, real_discussions or generator.real)


def auto_edit_from_synthesis(round_discussions: list,
                              target_experts: int = 800,
                              auto_enter: bool = False) -> Tuple[list, bool]:
    """
    在合成流程中自动集成编辑功能。

    用法:
        amplified_discussions, edited = auto_edit_from_synthesis(
            round_discussions, target_experts=500)
        if edited:
            print("用户修改了虚拟专家")

    Args:
        round_discussions: 本轮讨论列表
        target_experts: 目标专家数
        auto_enter: 是否自动进入编辑器（不询问）

    Returns:
        (amplified_discussions, was_edited)
    """
    from .emergence import VirtualExpertGenerator

    generator = VirtualExpertGenerator(round_discussions, target_experts=target_experts)
    amplified_discussions = generator.get_all_discussions()

    if auto_enter or _confirm("进入神经元编辑器? (y/n): "):
        editor = NeuronEditor(generator.virtual_discussions, round_discussions)
        was_edited = editor.run()
        # 重新构建 amplified_discussions
        amplified_discussions = generator.get_all_discussions()
        return amplified_discussions, was_edited

    return amplified_discussions, False