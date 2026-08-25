"""
反事实推演沙盘 —— "如果……会怎样？" 假设分析工具

核心功能：
- 加载检查点数据，允许用户修改关键参数
- 模拟"如果某条精华被移除/增强/弱化"会怎样
- 多场景分支管理：创建、比较、合并多个反事实场景
- 蒙特卡洛模拟：批量随机推演，统计模式发现
- 敏感性分析：量化各参数对结果的影响程度
- 因果链追踪：从操作到结果的完整因果路径
- 概率估计：基于多场景统计估算各结果的概率
- 差异引擎：两个场景之间的细粒度 diff
- 场景树可视化：所有场景的分支结构树
"""

import json
import os
import copy
import datetime
import random
import math
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter as pyCounter

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTextEdit, QWidget, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QScrollArea, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QRadioButton,
    QButtonGroup, QAbstractItemView, QMenu,
)


# ═══════════════════════════════════════════════════════════════
# 反事实场景 —— 一个独立的"what-if"分支
# ═══════════════════════════════════════════════════════════════

class CounterfactualScenario:
    """
    一个独立的反事实推演场景。

    每个场景有自己的：
    - 名称和描述
    - 修改后的精华池
    - 修改后的时间记忆
    - 操作历史
    - 父场景（从哪个场景分支而来）
    - 子场景列表
    - 标签（用于分类和组织）
    - 推演结果缓存
    """

    def __init__(self, name: str, engine: "CounterfactualEngine",
                 parent: Optional["CounterfactualScenario"] = None):
        self.name = name
        self.description = ""
        self.engine = engine
        self.parent = parent
        self.children: List["CounterfactualScenario"] = []
        self.tags: List[str] = []
        self.created_at = datetime.datetime.now().isoformat()
        self._result_cache: Optional[Dict] = None
        self._id = hashlib.md5(
            f"{name}_{self.created_at}_{random.random()}".encode()
        ).hexdigest()[:8]

    def duplicate(self, new_name: str) -> "CounterfactualScenario":
        """复制当前场景为一个新场景"""
        new_engine = copy.deepcopy(self.engine)
        new_engine.operations = copy.deepcopy(self.engine.operations)
        scenario = CounterfactualScenario(new_name, new_engine, parent=self)
        self.children.append(scenario)
        return scenario

    def branch(self, new_name: str, operations: List[Dict]) -> "CounterfactualScenario":
        """
        从当前场景分支出一个新场景，并应用一组操作。
        operations: 要应用的额外操作列表
        """
        # 从当前场景的修改状态开始
        current_items = copy.deepcopy(self.engine.modified_items)
        current_temporal = copy.deepcopy(self.engine.modified_temporal)

        # 创建一个新的引擎从原始数据开始，但有修改的精华池
        new_engine = copy.deepcopy(self.engine)
        new_engine.modified_items = current_items
        new_engine.modified_temporal = current_temporal
        new_engine.operations = copy.deepcopy(self.engine.operations)

        # 应用新操作
        for op in operations:
            op_type = op.get("type")
            if op_type == "boost":
                new_engine.boost_essence(op["item_id"], op.get("amount", 3.0))
            elif op_type == "suppress":
                new_engine.suppress_essence(op["item_id"], op.get("amount", 3.0))
            elif op_type == "remove":
                new_engine.remove_essence(op["item_id"])
            elif op_type == "add":
                new_engine.add_hypothetical(op.get("content", ""),
                                            score=op.get("score", 5.0))
            elif op_type == "temporal_boost":
                new_engine.boost_temporal_coupling(op.get("factor", 1.5))
            elif op_type == "temporal_weaken":
                new_engine.weaken_temporal_coupling(op.get("factor", 0.5))
            elif op_type == "temporal_reset":
                new_engine.reset_temporal_memory()

        scenario = CounterfactualScenario(new_name, new_engine, parent=self)
        self.children.append(scenario)
        return scenario

    def get_depth(self) -> int:
        """获取场景在树中的深度"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    def get_path(self) -> str:
        """获取从根到当前场景的路径"""
        parts = [self.name]
        current = self.parent
        while current:
            parts.append(current.name)
            current = current.parent
        return " → ".join(reversed(parts))

    def get_all_leaves(self) -> List["CounterfactualScenario"]:
        """获取所有叶子节点（终端场景）"""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

    def get_statistics_summary(self) -> Dict:
        """获取场景核心统计摘要"""
        stats = self.engine.get_statistics()
        o = stats["original"]
        m = stats["modified"]
        return {
            "id": self._id,
            "name": self.name,
            "depth": self.get_depth(),
            "items_original": o["count"],
            "items_modified": m["count"],
            "avg_score_original": o["avg_score"],
            "avg_score_modified": m["avg_score"],
            "score_delta": round(m["avg_score"] - o["avg_score"], 2),
            "operation_count": len(self.engine.operations),
            "tags": self.tags,
        }

    def to_dict(self) -> Dict:
        """序列化场景"""
        return {
            "id": self._id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "operations": self.engine.operations,
            "modified_items": self.engine.modified_items,
            "modified_temporal": self.engine.modified_temporal,
            "mode": self.engine.mode,
        }

    @classmethod
    def from_dict(cls, data: Dict, engine: "CounterfactualEngine") -> "CounterfactualScenario":
        """从字典恢复场景"""
        scenario = cls(data["name"], engine)
        scenario._id = data.get("id", scenario._id)
        scenario.description = data.get("description", "")
        scenario.tags = data.get("tags", [])
        scenario.created_at = data.get("created_at", scenario.created_at)
        if "modified_items" in data:
            engine.modified_items = data["modified_items"]
        if "modified_temporal" in data:
            engine.modified_temporal = data["modified_temporal"]
        if "operations" in data:
            engine.operations = data["operations"]
        if "mode" in data:
            engine.mode = data["mode"]
        return scenario


# ═══════════════════════════════════════════════════════════════
# 场景管理器 —— 管理多个场景的创建、分支、比较、树结构
# ═══════════════════════════════════════════════════════════════

class ScenarioManager:
    """
    多场景管理器。

    维护一个场景树，支持：
    - 创建根场景（基线）
    - 从任何场景分支出新场景
    - 场景之间的比较
    - 场景树遍历
    - 场景导入/导出
    - 批量场景统计
    """

    def __init__(self, base_engine: "CounterfactualEngine"):
        self.base_engine = base_engine
        self.root = CounterfactualScenario("基线", copy.deepcopy(base_engine))
        self._all_scenarios: Dict[str, CounterfactualScenario] = {}
        self._all_scenarios[self.root._id] = self.root
        self._active_scenario_id = self.root._id

    @property
    def active_scenario(self) -> CounterfactualScenario:
        return self._all_scenarios.get(self._active_scenario_id, self.root)

    def set_active(self, scenario_id: str) -> bool:
        if scenario_id in self._all_scenarios:
            self._active_scenario_id = scenario_id
            return True
        return False

    def create_branch(self, parent_id: str, name: str,
                      operations: List[Dict] = None) -> Optional[CounterfactualScenario]:
        """从指定场景创建分支"""
        parent = self._all_scenarios.get(parent_id)
        if not parent:
            return None
        if operations:
            new_scenario = parent.branch(name, operations)
        else:
            new_scenario = parent.duplicate(name)
        self._all_scenarios[new_scenario._id] = new_scenario
        return new_scenario

    def delete_scenario(self, scenario_id: str) -> bool:
        """删除一个场景（及其子场景）"""
        scenario = self._all_scenarios.get(scenario_id)
        if not scenario or scenario == self.root:
            return False
        # 先删除所有子场景
        for child in scenario.get_all_leaves():
            if child._id in self._all_scenarios and child != scenario:
                del self._all_scenarios[child._id]
        # 从父场景的子列表中移除
        if scenario.parent:
            scenario.parent.children = [
                c for c in scenario.parent.children if c._id != scenario_id
            ]
        del self._all_scenarios[scenario_id]
        if self._active_scenario_id == scenario_id:
            self._active_scenario_id = self.root._id
        return True

    def get_all_scenarios(self) -> List[CounterfactualScenario]:
        """获取所有场景"""
        return list(self._all_scenarios.values())

    def get_leaf_scenarios(self) -> List[CounterfactualScenario]:
        """获取所有叶子场景"""
        return self.root.get_all_leaves()

    def get_scenario_by_id(self, scenario_id: str) -> Optional[CounterfactualScenario]:
        return self._all_scenarios.get(scenario_id)

    def compare_all_leaves(self) -> List[Dict]:
        """比较所有叶子场景的统计摘要"""
        leaves = self.get_leaf_scenarios()
        summaries = [leaf.get_statistics_summary() for leaf in leaves]
        # 按评分增量排序
        summaries.sort(key=lambda s: s["score_delta"], reverse=True)
        return summaries

    def find_most_divergent(self) -> Tuple[CounterfactualScenario, CounterfactualScenario, float]:
        """
        找出与基线差异最大的叶子场景。

        返回 (场景, 基线推荐, 差异度)
        """
        leaves = self.get_leaf_scenarios()
        if len(leaves) <= 1:
            return (self.root, self.root, 0.0)

        baseline_stats = self.root.engine.get_statistics()
        baseline_avg = baseline_stats["modified"]["avg_score"]

        max_div = 0.0
        most_divergent = self.root
        for leaf in leaves:
            stats = leaf.engine.get_statistics()
            div = abs(stats["modified"]["avg_score"] - baseline_avg)
            if div > max_div:
                max_div = div
                most_divergent = leaf
        return (most_divergent, self.root, max_div)

    def find_best_scenario(self) -> CounterfactualScenario:
        """找出评分提升最大的场景"""
        leaves = self.get_leaf_scenarios()
        if not leaves:
            return self.root
        best = max(leaves, key=lambda s: s.get_statistics_summary()["score_delta"])
        return best

    def find_worst_scenario(self) -> CounterfactualScenario:
        """找出评分下降最大的场景"""
        leaves = self.get_leaf_scenarios()
        if not leaves:
            return self.root
        worst = min(leaves, key=lambda s: s.get_statistics_summary()["score_delta"])
        return worst

    def get_scenario_counts(self) -> Dict:
        """获取场景统计"""
        all_scenarios = self.get_all_scenarios()
        leaves = self.get_leaf_scenarios()
        return {
            "total": len(all_scenarios),
            "leaves": len(leaves),
            "max_depth": max(s.get_depth() for s in all_scenarios) if all_scenarios else 0,
            "branches": sum(1 for s in all_scenarios if s.children),
            "best_scenario": self.find_best_scenario().name,
            "worst_scenario": self.find_worst_scenario().name,
        }

    def export_scenarios(self) -> Dict:
        """导出所有场景"""
        return {
            "exported_at": datetime.datetime.now().isoformat(),
            "problem": self.base_engine.problem,
            "scenarios": [
                self._scenario_to_tree_dict(self.root)
            ],
        }

    def _scenario_to_tree_dict(self, scenario: CounterfactualScenario) -> Dict:
        data = scenario.to_dict()
        if scenario.children:
            data["children"] = [
                self._scenario_to_tree_dict(c) for c in scenario.children
            ]
        else:
            data["children"] = []
        return data

    def import_scenarios(self, data: Dict, parent_id: str = None) -> bool:
        """从字典导入场景"""
        try:
            parent = self._all_scenarios.get(parent_id, self.root) if parent_id else self.root
            self._import_scenario_tree(data, parent)
            return True
        except Exception as e:
            print(f"导入场景失败: {e}")
            return False

    def _import_scenario_tree(self, data: Dict, parent: CounterfactualScenario):
        new_engine = copy.deepcopy(self.base_engine)
        scenario = CounterfactualScenario.from_dict(data, new_engine)
        scenario.parent = parent
        parent.children.append(scenario)
        self._all_scenarios[scenario._id] = scenario
        for child_data in data.get("children", []):
            self._import_scenario_tree(child_data, scenario)


# ═══════════════════════════════════════════════════════════════
# 蒙特卡洛模拟器 —— 批量随机推演
# ═══════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    蒙特卡洛反事实模拟器。

    随机生成大量反事实操作，运行推演，统计结果分布。
    用于发现：
    - 哪些参数变化对结果影响最大
    - 结果分布的形状（正态/偏态/双峰）
    - 最可能的结果范围
    - 极端结果的条件
    """

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine
        self.results: List[Dict] = []
        self._rng = random.Random()

    def run_simulation(self, n_iterations: int = 100,
                       max_operations_per_run: int = 5,
                       progress_callback=None) -> Dict:
        """
        运行蒙特卡洛模拟。

        Args:
            n_iterations: 模拟次数
            max_operations_per_run: 每次最多随机操作数
            progress_callback: 进度回调函数(progress, total)

        Returns:
            模拟统计结果
        """
        self.results = []

        for i in range(n_iterations):
            engine = copy.deepcopy(self.base_engine)
            engine.operations = []

            # 随机决定操作数
            n_ops = self._rng.randint(1, max_operations_per_run)

            for _ in range(n_ops):
                op_type = self._rng.choice(["boost", "suppress", "remove", "add",
                                             "temporal_boost", "temporal_weaken"])

                if op_type in ("boost", "suppress"):
                    if not engine.modified_items:
                        continue
                    item = self._rng.choice(engine.modified_items)
                    amount = self._rng.uniform(1.0, 8.0)
                    if op_type == "boost":
                        engine.boost_essence(item["id"], amount)
                    else:
                        engine.suppress_essence(item["id"], amount)

                elif op_type == "remove":
                    if not engine.modified_items:
                        continue
                    item = self._rng.choice(engine.modified_items)
                    engine.remove_essence(item["id"])

                elif op_type == "add":
                    templates = [
                        "如果从另一个角度思考，或许我们应该考虑",
                        "一个被忽略的关键因素是",
                        "这个问题本质上涉及到",
                        "更深层次地看，这反映了",
                        "从全局角度出发，我们需要",
                    ]
                    template = self._rng.choice(templates)
                    content = f"{template}（蒙特卡洛假设 #{i}）"
                    score = self._rng.uniform(1.0, 10.0)
                    engine.add_hypothetical(content, score=score)

                elif op_type == "temporal_boost":
                    if engine.temporal_memory_data:
                        factor = self._rng.uniform(1.1, 3.0)
                        engine.boost_temporal_coupling(factor)

                elif op_type == "temporal_weaken":
                    if engine.temporal_memory_data:
                        factor = self._rng.uniform(0.1, 0.9)
                        engine.weaken_temporal_coupling(factor)

            # 收集结果
            stats = engine.get_statistics()
            comparison = engine.compare_rankings()
            o = stats["original"]
            m = stats["modified"]

            result = {
                "iteration": i,
                "n_operations": n_ops,
                "operation_types": [op["type"] for op in engine.operations],
                "items_before": o["count"],
                "items_after": m["count"],
                "avg_score_before": o["avg_score"],
                "avg_score_after": m["avg_score"],
                "score_delta": round(m["avg_score"] - o["avg_score"], 2),
                "max_score_before": o["max_score"],
                "max_score_after": m["max_score"],
                "std_dev_before": o["std_dev"],
                "std_dev_after": m["std_dev"],
                "gained_count": len(comparison["gained"]),
                "lost_count": len(comparison["lost"]),
                "new_count": len(comparison["new_entries"]),
                "removed_count": len(comparison["removed"]),
            }
            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, n_iterations)

        return self._compute_statistics()

    def _compute_statistics(self) -> Dict:
        """计算模拟统计"""
        if not self.results:
            return {"error": "no_results"}

        deltas = [r["score_delta"] for r in self.results]
        avg_delta = sum(deltas) / len(deltas)
        var_delta = sum((d - avg_delta) ** 2 for d in deltas) / len(deltas)
        std_delta = math.sqrt(var_delta) if var_delta > 0 else 0

        # 分布形状
        sorted_deltas = sorted(deltas)
        n = len(sorted_deltas)
        p25 = sorted_deltas[int(n * 0.25)]
        p50 = sorted_deltas[int(n * 0.50)]
        p75 = sorted_deltas[int(n * 0.75)]

        # 操作类型分布
        op_type_counter = pyCounter()
        for r in self.results:
            for ot in r["operation_types"]:
                op_type_counter[ot] += 1

        # 分布类型检测（简单启发式）
        median = p50
        skew = (p75 + p25 - 2 * median) / max(std_delta, 0.01)
        if abs(skew) < 0.3:
            distribution = "近似正态"
        elif skew > 0:
            distribution = "右偏态（正向结果更多）"
        else:
            distribution = "左偏态（负向结果更多）"

        # 最佳/最差场景
        best_idx = deltas.index(max(deltas))
        worst_idx = deltas.index(min(deltas))

        # 正向结果概率
        positive_ratio = sum(1 for d in deltas if d > 0) / n

        return {
            "n_iterations": len(self.results),
            "avg_delta": round(avg_delta, 4),
            "std_delta": round(std_delta, 4),
            "min_delta": round(min(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "p25": round(p25, 4),
            "p50": round(p50, 4),
            "p75": round(p75, 4),
            "skew": round(skew, 4),
            "distribution": distribution,
            "positive_ratio": round(positive_ratio, 4),
            "best_iteration": best_idx,
            "best_result": self.results[best_idx],
            "worst_iteration": worst_idx,
            "worst_result": self.results[worst_idx],
            "op_type_distribution": dict(op_type_counter),
            "all_results": self.results,
        }

    def find_most_influential_operations(self) -> List[Tuple[str, float]]:
        """
        找出最有影响力的操作类型。

        返回: [(操作类型, 平均影响力), ...]
        """
        if not self.results:
            return []

        op_impact = defaultdict(list)
        for r in self.results:
            for ot in set(r["operation_types"]):
                op_impact[ot].append(r["score_delta"])

        impacts = []
        for op_type, deltas in op_impact.items():
            avg_impact = sum(deltas) / len(deltas)
            impacts.append((op_type, round(avg_impact, 4)))

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        return impacts

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """获取置信区间"""
        if not self.results:
            return (0.0, 0.0)
        deltas = sorted([r["score_delta"] for r in self.results])
        n = len(deltas)
        lower_idx = int(n * (1 - confidence) / 2)
        upper_idx = int(n * (1 + confidence) / 2)
        return (deltas[lower_idx], deltas[upper_idx - 1])


# ═══════════════════════════════════════════════════════════════
# 敏感性分析器
# ═══════════════════════════════════════════════════════════════

class SensitivityAnalyzer:
    """
    敏感性分析器。

    量化每个参数变化对结果的敏感度。
    支持：
    - 单参数敏感性（每次改变一个参数）
    - 多参数交互敏感性（同时改变多个参数）
    - 敏感性排名
    - 最优参数组合
    """

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine

    def analyze_parameter_sensitivity(self, parameters: Dict[str, List]) -> Dict:
        """
        分析多参数敏感性。

        Args:
            parameters: {"参数名": [值列表]}

        Returns:
            {
                "param_name": {
                    "values": [...],
                    "results": [...],
                    "sensitivity": 0.0~1.0,
                    "optimal_value": ...,
                    "trend": "上升/下降/波动/未知",
                }
            }
        """
        results = {}

        for param_name, values in parameters.items():
            param_results = []
            for val in values:
                engine = copy.deepcopy(self.base_engine)
                engine.operations = []
                # 应用参数
                self._apply_parameter(engine, param_name, val)
                stats = engine.get_statistics()
                param_results.append({
                    "value": val,
                    "avg_score": stats["modified"]["avg_score"],
                    "max_score": stats["modified"]["max_score"],
                    "std_dev": stats["modified"]["std_dev"],
                    "count": stats["modified"]["count"],
                })

            # 计算敏感性（结果范围 / 参数范围）
            if len(param_results) >= 2:
                scores = [r["avg_score"] for r in param_results]
                score_range = max(scores) - min(scores)
                param_range = max(values) - min(values) if len(values) > 1 else 1.0
                sensitivity = score_range / max(param_range, 0.01)
                # 归一化到 0~1
                sensitivity = min(1.0, sensitivity * 2)

                # 趋势判断
                if len(scores) >= 3:
                    first_half = sum(scores[:len(scores)//2]) / max(len(scores)//2, 1)
                    second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)
                    if second_half > first_half * 1.05:
                        trend = "上升"
                    elif second_half < first_half * 0.95:
                        trend = "下降"
                    else:
                        trend = "波动"
                else:
                    trend = "未知"

                # 最优值
                best_idx = scores.index(max(scores))
                optimal_value = values[best_idx]
            else:
                sensitivity = 0.0
                trend = "未知"
                optimal_value = values[0] if values else None

            results[param_name] = {
                "values": values,
                "results": param_results,
                "sensitivity": round(sensitivity, 4),
                "optimal_value": optimal_value,
                "trend": trend,
            }

        return results

    def _apply_parameter(self, engine: "CounterfactualEngine",
                          param_name: str, value):
        """应用一个参数值到引擎"""
        if param_name == "boost_amount":
            if engine.modified_items:
                for item in engine.modified_items:
                    engine.boost_essence(item["id"], value)

        elif param_name == "suppress_amount":
            if engine.modified_items:
                for item in engine.modified_items:
                    engine.suppress_essence(item["id"], value)

        elif param_name == "temporal_factor":
            if engine.temporal_memory_data:
                if value > 1.0:
                    engine.boost_temporal_coupling(value)
                elif value < 1.0:
                    engine.weaken_temporal_coupling(value)

        elif param_name == "remove_count":
            for _ in range(int(value)):
                if engine.modified_items:
                    engine.remove_essence(engine.modified_items[0]["id"])

        elif param_name == "hypothetical_score":
            engine.add_hypothetical(
                "敏感性测试假设观点",
                score=value,
            )

    def get_sensitivity_ranking(self, parameters: Dict[str, List]) -> List[Tuple[str, float]]:
        """获取敏感性排名"""
        analysis = self.analyze_parameter_sensitivity(parameters)
        ranking = []
        for param_name, data in analysis.items():
            ranking.append((param_name, data["sensitivity"], data["trend"]))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking


# ═══════════════════════════════════════════════════════════════
# 因果链分析器
# ═══════════════════════════════════════════════════════════════

class CausalChainAnalyzer:
    """
    因果链分析器。

    从操作到结果追踪完整的因果路径。
    每个因果链包含：
    - 操作（cause）
    - 直接效果（immediate effect on items）
    - 间接效果（cascade effects on rankings）
    - 最终效果（final statistics change）
    """

    def __init__(self, engine: "CounterfactualEngine"):
        self.engine = engine

    def analyze(self) -> Dict:
        """分析完整的因果链"""
        chains = []
        for op in self.engine.operations:
            chain = self._trace_chain(op)
            chains.append(chain)

        return {
            "chains": chains,
            "total_chains": len(chains),
            "causal_density": self._compute_causal_density(chains),
            "dominant_chain": self._find_dominant_chain(chains),
        }

    def _trace_chain(self, operation: Dict) -> Dict:
        """追踪单个操作的因果链"""
        op_type = operation.get("type", "unknown")
        chain = {
            "operation": operation,
            "direct_effects": [],
            "cascade_effects": [],
            "final_impact": None,
        }

        if op_type in ("boost", "suppress"):
            item_id = operation.get("item_id")
            item = self.engine.get_essence_by_id(item_id)
            amount = operation.get("amount", 0)

            # 直接效果
            if item:
                chain["direct_effects"].append({
                    "target": f"精华 #{item_id}",
                    "action": "增强" if op_type == "boost" else "削弱",
                    "magnitude": amount,
                    "original_score": item.get("score", 0),
                    "new_score": item.get("score", 0) + (amount if op_type == "boost" else -amount),
                    "confidence": "高",
                })

            # 级联效果
            comparison = self.engine.compare_rankings()
            for entry in comparison.get("gained", []):
                if entry["id"] != item_id:
                    chain["cascade_effects"].append({
                        "target": f"精华 #{entry['id']}",
                        "effect": f"排名上升 #{entry['old_rank']} → #{entry['new_rank']}",
                        "score_change": f"{entry['old_score']:.1f} → {entry['new_score']:.1f}",
                        "confidence": "中",
                    })
            for entry in comparison.get("lost", []):
                if entry["id"] != item_id:
                    chain["cascade_effects"].append({
                        "target": f"精华 #{entry['id']}",
                        "effect": f"排名下降 #{entry['old_rank']} → #{entry['new_rank']}",
                        "score_change": f"{entry['old_score']:.1f} → {entry['new_score']:.1f}",
                        "confidence": "中",
                    })

        elif op_type == "remove":
            chain["direct_effects"].append({
                "target": f"精华 #{operation.get('item_id')}",
                "action": "移除",
                "magnitude": "完全删除",
                "confidence": "高",
            })
            chain["cascade_effects"].append({
                "target": "所有精华",
                "effect": "排名重新计算",
                "confidence": "中",
            })

        elif op_type == "add":
            chain["direct_effects"].append({
                "target": "新精华",
                "action": "添加",
                "magnitude": operation.get("description", "假设观点"),
                "confidence": "高",
            })
            chain["cascade_effects"].append({
                "target": "排名末尾",
                "effect": "后进可能影响整体分布",
                "confidence": "低",
            })

        elif "temporal" in op_type:
            chain["direct_effects"].append({
                "target": "时间耦合记忆",
                "action": "调整",
                "magnitude": f"×{operation.get('factor', 1.0)}",
                "confidence": "中",
            })
            chain["cascade_effects"].append({
                "target": "跨轮次认知协同",
                "effect": "间接影响涌现层级",
                "confidence": "低",
            })

        # 最终影响
        stats = self.engine.get_statistics()
        o = stats["original"]
        m = stats["modified"]
        chain["final_impact"] = {
            "avg_score_change": round(m["avg_score"] - o["avg_score"], 2),
            "count_change": m["count"] - o["count"],
            "std_dev_change": round(m["std_dev"] - o["std_dev"], 2),
        }

        return chain

    def _compute_causal_density(self, chains: List[Dict]) -> float:
        """计算因果密度（因果链的丰富程度）"""
        if not chains:
            return 0.0
        total_effects = sum(
            len(c["direct_effects"]) + len(c["cascade_effects"])
            for c in chains
        )
        total_possible = len(chains) * 3  # 每个操作至少1个直接+2个级联效果
        return min(1.0, total_effects / max(total_possible, 1))

    def _find_dominant_chain(self, chains: List[Dict]) -> Optional[Dict]:
        """找出主导因果链（影响力最大的）"""
        if not chains:
            return None
        best = max(chains, key=lambda c: abs(
            c["final_impact"]["avg_score_change"]
        ) if c["final_impact"] else 0)
        return best

    def generate_causal_summary(self, analysis: Dict) -> str:
        """生成因果链摘要文本"""
        lines = []
        lines.append("=" * 60)
        lines.append("🔗 因果链分析")
        lines.append("=" * 60)

        for i, chain in enumerate(analysis["chains"], 1):
            op = chain["operation"]
            lines.append(f"\n── 因果链 #{i}: {op.get('description', '')} ──")

            lines.append("  直接效果:")
            for effect in chain["direct_effects"]:
                lines.append(f"    · {effect['action']} {effect['target']} "
                             f"(幅度: {effect['magnitude']}, 置信度: {effect['confidence']})")

            if chain["cascade_effects"]:
                lines.append("  级联效果:")
                for effect in chain["cascade_effects"][:5]:
                    lines.append(f"    · {effect['target']}: {effect['effect']} "
                                 f"(置信度: {effect['confidence']})")

            if chain["final_impact"]:
                imp = chain["final_impact"]
                lines.append(f"  最终影响: 评分{'↑' if imp['avg_score_change'] > 0 else '↓'}"
                             f"{abs(imp['avg_score_change'])} | "
                             f"数量{'↑' if imp['count_change'] > 0 else '↓'}"
                             f"{abs(imp['count_change'])}")

        lines.append(f"\n因果密度: {analysis['causal_density']:.1%}")
        if analysis['dominant_chain']:
            op = analysis['dominant_chain']['operation']
            lines.append(f"主导因果链: {op.get('description', '')}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 概率估计器
# ═══════════════════════════════════════════════════════════════

class ProbabilityEstimator:
    """
    概率估计器。

    基于多场景统计，估算各结果的可能性。
    """

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def estimate_outcome_probabilities(self, n_bins: int = 5) -> Dict:
        """估算各结果区间概率"""
        leaves = self.manager.get_leaf_scenarios()
        if len(leaves) < 2:
            return {"error": "insufficient_scenarios"}

        deltas = [leaf.get_statistics_summary()["score_delta"] for leaf in leaves]
        min_delta = min(deltas)
        max_delta = max(deltas)
        bin_width = max((max_delta - min_delta) / n_bins, 0.01)

        bins = []
        for i in range(n_bins):
            lo = min_delta + i * bin_width
            hi = lo + bin_width
            count = sum(1 for d in deltas if lo <= d < hi)
            prob = count / len(deltas)
            bins.append({
                "range": f"[{lo:.2f}, {hi:.2f})",
                "count": count,
                "probability": round(prob, 4),
                "label": f"{'大幅提升' if lo > 0.5 else '小幅提升' if lo > 0 else '小幅下降' if hi > 0 else '大幅下降'}",
            })

        # 累积概率
        prob_positive = sum(1 for d in deltas if d > 0) / len(deltas)
        prob_negative = sum(1 for d in deltas if d < 0) / len(deltas)
        prob_neutral = sum(1 for d in deltas if d == 0) / len(deltas)

        return {
            "bins": bins,
            "prob_positive": round(prob_positive, 4),
            "prob_negative": round(prob_negative, 4),
            "prob_neutral": round(prob_neutral, 4),
            "expected_value": round(sum(deltas) / len(deltas), 4),
            "n_scenarios": len(leaves),
            "most_likely": max(bins, key=lambda b: b["probability"])["label"] if bins else "未知",
        }

    def estimate_success_probability(self, threshold: float = 0.5) -> float:
        """估算评分提升超过threshold的概率"""
        leaves = self.manager.get_leaf_scenarios()
        if not leaves:
            return 0.0
        successes = sum(1 for leaf in leaves
                        if leaf.get_statistics_summary()["score_delta"] > threshold)
        return successes / len(leaves)

    def estimate_risk(self, threshold: float = -0.5) -> float:
        """估算评分下降超过threshold的风险"""
        leaves = self.manager.get_leaf_scenarios()
        if not leaves:
            return 1.0
        failures = sum(1 for leaf in leaves
                       if leaf.get_statistics_summary()["score_delta"] < threshold)
        return failures / len(leaves)

    def generate_probability_report(self) -> str:
        """生成概率分析报告"""
        probs = self.estimate_outcome_probabilities()
        if "error" in probs:
            return f"概率估计失败: {probs['error']}"

        lines = []
        lines.append("=" * 60)
        lines.append("🎲 概率分析报告")
        lines.append("=" * 60)
        lines.append(f"基于 {probs['n_scenarios']} 个场景分析")
        lines.append("")
        lines.append("结果区间分布:")
        for bin_info in probs["bins"]:
            bar = "█" * int(bin_info["probability"] * 40)
            lines.append(f"  {bin_info['label']:<8} {bin_info['range']:>12}  "
                         f"{bar} {bin_info['probability']:.1%} ({bin_info['count']}次)")
        lines.append("")
        lines.append(f"正向概率: {probs['prob_positive']:.1%}")
        lines.append(f"负向概率: {probs['prob_negative']:.1%}")
        lines.append(f"中性概率: {probs['prob_neutral']:.1%}")
        lines.append(f"期望值: {probs['expected_value']:.2f}")
        lines.append(f"最可能结果: {probs['most_likely']}")

        success_p = self.estimate_success_probability(0.5)
        risk_p = self.estimate_risk(-0.5)
        lines.append(f"")
        lines.append(f"成功概率（评分↑ >0.5）: {success_p:.1%}")
        lines.append(f"风险概率（评分↓ >0.5）: {risk_p:.1%}")

        if success_p > risk_p:
            lines.append(f"\n📌 建议: 多场景推演表明，正向预期占优，可尝试实际执行。")
        elif risk_p > success_p:
            lines.append(f"\n📌 建议: 多场景推演表明，风险高于收益，建议谨慎。")
        else:
            lines.append(f"\n📌 建议: 多场景推演表明，结果不确定，建议增加更多分支。")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 差异引擎
# ═══════════════════════════════════════════════════════════════

class DiffEngine:
    """
    差异引擎 —— 细粒度比较两个场景的差异。
    """

    @staticmethod
    def diff_scenarios(scenario_a: CounterfactualScenario,
                       scenario_b: CounterfactualScenario) -> Dict:
        """
        比较两个场景的所有差异。

        返回: {
            "essence_diffs": [...],
            "ranking_diffs": [...],
            "stat_diffs": {...},
            "op_diffs": [...],
            "summary": "...",
            "similarity_score": 0.0~1.0,
        }
        """
        a_items = scenario_a.engine.modified_items
        b_items = scenario_b.engine.modified_items
        a_map = {e["id"]: e for e in a_items}
        b_map = {e["id"]: e for e in b_items}

        essence_diffs = []
        all_ids = set(list(a_map.keys()) + list(b_map.keys()))

        # 评分变化
        score_diffs = []
        for eid in all_ids:
            a_item = a_map.get(eid)
            b_item = b_map.get(eid)
            a_score = a_item["score"] if a_item else 0
            b_score = b_item["score"] if b_item else 0
            if a_score != b_score:
                score_diffs.append({
                    "id": eid,
                    "content": (a_item or b_item).get("content", "")[:50],
                    "score_a": a_score,
                    "score_b": b_score,
                    "delta": round(b_score - a_score, 2),
                    "type": "score_change",
                })

        # 新增/删除
        added = [e for eid, e in b_map.items() if eid not in a_map]
        removed = [e for eid, e in a_map.items() if eid not in b_map]

        for e in added:
            essence_diffs.append({
                "id": e["id"],
                "content": e.get("content", "")[:50],
                "score": e.get("score", 0),
                "type": "added",
            })
        for e in removed:
            essence_diffs.append({
                "id": e["id"],
                "content": e.get("content", "")[:50],
                "score": e.get("score", 0),
                "type": "removed",
            })

        # 排名变化
        a_ranking = scenario_a.engine.get_modified_ranking()
        b_ranking = scenario_b.engine.get_modified_ranking()
        a_rank_map = {e["id"]: i for i, e in enumerate(a_ranking)}
        b_rank_map = {e["id"]: i for i, e in enumerate(b_ranking)}

        ranking_diffs = []
        for eid in all_ids:
            if eid in a_rank_map and eid in b_rank_map:
                old_rank = a_rank_map[eid]
                new_rank = b_rank_map[eid]
                if old_rank != new_rank:
                    item = a_map.get(eid) or b_map.get(eid)
                    ranking_diffs.append({
                        "id": eid,
                        "content": item.get("content", "")[:50] if item else "",
                        "old_rank": old_rank + 1,
                        "new_rank": new_rank + 1,
                        "delta": old_rank - new_rank,
                    })

        # 统计差异
        a_stats = scenario_a.engine.get_statistics()
        b_stats = scenario_b.engine.get_statistics()
        stat_diffs = {}
        for key in ["count", "avg_score", "max_score", "min_score", "std_dev"]:
            a_val = a_stats["modified"].get(key, 0)
            b_val = b_stats["modified"].get(key, 0)
            diff_val = b_val - a_val if isinstance(a_val, (int, float)) else "N/A"
            stat_diffs[key] = {"a": a_val, "b": b_val, "diff": diff_val}

        # 操作差异
        op_diffs = []
        a_ops = set(op.get("description", "") for op in scenario_a.engine.operations)
        b_ops = set(op.get("description", "") for op in scenario_b.engine.operations)
        for op in b_ops - a_ops:
            op_diffs.append({"type": "added_in_b", "description": op})
        for op in a_ops - b_ops:
            op_diffs.append({"type": "removed_from_b", "description": op})

        # 相似度
        if essence_diffs:
            total_items = len(all_ids) + len(added) + len(removed)
            unchanged = total_items - len(essence_diffs)
            similarity = unchanged / max(total_items, 1)
        else:
            similarity = 1.0

        # 摘要
        summary = (
            f"场景 A '{scenario_a.name}' vs B '{scenario_b.name}': "
            f"{len(added)} 新增, {len(removed)} 移除, "
            f"{len(score_diffs)} 评分变化, {len(ranking_diffs)} 排名变化, "
            f"相似度 {similarity:.1%}"
        )

        return {
            "essence_diffs": essence_diffs,
            "score_diffs": score_diffs,
            "ranking_diffs": ranking_diffs,
            "stat_diffs": stat_diffs,
            "op_diffs": op_diffs,
            "summary": summary,
            "similarity_score": round(similarity, 4),
        }

    @staticmethod
    def generate_diff_report(diff: Dict) -> str:
        """生成差异报告文本"""
        lines = []
        lines.append("=" * 60)
        lines.append("📋 场景差异报告")
        lines.append("=" * 60)
        lines.append(diff["summary"])
        lines.append("")

        if diff["score_diffs"]:
            lines.append("─" * 40)
            lines.append("评分变化")
            lines.append("─" * 40)
            for d in sorted(diff["score_diffs"], key=lambda x: abs(x["delta"]), reverse=True)[:10]:
                lines.append(f"  #{d['id']} {d['content']}")
                lines.append(f"    {d['score_a']:.1f} → {d['score_b']:.1f} "
                             f"({'↑' if d['delta'] > 0 else '↓'}{abs(d['delta']):.1f})")

        if diff["ranking_diffs"]:
            lines.append("")
            lines.append("─" * 40)
            lines.append("排名变化")
            lines.append("─" * 40)
            for d in sorted(diff["ranking_diffs"], key=lambda x: abs(x["delta"]), reverse=True)[:10]:
                lines.append(f"  #{d['id']} {d['content']}")
                lines.append(f"    #{d['old_rank']} → #{d['new_rank']} "
                             f"({'↑' if d['delta'] > 0 else '↓'}{abs(d['delta'])})")

        if diff["essence_diffs"]:
            lines.append("")
            lines.append("─" * 40)
            lines.append("新增/移除")
            lines.append("─" * 40)
            for d in diff["essence_diffs"]:
                icon = "➕" if d["type"] == "added" else "🗑️"
                lines.append(f"  {icon} #{d['id']} {d['content']} (评分: {d['score']:.1f})")

        lines.append(f"\n相似度: {diff['similarity_score']:.1%}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 原反事实引擎（保留向后兼容）
# ═══════════════════════════════════════════════════════════════

class CounterfactualEngine:
    """反事实推演引擎（同原版，保持兼容）"""

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

        # 时间维度耦合记忆
        self.temporal_memory_data = data.get("temporal_memory", None)
        self.modified_temporal = copy.deepcopy(self.temporal_memory_data)

        # 操作历史
        self.operations: List[Dict] = []
        self._next_id = max((e.get("id", 0) for e in self.original_items), default=0) + 1

    # ── 修改操作（同原版） ──

    def boost_essence(self, item_id: int, amount: float = 3.0) -> bool:
        item = self._find_modified(item_id)
        if not item:
            return False
        item["score"] = item.get("score", 0) + amount
        item["tags"] = list(set(item.get("tags", []) + ["反事实增强"]))
        self.operations.append({
            "type": "boost", "item_id": item_id, "amount": amount,
            "description": f"增强 #{item_id} 评分 +{amount}",
        })
        return True

    def suppress_essence(self, item_id: int, amount: float = 3.0) -> bool:
        item = self._find_modified(item_id)
        if not item:
            return False
        item["score"] = item.get("score", 0) - amount
        item["tags"] = list(set(item.get("tags", []) + ["反事实削弱"]))
        self.operations.append({
            "type": "suppress", "item_id": item_id, "amount": amount,
            "description": f"削弱 #{item_id} 评分 -{amount}",
        })
        return True

    def remove_essence(self, item_id: int) -> bool:
        before = len(self.modified_items)
        self.modified_items = [e for e in self.modified_items if e.get("id") != item_id]
        if len(self.modified_items) < before:
            content = next((e.get("content", "")[:30] for e in self.original_items
                            if e.get("id") == item_id), "?")
            self.operations.append({
                "type": "remove", "item_id": item_id,
                "description": f"移除 #{item_id}: \"{content}...\"",
            })
            return True
        return False

    def add_hypothetical(self, content: str, contributor: str = "反事实推演",
                         score: float = 5.0, tags: List[str] = None) -> int:
        item = {
            "id": self._next_id, "content": content,
            "contributor": contributor,
            "source_round": self.rounds[-1].get("round_id", 1) if self.rounds else 1,
            "round": self.rounds[-1].get("round_id", 1) if self.rounds else 1,
            "score": score, "parent_id": None,
            "tags": tags or ["反事实假设"],
            "cited_by": [], "refined_by": [], "challenged_by": [],
            "approve_by": [], "reject_by": [], "abstain_by": [],
            "vote_reasons": [], "clarifications": [],
        }
        self._next_id += 1
        self.modified_items.append(item)
        self.operations.append({
            "type": "add", "item_id": item["id"],
            "description": f"添加假设: \"{content[:40]}...\"",
        })
        return item["id"]

    def change_mode(self, new_mode: str) -> None:
        self.mode = new_mode
        self.operations.append({
            "type": "mode_change", "description": f"切换模式: {new_mode}",
        })

    def reset(self) -> None:
        self.modified_items = copy.deepcopy(self.original_items)
        self.modified_temporal = copy.deepcopy(self.temporal_memory_data)
        self.operations = []
        self.mode = self.original_data.get("discussion_mode", "physical")

    # ── 时间记忆操作 ──

    def boost_temporal_coupling(self, factor: float = 1.5) -> bool:
        if self.modified_temporal is None:
            return False
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if not coupling:
            return False
        matrix = np.array(coupling) * factor
        self.modified_temporal["cumulative_coupling"] = np.clip(matrix, -1.0, 1.0).tolist()
        self.operations.append({
            "type": "temporal_boost", "factor": factor,
            "description": f"增强时间耦合 ×{factor}",
        })
        return True

    def weaken_temporal_coupling(self, factor: float = 0.5) -> bool:
        if self.modified_temporal is None:
            return False
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if not coupling:
            return False
        matrix = np.array(coupling) * factor
        self.modified_temporal["cumulative_coupling"] = matrix.tolist()
        self.operations.append({
            "type": "temporal_weaken", "factor": factor,
            "description": f"削弱时间耦合 ×{factor}",
        })
        return True

    def reset_temporal_memory(self) -> bool:
        if self.modified_temporal is None:
            return False
        coupling = self.modified_temporal.get("cumulative_coupling", None)
        if coupling:
            self.modified_temporal["cumulative_coupling"] = np.zeros_like(np.array(coupling)).tolist()
        self.modified_temporal["round"] = 0
        self.modified_temporal["topology_history"] = []
        self.modified_temporal["emergence_level_history"] = []
        self.operations.append({
            "type": "temporal_reset",
            "description": "清空时间记忆（模拟历史归零）",
        })
        return True

    def get_temporal_stats(self) -> Dict:
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
            "available": True, "shape": list(matrix.shape),
            "active_connections": int(active),
            "connection_density": round(density, 4),
            "rounds_recorded": self.modified_temporal.get("round", 0),
            "topology_entries": len(self.modified_temporal.get("topology_history", [])),
            "emergence_level_distribution": level_counts,
        }

    # ── 分析 & 比较 ──

    def get_original_ranking(self) -> List[Dict]:
        return sorted(self.original_items, key=lambda e: e.get("score", 0), reverse=True)

    def get_modified_ranking(self) -> List[Dict]:
        return sorted(self.modified_items, key=lambda e: e.get("score", 0), reverse=True)

    def compare_rankings(self) -> Dict:
        original = self.get_original_ranking()
        modified = self.get_modified_ranking()
        orig_map = {e["id"]: {"rank": i, "score": e.get("score", 0)}
                     for i, e in enumerate(original)}
        mod_map = {e["id"]: {"rank": i, "score": e.get("score", 0)}
                    for i, e in enumerate(modified)}
        gained, lost, unchanged, new_entries, removed = [], [], [], [], []
        for e in modified:
            eid = e["id"]
            if eid in orig_map:
                old_rank = orig_map[eid]["rank"]
                new_rank = mod_map[eid]["rank"]
                delta = old_rank - new_rank
                if delta > 0:
                    gained.append({"id": eid, "content": e.get("content", "")[:50],
                                   "old_rank": old_rank + 1, "new_rank": new_rank + 1,
                                   "old_score": orig_map[eid]["score"],
                                   "new_score": mod_map[eid]["score"], "delta": delta})
                elif delta < 0:
                    lost.append({"id": eid, "content": e.get("content", "")[:50],
                                 "old_rank": old_rank + 1, "new_rank": new_rank + 1,
                                 "old_score": orig_map[eid]["score"],
                                 "new_score": mod_map[eid]["score"], "delta": delta})
                else:
                    unchanged.append(eid)
            else:
                new_entries.append({"id": eid, "content": e.get("content", "")[:50],
                                    "score": e.get("score", 0)})
        for e in original:
            if e["id"] not in mod_map:
                removed.append({"id": e["id"], "content": e.get("content", "")[:50],
                                "old_score": e.get("score", 0)})
        gained.sort(key=lambda x: x["delta"], reverse=True)
        lost.sort(key=lambda x: x["delta"])
        return {"gained": gained, "lost": lost, "unchanged": unchanged,
                "new_entries": new_entries, "removed": removed}

    def get_statistics(self) -> Dict:
        def calc(items):
            if not items:
                return {"count": 0, "avg_score": 0, "max_score": 0, "min_score": 0, "std_dev": 0}
            scores = [e.get("score", 0) for e in items]
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            return {"count": len(items), "avg_score": round(avg, 2),
                    "max_score": max(scores), "min_score": min(scores),
                    "std_dev": round(variance ** 0.5, 2)}
        return {"original": calc(self.original_items), "modified": calc(self.modified_items)}

    def generate_counterfactual_synthesis(self) -> str:
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
        if self.operations:
            lines.append("─" * 40)
            lines.append("📝 推演操作")
            lines.append("─" * 40)
            for op in self.operations:
                lines.append(f"  · {op.get('description', '')}")
            lines.append("")
        tstats = self.get_temporal_stats()
        if tstats.get("available"):
            lines.append("─" * 40)
            lines.append("⏳ 时间耦合记忆")
            lines.append("─" * 40)
            lines.append(f"  记录轮次: {tstats['rounds_recorded']}")
            lines.append(f"  活跃连接: {tstats['active_connections']} "
                         f"/ {tstats['shape'][0]}×{tstats['shape'][1]}")
            lines.append(f"  连接密度: {tstats['connection_density']:.2%}")
            if tstats.get("emergence_level_distribution"):
                dist = tstats["emergence_level_distribution"]
                dist_str = ", ".join(f"L{k}: {v}次" for k, v in sorted(dist.items()))
                lines.append(f"  涌现层级分布: {dist_str}")
            temporal_ops = [op for op in self.operations if "temporal" in op["type"]]
            if temporal_ops:
                lines.append("  📌 可推演: 耦合强度变化将影响跨轮次认知协同效应")
                if tstats["connection_density"] > 0.3:
                    lines.append("  🔗 当前耦合密度较高，历史协同效应显著")
                elif tstats["connection_density"] < 0.1:
                    lines.append("  🔗 当前耦合密度较低，历史协同效应较弱")
            lines.append("")
        lines.append("─" * 40)
        lines.append("📊 统计对比")
        lines.append("─" * 40)
        o = stats["original"]
        m = stats["modified"]
        lines.append(f"  精华数量: {o['count']} → {m['count']}")
        delta_str = f"{'↑' if m['avg_score'] > o['avg_score'] else '↓'}{abs(m['avg_score'] - o['avg_score']):.1f}"
        lines.append(f"  平均评分: {o['avg_score']} → {m['avg_score']} ({delta_str})")
        lines.append(f"  最高评分: {o['max_score']} → {m['max_score']}")
        lines.append(f"  评分标准差: {o['std_dev']} → {m['std_dev']}")
        lines.append("")
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
        lines.append("=" * 60)
        lines.append("🔮 反事实综合方案（假设推演）")
        lines.append("=" * 60)
        if top_modified:
            lines.append(f"\n  在\"{self.problem[:40]}...\"的讨论中，")
            if comparison["gained"]:
                g = comparison["gained"][0]
                lines.append(f"  如果 {g['content'][:30]} 获得更多认可，")
                lines.append(f"  讨论的焦点将发生偏移。")
            lines.append("")
            lines.append("  💡 核心观点（按影响力排序）:")
            for i, e in enumerate(top_modified, 1):
                tags = ", ".join(e.get("tags", []))[:30]
                lines.append(f"    {i}. {e.get('content', '')[:60]}")
                lines.append(f"       (评分:{e.get('score', 0):.1f} | "
                             f"{e.get('contributor', '?')} | {tags})")
                lines.append("")
        delta = m["avg_score"] - o["avg_score"]
        if delta > 0.5:
            lines.append("  📌 结论: 反事实推演表明，该修改将提升整体讨论质量。")
        elif delta < -0.5:
            lines.append("  📌 结论: 反事实推演表明，该修改将降低讨论质量。")
        else:
            lines.append("  📌 结论: 反事实推演表明，该修改对整体讨论影响有限。")
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
    return engine.generate_counterfactual_synthesis()


# ═══════════════════════════════════════════════════════════════
# GUI 反事实推演对话框
# ═══════════════════════════════════════════════════════════════

class CounterfactualDialog(QDialog):
    """反事实推演沙盘对话框（含多场景管理）"""

    def __init__(self, engine: CounterfactualEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.scenario_manager = ScenarioManager(engine)
        self.monte_carlo = MonteCarloSimulator(engine)
        self.sensitivity = SensitivityAnalyzer(engine)
        self.causal_analyzer = CausalChainAnalyzer(engine)
        self.probability_estimator = ProbabilityEstimator(self.scenario_manager)
        self.diff_engine = DiffEngine()
        self._init_ui()
        self._refresh_essence_list()
        self._update_report()
        self._refresh_scenario_tree()
        self._refresh_compare_table()

    def _init_ui(self):
        self.setWindowTitle("🔮 反事实推演沙盘")
        self.setMinimumSize(1200, 800)
        self.resize(1300, 850)

        self.setStyleSheet("""
            QDialog { background-color: #0d0d0d; color: #d4d4d4; }
            QLabel { color: #c89b3c; }
            QGroupBox {
                color: #c89b3c; font-weight: bold;
                border: 1px solid #2a2a2a; border-radius: 4px;
                margin-top: 12px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
            }
            QTextEdit {
                background-color: #111111; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px;
                padding: 8px; font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #c89b3c; color: #0d0d0d;
                border: none; border-radius: 4px;
                padding: 6px 16px; font-weight: bold; font-size: 12px;
                min-height: 28px;
            }
            QPushButton:hover { background-color: #dbb052; }
            QPushButton:disabled { background-color: #333333; color: #666666; }
            QPushButton.danger { background-color: #c0392b; }
            QPushButton.danger:hover { background-color: #e74c3c; }
            QPushButton.secondary { background-color: #2a2a2a; color: #d4d4d4; }
            QPushButton.secondary:hover { background-color: #3a3a3a; }
            QListWidget {
                background-color: #111111; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px; font-size: 12px;
            }
            QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #1e1e1e; }
            QListWidget::item:selected { background-color: #c89b3c; color: #0d0d0d; }
            QComboBox {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px;
                padding: 4px 8px; min-height: 24px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px;
                padding: 4px 8px; min-height: 24px;
            }
            QTabWidget::pane { border: 1px solid #2a2a2a; background-color: #0d0d0d; }
            QTabBar::tab {
                background-color: #1e1e1e; color: #888888;
                padding: 6px 16px; border: 1px solid #2a2a2a;
                border-bottom: none; border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background-color: #0d0d0d; color: #c89b3c; }
            QTableWidget {
                background-color: #111111; color: #d4d4d4;
                border: 1px solid #2a2a2a; gridline-color: #1e1e1e;
                font-size: 12px;
            }
            QTableWidget::item:selected { background-color: #c89b3c; color: #0d0d0d; }
            QHeaderView::section {
                background-color: #1e1e1e; color: #c89b3c;
                padding: 4px; border: 1px solid #2a2a2a; font-weight: bold;
            }
            QTreeWidget {
                background-color: #111111; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px; font-size: 12px;
            }
            QTreeWidget::item:selected { background-color: #c89b3c; color: #0d0d0d; }
            QProgressBar {
                border: 1px solid #2a2a2a; border-radius: 4px;
                text-align: center; color: #d4d4d4;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk { background-color: #c89b3c; border-radius: 3px; }
            QLineEdit {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #2a2a2a; border-radius: 4px;
                padding: 4px 8px; min-height: 24px;
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
                      f"场景: {len(self.scenario_manager.get_all_scenarios())} 个")
        info.setStyleSheet("color: #888888; font-size: 12px;")
        main_layout.addWidget(info)

        # 标签页
        tabs = QTabWidget()
        tabs.setFont(QFont("Microsoft YaHei", 10))

        # ── Tab 1: 操作面板 ──
        op_tab = self._build_operation_tab()
        tabs.addTab(op_tab, "🔧 操作")

        # ── Tab 2: 推演报告 ──
        report_tab = self._build_report_tab()
        tabs.addTab(report_tab, "📄 推演报告")

        # ── Tab 3: 对比表 ──
        compare_tab = self._build_comparison_tab()
        tabs.addTab(compare_tab, "📊 对比表")

        # ── Tab 4: 统计 ──
        stats_tab = self._build_stats_tab()
        tabs.addTab(stats_tab, "📈 统计")

        # ── Tab 5: 场景树 ──
        tree_tab = self._build_scenario_tree_tab()
        tabs.addTab(tree_tab, "🌲 场景树")

        # ── Tab 6: 多场景对比 ──
        multi_compare_tab = self._build_multi_compare_tab()
        tabs.addTab(multi_compare_tab, "📊 多场景对比")

        # ── Tab 7: 蒙特卡洛 ──
        monte_tab = self._build_monte_carlo_tab()
        tabs.addTab(monte_tab, "🎲 蒙特卡洛")

        # ── Tab 8: 敏感性分析 ──
        sensitivity_tab = self._build_sensitivity_tab()
        tabs.addTab(sensitivity_tab, "📐 敏感性")

        # ── Tab 9: 因果链 ──
        causal_tab = self._build_causal_tab()
        tabs.addTab(causal_tab, "🔗 因果链")

        # ── Tab 10: 概率分析 ──
        prob_tab = self._build_probability_tab()
        tabs.addTab(prob_tab, "🎲 概率")

        main_layout.addWidget(tabs, 1)

        # 底部关闭按钮
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        bottom_row.addWidget(self.btn_close)
        main_layout.addLayout(bottom_row)

    # ── Tab 构建 ──

    def _build_operation_tab(self) -> QWidget:
        """Tab 1: 操作面板"""
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

        # 评分调整
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

        # 时间记忆操作
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

        # 创建场景分支按钮
        branch_group = QGroupBox("🌲 场景分支")
        branch_layout = QVBoxLayout(branch_group)
        branch_layout.addWidget(QLabel("当前操作可作为新场景分支保存:"))
        self.branch_name_input = QLineEdit()
        self.branch_name_input.setPlaceholderText("输入场景名称...")
        branch_layout.addWidget(self.branch_name_input)
        self.btn_create_branch = QPushButton("➕ 创建为新场景")
        self.btn_create_branch.clicked.connect(self._create_branch)
        branch_layout.addWidget(self.btn_create_branch)
        right_layout.addWidget(branch_group)

        # 重置
        self.btn_reset = QPushButton("🔄 重置所有操作")
        self.btn_reset.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        self.btn_reset.clicked.connect(self._reset_all)
        right_layout.addWidget(self.btn_reset)
        right_layout.addStretch()
        op_layout.addWidget(right_panel)

        return op_tab

    def _build_report_tab(self) -> QWidget:
        """Tab 2: 推演报告"""
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
        return report_tab

    def _build_comparison_tab(self) -> QWidget:
        """Tab 3: 对比表"""
        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        compare_layout.setContentsMargins(8, 8, 8, 8)
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(6)
        self.compare_table.setHorizontalHeaderLabels(["ID", "内容", "原评分", "新评分", "原排名", "新排名"])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.compare_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        compare_layout.addWidget(self.compare_table, 1)
        self.btn_refresh_compare = QPushButton("🔄 刷新对比表")
        self.btn_refresh_compare.clicked.connect(self._refresh_compare_table)
        compare_layout.addWidget(self.btn_refresh_compare)
        return compare_tab

    def _build_stats_tab(self) -> QWidget:
        """Tab 4: 统计"""
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text, 1)
        self.btn_refresh_stats = QPushButton("🔄 刷新统计")
        self.btn_refresh_stats.clicked.connect(self._refresh_stats)
        stats_layout.addWidget(self.btn_refresh_stats)
        return stats_tab

    def _build_scenario_tree_tab(self) -> QWidget:
        """Tab 5: 场景树"""
        tree_tab = QWidget()
        tree_layout = QVBoxLayout(tree_tab)
        tree_layout.setContentsMargins(8, 8, 8, 8)

        # 场景树控件
        self.scenario_tree = QTreeWidget()
        self.scenario_tree.setHeaderLabels(["场景名称", "ID", "深度", "评分变化", "操作数", "场景数"])
        self.scenario_tree.setColumnWidth(0, 250)
        self.scenario_tree.setColumnWidth(1, 100)
        self.scenario_tree.setColumnWidth(2, 60)
        self.scenario_tree.setColumnWidth(3, 100)
        self.scenario_tree.setColumnWidth(4, 80)
        self.scenario_tree.setColumnWidth(5, 80)
        self.scenario_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.scenario_tree.itemClicked.connect(self._on_scenario_tree_click)
        tree_layout.addWidget(self.scenario_tree, 1)

        # 操作按钮
        btn_row = QHBoxLayout()
        self.btn_refresh_tree = QPushButton("🔄 刷新场景树")
        self.btn_refresh_tree.clicked.connect(self._refresh_scenario_tree)
        btn_row.addWidget(self.btn_refresh_tree)

        self.btn_activate_scenario = QPushButton("▶ 激活选中场景")
        self.btn_activate_scenario.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_activate_scenario.clicked.connect(self._activate_scenario)
        btn_row.addWidget(self.btn_activate_scenario)

        self.btn_delete_scenario = QPushButton("🗑️ 删除场景")
        self.btn_delete_scenario.setStyleSheet("background-color: #c0392b; color: white;")
        self.btn_delete_scenario.clicked.connect(self._delete_scenario)
        btn_row.addWidget(self.btn_delete_scenario)

        self.btn_diff_scenarios = QPushButton("📋 差异对比")
        self.btn_diff_scenarios.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        self.btn_diff_scenarios.clicked.connect(self._diff_selected_scenarios)
        btn_row.addWidget(self.btn_diff_scenarios)

        btn_row.addStretch()
        tree_layout.addLayout(btn_row)

        # 场景摘要
        self.scenario_info = QLabel("选择场景查看详情")
        self.scenario_info.setStyleSheet("color: #888888; font-size: 11px; padding: 4px;")
        self.scenario_info.setWordWrap(True)
        tree_layout.addWidget(self.scenario_info)

        return tree_tab

    def _build_multi_compare_tab(self) -> QWidget:
        """Tab 6: 多场景对比"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        self.multi_compare_table = QTableWidget()
        self.multi_compare_table.setColumnCount(7)
        self.multi_compare_table.setHorizontalHeaderLabels(
            ["场景", "深度", "操作数", "精华数", "平均评分", "评分变化", "标签"]
        )
        self.multi_compare_table.horizontalHeader().setStretchLastSection(True)
        self.multi_compare_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.multi_compare_table, 1)

        btn_row = QHBoxLayout()
        self.btn_refresh_multi = QPushButton("🔄 刷新多场景对比")
        self.btn_refresh_multi.clicked.connect(self._refresh_multi_compare)
        btn_row.addWidget(self.btn_refresh_multi)
        self.btn_export_scenarios = QPushButton("💾 导出所有场景")
        self.btn_export_scenarios.setStyleSheet("background-color: #2a2a2a; color: #d4d4d4;")
        self.btn_export_scenarios.clicked.connect(self._export_scenarios)
        btn_row.addWidget(self.btn_export_scenarios)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return tab

    def _build_monte_carlo_tab(self) -> QWidget:
        """Tab 7: 蒙特卡洛模拟"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        # 控制面板
        ctrl_group = QGroupBox("模拟参数")
        ctrl_layout = QHBoxLayout(ctrl_group)
        ctrl_layout.addWidget(QLabel("迭代次数:"))
        self.mc_iterations = QSpinBox()
        self.mc_iterations.setRange(10, 1000)
        self.mc_iterations.setValue(100)
        self.mc_iterations.setSingleStep(10)
        ctrl_layout.addWidget(self.mc_iterations)
        ctrl_layout.addWidget(QLabel("每次操作数:"))
        self.mc_ops = QSpinBox()
        self.mc_ops.setRange(1, 10)
        self.mc_ops.setValue(5)
        ctrl_layout.addWidget(self.mc_ops)
        self.btn_run_mc = QPushButton("▶ 运行蒙特卡洛模拟")
        self.btn_run_mc.clicked.connect(self._run_monte_carlo)
        ctrl_layout.addWidget(self.btn_run_mc)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl_group)

        # 进度条
        self.mc_progress = QProgressBar()
        self.mc_progress.setVisible(False)
        layout.addWidget(self.mc_progress)

        # 结果展示
        self.mc_text = QTextEdit()
        self.mc_text.setReadOnly(True)
        layout.addWidget(self.mc_text, 1)

        return tab

    def _build_sensitivity_tab(self) -> QWidget:
        """Tab 8: 敏感性分析"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_group = QGroupBox("参数选择")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_run_sensitivity = QPushButton("▶ 运行敏感性分析")
        self.btn_run_sensitivity.clicked.connect(self._run_sensitivity)
        ctrl_layout.addWidget(self.btn_run_sensitivity)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl_group)

        self.sensitivity_text = QTextEdit()
        self.sensitivity_text.setReadOnly(True)
        layout.addWidget(self.sensitivity_text, 1)

        return tab

    def _build_causal_tab(self) -> QWidget:
        """Tab 9: 因果链分析"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_group = QGroupBox("因果链分析")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_run_causal = QPushButton("▶ 运行因果链分析")
        self.btn_run_causal.clicked.connect(self._run_causal)
        ctrl_layout.addWidget(self.btn_run_causal)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl_group)

        self.causal_text = QTextEdit()
        self.causal_text.setReadOnly(True)
        layout.addWidget(self.causal_text, 1)

        return tab

    def _build_probability_tab(self) -> QWidget:
        """Tab 10: 概率分析"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_group = QGroupBox("概率估计")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.btn_run_prob = QPushButton("▶ 运行概率分析")
        self.btn_run_prob.clicked.connect(self._run_probability)
        ctrl_layout.addWidget(self.btn_run_prob)
        ctrl_layout.addStretch()
        layout.addWidget(ctrl_group)

        self.prob_text = QTextEdit()
        self.prob_text.setReadOnly(True)
        layout.addWidget(self.prob_text, 1)

        return tab

    # ── GUI 回调 ──

    def _refresh_essence_list(self):
        self.essence_list.clear()
        items = self.engine.get_modified_ranking()
        for e in items:
            eid = e.get("id", 0)
            content = e.get("content", "")[:45]
            score = e.get("score", 0)
            tags = e.get("tags", [])
            tag_str = ""
            if "反事实增强" in tags: tag_str = " ⬆"
            elif "反事实削弱" in tags: tag_str = " ⬇"
            elif "反事实假设" in tags: tag_str = " ✨"
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
        items = self.engine.get_modified_ranking()
        if 0 <= row < len(items):
            e = items[row]
            tags = ", ".join(e.get("tags", []))
            self.selected_info.setText(
                f"#{e['id']} | {e.get('contributor', '?')} | 评分: {e.get('score', 0):.1f}\n"
                f"标签: {tags}\n{e.get('content', '')[:80]}"
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
        self.engine.boost_essence(eid, self.amount_spin.value())
        self._refresh_essence_list()
        self._update_report()

    def _suppress_selected(self):
        eid = self._get_selected_id()
        if eid is None:
            return
        self.engine.suppress_essence(eid, self.amount_spin.value())
        self._refresh_essence_list()
        self._update_report()

    def _remove_selected(self):
        eid = self._get_selected_id()
        if eid is None:
            return
        reply = QMessageBox.question(self, "确认移除",
            f"确定要移除精华 #{eid} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.remove_essence(eid)
            self._refresh_essence_list()
            self._update_report()

    def _add_hypothetical(self):
        content = self.hypo_input.toPlainText().strip()
        if not content:
            return
        self.engine.add_hypothetical(content, score=self.hypo_score.value())
        self.hypo_input.clear()
        self._refresh_essence_list()
        self._update_report()

    def _apply_mode(self):
        self.engine.change_mode(self.mode_combo.currentText())
        self._update_report()

    def _reset_all(self):
        reply = QMessageBox.question(self, "确认重置",
            "确定要重置所有操作吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.reset()
            self.mode_combo.setCurrentText(self.engine.mode)
            self._refresh_essence_list()
            self._update_report()

    def _boost_temporal(self):
        self.engine.boost_temporal_coupling(factor=1.5)
        self._update_report()

    def _weaken_temporal(self):
        self.engine.weaken_temporal_coupling(factor=0.5)
        self._update_report()

    def _reset_temporal(self):
        reply = QMessageBox.question(self, "确认清空",
            "确定要清空时间记忆吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.engine.reset_temporal_memory()
            self._update_report()

    # ── 场景管理 ──

    def _create_branch(self):
        name = self.branch_name_input.text().strip()
        if not name:
            name = f"场景 {len(self.scenario_manager.get_all_scenarios())}"
        # 收集当前操作
        operations = [copy.deepcopy(op) for op in self.engine.operations]
        if not operations:
            QMessageBox.information(self, "提示", "请在创建场景前先执行一些操作。")
            return
        scenario = self.scenario_manager.create_branch(
            self.scenario_manager.root._id, name, operations
        )
        if scenario:
            self.branch_name_input.clear()
            self._refresh_scenario_tree()
            self._refresh_multi_compare()
            self._update_info()

    def _refresh_scenario_tree(self):
        self.scenario_tree.clear()
        self._add_scenario_to_tree(self.scenario_manager.root, None)

    def _add_scenario_to_tree(self, scenario: CounterfactualScenario, parent_item):
        stats = scenario.get_statistics_summary()
        delta_str = f"{'+' if stats['score_delta'] >= 0 else ''}{stats['score_delta']:.2f}"
        tag_str = ", ".join(scenario.tags) if scenario.tags else ""
        item = QTreeWidgetItem(parent_item, [
            scenario.name, scenario._id, str(stats["depth"]),
            delta_str, str(stats["operation_count"]),
            str(len(scenario.get_all_leaves())),
        ])
        item.setData(0, Qt.ItemDataRole.UserRole, scenario._id)
        if stats["score_delta"] > 0:
            item.setForeground(3, QColor(52, 168, 83))
        elif stats["score_delta"] < 0:
            item.setForeground(3, QColor(234, 67, 53))
        for child in scenario.children:
            self._add_scenario_to_tree(child, item)

    def _on_scenario_tree_click(self, item: QTreeWidgetItem, column: int):
        scenario_id = item.data(0, Qt.ItemDataRole.UserRole)
        if scenario_id:
            scenario = self.scenario_manager.get_scenario_by_id(scenario_id)
            if scenario:
                stats = scenario.get_statistics_summary()
                self.scenario_info.setText(
                    f"场景: {scenario.name}  |  "
                    f"深度: {stats['depth']}  |  "
                    f"操作: {stats['operation_count']}  |  "
                    f"原始评分: {stats['avg_score_original']}  →  "
                    f"修改后: {stats['avg_score_modified']}  "
                    f"({'↑' if stats['score_delta'] >= 0 else '↓'}{abs(stats['score_delta'])})\n"
                    f"路径: {scenario.get_path()}\n"
                    f"标签: {', '.join(scenario.tags) if scenario.tags else '无'}"
                )

    def _activate_scenario(self):
        selected = self.scenario_tree.currentItem()
        if not selected:
            return
        scenario_id = selected.data(0, Qt.ItemDataRole.UserRole)
        if scenario_id and self.scenario_manager.set_active(scenario_id):
            scenario = self.scenario_manager.active_scenario
            # 同步引擎状态
            self.engine.modified_items = scenario.engine.modified_items
            self.engine.modified_temporal = scenario.engine.modified_temporal
            self.engine.operations = scenario.engine.operations
            self.engine.mode = scenario.engine.mode
            self.mode_combo.setCurrentText(self.engine.mode)
            self._refresh_essence_list()
            self._update_report()
            QMessageBox.information(self, "激活成功",
                f"已激活场景: {scenario.name}")

    def _delete_scenario(self):
        selected = self.scenario_tree.currentItem()
        if not selected:
            return
        scenario_id = selected.data(0, Qt.ItemDataRole.UserRole)
        if not scenario_id:
            return
        scenario = self.scenario_manager.get_scenario_by_id(scenario_id)
        if scenario == self.scenario_manager.root:
            QMessageBox.warning(self, "警告", "不能删除基线场景。")
            return
        reply = QMessageBox.question(self, "确认删除",
            f"确定要删除场景 '{scenario.name}' 及其所有子场景吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.scenario_manager.delete_scenario(scenario_id)
            self._refresh_scenario_tree()
            self._refresh_multi_compare()
            self._update_info()

    def _diff_selected_scenarios(self):
        """选中两个场景进行差异对比"""
        from PySide6.QtWidgets import QInputDialog
        all_scenarios = self.scenario_manager.get_all_scenarios()
        if len(all_scenarios) < 2:
            QMessageBox.information(self, "提示", "需要至少两个场景才能进行差异对比。")
            return
        names = [s.name for s in all_scenarios]
        name_a, ok_a = QInputDialog.getItem(self, "选择场景 A", "场景 A:", names, 0, False)
        if not ok_a:
            return
        name_b, ok_b = QInputDialog.getItem(self, "选择场景 B", "场景 B:", names, 1, False)
        if not ok_b:
            return
        scenario_a = next((s for s in all_scenarios if s.name == name_a), None)
        scenario_b = next((s for s in all_scenarios if s.name == name_b), None)
        if not scenario_a or not scenario_b:
            return
        diff = DiffEngine.diff_scenarios(scenario_a, scenario_b)
        report = DiffEngine.generate_diff_report(diff)
        dialog = QDialog(self)
        dialog.setWindowTitle("📋 场景差异报告")
        dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(report)
        layout.addWidget(text, 1)
        btn = QPushButton("关闭")
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        dialog.exec_()

    # ── 多场景对比 ──

    def _refresh_multi_compare(self):
        summaries = self.scenario_manager.compare_all_leaves()
        self.multi_compare_table.setRowCount(len(summaries))
        for row, s in enumerate(summaries):
            self.multi_compare_table.setItem(row, 0, QTableWidgetItem(s["name"]))
            self.multi_compare_table.setItem(row, 1, QTableWidgetItem(str(s["depth"])))
            self.multi_compare_table.setItem(row, 2, QTableWidgetItem(str(s["operation_count"])))
            self.multi_compare_table.setItem(row, 3, QTableWidgetItem(str(s["items_modified"])))
            self.multi_compare_table.setItem(row, 4, QTableWidgetItem(str(s["avg_score_modified"])))
            delta_str = f"{'+' if s['score_delta'] >= 0 else ''}{s['score_delta']:.2f}"
            item = QTableWidgetItem(delta_str)
            if s["score_delta"] > 0:
                item.setForeground(QColor(52, 168, 83))
            elif s["score_delta"] < 0:
                item.setForeground(QColor(234, 67, 53))
            self.multi_compare_table.setItem(row, 5, item)
            self.multi_compare_table.setItem(row, 6, QTableWidgetItem(", ".join(s["tags"])))
        self.multi_compare_table.resizeColumnsToContents()

    # ── 蒙特卡洛 ──

    def _run_monte_carlo(self):
        self.mc_progress.setVisible(True)
        self.mc_progress.setValue(0)
        n_iter = self.mc_iterations.value()
        n_ops = self.mc_ops.value()

        def progress(curr, total):
            self.mc_progress.setMaximum(total)
            self.mc_progress.setValue(curr)
            QTimer.singleShot(0, lambda: None)

        stats = self.monte_carlo.run_simulation(n_iter, n_ops, progress)

        lines = []
        lines.append("=" * 60)
        lines.append(f"🎲 蒙特卡洛模拟结果 ({n_iter} 次迭代)")
        lines.append("=" * 60)
        lines.append(f"   每次操作数: 1~{n_ops}")
        lines.append(f"   平均评分变化: {stats['avg_delta']:.4f}")
        lines.append(f"   标准差: {stats['std_delta']:.4f}")
        lines.append(f"   最小值: {stats['min_delta']:.4f}")
        lines.append(f"   最大值: {stats['max_delta']:.4f}")
        lines.append(f"   中位数 (P50): {stats['p50']:.4f}")
        lines.append(f"   四分位 (P25~P75): {stats['p25']:.4f} ~ {stats['p75']:.4f}")
        lines.append(f"   偏度: {stats['skew']:.4f}")
        lines.append(f"   分布形态: {stats['distribution']}")
        lines.append(f"   正向结果比例: {stats['positive_ratio']:.1%}")
        lines.append(f"   95% 置信区间: {self.monte_carlo.get_confidence_interval(0.95)}")
        lines.append("")
        lines.append("操作类型影响力排名:")
        impacts = self.monte_carlo.find_most_influential_operations()
        for op_type, impact in impacts:
            lines.append(f"  {op_type}: {impact:.4f}")
        lines.append("")
        lines.append("操作类型分布:")
        for op_type, count in sorted(stats["op_type_distribution"].items(),
                                      key=lambda x: x[1], reverse=True):
            bar = "█" * int(count / max(stats["n_iterations"], 1) * 40)
            lines.append(f"  {op_type:<20} {bar} {count}次")
        lines.append("")
        lines.append(f"最佳迭代 #{stats['best_iteration']}: "
                     f"评分变化 {stats['best_result']['score_delta']}")
        lines.append(f"最差迭代 #{stats['worst_iteration']}: "
                     f"评分变化 {stats['worst_result']['score_delta']}")

        self.mc_text.setText("\n".join(lines))
        self.mc_progress.setVisible(False)

    # ── 敏感性分析 ──

    def _run_sensitivity(self):
        params = {
            "boost_amount": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
            "suppress_amount": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
            "temporal_factor": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            "hypothetical_score": [1.0, 3.0, 5.0, 7.0, 10.0],
        }
        analysis = self.sensitivity.analyze_parameter_sensitivity(params)
        ranking = self.sensitivity.get_sensitivity_ranking(params)

        lines = []
        lines.append("=" * 60)
        lines.append("📐 敏感性分析")
        lines.append("=" * 60)
        lines.append("")
        lines.append("参数敏感性排名:")
        for i, (param, sens, trend) in enumerate(ranking, 1):
            bar = "█" * int(sens * 30)
            lines.append(f"  {i}. {param:<20} {bar} {sens:.2%} ({trend})")
        lines.append("")
        for param_name, data in analysis.items():
            lines.append(f"── {param_name} ──")
            lines.append(f"  敏感性: {data['sensitivity']:.2%}")
            lines.append(f"  最优值: {data['optimal_value']}")
            lines.append(f"  趋势: {data['trend']}")
            lines.append(f"  值→评分:")
            for r in data["results"]:
                lines.append(f"    {r['value']:<8} → {r['avg_score']:.2f}")
            lines.append("")

        self.sensitivity_text.setText("\n".join(lines))

    # ── 因果链 ──

    def _run_causal(self):
        if not self.engine.operations:
            self.causal_text.setText("尚无操作，请先在操作面板执行一些操作。")
            return
        analysis = self.causal_analyzer.analyze()
        report = self.causal_analyzer.generate_causal_summary(analysis)
        self.causal_text.setText(report)

    # ── 概率分析 ──

    def _run_probability(self):
        if len(self.scenario_manager.get_leaf_scenarios()) < 2:
            self.prob_text.setText("需要至少 2 个叶子场景才能进行概率分析。\n"
                                    "请先在操作面板创建更多场景分支。")
            return
        report = self.probability_estimator.generate_probability_report()
        self.prob_text.setText(report)

    # ── 通用更新 ──

    def _update_report(self):
        report = self.engine.generate_counterfactual_synthesis()
        self.report_text.setText(report)
        self._refresh_compare_table()
        self._refresh_stats()

    def _refresh_compare_table(self):
        comparison = self.engine.compare_rankings()
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
            rows.append((e["id"], e["content"], e["old_score"], "-", "-", "-", "🗑️"))
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
        tstats = self.engine.get_temporal_stats()
        if tstats.get("available"):
            lines.append("")
            lines.append("─" * 50)
            lines.append("⏳ 时间耦合记忆统计")
            lines.append("─" * 50)
            lines.append(f"记录轮次: {tstats['rounds_recorded']}")
            lines.append(f"活跃连接: {tstats['active_connections']} "
                         f"/ {tstats['shape'][0]}×{tstats['shape'][1]}")
            lines.append(f"连接密度: {tstats['connection_density']:.2%}")
            if tstats.get("emergence_level_distribution"):
                for k, v in sorted(tstats["emergence_level_distribution"].items()):
                    lines.append(f"  L{k} 出现次数: {v}")
        self.stats_text.setText("\n".join(lines))
        # 同时更新多场景对比
        self._refresh_multi_compare()

    def _update_info(self):
        n = len(self.scenario_manager.get_all_scenarios())
        # 更新标题栏信息
        self.window().findChild(QLabel, "").setText(
            f"📊 精华池: {len(self.engine.original_items)} 条  |  "
            f"👥 {len(self.engine.player_names)} 位专家  |  "
            f"场景: {n} 个")

    def _export_report(self):
        report = self.engine.generate_counterfactual_synthesis()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"counterfactual_{timestamp}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出推演报告", default_name,
            "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*.*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            QMessageBox.information(self, "导出成功", f"报告已导出到:\n{path}")

    def _export_scenarios(self):
        data = self.scenario_manager.export_scenarios()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"scenarios_{timestamp}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出场景", default_name,
            "JSON文件 (*.json);;所有文件 (*.*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "导出成功", f"场景已导出到:\n{path}")


def open_counterfactual_dialog(checkpoint_path: str, parent=None) -> bool:
    """打开反事实推演对话框（便利函数）"""
    engine = load_counterfactual_from_checkpoint(checkpoint_path)
    if not engine:
        return False
    dialog = CounterfactualDialog(engine, parent)
    dialog.exec_()
    return True