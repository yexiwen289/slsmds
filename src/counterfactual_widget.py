"""
反事实推演沙盘 —— "如果……会怎样？" 假设分析
（纯引擎版，无GUI依赖）

核心功能：
- 加载检查点数据，允许用户修改关键参数
- 模拟"如果某条精华被移除/增强/弱化"会怎样
- 多场景分支管理：创建、比较、合并多个反事实场景
- 蒙特卡洛模拟：批量随机推演，统计模式发现
- 敏感性分析：量化各参数对结果的影响程度
- 因果链追踪：从操作到结果的完整因果路径
- 概率估计：基于多场景统计估算各结果的概率
- 差异引擎：两个场景之间的细粒度 diff
- 场景聚类分析：将相似场景自动分组
- 场景优化器：自动搜索最优参数组合
- 因果网络构建：构建操作与结果之间的因果图
- 稳健性分析：测量结果对不同参数扰动的敏感度
- 趋势外推：预测跨场景的评分变化趋势
- 异常检测：自动标记统计异常场景
"""

import json
import os
import copy
import datetime
import random
import math
import hashlib
from typing import List, Dict, Optional, Tuple, Set, Callable
from collections import defaultdict, Counter as pyCounter

import numpy as np


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
        """从当前场景分支出一个新场景，并应用一组操作"""
        current_items = copy.deepcopy(self.engine.modified_items)
        current_temporal = copy.deepcopy(self.engine.modified_temporal)
        new_engine = copy.deepcopy(self.engine)
        new_engine.modified_items = current_items
        new_engine.modified_temporal = current_temporal
        new_engine.operations = copy.deepcopy(self.engine.operations)
        for op in operations:
            op_type = op.get("type")
            if op_type == "boost":
                new_engine.boost_essence(op["item_id"], op.get("amount", 3.0))
            elif op_type == "suppress":
                new_engine.suppress_essence(op["item_id"], op.get("amount", 3.0))
            elif op_type == "remove":
                new_engine.remove_essence(op["item_id"])
            elif op_type == "add":
                new_engine.add_hypothetical(
                    op.get("content", ""), score=op.get("score", 5.0))
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
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    def get_path(self) -> str:
        parts = [self.name]
        current = self.parent
        while current:
            parts.append(current.name)
            current = current.parent
        return " → ".join(reversed(parts))

    def get_all_leaves(self) -> List["CounterfactualScenario"]:
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

    def get_statistics_summary(self) -> Dict:
        stats = self.engine.get_statistics()
        o = stats["original"]
        m = stats["modified"]
        return {
            "id": self._id, "name": self.name,
            "depth": self.get_depth(),
            "items_original": o["count"], "items_modified": m["count"],
            "avg_score_original": o["avg_score"],
            "avg_score_modified": m["avg_score"],
            "score_delta": round(m["avg_score"] - o["avg_score"], 2),
            "operation_count": len(self.engine.operations),
            "tags": self.tags,
        }

    def to_dict(self) -> Dict:
        return {
            "id": self._id, "name": self.name, "description": self.description,
            "tags": self.tags, "created_at": self.created_at,
            "operations": self.engine.operations,
            "modified_items": self.engine.modified_items,
            "modified_temporal": self.engine.modified_temporal,
            "mode": self.engine.mode,
        }

    @classmethod
    def from_dict(cls, data: Dict, engine: "CounterfactualEngine") -> "CounterfactualScenario":
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
    """多场景管理器，维护一个场景树"""

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
        scenario = self._all_scenarios.get(scenario_id)
        if not scenario or scenario == self.root:
            return False
        for child in scenario.get_all_leaves():
            if child._id in self._all_scenarios and child != scenario:
                del self._all_scenarios[child._id]
        if scenario.parent:
            scenario.parent.children = [
                c for c in scenario.parent.children if c._id != scenario_id
            ]
        del self._all_scenarios[scenario_id]
        if self._active_scenario_id == scenario_id:
            self._active_scenario_id = self.root._id
        return True

    def get_all_scenarios(self) -> List[CounterfactualScenario]:
        return list(self._all_scenarios.values())

    def get_leaf_scenarios(self) -> List[CounterfactualScenario]:
        return self.root.get_all_leaves()

    def get_scenario_by_id(self, scenario_id: str) -> Optional[CounterfactualScenario]:
        return self._all_scenarios.get(scenario_id)

    def compare_all_leaves(self) -> List[Dict]:
        leaves = self.get_leaf_scenarios()
        summaries = [leaf.get_statistics_summary() for leaf in leaves]
        summaries.sort(key=lambda s: s["score_delta"], reverse=True)
        return summaries

    def find_most_divergent(self) -> Tuple[CounterfactualScenario, CounterfactualScenario, float]:
        leaves = self.get_leaf_scenarios()
        if len(leaves) <= 1:
            return (self.root, self.root, 0.0)
        baseline_avg = self.root.engine.get_statistics()["modified"]["avg_score"]
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
        leaves = self.get_leaf_scenarios()
        if not leaves:
            return self.root
        return max(leaves, key=lambda s: s.get_statistics_summary()["score_delta"])

    def find_worst_scenario(self) -> CounterfactualScenario:
        leaves = self.get_leaf_scenarios()
        if not leaves:
            return self.root
        return min(leaves, key=lambda s: s.get_statistics_summary()["score_delta"])

    def get_scenario_counts(self) -> Dict:
        all_scenarios = self.get_all_scenarios()
        leaves = self.get_leaf_scenarios()
        return {
            "total": len(all_scenarios), "leaves": len(leaves),
            "max_depth": max(s.get_depth() for s in all_scenarios) if all_scenarios else 0,
            "branches": sum(1 for s in all_scenarios if s.children),
            "best_scenario": self.find_best_scenario().name,
            "worst_scenario": self.find_worst_scenario().name,
        }

    def export_scenarios(self) -> Dict:
        return {
            "exported_at": datetime.datetime.now().isoformat(),
            "problem": self.base_engine.problem,
            "scenarios": [self._scenario_to_tree_dict(self.root)],
        }

    def _scenario_to_tree_dict(self, scenario: CounterfactualScenario) -> Dict:
        data = scenario.to_dict()
        data["children"] = [
            self._scenario_to_tree_dict(c) for c in scenario.children
        ]
        return data

    def import_scenarios(self, data: Dict, parent_id: str = None) -> bool:
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
    """蒙特卡洛反事实模拟器"""

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine
        self.results: List[Dict] = []
        self._rng = random.Random()

    def run_simulation(self, n_iterations: int = 100,
                       max_operations_per_run: int = 5,
                       progress_callback: Callable = None) -> Dict:
        self.results = []
        for i in range(n_iterations):
            engine = copy.deepcopy(self.base_engine)
            engine.operations = []
            n_ops = self._rng.randint(1, max_operations_per_run)
            for _ in range(n_ops):
                op_type = self._rng.choice(
                    ["boost", "suppress", "remove", "add",
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
                    engine.add_hypothetical(
                        f"{template}（蒙特卡洛假设 #{i}）",
                        score=self._rng.uniform(1.0, 10.0))
                elif op_type == "temporal_boost":
                    if engine.temporal_memory_data:
                        engine.boost_temporal_coupling(self._rng.uniform(1.1, 3.0))
                elif op_type == "temporal_weaken":
                    if engine.temporal_memory_data:
                        engine.weaken_temporal_coupling(self._rng.uniform(0.1, 0.9))
            stats = engine.get_statistics()
            comparison = engine.compare_rankings()
            o, m = stats["original"], stats["modified"]
            self.results.append({
                "iteration": i, "n_operations": n_ops,
                "operation_types": [op["type"] for op in engine.operations],
                "items_before": o["count"], "items_after": m["count"],
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
            })
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        return self._compute_statistics()

    def _compute_statistics(self) -> Dict:
        if not self.results:
            return {"error": "no_results"}
        deltas = [r["score_delta"] for r in self.results]
        n = len(deltas)
        avg_delta = sum(deltas) / n
        var_delta = sum((d - avg_delta) ** 2 for d in deltas) / n
        std_delta = math.sqrt(var_delta) if var_delta > 0 else 0
        sorted_deltas = sorted(deltas)
        p25 = sorted_deltas[int(n * 0.25)]
        p50 = sorted_deltas[int(n * 0.50)]
        p75 = sorted_deltas[int(n * 0.75)]
        skew = (p75 + p25 - 2 * p50) / max(std_delta, 0.01)
        if abs(skew) < 0.3:
            distribution = "近似正态"
        elif skew > 0:
            distribution = "右偏态（正向结果更多）"
        else:
            distribution = "左偏态（负向结果更多）"
        best_idx = deltas.index(max(deltas))
        worst_idx = deltas.index(min(deltas))
        positive_ratio = sum(1 for d in deltas if d > 0) / n
        op_type_counter = pyCounter()
        for r in self.results:
            for ot in r["operation_types"]:
                op_type_counter[ot] += 1
        return {
            "n_iterations": n, "avg_delta": round(avg_delta, 4),
            "std_delta": round(std_delta, 4),
            "min_delta": round(min(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "p25": round(p25, 4), "p50": round(p50, 4), "p75": round(p75, 4),
            "skew": round(skew, 4), "distribution": distribution,
            "positive_ratio": round(positive_ratio, 4),
            "best_iteration": best_idx, "best_result": self.results[best_idx],
            "worst_iteration": worst_idx, "worst_result": self.results[worst_idx],
            "op_type_distribution": dict(op_type_counter),
            "all_results": self.results,
        }

    def find_most_influential_operations(self) -> List[Tuple[str, float]]:
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
        if not self.results:
            return (0.0, 0.0)
        deltas = sorted([r["score_delta"] for r in self.results])
        n = len(deltas)
        lower_idx = int(n * (1 - confidence) / 2)
        upper_idx = int(n * (1 + confidence) / 2)
        return (deltas[lower_idx], deltas[upper_idx - 1])

    def get_monte_carlo_report(self) -> str:
        """生成蒙特卡洛分析报告文本"""
        stats = self._compute_statistics()
        if "error" in stats:
            return f"模拟失败: {stats['error']}"
        ci = self.get_confidence_interval(0.95)
        lines = []
        lines.append("=" * 60)
        lines.append(f"🎲 蒙特卡洛模拟报告 ({stats['n_iterations']} 次迭代)")
        lines.append("=" * 60)
        lines.append(f"  平均评分变化: {stats['avg_delta']:.4f}")
        lines.append(f"  标准差: {stats['std_delta']:.4f}")
        lines.append(f"  范围: [{stats['min_delta']:.4f}, {stats['max_delta']:.4f}]")
        lines.append(f"  中位数: {stats['p50']:.4f}")
        lines.append(f"  四分位: P25={stats['p25']:.4f}, P75={stats['p75']:.4f}")
        lines.append(f"  95%置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  偏度: {stats['skew']:.4f} ({stats['distribution']})")
        lines.append(f"  正向结果概率: {stats['positive_ratio']:.1%}")
        lines.append("")
        lines.append("操作影响力排名:")
        for op_type, impact in self.find_most_influential_operations():
            bar = "█" * int(abs(impact) * 10)
            lines.append(f"  {op_type:<20} {bar} {impact:+.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 敏感性分析器
# ═══════════════════════════════════════════════════════════════

class SensitivityAnalyzer:
    """敏感性分析器"""

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine

    def analyze_parameter_sensitivity(self, parameters: Dict[str, List]) -> Dict:
        results = {}
        for param_name, values in parameters.items():
            param_results = []
            for val in values:
                engine = copy.deepcopy(self.base_engine)
                engine.operations = []
                self._apply_parameter(engine, param_name, val)
                stats = engine.get_statistics()
                param_results.append({
                    "value": val,
                    "avg_score": stats["modified"]["avg_score"],
                    "max_score": stats["modified"]["max_score"],
                    "std_dev": stats["modified"]["std_dev"],
                    "count": stats["modified"]["count"],
                })
            if len(param_results) >= 2:
                scores = [r["avg_score"] for r in param_results]
                score_range = max(scores) - min(scores)
                sensitivity = min(1.0, score_range * 2)
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
                best_idx = scores.index(max(scores))
                optimal_value = values[best_idx]
            else:
                sensitivity = 0.0
                trend = "未知"
                optimal_value = values[0] if values else None
            results[param_name] = {
                "values": values, "results": param_results,
                "sensitivity": round(sensitivity, 4),
                "optimal_value": optimal_value, "trend": trend,
            }
        return results

    def _apply_parameter(self, engine, param_name, value):
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
            engine.add_hypothetical("敏感性测试假设观点", score=value)

    def get_sensitivity_ranking(self, parameters: Dict[str, List]) -> List[Tuple[str, float, str]]:
        analysis = self.analyze_parameter_sensitivity(parameters)
        ranking = []
        for param_name, data in analysis.items():
            ranking.append((param_name, data["sensitivity"], data["trend"]))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_sensitivity_report(self, parameters: Dict[str, List] = None) -> str:
        """生成敏感性分析报告文本"""
        if parameters is None:
            parameters = {
                "boost_amount": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
                "suppress_amount": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
                "temporal_factor": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
                "hypothetical_score": [1.0, 3.0, 5.0, 7.0, 10.0],
            }
        analysis = self.analyze_parameter_sensitivity(parameters)
        ranking = self.get_sensitivity_ranking(parameters)
        lines = []
        lines.append("=" * 60)
        lines.append("📐 敏感性分析报告")
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
            for r in data["results"]:
                lines.append(f"    {r['value']:<8} → 评分 {r['avg_score']:.2f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 因果链分析器
# ═══════════════════════════════════════════════════════════════

class CausalChainAnalyzer:
    """因果链分析器"""

    def __init__(self, engine: "CounterfactualEngine"):
        self.engine = engine

    def analyze(self) -> Dict:
        chains = []
        for op in self.engine.operations:
            chain = self._trace_chain(op)
            chains.append(chain)
        return {
            "chains": chains, "total_chains": len(chains),
            "causal_density": self._compute_causal_density(chains),
            "dominant_chain": self._find_dominant_chain(chains),
        }

    def _trace_chain(self, operation: Dict) -> Dict:
        op_type = operation.get("type", "unknown")
        chain = {"operation": operation, "direct_effects": [],
                 "cascade_effects": [], "final_impact": None}
        if op_type in ("boost", "suppress"):
            item_id = operation.get("item_id")
            item = self.engine.get_essence_by_id(item_id)
            amount = operation.get("amount", 0)
            if item:
                chain["direct_effects"].append({
                    "target": f"精华 #{item_id}",
                    "action": "增强" if op_type == "boost" else "削弱",
                    "magnitude": amount,
                    "original_score": item.get("score", 0),
                    "new_score": item.get("score", 0) + (
                        amount if op_type == "boost" else -amount),
                    "confidence": "高",
                })
            for entry in self.engine.compare_rankings().get("gained", []):
                if entry["id"] != item_id:
                    chain["cascade_effects"].append({
                        "target": f"精华 #{entry['id']}",
                        "effect": f"上升 #{entry['old_rank']}→#{entry['new_rank']}",
                        "confidence": "中",
                    })
            for entry in self.engine.compare_rankings().get("lost", []):
                if entry["id"] != item_id:
                    chain["cascade_effects"].append({
                        "target": f"精华 #{entry['id']}",
                        "effect": f"下降 #{entry['old_rank']}→#{entry['new_rank']}",
                        "confidence": "中",
                    })
        elif op_type == "remove":
            chain["direct_effects"].append({
                "target": f"精华 #{operation.get('item_id')}",
                "action": "移除", "magnitude": "完全删除", "confidence": "高",
            })
            chain["cascade_effects"].append({
                "target": "所有精华", "effect": "排名重新计算", "confidence": "中",
            })
        elif op_type == "add":
            chain["direct_effects"].append({
                "target": "新精华", "action": "添加",
                "magnitude": operation.get("description", "假设观点"),
                "confidence": "高",
            })
            chain["cascade_effects"].append({
                "target": "排名末尾", "effect": "后进影响整体分布", "confidence": "低",
            })
        elif "temporal" in op_type:
            chain["direct_effects"].append({
                "target": "时间耦合记忆", "action": "调整",
                "magnitude": f"×{operation.get('factor', 1.0)}",
                "confidence": "中",
            })
            chain["cascade_effects"].append({
                "target": "跨轮次认知协同",
                "effect": "间接影响涌现层级", "confidence": "低",
            })
        stats = self.engine.get_statistics()
        o, m = stats["original"], stats["modified"]
        chain["final_impact"] = {
            "avg_score_change": round(m["avg_score"] - o["avg_score"], 2),
            "count_change": m["count"] - o["count"],
            "std_dev_change": round(m["std_dev"] - o["std_dev"], 2),
        }
        return chain

    def _compute_causal_density(self, chains: List[Dict]) -> float:
        if not chains:
            return 0.0
        total_effects = sum(len(c["direct_effects"]) + len(c["cascade_effects"])
                            for c in chains)
        total_possible = len(chains) * 3
        return min(1.0, total_effects / max(total_possible, 1))

    def _find_dominant_chain(self, chains: List[Dict]) -> Optional[Dict]:
        if not chains:
            return None
        return max(chains, key=lambda c: abs(
            c["final_impact"]["avg_score_change"]) if c["final_impact"] else 0)

    def get_causal_report(self) -> str:
        """生成因果链分析报告文本"""
        analysis = self.analyze()
        lines = []
        lines.append("=" * 60)
        lines.append("🔗 因果链分析报告")
        lines.append("=" * 60)
        for i, chain in enumerate(analysis["chains"], 1):
            op = chain["operation"]
            lines.append(f"\n── 因果链 #{i}: {op.get('description', '')} ──")
            for effect in chain["direct_effects"]:
                lines.append(f"  · {effect['action']} {effect['target']} "
                             f"(幅度: {effect['magnitude']}, 置信度: {effect['confidence']})")
            for effect in chain["cascade_effects"][:5]:
                lines.append(f"  ↳ {effect['target']}: {effect['effect']} "
                             f"(置信度: {effect['confidence']})")
            if chain["final_impact"]:
                imp = chain["final_impact"]
                lines.append(f"  最终影响: 评分{imp['avg_score_change']:+.2f} | "
                             f"数量{imp['count_change']:+d} | "
                             f"标准差{imp['std_dev_change']:+.2f}")
        lines.append(f"\n因果密度: {analysis['causal_density']:.1%}")
        if analysis['dominant_chain']:
            lines.append(
                f"主导因果链: {analysis['dominant_chain']['operation'].get('description', '')}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 概率估计器
# ═══════════════════════════════════════════════════════════════

class ProbabilityEstimator:
    """概率估计器"""

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def estimate_outcome_probabilities(self, n_bins: int = 5) -> Dict:
        leaves = self.manager.get_leaf_scenarios()
        if len(leaves) < 2:
            return {"error": "insufficient_scenarios"}
        deltas = [leaf.get_statistics_summary()["score_delta"] for leaf in leaves]
        min_delta, max_delta = min(deltas), max(deltas)
        bin_width = max((max_delta - min_delta) / n_bins, 0.01)
        bins = []
        for i in range(n_bins):
            lo = min_delta + i * bin_width
            hi = lo + bin_width
            count = sum(1 for d in deltas if lo <= d < hi)
            prob = count / len(deltas)
            bins.append({
                "range": f"[{lo:.2f}, {hi:.2f})", "count": count,
                "probability": round(prob, 4),
                "label": ("大幅提升" if lo > 0.5 else "小幅提升" if lo > 0
                          else "小幅下降" if hi > 0 else "大幅下降"),
            })
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
        leaves = self.manager.get_leaf_scenarios()
        if not leaves:
            return 0.0
        successes = sum(1 for leaf in leaves
                        if leaf.get_statistics_summary()["score_delta"] > threshold)
        return successes / len(leaves)

    def estimate_risk(self, threshold: float = -0.5) -> float:
        leaves = self.manager.get_leaf_scenarios()
        if not leaves:
            return 1.0
        failures = sum(1 for leaf in leaves
                       if leaf.get_statistics_summary()["score_delta"] < threshold)
        return failures / len(leaves)

    def get_probability_report(self) -> str:
        """生成概率分析报告文本"""
        probs = self.estimate_outcome_probabilities()
        if "error" in probs:
            return f"概率估计失败: {probs['error']}"
        lines = []
        lines.append("=" * 60)
        lines.append("🎲 概率分析报告")
        lines.append("=" * 60)
        lines.append(f"基于 {probs['n_scenarios']} 个场景分析")
        for bin_info in probs["bins"]:
            bar = "█" * int(bin_info["probability"] * 40)
            lines.append(f"  {bin_info['label']:<8} {bin_info['range']:>12}  "
                         f"{bar} {bin_info['probability']:.1%} ({bin_info['count']}次)")
        lines.append(f"\n正向概率: {probs['prob_positive']:.1%}")
        lines.append(f"负向概率: {probs['prob_negative']:.1%}")
        lines.append(f"期望值: {probs['expected_value']:.2f}")
        success_p = self.estimate_success_probability(0.5)
        risk_p = self.estimate_risk(-0.5)
        lines.append(f"成功概率: {success_p:.1%}")
        lines.append(f"风险概率: {risk_p:.1%}")
        if success_p > risk_p:
            lines.append("\n建议: 正向预期占优，可尝试实际执行。")
        elif risk_p > success_p:
            lines.append("\n建议: 风险高于收益，建议谨慎。")
        else:
            lines.append("\n建议: 结果不确定，建议增加更多分支。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 差异引擎
# ═══════════════════════════════════════════════════════════════

class DiffEngine:
    """差异引擎 —— 细粒度比较两个场景的差异"""

    @staticmethod
    def diff_scenarios(scenario_a: CounterfactualScenario,
                       scenario_b: CounterfactualScenario) -> Dict:
        a_items = scenario_a.engine.modified_items
        b_items = scenario_b.engine.modified_items
        a_map = {e["id"]: e for e in a_items}
        b_map = {e["id"]: e for e in b_items}
        all_ids = set(list(a_map.keys()) + list(b_map.keys()))

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
                    "score_a": a_score, "score_b": b_score,
                    "delta": round(b_score - a_score, 2),
                    "type": "score_change",
                })
        added = [e for eid, e in b_map.items() if eid not in a_map]
        removed = [e for eid, e in a_map.items() if eid not in b_map]
        essence_diffs = []
        for e in added:
            essence_diffs.append({
                "id": e["id"], "content": e.get("content", "")[:50],
                "score": e.get("score", 0), "type": "added",
            })
        for e in removed:
            essence_diffs.append({
                "id": e["id"], "content": e.get("content", "")[:50],
                "score": e.get("score", 0), "type": "removed",
            })

        a_ranking = scenario_a.engine.get_modified_ranking()
        b_ranking = scenario_b.engine.get_modified_ranking()
        a_rank_map = {e["id"]: i for i, e in enumerate(a_ranking)}
        b_rank_map = {e["id"]: i for i, e in enumerate(b_ranking)}
        ranking_diffs = []
        for eid in all_ids:
            if eid in a_rank_map and eid in b_rank_map:
                old_rank, new_rank = a_rank_map[eid], b_rank_map[eid]
                if old_rank != new_rank:
                    item = a_map.get(eid) or b_map.get(eid)
                    ranking_diffs.append({
                        "id": eid, "content": item.get("content", "")[:50] if item else "",
                        "old_rank": old_rank + 1, "new_rank": new_rank + 1,
                        "delta": old_rank - new_rank,
                    })

        a_stats = scenario_a.engine.get_statistics()
        b_stats = scenario_b.engine.get_statistics()
        stat_diffs = {}
        for key in ["count", "avg_score", "max_score", "min_score", "std_dev"]:
            a_val = a_stats["modified"].get(key, 0)
            b_val = b_stats["modified"].get(key, 0)
            diff_val = b_val - a_val if isinstance(a_val, (int, float)) else "N/A"
            stat_diffs[key] = {"a": a_val, "b": b_val, "diff": diff_val}

        total_items = len(all_ids) + len(added) + len(removed)
        unchanged = total_items - len(essence_diffs)
        similarity = unchanged / max(total_items, 1)
        summary = (
            f"场景 A '{scenario_a.name}' vs B '{scenario_b.name}': "
            f"{len(added)} 新增, {len(removed)} 移除, "
            f"{len(score_diffs)} 评分变化, {len(ranking_diffs)} 排名变化, "
            f"相似度 {similarity:.1%}"
        )
        return {
            "essence_diffs": essence_diffs, "score_diffs": score_diffs,
            "ranking_diffs": ranking_diffs, "stat_diffs": stat_diffs,
            "summary": summary, "similarity_score": round(similarity, 4),
        }

    @staticmethod
    def generate_diff_report(diff: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("📋 场景差异报告")
        lines.append("=" * 60)
        lines.append(diff["summary"])
        if diff["score_diffs"]:
            lines.append("\n评分变化:")
            for d in sorted(diff["score_diffs"], key=lambda x: abs(x["delta"]), reverse=True)[:10]:
                lines.append(f"  #{d['id']} {d['content']}")
                lines.append(f"    {d['score_a']:.1f} → {d['score_b']:.1f} "
                             f"({'↑' if d['delta'] > 0 else '↓'}{abs(d['delta']):.1f})")
        if diff["ranking_diffs"]:
            lines.append("\n排名变化:")
            for d in sorted(diff["ranking_diffs"], key=lambda x: abs(x["delta"]), reverse=True)[:10]:
                lines.append(f"  #{d['id']} {d['content']}")
                lines.append(f"    #{d['old_rank']} → #{d['new_rank']} "
                             f"({'↑' if d['delta'] > 0 else '↓'}{abs(d['delta'])})")
        if diff["essence_diffs"]:
            lines.append("\n新增/移除:")
            for d in diff["essence_diffs"]:
                icon = "➕" if d["type"] == "added" else "🗑️"
                lines.append(f"  {icon} #{d['id']} {d['content']} (评分: {d['score']:.1f})")
        lines.append(f"\n相似度: {diff['similarity_score']:.1%}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 场景聚类分析器
# ═══════════════════════════════════════════════════════════════

class ScenarioClusterAnalyzer:
    """
    场景聚类分析器。

    基于评分变化、操作类型分布、精华池统计等特征，
    将相似场景自动分组，发现模式。
    """

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def cluster_by_outcome(self, n_clusters: int = 3) -> List[List[CounterfactualScenario]]:
        """按评分变化聚类场景"""
        leaves = self.manager.get_leaf_scenarios()
        if len(leaves) < n_clusters:
            return [leaves]

        deltas = [leaf.get_statistics_summary()["score_delta"] for leaf in leaves]
        sorted_idx = sorted(range(len(deltas)), key=lambda i: deltas[i])

        clusters = []
        cluster_size = len(sorted_idx) // n_clusters
        for i in range(n_clusters):
            if i == n_clusters - 1:
                cluster_indices = sorted_idx[i * cluster_size:]
            else:
                cluster_indices = sorted_idx[i * cluster_size:(i + 1) * cluster_size]
            clusters.append([leaves[idx] for idx in cluster_indices])

        return clusters

    def get_cluster_labels(self, n_clusters: int = 3) -> List[str]:
        """为每个聚类生成标签"""
        clusters = self.cluster_by_outcome(n_clusters)
        labels = []
        for cluster in clusters:
            if not cluster:
                labels.append("空聚类")
                continue
            deltas = [s.get_statistics_summary()["score_delta"] for s in cluster]
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta > 0.5:
                labels.append(f"高收益 (Δ={avg_delta:.2f})")
            elif avg_delta > 0:
                labels.append(f"小幅提升 (Δ={avg_delta:.2f})")
            elif avg_delta > -0.5:
                labels.append(f"小幅下降 (Δ={avg_delta:.2f})")
            else:
                labels.append(f"高风险 (Δ={avg_delta:.2f})")
        return labels

    def get_cluster_report(self, n_clusters: int = 3) -> str:
        """生成聚类报告"""
        clusters = self.cluster_by_outcome(n_clusters)
        labels = self.get_cluster_labels(n_clusters)
        lines = []
        lines.append("=" * 60)
        lines.append(f"🗂️ 场景聚类分析 ({n_clusters} 类)")
        lines.append("=" * 60)
        for i, (cluster, label) in enumerate(zip(clusters, labels)):
            lines.append(f"\n── 聚类 {i + 1}: {label} ({len(cluster)} 个场景) ──")
            for s in cluster:
                stats = s.get_statistics_summary()
                lines.append(f"  · {s.name:<20} Δ={stats['score_delta']:+.2f}  "
                             f"操作={stats['operation_count']} 深度={stats['depth']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 场景优化器 —— 自动搜索最优参数组合
# ═══════════════════════════════════════════════════════════════

class ScenarioOptimizer:
    """
    场景优化器。

    通过自动搜索参数空间，寻找评分提升最大的操作组合。
    支持：
    - 随机搜索（Random Search）
    - 网格搜索（Grid Search）
    - 模拟退火（Simulated Annealing）
    """

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine

    def random_search(self, n_trials: int = 50,
                      max_ops: int = 5,
                      target: str = "avg_score") -> Dict:
        """
        随机搜索最优参数组合。

        Args:
            n_trials: 搜索次数
            max_ops: 每次最多操作数
            target: 优化目标 ("avg_score" | "max_score" | "diversity")

        Returns:
            最优结果
        """
        best_score = -float("inf")
        best_result = None
        best_engine = None

        for trial in range(n_trials):
            engine = copy.deepcopy(self.base_engine)
            engine.operations = []
            n_ops = random.randint(1, max_ops)

            for _ in range(n_ops):
                op_type = random.choice(["boost", "suppress", "add"])
                if op_type == "boost":
                    if not engine.modified_items:
                        continue
                    item = random.choice(engine.modified_items)
                    amount = random.uniform(1.0, 10.0)
                    engine.boost_essence(item["id"], amount)
                elif op_type == "suppress":
                    if not engine.modified_items:
                        continue
                    item = random.choice(engine.modified_items)
                    # 只削弱低分精华
                    if item.get("score", 0) < 5.0:
                        engine.suppress_essence(item["id"], random.uniform(1.0, 5.0))
                elif op_type == "add":
                    engine.add_hypothetical(
                        f"优化生成假设 #{trial}",
                        score=random.uniform(3.0, 9.0))

            stats = engine.get_statistics()
            modified = stats["modified"]
            if target == "avg_score":
                score = modified["avg_score"]
            elif target == "max_score":
                score = modified["max_score"]
            elif target == "diversity":
                score = modified["std_dev"]
            else:
                score = modified["avg_score"]

            if score > best_score:
                best_score = score
                best_result = stats
                best_engine = engine

        return {
            "best_score": round(best_score, 2),
            "target": target,
            "n_trials": n_trials,
            "best_statistics": best_result,
            "best_operations": best_engine.operations if best_engine else [],
            "best_ranking": best_engine.get_modified_ranking()[:5] if best_engine else [],
        }

    def grid_search(self, item_ids: List[int],
                    boost_values: List[float] = None,
                    suppress_values: List[float] = None) -> List[Dict]:
        """
        网格搜索 —— 对指定精华尝试所有评分组合。

        Args:
            item_ids: 要调整的精华ID列表
            boost_values: 要尝试的增强幅度
            suppress_values: 要尝试的削弱幅度

        Returns:
            按评分降序排列的结果列表
        """
        if boost_values is None:
            boost_values = [1.0, 3.0, 5.0]
        if suppress_values is None:
            suppress_values = [1.0, 3.0, 5.0]

        results = []
        for item_id in item_ids:
            for val in boost_values:
                engine = copy.deepcopy(self.base_engine)
                engine.operations = []
                engine.boost_essence(item_id, val)
                stats = engine.get_statistics()
                results.append({
                    "item_id": item_id, "action": "boost", "value": val,
                    "avg_score": stats["modified"]["avg_score"],
                    "max_score": stats["modified"]["max_score"],
                    "std_dev": stats["modified"]["std_dev"],
                    "operations": len(engine.operations),
                })
            for val in suppress_values:
                engine = copy.deepcopy(self.base_engine)
                engine.operations = []
                engine.suppress_essence(item_id, val)
                stats = engine.get_statistics()
                results.append({
                    "item_id": item_id, "action": "suppress", "value": val,
                    "avg_score": stats["modified"]["avg_score"],
                    "max_score": stats["modified"]["max_score"],
                    "std_dev": stats["modified"]["std_dev"],
                    "operations": len(engine.operations),
                })

        results.sort(key=lambda r: r["avg_score"], reverse=True)
        return results

    def get_optimization_report(self, n_trials: int = 50) -> str:
        """生成优化报告"""
        best = self.random_search(n_trials=n_trials)
        lines = []
        lines.append("=" * 60)
        lines.append(f"🎯 场景优化报告 ({n_trials} 次搜索)")
        lines.append("=" * 60)
        lines.append(f"  优化目标: {best['target']}")
        lines.append(f"  最优评分: {best['best_score']:.2f}")
        lines.append(f"  搜索次数: {best['n_trials']}")
        if best["best_statistics"]:
            bs = best["best_statistics"]
            o, m = bs["original"], bs["modified"]
            lines.append(f"  原始评分: {o['avg_score']} → {m['avg_score']} "
                         f"(Δ={m['avg_score'] - o['avg_score']:+.2f})")
        if best["best_operations"]:
            lines.append(f"\n最优操作组合:")
            for op in best["best_operations"]:
                lines.append(f"  · {op.get('description', '')}")
        if best["best_ranking"]:
            lines.append(f"\n优化后 Top 5:")
            for i, e in enumerate(best["best_ranking"], 1):
                lines.append(f"  {i}. [{e.get('score', 0):.1f}] {e.get('content', '')[:50]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 因果网络构建器
# ═══════════════════════════════════════════════════════════════

class CausalNetworkBuilder:
    """
    因果网络构建器。

    从多个场景的操作和结果中构建因果图。
    节点 = 操作类型/精华条目
    边 = 因果关系（操作 → 影响到的精华）
    """

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def build_network(self) -> Dict:
        """
        构建因果网络。

        返回: {
            "nodes": [{"id": ..., "type": ..., "label": ..., "weight": ...}],
            "edges": [{"source": ..., "target": ..., "weight": ..., "effect": ...}],
            "metrics": {...},
        }
        """
        nodes = {}
        edges = []
        node_weights = defaultdict(float)

        # 收集所有场景的操作
        for scenario in self.manager.get_all_scenarios():
            delta = scenario.get_statistics_summary()["score_delta"]

            for op in scenario.engine.operations:
                op_type = op.get("type", "unknown")
                op_desc = op.get("description", op_type)

                # 操作节点
                op_node_id = f"op_{op_type}_{hash(op_desc) % 10000}"
                if op_node_id not in nodes:
                    nodes[op_node_id] = {
                        "id": op_node_id, "type": "operation",
                        "label": op_desc, "weight": 0,
                    }
                nodes[op_node_id]["weight"] += 1
                node_weights[op_node_id] += delta

                # 影响的精华节点
                item_id = op.get("item_id")
                if item_id:
                    item_node_id = f"item_{item_id}"
                    item = scenario.engine.get_essence_by_id(item_id)
                    if item_node_id not in nodes:
                        nodes[item_node_id] = {
                            "id": item_node_id, "type": "essence",
                            "label": item.get("content", "")[:40] if item else f"Item #{item_id}",
                            "weight": 0,
                        }
                    nodes[item_node_id]["weight"] += 1
                    node_weights[item_node_id] += delta

                    edges.append({
                        "source": op_node_id, "target": item_node_id,
                        "weight": abs(delta), "effect": "positive" if delta > 0 else "negative",
                    })

        # 中心度分析
        edge_targets = defaultdict(int)
        for edge in edges:
            edge_targets[edge["target"]] += 1
            edge_targets[edge["source"]] += 1

        metrics = {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "density": len(edges) / max(len(nodes) * (len(nodes) - 1), 1) * 2,
            "hub_nodes": sorted(
                [{"id": nid, "connections": edge_targets.get(nid, 0),
                  "impact": round(node_weights.get(nid, 0), 2)}
                 for nid in nodes],
                key=lambda x: x["connections"], reverse=True
            )[:10],
        }

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "metrics": metrics,
        }

    def get_network_report(self) -> str:
        """生成因果网络报告"""
        network = self.build_network()
        metrics = network["metrics"]
        lines = []
        lines.append("=" * 60)
        lines.append("🕸️ 因果网络分析报告")
        lines.append("=" * 60)
        lines.append(f"  节点数: {metrics['n_nodes']}")
        lines.append(f"  边数: {metrics['n_edges']}")
        lines.append(f"  网络密度: {metrics['density']:.4f}")
        lines.append(f"\n枢纽节点 (影响最大的操作/精华):")
        for node in metrics["hub_nodes"][:10]:
            node_data = next((n for n in network["nodes"] if n["id"] == node["id"]), None)
            label = node_data["label"] if node_data else node["id"]
            lines.append(f"  · {label:<30} 连接={node['connections']}  "
                         f"累积影响={node['impact']:+.2f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 稳健性分析器
# ═══════════════════════════════════════════════════════════════

class RobustnessAnalyzer:
    """
    稳健性分析器。

    测量结果对不同参数扰动的敏感度。
    稳健性越高的结果，越值得信赖。
    """

    def __init__(self, engine: "CounterfactualEngine"):
        self.base_engine = engine

    def analyze_robustness(self, perturbations: int = 10,
                           noise_level: float = 0.2) -> Dict:
        """
        分析结果稳健性。

        Args:
            perturbations: 扰动次数
            noise_level: 扰动幅度 (0~1)

        Returns:
            {"baseline": ..., "perturbations": [...], "stability": 0.0~1.0, ...}
        """
        baseline = self.base_engine.get_statistics()
        baseline_avg = baseline["modified"]["avg_score"]

        perturbed_scores = []
        for _ in range(perturbations):
            engine = copy.deepcopy(self.base_engine)
            # 对每个精华评分添加随机扰动
            for item in engine.modified_items:
                noise = random.uniform(-noise_level, noise_level) * item.get("score", 5)
                item["score"] = max(0.1, item.get("score", 0) + noise)
            stats = engine.get_statistics()
            perturbed_scores.append(stats["modified"]["avg_score"])

        if perturbed_scores:
            avg_perturbed = sum(perturbed_scores) / len(perturbed_scores)
            variance = sum((s - avg_perturbed) ** 2 for s in perturbed_scores) / len(perturbed_scores)
            std_dev = math.sqrt(variance)
            # 稳健性 = 1 - (变异系数 / 最大变异系数)
            cv = std_dev / max(abs(avg_perturbed), 0.01)
            stability = max(0.0, min(1.0, 1.0 - cv * 5))
        else:
            avg_perturbed = baseline_avg
            std_dev = 0
            stability = 1.0

        return {
            "baseline_avg": baseline_avg,
            "perturbed_avg": round(avg_perturbed, 4),
            "std_dev": round(std_dev, 4),
            "stability": round(stability, 4),
            "n_perturbations": perturbations,
            "noise_level": noise_level,
            "perturbed_scores": perturbed_scores,
            "min_perturbed": min(perturbed_scores) if perturbed_scores else baseline_avg,
            "max_perturbed": max(perturbed_scores) if perturbed_scores else baseline_avg,
        }

    def get_robustness_report(self, perturbations: int = 10,
                               noise_level: float = 0.2) -> str:
        """生成稳健性分析报告"""
        result = self.analyze_robustness(perturbations, noise_level)
        lines = []
        lines.append("=" * 60)
        lines.append("🛡️ 稳健性分析报告")
        lines.append("=" * 60)
        lines.append(f"  扰动次数: {result['n_perturbations']}")
        lines.append(f"  扰动幅度: ±{result['noise_level']:.0%}")
        lines.append(f"  基线评分: {result['baseline_avg']:.2f}")
        lines.append(f"  扰动后均值: {result['perturbed_avg']:.2f}")
        lines.append(f"  扰动后标准差: {result['std_dev']:.4f}")
        lines.append(f"  范围: [{result['min_perturbed']:.2f}, {result['max_perturbed']:.2f}]")
        stability = result['stability']
        bar = "🟩" * int(stability * 20) + "🟥" * (20 - int(stability * 20))
        lines.append(f"  稳健性: {stability:.1%} {bar}")
        if stability > 0.8:
            lines.append("\n结论: 高度稳健，结果可信赖。")
        elif stability > 0.5:
            lines.append("\n结论: 中等稳健，结果可参考但需注意波动。")
        else:
            lines.append("\n结论: 低稳健性，结果高度依赖参数选择。")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 趋势外推器
# ═══════════════════════════════════════════════════════════════

class TrendExtrapolator:
    """
    趋势外推器。

    基于场景树中的路径，预测评分变化趋势。
    支持线性外推和指数外推。
    """

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def extrapolate_linear(self, scenario: CounterfactualScenario,
                           n_steps: int = 3) -> List[float]:
        """线性外推：沿场景路径预测未来评分变化"""
        path = []
        current = scenario
        while current:
            stats = current.get_statistics_summary()
            path.append(stats["score_delta"])
            current = current.parent
        path.reverse()

        if len(path) < 2:
            return [path[-1] if path else 0] * n_steps

        # 线性回归
        x = list(range(len(path)))
        y = path
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)
        intercept = (sum(y) - slope * sum(x)) / n

        predictions = []
        for step in range(1, n_steps + 1):
            pred = slope * (len(path) - 1 + step) + intercept
            predictions.append(pred)
        return predictions

    def extrapolate_exponential(self, scenario: CounterfactualScenario,
                                n_steps: int = 3) -> List[float]:
        """指数外推：假设变化率逐步衰减"""
        path = []
        current = scenario
        while current:
            stats = current.get_statistics_summary()
            path.append(stats["score_delta"])
            current = current.parent
        path.reverse()

        if len(path) < 2:
            return [path[-1] if path else 0] * n_steps

        # 计算变化率
        rates = []
        for i in range(1, len(path)):
            if path[i - 1] != 0:
                rates.append(path[i] / path[i - 1])
            else:
                rates.append(1.0)
        avg_rate = sum(rates) / len(rates) if rates else 1.0

        predictions = []
        last_val = path[-1]
        for step in range(1, n_steps + 1):
            last_val *= avg_rate * (0.85 ** step)  # 衰减因子
            predictions.append(last_val)
        return predictions

    def get_trend_report(self, n_steps: int = 3) -> str:
        """生成趋势外推报告"""
        leaves = self.manager.get_leaf_scenarios()
        if not leaves:
            return "无场景可用于趋势分析。"

        lines = []
        lines.append("=" * 60)
        lines.append(f"📈 趋势外推报告 ({n_steps} 步)")
        lines.append("=" * 60)

        for leaf in leaves[:10]:
            linear = self.extrapolate_linear(leaf, n_steps)
            exponential = self.extrapolate_exponential(leaf, n_steps)
            lines.append(f"\n── {leaf.name} ──")
            current = leaf.get_statistics_summary()["score_delta"]
            lines.append(f"  当前: {current:+.2f}")
            lines.append(f"  线性外推: {', '.join(f'{p:+.2f}' for p in linear)}")
            lines.append(f"  指数外推: {', '.join(f'{p:+.2f}' for p in exponential)}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 异常检测器
# ═══════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    异常检测器。

    自动标记统计异常的场景：
    - 评分突变（与父场景差异过大）
    - 评分反转（正向变负向或反之）
    - 操作效率异常（操作数/结果比异常）
    - 离群值（偏离均值3个标准差以上）
    """

    def __init__(self, manager: ScenarioManager):
        self.manager = manager

    def detect_anomalies(self) -> List[Dict]:
        """检测所有异常场景"""
        anomalies = []
        all_scenarios = self.manager.get_all_scenarios()

        # 1. 评分突变
        for scenario in all_scenarios:
            if scenario.parent:
                child_stats = scenario.get_statistics_summary()
                parent_stats = scenario.parent.get_statistics_summary()
                delta_diff = abs(child_stats["score_delta"] - parent_stats["score_delta"])
                if delta_diff > 2.0:
                    anomalies.append({
                        "scenario_name": scenario.name,
                        "type": "评分突变",
                        "severity": "高" if delta_diff > 5.0 else "中",
                        "detail": f"与父场景 '{scenario.parent.name}' 评分差异 Δ={delta_diff:.2f}",
                    })

        # 2. 评分反转
        for scenario in all_scenarios:
            if scenario.parent:
                child_delta = scenario.get_statistics_summary()["score_delta"]
                parent_delta = scenario.parent.get_statistics_summary()["score_delta"]
                if child_delta * parent_delta < 0 and abs(child_delta) > 0.5:
                    anomalies.append({
                        "scenario_name": scenario.name,
                        "type": "评分反转",
                        "severity": "高",
                        "detail": f"父场景 {parent_delta:+.2f} → 当前 {child_delta:+.2f}",
                    })

        # 3. 操作效率异常
        for scenario in all_scenarios:
            stats = scenario.get_statistics_summary()
            if stats["operation_count"] > 0 and stats["depth"] > 0:
                efficiency = abs(stats["score_delta"]) / stats["operation_count"]
                if efficiency > 2.0:
                    anomalies.append({
                        "scenario_name": scenario.name,
                        "type": "操作效率异常",
                        "severity": "低",
                        "detail": f"单操作效率 {efficiency:.2f}（操作数={stats['operation_count']}）",
                    })

        # 4. 离群值
        leaf_deltas = [s.get_statistics_summary()["score_delta"]
                       for s in self.manager.get_leaf_scenarios()]
        if len(leaf_deltas) > 3:
            mean = sum(leaf_deltas) / len(leaf_deltas)
            var = sum((d - mean) ** 2 for d in leaf_deltas) / len(leaf_deltas)
            std = math.sqrt(var)
            for scenario in self.manager.get_leaf_scenarios():
                delta = scenario.get_statistics_summary()["score_delta"]
                z_score = abs(delta - mean) / max(std, 0.01)
                if z_score > 2.0:
                    anomalies.append({
                        "scenario_name": scenario.name,
                        "type": "离群值",
                        "severity": "高" if z_score > 3.0 else "中",
                        "detail": f"Z-score={z_score:.2f} (Δ={delta:+.2f}, 均值={mean:.2f})",
                    })

        anomalies.sort(key=lambda a: {"高": 0, "中": 1, "低": 2}[a["severity"]])
        return anomalies

    def get_anomaly_report(self) -> str:
        """生成异常检测报告"""
        anomalies = self.detect_anomalies()
        if not anomalies:
            return "未检测到异常场景，所有场景表现正常。"
        lines = []
        lines.append("=" * 60)
        lines.append(f"⚠️ 异常检测报告 ({len(anomalies)} 个异常)")
        lines.append("=" * 60)
        for a in anomalies:
            severity_icon = "🔴" if a["severity"] == "高" else "🟡" if a["severity"] == "中" else "🟢"
            lines.append(f"\n{severity_icon} [{a['severity']}] {a['type']} — {a['scenario_name']}")
            lines.append(f"  {a['detail']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 原反事实引擎（保留向后兼容）
# ═══════════════════════════════════════════════════════════════

class CounterfactualEngine:
    """反事实推演引擎"""

    def __init__(self, data: Dict):
        self.original_data = data
        self.essence_pool_data = data.get("essence_pool", {})
        self.original_items: List[Dict] = copy.deepcopy(
            self.essence_pool_data.get("items", []))
        self.modified_items: List[Dict] = copy.deepcopy(self.original_items)
        self.player_names = data.get("game_record", {}).get("player_names", [])
        self.problem = data.get("problem", data.get("game_record", {}).get("problem", ""))
        self.mode = data.get("discussion_mode", "physical")
        self.rounds = data.get("game_record", {}).get("rounds", [])
        self.temporal_memory_data = data.get("temporal_memory", None)
        self.modified_temporal = copy.deepcopy(self.temporal_memory_data)
        self.operations: List[Dict] = []
        self._next_id = max((e.get("id", 0) for e in self.original_items), default=0) + 1

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
            "id": self._next_id, "content": content, "contributor": contributor,
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
            "type": "temporal_reset", "description": "清空时间记忆（模拟历史归零）",
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
        top_modified = self.get_modified_ranking()[:5]
        lines = []
        lines.append("=" * 60)
        lines.append("🔮 反事实推演报告")
        lines.append("=" * 60)
        lines.append(f"   问题: {self.problem[:60]}")
        lines.append(f"   模式: {self.mode}")
        lines.append(f"   操作数: {len(self.operations)}")
        if self.operations:
            lines.append("\n📝 推演操作:")
            for op in self.operations:
                lines.append(f"  · {op.get('description', '')}")
        tstats = self.get_temporal_stats()
        if tstats.get("available"):
            lines.append("\n⏳ 时间耦合记忆")
            lines.append(f"  记录轮次: {tstats['rounds_recorded']}")
            lines.append(f"  活跃连接: {tstats['active_connections']} "
                         f"/ {tstats['shape'][0]}×{tstats['shape'][1]}")
            lines.append(f"  连接密度: {tstats['connection_density']:.2%}")
        lines.append("\n📊 统计对比")
        o, m = stats["original"], stats["modified"]
        lines.append(f"  精华数量: {o['count']} → {m['count']}")
        delta_str = f"{'↑' if m['avg_score'] > o['avg_score'] else '↓'}{abs(m['avg_score'] - o['avg_score']):.1f}"
        lines.append(f"  平均评分: {o['avg_score']} → {m['avg_score']} ({delta_str})")
        lines.append(f"  最高评分: {o['max_score']} → {m['max_score']}")
        lines.append(f"  评分标准差: {o['std_dev']} → {m['std_dev']}")
        if comparison["gained"]:
            lines.append("\n🔼 排名上升:")
            for e in comparison["gained"][:5]:
                lines.append(f"  #{e['id']} \"{e['content']}\"  "
                             f"#{e['old_rank']} → #{e['new_rank']}")
        if comparison["lost"]:
            lines.append("\n🔽 排名下降:")
            for e in comparison["lost"][:5]:
                lines.append(f"  #{e['id']} \"{e['content']}\"  "
                             f"#{e['old_rank']} → #{e['new_rank']}")
        if comparison["new_entries"]:
            lines.append("\n✨ 新增:")
            for e in comparison["new_entries"]:
                lines.append(f"  #{e['id']} \"{e['content']}\"  (评分 {e['score']:.1f})")
        if comparison["removed"]:
            lines.append("\n🗑️ 移除:")
            for e in comparison["removed"]:
                lines.append(f"  #{e['id']} \"{e['content']}\"")
        lines.append("\n💡 Top 5 观点（按影响力排序）:")
        for i, e in enumerate(top_modified, 1):
            lines.append(f"  {i}. [{e.get('score', 0):.1f}] {e.get('content', '')[:60]}")
        delta = m["avg_score"] - o["avg_score"]
        if delta > 0.5:
            lines.append("\n结论: 反事实推演表明，该修改将提升整体讨论质量。")
        elif delta < -0.5:
            lines.append("\n结论: 反事实推演表明，该修改将降低讨论质量。")
        else:
            lines.append("\n结论: 反事实推演表明，该修改对整体讨论影响有限。")
        lines.append("\n" + "=" * 60)
        lines.append("🔚 推演结束")
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