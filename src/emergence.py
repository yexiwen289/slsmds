"""
相变拓扑引擎（Phase Transition Topology Engine）v2.0
—— 真正的非线性贡献质变引擎

============================================================
核心理念：基于复杂系统理论的相变动力学
============================================================

本引擎不再使用简单的加权平均 + 阈值判定，而是构建了完整的
复杂系统计算框架，真正实现"量变→质变"的非线性相变：

  1. 相空间嵌入（Phase Space Embedding）
     将专家观点映射到 6 维认知相空间，每个维度代表一种认知特质

  2. 非线性耦合矩阵（Non-linear Coupling Matrix）
     基于 Ising 模型的专家观点相互作用，谱半径检测临界点

  3. 序参量动力学（Order Parameter Dynamics）
     Metropolis 蒙特卡洛模拟，临界慢化效应检测相变前兆

  4. 自组织临界性（Self-Organized Criticality）
     Bak-Tang-Wiesenfeld 沙堆模型，沙崩幂律分布检测

  5. 混沌边缘检测（Edge of Chaos Detection）
     最大李雅普诺夫指数估计，维持系统在涌现最优区域

  6. 量子叠加态（Quantum Superposition State）
     矛盾观点量子叠加，干涉模式计算，测量坍缩

涌现层级（5 级，真正的非线性阶跃）：
  Level 0: 直接综合（线性保留）
  Level 1: 交叉耦合综合（非线性耦合矩阵 + 吸引子收敛）
  Level 2: 序参量涌现（临界慢化检测 + 相变触发）
  Level 3: 自组织临界综合（沙堆模型 + 沙崩涌现）
  Level 4: 量子叠加与混沌边缘（叠加态坍缩 + 深度质变）
"""

import math
import random
import numpy as np
from collections import Counter
from .prompts_b64 import _get_b64_prompt


# ═══════════════════════════════════════════════════════════════
# 1. 相空间表示与嵌入
# ═══════════════════════════════════════════════════════════════

class OpinionPhaseVector:
    """
    将专家观点映射到多维相空间向量。

    认知维度（6 维）：
    - coherence:      逻辑一致性（句子结构稳定性）
    - novelty:        新颖性（罕见概念密度）
    - depth:          认知深度（复杂逻辑链密度）
    - divergence:     分歧度（与主流观点的偏离程度）
    - specificity:    具体程度（可操作性指标密度）
    - emotional:      情感强度（情绪标记密度）

    这些维度构成一个 6 维相空间，每个专家的观点是该空间中的一个点。
    观点之间的"距离"和"相互作用"由该空间中的几何关系决定。
    """

    DIMENSIONS = ['coherence', 'novelty', 'depth', 'divergence', 'specificity', 'emotional']
    DIMENSION_NAMES_CN = {
        'coherence': '逻辑一致性',
        'novelty': '新颖性',
        'depth': '认知深度',
        'divergence': '分歧度',
        'specificity': '具体程度',
        'emotional': '情感强度',
    }

    # 认知特征标记词库
    _COHERENCE_INDICATORS = ['因为', '所以', '因此', '于是', '从而', '基于', '根据', '由此']
    _NOVELTY_INDICATORS = ['本质上', '悖论', '辩证', '超越', '涌现', '全新', '重构', '范式', '颠覆']
    _DEPTH_INDICATORS = ['如果', '那么', '虽然', '但是', '不仅', '而且', '另一方面', '从长远看', '根本上']
    _DIVERGENCE_INDICATORS = ['不是', '不对', '不同意', '相反', '然而', '但是', '问题在于', '缺陷', '局限', '误区']
    _SPECIFICITY_INDICATORS = ['%', '数据', '案例', '例子', '比如', '例如', '具体', '实际', '指标', '方案']
    _EMOTIONAL_INDICATORS = ['！', '？', '令人', '惊讶', '遗憾', '关键', '重要', '必须', '绝对']

    def __init__(self, text: str, player_name: str = ""):
        self.text = text
        self.player_name = player_name
        self.vector = self._embed(text)
        self.energy = float(np.linalg.norm(self.vector))  # 观点能量

    def _embed(self, text: str) -> np.ndarray:
        """基于认知特征词典的经验嵌入函数"""
        if not text:
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        words = len(text)
        sentences = max(1, text.count('。') + text.count('！') + text.count('？') +
                        text.count('\n') + text.count(';') + text.count('；'))
        avg_sentence_len = words / sentences

        # coherence: 句子长度适中且稳定表示逻辑完整
        coherence = 1.0 - min(1.0, abs(avg_sentence_len - 35) / 70)
        coherence = 0.3 + 0.7 * coherence * (1 + 0.3 * self._count_indicators(text, self._COHERENCE_INDICATORS) / 5)

        # novelty: 新颖标记词密度
        novelty = 0.2 + 0.8 * min(1.0, self._count_indicators(text, self._NOVELTY_INDICATORS) / 4)

        # depth: 逻辑连接词密度
        depth = 0.2 + 0.8 * min(1.0, self._count_indicators(text, self._DEPTH_INDICATORS) / 6)

        # divergence: 否定/对比词密度
        divergence = 0.1 + 0.9 * min(1.0, self._count_indicators(text, self._DIVERGENCE_INDICATORS) / 5)

        # specificity: 具体指标密度
        specificity = 0.15 + 0.85 * min(1.0, self._count_indicators(text, self._SPECIFICITY_INDICATORS) / 5)

        # emotional: 情感标记密度
        emotional = 0.1 + 0.9 * min(1.0, self._count_indicators(text, self._EMOTIONAL_INDICATORS) / 4)

        return np.array([coherence, novelty, depth, divergence, specificity, emotional])

    @staticmethod
    def _count_indicators(text: str, indicators: list) -> int:
        """统计文本中特征标记词的出现次数"""
        return sum(text.count(ind) for ind in indicators)

    def similarity(self, other: 'OpinionPhaseVector') -> float:
        """计算余弦相似度（观点在相空间中的夹角）"""
        v1, v2 = self.vector, other.vector
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def distance(self, other: 'OpinionPhaseVector') -> float:
        """计算欧氏距离（观点在相空间中的距离）"""
        return float(np.linalg.norm(self.vector - other.vector))

    def __repr__(self) -> str:
        return f"<OpinionPhaseVector {self.player_name}: [{', '.join(f'{v:.2f}' for v in self.vector)}]>"


# ═══════════════════════════════════════════════════════════════
# 2. 非线性耦合矩阵
# ═══════════════════════════════════════════════════════════════

class NonLinearCouplingMatrix:
    """
    专家观点之间的非线性耦合矩阵。

    基于统计物理中 Ising 模型的启发：
    - 每个专家观点是一个"自旋"（spin）
    - 耦合矩阵 W_ij 定义了两个自旋之间的相互作用强度
    - 系统的总哈密顿量 H = -∑W_ij * s_i * s_j

    非线性特性：
    - 耦合强度由 tanh 非线性变换决定
    - 当观点相似度处于中等水平时，耦合强度变化最剧烈
    - 谱半径（最大特征值）接近 1 时，系统处于临界点

    临界点意义：
    - 在临界点附近，微小扰动可导致系统发生相变
    - 这正是"量变→质变"的数学基础
    """

    def __init__(self, phase_vectors: list):
        self.n = len(phase_vectors)
        self.vectors = phase_vectors
        self.matrix, self.coupling_histogram = self._build_coupling_matrix()
        self.eigenvalues = self._compute_eigenvalues()
        self._criticality_memory = []  # 临界性历史

    def _build_coupling_matrix(self) -> tuple:
        """
        构建非线性耦合矩阵。

        使用 tanh 非线性变换：
        W_ij = tanh(α * (sim_ij - β))

        其中 α=2.5 控制非线性陡峭度，β=0.3 控制偏移。
        当 sim_ij = β 时，W_ij = 0（无耦合）。
        当 sim_ij 偏离 β 时，耦合强度非线性增长。
        """
        W = np.zeros((self.n, self.n))
        pairs = []

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    W[i, j] = 0.0
                else:
                    sim = self.vectors[i].similarity(self.vectors[j])
                    # 非线性变换：tanh 使耦合强度在中等相似度时变化最剧烈
                    coupling = math.tanh(2.5 * (sim - 0.3))
                    W[i, j] = coupling
                    pairs.append(coupling)

        # 耦合强度直方图
        hist = Counter()
        for c in pairs:
            hist[int(c * 10) / 10] += 1

        return W, hist

    def _compute_eigenvalues(self) -> np.ndarray:
        """计算耦合矩阵的特征值谱"""
        if self.n == 0:
            return np.array([])
        return np.linalg.eigvalsh(self.matrix)

    @property
    def spectral_radius(self) -> float:
        """
        谱半径：最大特征值的绝对值（经 n 归一化）。

        物理意义（归一化后）：
        - 归一化谱半径 < 0.1：系统稳定，有序态
        - 归一化谱半径 ≈ 0.15~0.30：系统处于临界点，相变即将发生
        - 归一化谱半径 > 0.50：系统失稳，进入混沌态
        """
        if len(self.eigenvalues) == 0:
            return 0.0
        sr = float(max(abs(ev) for ev in self.eigenvalues))
        # 按专家数归一化，使谱半径在不同规模的系统间可比
        if self.n > 1:
            sr = sr / self.n
        self._criticality_memory.append(sr)
        return sr

    @property
    def is_critical(self) -> bool:
        """系统是否接近相变临界点（归一化后）"""
        sr = self.spectral_radius
        return 0.10 < sr < 0.40

    @property
    def criticality_trend(self) -> str:
        """临界性趋势：上升/下降/稳定"""
        if len(self._criticality_memory) < 3:
            return "稳定"
        recent = self._criticality_memory[-3:]
        if recent[-1] > recent[0] * 1.05:
            return "↑ 趋近临界"
        elif recent[-1] < recent[0] * 0.95:
            return "↓ 远离临界"
        return "— 稳定"

    @property
    def eigenvalue_gap(self) -> float:
        """
        特征值间隙（最大与次大特征值之差）。

        物理意义：
        - 间隙大：系统有明确的"基态"，容易收敛
        - 间隙小：系统有多个竞争态，容易产生相变
        - 间隙消失：简并，相变点
        """
        if len(self.eigenvalues) < 2:
            return 0.0
        sorted_ev = sorted(self.eigenvalues, reverse=True)
        return float(sorted_ev[0] - sorted_ev[1])

    def coupling_energy(self, state: np.ndarray) -> float:
        """计算系统耦合能量 E = -∑W_ij * s_i * s_j"""
        return -float(state @ self.matrix @ state)

    def frustration_index(self) -> float:
        """
        阻挫指数：测量系统内在矛盾程度。

        当耦合矩阵中同时存在正负耦合时，系统无法同时满足所有约束，
        产生"阻挫"（frustration）——这是复杂系统非线性的重要来源。
        """
        if self.n < 2:
            return 0.0
        pos = np.sum(self.matrix > 0.01)
        neg = np.sum(self.matrix < -0.01)
        total = pos + neg
        if total == 0:
            return 0.0
        # 阻挫 = 正负耦合的冲突程度
        return min(1.0, 2 * min(pos, neg) / total)

    def dominant_clusters(self) -> list:
        """
        基于耦合矩阵的谱聚类检测主导观点簇。

        返回：观点簇列表，每个簇包含专家索引列表
        """
        if self.n < 3:
            return [[i] for i in range(self.n)]

        # 使用特征向量进行简单聚类
        try:
            eigvals, eigvecs = np.linalg.eigh(self.matrix)
            # 取最大的两个特征向量
            if len(eigvals) >= 2:
                vecs = eigvecs[:, -2:]
                # 基于符号聚类
                clusters = {}
                for i in range(self.n):
                    # 将特征向量符号作为聚类签名
                    sig = tuple(1 if v > 0 else -1 for v in vecs[i])
                    clusters.setdefault(sig, []).append(i)
                return list(clusters.values())
        except Exception:
            pass
        return [[i] for i in range(self.n)]


# ═══════════════════════════════════════════════════════════════
# 3. 序参量动力学（临界慢化检测）
# ═══════════════════════════════════════════════════════════════

class OrderParameterDynamics:
    """
    序参量动力学——真正的相变检测引擎。

    基于统计物理的序参量理论：

    序参量 M = |∑s_i| / N

    其中 s_i ∈ {-1, +1} 是专家观点的二值化表示。
    序参量衡量系统的"有序度"：
    - M ≈ 1：所有观点高度一致（铁磁态）
    - M ≈ 0：观点完全分散（顺磁态/混沌态）

    临界现象检测：
    1. 涨落增强（Fluctuation Enhancement）
       接近临界点时，序参量的方差急剧增大
       ⟨δM²⟩ ∝ |T - Tc|^(-γ)

    2. 临界慢化（Critical Slowing Down）
       接近临界点时，序参量的自相关时间 diverges
       τ ∝ |T - Tc|^(-zν)

    3. 有限尺寸标度（Finite Size Scaling）
       在小系统中，临界点附近的峰会被展宽
       但峰的位置随系统尺寸变化
    """

    def __init__(self, coupling_matrix: NonLinearCouplingMatrix):
        self.coupling = coupling_matrix
        self.n = coupling_matrix.n
        self.history = []       # 序参量历史
        self.fluctuation_history = []    # 涨落历史
        self.slowing_history = []        # 慢化历史
        self._rng = random.Random(42)

    @staticmethod
    def _random_state(n: int, rng: random.Random = None) -> np.ndarray:
        """生成随机自旋状态（s_i ∈ {-1, +1}）"""
        if rng:
            return np.array([1 if rng.random() > 0.5 else -1 for _ in range(n)])
        return np.random.choice([-1.0, 1.0], size=n)

    def compute_order_parameter(self, state: np.ndarray) -> float:
        """计算序参量 M = |∑s_i| / N"""
        if self.n == 0:
            return 0.0
        return float(abs(np.sum(state)) / self.n)

    def metropolis_step(self, state: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Metropolis-Hastings 蒙特卡洛一步。

        以概率 min(1, exp(-ΔE/T)) 接受自旋翻转。
        温度 T 控制探索性：
        - T → 0：纯贪心，系统快速收敛到基态
        - T → ∞：纯随机，系统在态空间随机游走
        """
        new_state = state.copy()
        for i in range(self.n):
            delta_E = 2 * new_state[i] * (self.coupling.matrix[i] @ new_state)
            if delta_E < 0 or self._rng.random() < math.exp(-delta_E / max(temperature, 0.01)):
                new_state[i] = -new_state[i]
        return new_state

    def simulate(self, steps: int = 20, temperature: float = 1.0, is_amplified: bool = False) -> dict:
        """
        模拟序参量在耦合矩阵下的演化。

        大系统（n>>1）时自动增加步数保证足够的统计采样。
        对大系统，额外运行多组独立模拟以估算涨落和慢化。

        返回：
        - order_parameter: 平均序参量
        - fluctuation: 序参量涨落（方差）
        - critical_slowing_down: 临界慢化指数（0~1）
        - final_state: 最终自旋状态
        - states: 状态演化历史
        - convergence_speed: 收敛速度
        """
        # 大系统需要更多步数才能检测到涨落和慢化
        # 当 n >= 20 时，步数随 n 线性增长
        actual_steps = steps
        if self.n >= 20:
            scale_factor = max(1.0, self.n / 8.0)
            actual_steps = max(steps, int(steps * scale_factor))

        # 多组独立模拟，从不同初始条件出发
        n_runs = 1
        if is_amplified or self.n >= 20:
            n_runs = min(10, max(3, self.n // 10))

        all_order_params = []

        for run in range(n_runs):
            run_rng = random.Random(42 + run * 7)
            state = self._random_state(self.n, run_rng)
            run_order_params = []

            for step in range(actual_steps):
                t = temperature * (1.0 - 0.5 * step / max(actual_steps, 1))

                # 大规模系统：每步执行多次遍历
                n_sweeps = max(1, self.n // 10)
                for _ in range(n_sweeps):
                    state = self.metropolis_step(state, t)

                run_order_params.append(self.compute_order_parameter(state))

            if run == 0:
                self.history = run_order_params
                all_order_params = run_order_params
            else:
                all_order_params.extend(run_order_params)

        # 涨落计算：在多组模拟的平均值上计算方差
        # 对大系统，使用组间方差更准确
        if n_runs > 1:
            # 计算每组末尾 1/3 步的平均序参量
            trailing = max(1, actual_steps // 3)
            run_means = []
            for run in range(n_runs):
                start = run * actual_steps
                end = (run + 1) * actual_steps
                run_vals = all_order_params[start:end]
                run_means.append(float(np.mean(run_vals[-trailing:])))
            fluctuation = float(np.var(run_means)) if len(run_means) > 1 else 0.0
        else:
            fluctuation = float(np.var(all_order_params)) if len(all_order_params) > 1 else 0.0

        self.fluctuation_history.append(fluctuation)

        # 收敛速度：最后几步序参量的变化幅度
        if len(self.history) > 3:
            convergence = float(np.std(self.history[-3:]))
        else:
            convergence = 1.0

        # 临界慢化指数：基于谱半径 + 系统规模估计
        # 物理直觉：谱半径越接近 1、系统越大，慢化越严重
        slowing_down = self._estimate_slowing_down_structural()

        # 平均序参量
        mean_order = float(np.mean(self.history))

        return {
            "order_parameter": mean_order,
            "fluctuation": fluctuation,
            "critical_slowing_down": slowing_down,
            "convergence_speed": convergence,
            "final_state": None,
            "states": None,
        }

    def _estimate_slowing_down_structural(self) -> float:
        """
        基于系统结构特征估计临界慢化指数。

        对于大系统（n >> 1），序参量自相关时间难以直接测量，
        因为单次模拟中序参量变化极小。本方法通过系统结构特征
        来估计慢化程度：

        1. 谱半径越接近 1 → 慢化越严重
        2. 系统规模越大 → 慢化越显著（大系统需要更多时间一致）
        3. 阻挫越高 → 慢化越严重（系统难以达到基态）
        """
        # 谱半径贡献：越接近 1 越慢化
        sr = min(1.0, self.coupling.spectral_radius)
        sr_contrib = 1.0 - abs(1.0 - sr)  # 1.0 时最大

        # 系统规模贡献：大系统慢化更明显
        size_contrib = min(1.0, math.log2(self.n) / 10.0) if self.n > 1 else 0.0

        # 阻挫贡献：高阻挫 = 难以收敛 = 慢化
        frust = min(1.0, self.coupling.frustration_index() * 2.0)
        frust_contrib = frust

        # 加权综合
        slowing = 0.5 * sr_contrib + 0.3 * size_contrib + 0.2 * frust_contrib
        self.slowing_history.append(slowing)
        return min(1.0, max(0.0, slowing))

    def _estimate_slowing_down(self, order_params: list) -> float:
        """
        估计临界慢化指数。

        通过自相关函数的衰减时间来判断：
        - 快速衰减（τ 小）：远离临界点
        - 慢速衰减（τ 大）：靠近临界点（临界慢化）
        """
        if len(order_params) < 4:
            return 0.0

        arr = np.array(order_params) - np.mean(order_params)
        var = np.var(arr)
        if var < 1e-10:
            return 0.0

        # 自相关
        ac = np.correlate(arr, arr, mode='full')
        ac = ac[len(ac)//2:] / ac[len(ac)//2]

        # 找到自相关降到 1/e 的滞后时间
        tau = 1
        for t in range(1, len(ac)):
            if ac[t] < 1/math.e:
                tau = t
                break

        # 归一化到 0~1
        max_tau = max(1, len(order_params) // 2)
        slowing = min(1.0, tau / max_tau)
        self.slowing_history.append(slowing)
        return slowing


# ═══════════════════════════════════════════════════════════════
# 4. 自组织临界性（沙堆模型）
# ═══════════════════════════════════════════════════════════════

class SelfOrganizedCriticality:
    """
    自组织临界性（SOC）引擎。

    基于 Bak-Tang-Wiesenfeld (BTW) 沙堆模型：

    核心机制：
    - 每个专家观点是一粒"沙子"（能量量子）
    - 沙堆通过添加沙子自然演化到临界状态
    - 在临界点，单个沙粒的添加可能触发大规模"沙崩"
    - 沙崩大小服从幂律分布（P(s) ∝ s^(-τ)）
    - 幂律分布是临界性的标志

    为什么是沙堆模型：
    - 与观点积累过程同构：每个新观点都是"一粒沙"
    - 沙崩对应"涌现综合"：大量观点积累到临界点后突然爆发
    - 沙崩大小不可预测：这正体现了"量变→质变"的非线性
    """

    def __init__(self, grid_size: int = 5):
        self.size = max(3, grid_size)
        self.grid = np.zeros((self.size, self.size), dtype=np.float64)
        self.avalanche_sizes = []
        self.avalanche_timeline = []
        self.total_grains = 0
        self.critical_threshold = 4.0  # 沙崩阈值
        self._rng = random.Random(42)  # 固定随机种子，确保可复现

    def add_grain(self, x: int = None, y: int = None, energy: float = 1.0) -> int:
        """
        添加一粒沙子，返回触发的沙崩大小。

        参数：
        - energy: 沙粒能量（可关联观点能量）
        """
        if x is None:
            x = self._rng.randint(0, self.size - 1)
        if y is None:
            y = self._rng.randint(0, self.size - 1)

        self.grid[x, y] += energy
        self.total_grains += 1

        return self._topple()

    def _topple(self) -> int:
        """执行沙崩传播（Bak-Tang-Wiesenfeld 算法）"""
        avalanche = 0
        toppling = []

        while True:
            # 找到所有超阈值位置
            critical = np.where(self.grid >= self.critical_threshold)
            if len(critical[0]) == 0:
                break

            for i, j in zip(critical[0], critical[1]):
                self.grid[i, j] -= self.critical_threshold
                avalanche += 1
                toppling.append((i, j))

                # 向四个邻居传播
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        self.grid[ni, nj] += 1.0

        if avalanche > 0:
            self.avalanche_sizes.append(avalanche)
            self.avalanche_timeline.append(avalanche)

        return avalanche

    @property
    def is_at_criticality(self) -> bool:
        """
        系统是否处于临界状态。

        检测方法：如果沙崩大小分布存在"大沙崩"（>3），
        说明系统处于临界状态。
        """
        if len(self.avalanche_sizes) < 8:
            return False
        large = sum(1 for a in self.avalanche_sizes[-20:] if a >= 3)
        return large / max(1, len(self.avalanche_sizes[-20:])) > 0.15

    @property
    def criticality_level(self) -> float:
        """
        临界程度（0~1）。

        基于：
        1. 沙崩大小的方差（临界点附近沙崩大小分布更广）
        2. 大沙崩比例
        3. 沙堆总能量
        """
        # 沙崩方差
        recent = self.avalanche_sizes[-20:] if len(self.avalanche_sizes) >= 20 else self.avalanche_sizes
        if len(recent) >= 2:
            var = float(np.var(recent))
            var_factor = min(1.0, var / 8)
        else:
            var_factor = 0.0

        # 大沙崩比例
        large = sum(1 for a in recent if a >= 3) if recent else 0
        large_factor = large / max(1, len(recent))

        # 能量因子
        energy = float(np.sum(self.grid))
        energy_factor = min(1.0, energy / (self.size * self.size * 2))

        # 加权综合
        return 0.4 * var_factor + 0.4 * large_factor + 0.2 * energy_factor

    @property
    def power_law_fit(self) -> float:
        """
        幂律拟合优度（R²）。

        如果沙崩大小分布很好地服从幂律，说明系统处于真正的临界状态。
        这里用简化的方法估计。
        """
        if len(self.avalanche_sizes) < 5:
            return 0.0

        sizes = self.avalanche_sizes[-30:]
        if not sizes:
            return 0.0

        # 统计各大小沙崩的频次
        max_size = max(sizes)
        if max_size < 2:
            return 0.0

        hist = {}
        for s in sizes:
            hist[s] = hist.get(s, 0) + 1

        # 检查是否呈现幂律特征（小沙崩多，大沙崩少）
        small = sum(v for k, v in hist.items() if k <= 2)
        large = sum(v for k, v in hist.items() if k >= 4)
        total = len(sizes)

        return small / max(1, total) - large / max(1, total)


# ═══════════════════════════════════════════════════════════════
# 5. 混沌边缘检测
# ═══════════════════════════════════════════════════════════════

class EdgeOfChaosDetector:
    """
    混沌边缘检测器。

    基于李雅普诺夫指数（Lyapunov Exponent）的混沌检测：

    λ = lim(n→∞) (1/n) * ln|δx(n)/δx(0)|

    - λ < 0：系统稳定（有序态）
    - λ ≈ 0：系统处于混沌边缘（涌现最优区域）
    - λ > 0：系统混沌（混沌态）

    为什么混沌边缘是涌现的最佳区域：
    - 在混沌边缘，系统既不过于有序（死板）也不过于混沌（随机）
    - 信息在"稳定"和"变化"之间达到最佳平衡
    - 这是复杂系统产生真正新颖性的区域
    """

    def __init__(self, coupling_matrix: NonLinearCouplingMatrix):
        self.coupling = coupling_matrix
        self.lyapunov_estimates = []
        self.entropy_estimates = []

    def estimate_lyapunov(self, steps: int = 10) -> float:
        """
        估计最大李雅普诺夫指数。

        使用标准轨道扰动法：
        1. 初始化主轨道和扰动轨道
        2. 每一步计算轨道分离率
        3. 平均后得到李雅普诺夫指数估计
        """
        if self.coupling.n < 2:
            return -1.0

        n = self.coupling.n
        W = self.coupling.matrix

        # 主轨道初始化
        x = np.random.randn(n) * 0.1
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x

        # 扰动轨道
        delta = np.random.randn(n) * 1e-6
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0:
            delta = delta / delta_norm * 1e-6

        lyap_sum = 0.0
        count = 0

        for _ in range(steps):
            # 非线性映射（带 tanh 激活）
            x_new = np.tanh(W @ x)
            delta_new = np.tanh(W @ (x + delta)) - x_new

            d_norm = np.linalg.norm(delta_new)
            if d_norm > 1e-15:
                lyap_sum += math.log(d_norm / 1e-6)
                delta = delta_new / d_norm * 1e-6
                count += 1

            # 重新归一化主轨道
            x_new_norm = np.linalg.norm(x_new)
            if x_new_norm > 0:
                x = x_new / x_new_norm * 0.1
            else:
                x = x_new

        estimate = lyap_sum / max(1, count)
        self.lyapunov_estimates.append(estimate)
        return estimate

    @property
    def at_edge_of_chaos(self) -> bool:
        """是否处于混沌边缘（-0.15 < λ < 0.15）"""
        if not self.lyapunov_estimates:
            return False
        latest = self.lyapunov_estimates[-1]
        return -0.15 < latest < 0.15

    @property
    def chaos_depth(self) -> float:
        """
        混沌深度（李雅普诺夫指数最新值）。

        - 正数越大 → 越混沌
        - 负数越大绝对值 → 越有序
        - 接近零 → 混沌边缘
        """
        if not self.lyapunov_estimates:
            return 0.0
        return self.lyapunov_estimates[-1]

    @property
    def chaos_regime(self) -> str:
        """混沌状态描述"""
        if not self.lyapunov_estimates:
            return "未知"
        ly = self.lyapunov_estimates[-1]
        if ly < -0.3:
            return "❄ 高度有序"
        elif ly < -0.15:
            return "→ 趋向有序"
        elif ly < 0.15:
            return "✦ 混沌边缘 ✦"
        elif ly < 0.3:
            return "∼ 轻度混沌"
        else:
            return "★ 深度混沌"

    def estimate_entropy(self, states: list) -> float:
        """
        估计系统的香农熵。

        熵越高，系统的多样性/不确定性越高。
        在混沌边缘，熵通常处于中等水平。
        """
        if not states:
            return 0.0

        # 将状态离散化
        bins = 10
        hist = {}
        for s in states:
            for val in np.round(s, 1):
                key = int(val * bins)
                hist[key] = hist.get(key, 0) + 1

        total = sum(hist.values())
        if total == 0:
            return 0.0

        probs = [c / total for c in hist.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        self.entropy_estimates.append(entropy)
        return entropy


# ═══════════════════════════════════════════════════════════════
# 6. 量子叠加态
# ═══════════════════════════════════════════════════════════════

class QuantumSuperposition:
    """
    观点量子叠加态引擎。

    基于量子认知科学（Quantum Cognition）的框架：

    核心概念：
    1. 叠加态（Superposition）
       矛盾的专家观点同时存在于叠加态中，类似于薛定谔的猫。
       |ψ⟩ = Σ α_i |观点_i⟩

    2. 干涉（Interference）
       相似观点产生建设性干涉（振幅增强）
       冲突观点产生破坏性干涉（振幅减弱）

    3. 测量坍缩（Measurement Collapse）
       当"测量"（综合/决策）发生时，叠加态坍缩为确定输出。
       坍缩概率由玻恩规则决定：P(i) = |α_i|²

    4. 量子纠缠（Quantum Entanglement）
       高度相关的观点之间产生纠缠，一个观点的"测量"影响其他观点。

    5. 多路径探索（Multi-path Exploration）
       在坍缩前，系统同时探索所有可能的综合路径。
       这确保了输出不遗漏任何可能的洞见。
    """

    def __init__(self, phase_vectors: list):
        self.vectors = phase_vectors
        self.n = len(phase_vectors)
        # 振幅：每个观点在叠加态中的"概率幅"
        self.amplitudes = np.ones(self.n) / math.sqrt(max(1, self.n))
        self.collapsed = False
        self.measurement_history = []

    def interference_pattern(self) -> np.ndarray:
        """
        计算量子干涉模式。

        干涉强度决定了观点之间的"共振"程度：
        - 建设性干涉：观点高度相似，振幅增强（同相）
        - 破坏性干涉：观点高度冲突，振幅减弱（反相）
        - 无干涉：观点独立，振幅不变
        """
        interference = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    sim = self.vectors[i].similarity(self.vectors[j])
                    # 干涉强度：-1（完全破坏性）到 +1（完全建设性）
                    interference[i] += (sim - 0.5) * 2

        # 归一化
        max_val = max(abs(interference)) or 1
        return interference / max_val

    def entanglement_entropy(self) -> float:
        """
        纠缠熵：测量观点之间的量子纠缠程度。

        高纠缠熵意味着观点高度关联，无法独立处理。
        这是涌现综合的必要条件。
        """
        if self.n < 2:
            return 0.0

        # 用相似度矩阵的熵来估计纠缠
        sim_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                sim_matrix[i, j] = self.vectors[i].similarity(self.vectors[j])

        # 归一化得到概率分布
        flat = sim_matrix.flatten()
        flat = flat / np.sum(flat) if np.sum(flat) > 0 else flat

        entropy = -sum(p * math.log(p + 1e-10) for p in flat)
        # 归一化到 0~1
        max_entropy = math.log(self.n * self.n)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def measure(self, temperature: float = 1.0) -> np.ndarray:
        """
        "测量"叠加态，坍缩为确定概率分布。

        玻恩规则：P(i) = |α_i|² × 干涉修正

        温度参数：
        - temperature → 0：确定性坍缩（纯利用）
        - temperature → ∞：随机坍缩（纯探索，多样性强）
        """
        interference = self.interference_pattern()

        # 玻恩规则：概率 = |振幅|²
        probs = np.abs(self.amplitudes) ** 2

        # 干涉修正
        probs = probs * (1 + 0.3 * interference)
        probs = np.maximum(probs, 0.01)
        probs = probs / np.sum(probs)

        # 温度调整（模拟量子退火）
        if temperature < 0.99 or temperature > 1.01:
            probs = probs ** (1.0 / max(temperature, 0.1))
            probs = probs / np.sum(probs)

        self.collapsed = True
        self.measurement_history.append(probs.copy())
        return probs

    def superposition_depth(self) -> float:
        """
        叠加深度：测量观点之间的量子叠加程度。

        高叠加深度 = 高度矛盾的观点共存，
        这通常意味着系统处于"临界叠加态"，
        即将发生质变。
        """
        if self.n < 2:
            return 0.0

        # 测量观点之间的平均距离
        distances = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                distances.append(self.vectors[i].distance(self.vectors[j]))

        if not distances:
            return 0.0

        avg_dist = np.mean(distances)
        return min(1.0, avg_dist / 2.0)


# ═══════════════════════════════════════════════════════════════
# 7. 虚拟专家生成器（10人→100人效果的核心引擎）
# ═══════════════════════════════════════════════════════════════

class VirtualExpertGenerator:
    """
    虚拟专家生成器——从少量真实专家中生成大量虚拟专家。

    核心思想：通过相空间采样、插值和扰动，从 N 个真实专家
    生成 M 个虚拟专家，使得系统在相空间中的覆盖密度等同于
    大量专家的效果。

    生成策略（4 种非线性扩充）：
    1. 成对插值（Pairwise Interpolation）
       在每对真实专家的相空间向量之间进行非线性插值，
       生成中间态观点，填补相空间中的"认知空隙"。

    2. 相空间扰动（Phase Space Perturbation）
       对每个真实专家的向量在各维度上添加不同幅度的噪声，
       模拟专家内部的不同思考变体。

    3. 边界外推（Boundary Extrapolation）
       在相空间边界外推，生成极端化观点，
       模拟"极端派专家"的立场。

    4. 吸引子采样（Attractor Sampling）
       基于耦合矩阵的动力学系统，迭代收敛到吸引子，
       从吸引子附近采样生成"虚拟集体无意识"观点。

    使用方式：
        generator = VirtualExpertGenerator(real_discussions, target=100)
        all_experts = generator.get_all_discussions()
        # 现在有 100 个专家的效果
    """

    # 认知维度映射到发言模板（多变体，高质量）
    _SPEECH_TEMPLATES = {
        # ── 连贯性/逻辑性 ──
        "high_coherence": [
            "从系统论的角度来看，这个问题本质上是一个多层次、多变量的复杂系统，需要我们用整体性思维来把握各要素之间的非线性耦合关系。",
            "如果我们把这个问题的各个维度放在一个统一的框架里审视，就会发现它们之间存在着深层的结构同构性——解决其中一个维度必然牵动其他维度。",
            "这并非孤立的问题，而是一个嵌套的系统。任何试图只解决局部而不考虑整体涌现性的方案，都会在更长的时间尺度上失效。",
        ],
        "mid_coherence": [
            "经过多角度分析，我认为这个问题可以从不同的层面来理解，每个层面都有其内在的逻辑自洽性。",
            "综合各方观点，这个问题既有结构性的面向，也有过程性的面向，需要在不同层面上分别处理。",
            "我的判断是：这个问题并非非此即彼，而是一个需要在不同维度上取得动态平衡的复合问题。",
        ],
        "low_coherence": [
            "这个问题牵涉的因素太多，需要我们逐一拆解分析，不能急于给出统一结论。",
            "我注意到各方观点之间存在一些尚未被充分梳理的交叉地带，这些交叉地带恰恰可能是突破口。",
            "在给出判断之前，我想先厘清问题的边界条件——哪些是我们确知的，哪些仍是假设。",
        ],

        # ── 新颖性 ──
        "high_novelty": [
            "我认为这里隐藏着一个范式级的认知转换：我们一直在用旧的框架描述一个本质上不同的新现象，这就像用牛顿力学描述量子效应一样注定失真。",
            "让我提出一个可能令人不安的视角——如果这个问题的'答案'本身就是一个递归的自指结构呢？不是我们在解决问题，而是问题在重构我们。",
            "我想指出一个被所有人忽略的可能性：我们讨论的并不是同一个问题，而是在用同一个词语指代完全不同的认知对象。",
        ],
        "mid_novelty": [
            "这个问题的本质可能和表面呈现的不太一样，有一些隐藏的维度值得我们花时间深入探索。",
            "我想引入一个跨学科的视角，虽然它不一定直接解决问题，但可能为我们的思考打开新的路径。",
            "除了已有的分析路径，我认为还存在一条很少有人走但可能富有成效的思路，值得认真考虑。",
        ],
        "low_novelty": [
            "基于已有的理论和实践经验，我认为应该采用成熟的方法论来处理，创新不一定是最优策略。",
            "在已有的认知框架内，这个问题是可以被系统性地解决的，关键在于执行的精度而非方向的改变。",
            "我倾向于沿用经过验证的分析路径，同时对一些细节进行必要的校准和优化。",
        ],

        # ── 深度 ──
        "high_depth": [
            "如果从更根本的层面来看，这不是一个技术问题而是一个本体论问题——它在追问我们赖以思考的底层架构本身是否足够自洽。",
            "深层来看，这里涉及一个根本性的悖论：我们用来解决问题的工具本身就是问题的一部分，这意味着任何线性的解决方案都会在某个临界点自我翻转。",
            "我试图触及这个问题最深层的结构。在剥去所有表面因素后，剩下的核心矛盾是：存在与认知之间的鸿沟，以及跨越这个鸿沟的冲动本身是否构成了一种新的存在方式。",
        ],
        "mid_depth": [
            "深层来看，这里涉及到一个根本性的矛盾，需要我们辩证地看待对立面之间的张力关系。",
            "如果我们往深处挖掘，会发现这个问题的底层逻辑包含一个隐含的前提假设，而这个假设本身就值得被审视。",
            "在表面层次之下，我认为存在一个结构性的因果链条，它解释了为什么不同的人会对同一现象得出如此不同的结论。",
        ],
        "low_depth": [
            "从表面现象来看，这个问题的核心在于一些可以具体识别和处理的操作性因素。",
            "我认为不必过度复杂化——识别出关键变量并采取针对性的措施，就可以取得实质进展。",
            "在实践层面，这个问题可以被分解为若干子问题，逐一处理后即可得到整体改善。",
        ],

        # ── 分歧度 ──
        "high_divergence": [
            "我必须直言不讳地反对当前的主流共识。问题在于，我们在一个未经充分检验的前提下搭建了整座论证大厦，而这个前提本身可能就是错的。",
            "我不同意主流观点，并不是为了唱反调，而是因为我认为大家忽略了一个关键的否定性证据——这个证据一旦被正视，将颠覆整个讨论的方向。",
            "让我指出一个不舒服的事实：我们所谓的共识，可能只是群体思维的产物。真正的洞见往往出现在被集体沉默覆盖的角落。",
        ],
        "mid_divergence": [
            "虽然主流观点有其道理，但我想补充一个不同的视角，这个视角不否定已有观点，而是试图拓展讨论的边界。",
            "我基本同意大方向，但对其中一个关键环节持保留意见——那里存在一个我认为被过度简化的复杂地带。",
            "我想提出一个折中的但略有偏移的立场：既不完全接受也不完全否定当前观点，而是重新框定问题的范围。",
        ],
        "low_divergence": [
            "我基本上同意前面各位的分析框架和核心结论，这里做一些细节性的补充和验证。",
            "综合已有的讨论，我认为方向是正确的，我想在执行层面和验证方法上做一些补充。",
            "我赞同当前的分析路径，同时提供一些来自不同数据来源的交叉验证结果。",
        ],

        # ── 具体性 ──
        "high_specificity": [
            "具体来说，根据已有的实证数据显示，在可比较的案例中，采用系统性方法的成功率比单一维度干预高出约40%，而且这个差距会随系统复杂度的提升而扩大。",
            "我可以给出一个具体的分析框架：首先识别系统的反馈回路结构，然后标注每个回路的时间延迟和增益系数，最后在关键耦合点施加干预。",
            "举一个具体的案例来说明：在某实际项目中，当我们把非线性耦合因素纳入考量后，原来的线性模型预测误差从15%上升到了47%——这说明了非线性的重要性。",
        ],
        "mid_specificity": [
            "从实际案例来看，这种方法有其适用场景和局限性，关键在于识别系统的复杂度等级。",
            "可以参考一些中间层面的方法论，它们在理论严格性和操作可行性之间取得了较好的平衡。",
            "我建议采用一种分层的方法：先处理可直接观测的表层结构，再逐步深入到需要间接推断的深层结构。",
        ],
        "low_specificity": [
            "总的来说，这种方法在理论上是可行的，但具体实施需要更多的前置条件分析。",
            "在方向确定之后，具体的执行路径有多种选择，需要根据实际情况灵活调整。",
            "框架层面我认为已经比较清晰，接下来需要在实践中检验和校准。",
        ],

        # ── 情感 ──
        "high_emotional": [
            "这太令人振奋了！如果我们的判断是正确的，这将不仅仅是一个技术突破，而是一种全新的认知世界的方式——我们必须以最大的紧迫感推进。",
            "我必须承认，当我意识到这个问题的深层含义时，我感到一种近乎敬畏的震撼——我们触碰到的可能是认知本身的基础结构。",
            "这让我感到不安，但也让我感到兴奋——不安是因为我们可能正在踏入未知的领域，兴奋是因为这正是认知突破发生的前兆。",
        ],
        "low_emotional": [
            "这是一个技术性问题，需要理性分析和客观判断，不宜过度解读其象征意义。",
            "从冷静分析的角度来看，这个发现的重要性需要更多数据的支撑才能最终确认。",
            "我倾向于保持审慎的态度：在证据尚不充分的情况下，避免过早的情绪化判断。",
        ],
    }

    # 逻辑连接词池
    _TRANSITIONS = [
        "进一步说，", "从另一个角度来看，", "值得指出的是，",
        "在此基础上，", "更深层地看，", "与此相关的是，",
        "如果沿此思路推演，", "一个容易被忽略的点是，", "把视线拉远一些，",
    ]
    _OPENINGS = [
        "关于这个问题，我的看法是：", "经过思考，我认为：",
        "在这个问题上，我持有以下立场：", "让我直接陈述我的判断：",
        "我的分析路径如下：", "如果允许我从另一个层面切入：",
    ]

    def __init__(self, real_discussions: list, target_experts: int = 2000):
        """
        参数:
            real_discussions: [{"player_name", "speech", "key_insight"}, ...]
            target_experts: 目标专家总数（含真实专家），默认 2000
        """
        self.real = real_discussions
        self.n_real = len(real_discussions)
        self.target = max(target_experts, self.n_real + 10)

        # 构建真实专家的相空间向量
        self.real_vectors = [
            OpinionPhaseVector(d.get('speech', ''), d.get('player_name', ''))
            for d in real_discussions
        ]

        # 生成虚拟专家
        self.virtual_discussions = []
        if self.n_real >= 2:
            self._generate_all()

    def _generate_all(self):
        """按策略配额生成所有虚拟专家（两阶段：先生成向量，选完后生成文本）"""
        n_needed = self.target - self.n_real
        if n_needed <= 0:
            return

        # 各策略配额（分布均衡，覆盖不同相空间区域）
        # 探索策略占 20%，确保突破原始专家凸包
        strategy_pool_size = n_needed * 2  # 预生成2倍，再筛选
        n_explore = strategy_pool_size // 5
        remaining = strategy_pool_size - n_explore
        n_interp = remaining // 4
        n_perturb = remaining // 4
        n_extra = remaining // 4
        n_attractor = remaining - n_interp - n_perturb - n_extra

        # 第一阶段：只生成向量（不生成文本，大幅加速）
        pool = []
        pool.extend(self._generate_exploration(n_explore))
        pool.extend(self._generate_interpolation(n_interp))
        pool.extend(self._generate_perturbation(n_perturb))
        pool.extend(self._generate_extrapolation(n_extra))
        pool.extend(self._generate_attractor(n_attractor))

        # 相空间多样性筛选
        selected = self._select_diverse(pool, n_needed)

        # 第二阶段：只为选中的虚拟专家生成文本
        self.virtual_discussions = []
        for item in selected:
            vec = item.get('_vector')
            if vec is not None:
                speech = self._vector_to_speech(vec)
                item['speech'] = speech
                item['key_insight'] = item.get('key_insight', '虚拟观点')
                item.pop('_vector', None)
                self.virtual_discussions.append(item)

    def _select_diverse(self, candidates: list, n: int) -> list:
        """
        贪婪多样性筛选（增量更新版，O(M×N) 复杂度）。

        每次选择与已有集合最不相似的一个候选，确保最大化相空间覆盖度。
        维护 min_dist 数组避免重复计算，支持大规模选择（2000+）。
        """
        if not candidates or n <= 0:
            return []

        M = len(candidates)

        # 构建候选向量矩阵 (M × 6)
        cand_matrix = np.array([
            c['_vector'] if '_vector' in c and isinstance(c['_vector'], np.ndarray)
            else OpinionPhaseVector(c.get('speech', ''), c.get('player_name', '')).vector
            for c in candidates
        ], dtype=np.float64)

        # 真实专家向量矩阵 (N × 6)
        real_matrix = np.array([v.vector for v in self.real_vectors], dtype=np.float64)

        # 初始化 min_dist：每个候选到最近真实专家的距离
        if real_matrix.shape[0] > 0:
            # (M × N) 距离矩阵
            diff = cand_matrix[:, np.newaxis, :] - real_matrix[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            min_dists = np.min(dists, axis=1)  # (M,)
        else:
            min_dists = np.full(M, float('inf'))

        selected_indices = []
        selected_items = []
        n_select = min(n, M)

        for _ in range(n_select):
            # 选 min_dist 最大的候选
            best_idx = int(np.argmax(min_dists))
            if min_dists[best_idx] < 0:
                break

            selected_indices.append(best_idx)
            selected_items.append(candidates[best_idx])

            # 增量更新：只需计算到新选点的距离，然后取 min
            new_vec = cand_matrix[best_idx]  # (6,)
            diff_new = cand_matrix - new_vec[np.newaxis, :]  # (M × 6)
            dist_new = np.sqrt(np.sum(diff_new ** 2, axis=1))  # (M,)
            min_dists = np.minimum(min_dists, dist_new)

            # 排除已选
            min_dists[best_idx] = -1

        return selected_items

    def select_neuron_representatives(self, n: int = 20) -> list:
        """
        从全部专家（真实+虚拟）中选出 n 个代表性"神经元"（增量更新版）。

        使用贪婪最远点采样，确保选出的代表最大化覆盖相空间。
        这些代表将作为虚拟专家群体的"突触输出"注入 LLM 综合prompt。
        """
        all_discussions = self.get_all_discussions()
        M = len(all_discussions)
        if M <= n:
            return all_discussions

        # 构建所有专家的向量矩阵
        all_vecs = []
        for d in all_discussions:
            if '_vector' in d and isinstance(d['_vector'], np.ndarray):
                all_vecs.append(d['_vector'])
            else:
                all_vecs.append(OpinionPhaseVector(d.get('speech', ''), d.get('player_name', '')).vector)
        all_matrix = np.array(all_vecs, dtype=np.float64)  # (M × 6)

        # 从第一个开始
        selected_idx = [0]
        min_dists = np.sqrt(np.sum((all_matrix - all_matrix[0:1]) ** 2, axis=1))
        min_dists[0] = -1

        for _ in range(n - 1):
            best_idx = int(np.argmax(min_dists))
            if min_dists[best_idx] < 0:
                break
            selected_idx.append(best_idx)
            # 增量更新
            diff_new = all_matrix - all_matrix[best_idx:best_idx+1]
            dist_new = np.sqrt(np.sum(diff_new ** 2, axis=1))
            min_dists = np.minimum(min_dists, dist_new)
            min_dists[best_idx] = -1

        return [all_discussions[i] for i in selected_idx]

    def _generate_interpolation(self, count: int) -> list:
        """成对非线性插值生成虚拟专家"""
        if self.n_real < 2 or count <= 0:
            return []

        pairs = []
        for i in range(self.n_real):
            for j in range(i + 1, self.n_real):
                pairs.append((i, j))

        if not pairs:
            return []

        rng = random.Random(42)
        results = []
        generated = 0
        attempts = 0

        while generated < count and attempts < count * 5:
            attempts += 1
            i, j = rng.choice(pairs)
            v1, v2 = self.real_vectors[i].vector, self.real_vectors[j].vector

            # 非线性插值系数（非均匀分布，探索非线性区域）
            alpha = rng.uniform(0.15, 0.85)
            beta = 1 - alpha

            # 引入非线性扭曲：在中间区域增加扰动
            nonlinear_bump = 0.08 * math.sin(alpha * math.pi)
            noise = np.array([rng.uniform(-1, 1) for _ in range(6)]) * 0.06

            virtual_vec = alpha * v1 + beta * v2 + nonlinear_bump * noise
            virtual_vec = np.clip(virtual_vec, 0.05, 0.95)

            results.append({
                "player_name": f"虚拟-内插{generated+1}",
                "speech": "",  # 第二阶段生成
                "key_insight": f"内插观点 (α={alpha:.2f})",
                "_vector": virtual_vec.copy(),
            })
            generated += 1

        return results

    def _generate_perturbation(self, count: int) -> list:
        """相空间各维度差异化扰动生成"""
        if self.n_real < 1 or count <= 0:
            return []

        rng = random.Random(43)
        results = []
        # 各维度的扰动幅度（不同维度不同扰动水平）
        dim_noise_scale = [0.12, 0.22, 0.18, 0.28, 0.18, 0.14]

        for k in range(count):
            src = rng.randint(0, self.n_real - 1)
            orig = self.real_vectors[src].vector

            noise = np.array([rng.gauss(0, s) for s in dim_noise_scale])
            virtual_vec = orig + noise
            virtual_vec = np.clip(virtual_vec, 0.05, 0.95)

            results.append({
                "player_name": f"虚拟-扰动{k+1}",
                "speech": "",
                "key_insight": "扰动生成观点",
                "_vector": virtual_vec.copy(),
            })

        return results

    def _generate_extrapolation(self, count: int) -> list:
        """边界外推生成极端化观点"""
        if self.n_real < 1 or count <= 0:
            return []

        rng = random.Random(44)
        dim_names = ['逻辑性', '新颖性', '深度', '分歧度', '具体性', '情感']
        results = []

        for k in range(count):
            src = rng.randint(0, self.n_real - 1)
            orig = self.real_vectors[src].vector

            # 选择 1-2 个维度进行极端化
            n_dims = 1 if rng.random() < 0.6 else 2
            dims = rng.sample(range(6), n_dims)

            virtual_vec = orig.copy()
            dim_labels = []
            for d in dims:
                direction = 1 if rng.random() > 0.5 else -1
                factor = 1.0 + rng.uniform(0.3, 0.8)
                if direction > 0:
                    virtual_vec[d] = min(0.95, orig[d] * factor)
                else:
                    virtual_vec[d] = max(0.05, orig[d] / factor)
                dim_labels.append(dim_names[d])

            results.append({
                "player_name": f"虚拟-极端{k+1}",
                "speech": "",
                "key_insight": f"极端{'/'.join(dim_labels)}",
                "_vector": virtual_vec.copy(),
            })

        return results

    def _generate_attractor(self, count: int) -> list:
        """耦合矩阵吸引子采样生成"""
        if self.n_real < 2 or count <= 0:
            return []

        results = []
        rng = random.Random(45)

        try:
            # 构建耦合矩阵
            W = np.zeros((self.n_real, self.n_real))
            for i in range(self.n_real):
                for j in range(self.n_real):
                    if i != j:
                        sim = self.real_vectors[i].similarity(self.real_vectors[j])
                        W[i, j] = math.tanh(2.5 * (sim - 0.3))

            for _ in range(count):
                # 随机初始状态
                state = np.random.randn(self.n_real) * 0.1
                # 迭代到吸引子（非线性动力学）
                for _ in range(30):
                    state = np.tanh(1.5 * W @ state)

                # 从吸引子状态生成虚拟专家
                # 选择贡献最大的几个真实专家，混合他们的相空间向量
                weights = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
                virtual_vec = np.zeros(6)
                for ei in range(self.n_real):
                    virtual_vec += weights[ei] * self.real_vectors[ei].vector * (1 + 0.2 * state[ei])

                virtual_vec = np.clip(virtual_vec, 0.05, 0.95)

                results.append({
                    "player_name": f"虚拟-吸引子{rng.randint(100,999)}",
                    "speech": "",
                    "key_insight": "吸引子收敛观点",
                    "_vector": virtual_vec.copy(),
                })
        except Exception:
            pass

        return results

    def _generate_exploration(self, count: int) -> list:
        """
        相空间自由探索：在全 [0.05, 0.95]^6 空间中随机采样。

        这是唯一能突破原始专家凸包的策略，确保虚拟专家覆盖
        原始专家未曾触及的认知区域，从而显著提升相空间多样性。
        """
        if count <= 0:
            return []

        rng = random.Random(46)
        dim_names = ['逻辑性', '新颖性', '深度', '分歧度', '具体性', '情感']
        results = []

        for k in range(count):
            # 完全随机向量（均匀采样全相空间）
            virtual_vec = np.array([rng.uniform(0.05, 0.95) for _ in range(6)])

            # 部分探索点与原始专家分布边界对齐，但向外延伸
            if rng.random() < 0.3:
                # 在对角线方向向外延伸
                src = rng.randint(0, self.n_real - 1)
                orig = self.real_vectors[src].vector
                direction = virtual_vec - orig
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                step = rng.uniform(0.5, 2.0)
                virtual_vec = orig + direction * step
                virtual_vec = np.clip(virtual_vec, 0.05, 0.95)

            # 标记主导维度
            dominant = np.argmax(virtual_vec)
            results.append({
                "player_name": f"虚拟-探索{k+1}",
                "speech": "",
                "key_insight": f"探索{dim_names[dominant]}",
                "_vector": virtual_vec.copy(),
            })

        return results

    def _vector_to_speech(self, vector: np.ndarray) -> str:
        """
        将相空间向量映射为高质量合成发言。

        生成结构化多段发言：开场白 → 核心论证 → 深度延伸 → 结论
        每个维度从多变体模板中随机选取，确保发言多样性和连贯性。
        """
        coherence, novelty, depth, divergence, specificity, emotional = vector
        rng = random.Random(int(abs(np.sum(vector) * 10000)) % 2**31)

        def _pick(key: str, threshold_high: float = 0.6,
                  threshold_mid: float = 0.35) -> str:
            val = {
                'coherence': coherence, 'novelty': novelty,
                'depth': depth, 'divergence': divergence,
                'specificity': specificity, 'emotional': emotional,
            }.get(key, 0.5)

            if val > threshold_high:
                hk = f"high_{key}"
            elif val > threshold_mid:
                hk = f"mid_{key}"
            else:
                hk = f"low_{key}"

            templates = self._SPEECH_TEMPLATES.get(hk, [])
            if isinstance(templates, list) and templates:
                return rng.choice(templates)
            return ""

        # ── 构建结构化发言 ──
        opening = rng.choice(self._OPENINGS)
        main_body = _pick('coherence', 0.6, 0.35)
        novelty_part = _pick('novelty', 0.65, 0.35)

        # 过渡 + 深度延伸
        transition1 = rng.choice(self._TRANSITIONS)
        depth_part = _pick('depth', 0.6, 0.35)

        # 分歧/共识视角
        transition2 = rng.choice(self._TRANSITIONS)
        divergence_part = _pick('divergence', 0.6, 0.3)

        # 具体性补充
        specificity_part = _pick('specificity', 0.6, 0.35)

        # 情感结尾（仅当情感维度显著时添加）
        emotional_part = ""
        if emotional > 0.6:
            emotional_part = _pick('emotional', 0.65, 0.35, )

        # 组装发言
        parts = [opening + main_body]
        if novelty_part:
            parts.append(novelty_part)
        if depth_part:
            parts.append(transition1 + depth_part)
        if divergence_part:
            parts.append(transition2 + divergence_part)
        if specificity_part:
            parts.append(specificity_part)
        if emotional_part:
            parts.append(emotional_part)

        return " ".join(parts)

    def get_all_discussions(self) -> list:
        """返回所有专家（真实+虚拟）的讨论列表"""
        return self.real + self.virtual_discussions

    @property
    def virtual_vectors(self) -> np.ndarray:
        """
        返回所有虚拟专家的相空间向量矩阵 (M, 6)。

        用于神经认知反馈闭环中的距离计算与 Hebbian 更新。
        """
        vecs = []
        for d in self.virtual_discussions:
            if '_vector' in d and isinstance(d['_vector'], np.ndarray):
                vecs.append(d['_vector'])
            else:
                vecs.append(OpinionPhaseVector(d.get('speech', ''), d.get('player_name', '')).vector)
        return np.array(vecs, dtype=np.float64) if vecs else np.empty((0, 6))

    @property
    def amplification_ratio(self) -> float:
        """放大比例：虚拟专家数 / 真实专家数"""
        total = self.n_real + len(self.virtual_discussions)
        return total / max(1, self.n_real)


# ═══════════════════════════════════════════════════════════════
# 8. 时间维度耦合记忆（让引擎在时间中演化）
# ═══════════════════════════════════════════════════════════════

class TemporalCouplingMemory:
    """
    时间维度耦合记忆——让相变引擎在时间中演化。

    核心机制：
    1. 累积耦合：每轮观点交互后更新耦合矩阵，形成"认知记忆"
    2. 连接净化：弱连接随时间衰减，强连接被强化（突触可塑性）
    3. 拓扑重构：根据交互频率和相似度演化，自动调整连接结构
    4. 时间衰减：未被强化的连接随时间指数衰减（遗忘曲线）

    物理直觉：
    - 大脑的突触可塑性：经常同时激活的神经元之间连接增强
    - 遗忘曲线：长期不用的连接自然衰减
    - 修剪机制：冗余连接被剪除，保留高效连接
    """

    def __init__(self, n_experts_max: int = 200, decay_rate: float = 0.15,
                 prune_threshold: float = 0.05, hebbian_strength: float = 0.1):
        self.n_max = n_experts_max
        self.decay_rate = decay_rate          # 时间衰减率（遗忘速度）
        self.prune_threshold = prune_threshold  # 剪枝阈值
        self.hebbian_strength = hebbian_strength  # Hebbian 强化系数
        self.round = 0

        # 累积耦合矩阵 (n_max × n_max)：跨轮次的"认知记忆"
        self.cumulative_coupling = np.zeros((n_experts_max, n_experts_max))
        # 交互频率矩阵（每对专家互动次数）
        self.interaction_frequency = np.zeros((n_experts_max, n_experts_max), dtype=int)
        # 连接年龄（记录连接存在了多少轮未被强化）
        self.connection_age = np.zeros((n_experts_max, n_experts_max), dtype=int)
        # 拓扑演化历史记录
        self.topology_history = []
        # 连接质量评分（基于历史交互的加权评分）
        self.connection_quality = np.zeros((n_experts_max, n_experts_max))
        # ── 虚拟专家权重矩阵 (n_max × 6)：存储虚拟专家在相空间中的位置 ──
        # 用于神经认知反馈闭环的 Hebbian 累积更新
        self.virtual_weight_matrix = np.zeros((n_experts_max, 6))

    def update(self, phase_vectors: list, round_count: int):
        """
        用本轮专家观点更新耦合记忆。

        流程：
        1. 计算本轮即时耦合矩阵
        2. 时间衰减：所有旧连接按遗忘曲线衰减
        3. 融合新耦合：新交互注入累积记忆
        4. Hebbian 强化：高频交互对连接增强
        5. 连接净化：剪枝弱连接，淘汰冗余
        6. 记录拓扑状态
        """
        n = len(phase_vectors)
        self.round = round_count

        # 1. 计算本轮即时耦合
        current_coupling = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = phase_vectors[i].similarity(phase_vectors[j])
                coupling = math.tanh(2.5 * (sim - 0.3))
                current_coupling[i, j] = coupling

        # 2. 时间衰减（遗忘曲线）：所有旧连接衰减
        #    衰减因子 = e^(-decay_rate * delta_t)
        decay_factor = math.exp(-self.decay_rate)
        self.cumulative_coupling[:n, :n] *= decay_factor
        self.connection_age[:n, :n] += 1

        # 3. 融合本轮耦合（新信息注入）
        #    新耦合以 0.3 的学习率融入累积记忆
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.cumulative_coupling[i, j] += current_coupling[i, j] * 0.3
                self.interaction_frequency[i, j] += 1
                self.connection_age[i, j] = 0  # 重置年龄（被强化了）

        # 4. Hebbian 强化：高频交互对连接增强
        #    经常同时说话/互评的专家之间连接更强
        max_freq = max(1, np.max(self.interaction_frequency[:n, :n]))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                freq_ratio = self.interaction_frequency[i, j] / max_freq
                if freq_ratio > 0.3:
                    # 高频交互对 → 连接强化
                    self.cumulative_coupling[i, j] *= (1.0 + self.hebbian_strength * freq_ratio)
                    # 更新连接质量评分
                    self.connection_quality[i, j] = min(1.0,
                        self.connection_quality[i, j] + 0.1 * freq_ratio)

        # 5. 连接净化（自动剪枝 + 淘汰）
        self._purify(n)

        # 6. 记录拓扑状态
        self._record_topology(n)

    def _purify(self, n: int):
        """
        自动净化连接网络。

        三个净化操作：
        a) 弱连接剪枝：低于阈值的连接直接归零
        b) 强连接强化：高强度连接进一步突触增强
        c) 老连接淘汰：超过 5 轮未被强化的连接大幅衰减
        """
        # a) 弱连接剪枝
        weak_mask = np.abs(self.cumulative_coupling[:n, :n]) < self.prune_threshold
        self.cumulative_coupling[:n, :n][weak_mask] = 0.0
        self.connection_quality[:n, :n][weak_mask] = 0.0

        # b) 强连接强化（突触可塑性）
        strong_mask = np.abs(self.cumulative_coupling[:n, :n]) > 0.5
        self.cumulative_coupling[:n, :n][strong_mask] *= 1.05  # 5% 强化
        if np.any(strong_mask):
            self.connection_quality[:n, :n][strong_mask] = np.minimum(
                self.connection_quality[:n, :n][strong_mask] + 0.05, 1.0)

        # c) 老连接淘汰（遗忘）
        aged_mask = self.connection_age[:n, :n] > 5
        self.cumulative_coupling[:n, :n][aged_mask] *= 0.3  # 大幅衰减
        self.connection_quality[:n, :n][aged_mask] *= 0.5

        # 确保对角线为零
        for i in range(n):
            self.cumulative_coupling[i, i] = 0.0

    def get_coupling_matrix(self, n: int) -> np.ndarray:
        """获取当前 n×n 大小的累积耦合矩阵"""
        return self.cumulative_coupling[:n, :n].copy()

    def get_network_stats(self, n: int) -> dict:
        """
        获取当前网络统计信息。
        用于生成拓扑状态报告。
        """
        matrix = self.cumulative_coupling[:n, :n]
        active = np.sum(np.abs(matrix) > self.prune_threshold)
        total = n * (n - 1)
        density = active / total if total > 0 else 0
        pos = np.sum(matrix > self.prune_threshold)
        neg = np.sum(matrix < -self.prune_threshold)

        return {
            "round": self.round,
            "n_experts": n,
            "active_connections": int(active),
            "connection_density": round(density, 4),
            "positive_connections": int(pos),
            "negative_connections": int(neg),
            "avg_strength": round(float(np.mean(np.abs(matrix[matrix != 0]))) if np.any(matrix != 0) else 0, 4),
            "avg_quality": round(float(np.mean(self.connection_quality[:n, :n])), 4),
            "purification_ratio": round(1.0 - density, 4),
        }

    def _record_topology(self, n: int):
        """记录拓扑状态到历史"""
        stats = self.get_network_stats(n)
        self.topology_history.append(stats)
        # 保留最近 50 轮历史
        if len(self.topology_history) > 50:
            self.topology_history = self.topology_history[-50:]

    def get_topology_trend(self) -> str:
        """
        拓扑演化趋势分析。
        返回拓扑变化方向的文本描述。
        """
        if len(self.topology_history) < 3:
            return "— 积累中（需≥3轮）"

        recent = self.topology_history[-3:]
        density_trend = [r["connection_density"] for r in recent]
        quality_trend = [r["avg_quality"] for r in recent]

        # 密度趋势
        if density_trend[-1] > density_trend[0] * 1.1:
            density_dir = "↑ 连接密度上升（网络生长）"
        elif density_trend[-1] < density_trend[0] * 0.9:
            density_dir = "↓ 连接密度下降（网络修剪）"
        else:
            density_dir = "— 连接密度稳定"

        # 质量趋势
        if quality_trend[-1] > quality_trend[0] * 1.05:
            quality_dir = "↑ 连接质量提升（突触强化）"
        elif quality_trend[-1] < quality_trend[0] * 0.95:
            quality_dir = "↓ 连接质量下降（突触衰退）"
        else:
            quality_dir = "— 质量稳定"

        return f"{density_dir} | {quality_dir}"

    # ────────────────────────────────────────────────────────────
    # 虚拟专家权重矩阵（神经认知反馈闭环支持）
    # ────────────────────────────────────────────────────────────

    def update_virtual_weights(self, virtual_vectors: np.ndarray):
        """
        存储更新后的虚拟专家相空间向量。

        参数:
            virtual_vectors: (M, 6) numpy array，虚拟专家在相空间中的位置
        """
        n = min(len(virtual_vectors), self.n_max)
        self.virtual_weight_matrix[:n] = virtual_vectors[:n]

    def get_virtual_weights(self, n: int = None) -> np.ndarray:
        """
        获取存储的虚拟专家向量矩阵。

        参数:
            n: 返回前 n 个虚拟专家向量（默认全部）
        """
        if n is None:
            n = self.n_max
        return self.virtual_weight_matrix[:n].copy()

    def to_dict(self) -> dict:
        """
        序列化时间记忆为可 JSON 序列化的字典。

        用于保存到检查点，实现跨对话的连续性。
        """
        return {
            "n_max": self.n_max,
            "decay_rate": self.decay_rate,
            "prune_threshold": self.prune_threshold,
            "hebbian_strength": self.hebbian_strength,
            "round": self.round,
            "cumulative_coupling": self.cumulative_coupling.tolist(),
            "interaction_frequency": self.interaction_frequency.tolist(),
            "connection_age": self.connection_age.tolist(),
            "connection_quality": self.connection_quality.tolist(),
            "topology_history": self.topology_history,
            "virtual_weight_matrix": self.virtual_weight_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TemporalCouplingMemory':
        """从字典恢复时间记忆实例"""
        tm = cls(
            n_experts_max=data.get("n_max", 200),
            decay_rate=data.get("decay_rate", 0.15),
            prune_threshold=data.get("prune_threshold", 0.05),
            hebbian_strength=data.get("hebbian_strength", 0.1),
        )
        tm.round = data.get("round", 0)
        tm.cumulative_coupling = np.array(data.get("cumulative_coupling", [[0]]))
        tm.interaction_frequency = np.array(data.get("interaction_frequency", [[0]]), dtype=int)
        tm.connection_age = np.array(data.get("connection_age", [[0]]), dtype=int)
        tm.connection_quality = np.array(data.get("connection_quality", [[0]]))
        tm.topology_history = data.get("topology_history", [])
        vwm = data.get("virtual_weight_matrix", None)
        if vwm:
            arr = np.array(vwm)
            # 确保矩阵大小与 n_max 一致，截断或填充
            if arr.shape[0] <= tm.n_max:
                tm.virtual_weight_matrix[:arr.shape[0]] = arr
            else:
                tm.virtual_weight_matrix = arr[:tm.n_max]
        return tm


# ═══════════════════════════════════════════════════════════════
# 9. 相变拓扑引擎主类（增强版）
# ═══════════════════════════════════════════════════════════════

class PhaseTransitionEngine:
    """
    相变拓扑引擎主类。

    整合所有子模块，提供统一的涌现计算接口。
    这是整个系统的核心计算单元。

    涌现层级（5 级非线性阶跃）：
    Level 0: 直接综合（线性保留）
    Level 1: 交叉耦合综合（非线性耦合矩阵 + 吸引子收敛）
    Level 2: 序参量涌现（临界慢化检测 + 相变触发）
    Level 3: 自组织临界综合（沙堆模型 + 沙崩涌现）
    Level 4: 量子叠加与混沌边缘（叠加态坍缩 + 深度质变）
    """

    LEVELS = {
        0: "直接综合（线性保留）",
        1: "交叉耦合综合（非线性耦合矩阵）",
        2: "序参量涌现（相变触发）",
        3: "自组织临界综合（沙崩涌现）",
        4: "量子叠加与混沌边缘（深度质变）",
    }

    LEVEL_DESCRIPTIONS = {
        0: "观点直接拼接，保持原有行为",
        1: "通过非线性耦合矩阵检测观点间的相互作用，产生交叉视角",
        2: "序参量接近临界点，临界慢化效应触发认知相变",
        3: "沙堆模型达到临界状态，大规模沙崩（涌现）自然发生",
        4: "量子叠加态坍缩，混沌边缘的深度质变，产生真正原创性洞见",
    }

    def __init__(self, round_discussions: list, essence_pool, round_count: int,
                 amplification_ratio: float = 1.0, diversity_index: float = 0.0,
                 is_amplified: bool = False,
                 precomputed_vectors: list = None,
                 temporal_memory: 'TemporalCouplingMemory' = None):
        """
        参数:
            round_discussions: 讨论数据
            essence_pool: 精华池
            round_count: 轮次
            amplification_ratio: 虚拟专家放大比例（1.0 = 无放大，10.0 = 10倍放大）
            diversity_index: 相空间多样性指数（0~1）
            is_amplified: 是否经过虚拟专家放大
            precomputed_vectors: 预计算的相空间向量列表（可选）。
                当提供时，直接用这些向量构建耦合矩阵，而非从 speech 文本重新嵌入。
                这避免了虚拟专家文本→向量往返的信息损失。
            temporal_memory: 时间维度耦合记忆（可选）。
                提供时，引擎的耦合矩阵会融合历史累积的耦合信息，
                实现跨轮次的知识累积和连接净化。
        """
        self.discussions = round_discussions
        self.essence_pool = essence_pool
        self.round_count = round_count
        self.n_experts = len(round_discussions)
        self.amplification_ratio = max(1.0, amplification_ratio)
        self.diversity_index = max(0.0, min(1.0, diversity_index))
        self.is_amplified = is_amplified
        self.temporal_memory = temporal_memory  # 时间维度耦合记忆

        # 构建相空间
        # 如果提供预计算向量，直接使用（避免虚拟专家文本→向量往返的信息损失）
        if precomputed_vectors is not None and len(precomputed_vectors) == len(round_discussions):
            self.phase_vectors = precomputed_vectors
        else:
            self.phase_vectors = [
                OpinionPhaseVector(d.get('speech', ''), d.get('player_name', ''))
                for d in round_discussions
            ]

        # 构建耦合矩阵
        self.coupling = NonLinearCouplingMatrix(self.phase_vectors)

        # ── 时间维度融合：将历史累积耦合注入当前耦合矩阵 ──
        # 这使引擎的"观点相互作用"受到跨轮次记忆的影响
        if temporal_memory is not None:
            n = self.n_experts
            hist_coupling = temporal_memory.get_coupling_matrix(n)
            if np.any(hist_coupling != 0):
                # 历史耦合以 0.4 的权重融合进当前耦合
                # 融合后重新计算特征值谱
                self.coupling.matrix = (
                    0.6 * self.coupling.matrix +
                    0.4 * hist_coupling
                )
                self.coupling.eigenvalues = self.coupling._compute_eigenvalues()
                # 更新耦合直方图
                pairs = [self.coupling.matrix[i, j]
                         for i in range(n) for j in range(n) if i != j]
                self.coupling.coupling_histogram = Counter()
                for c in pairs:
                    self.coupling.coupling_histogram[int(c * 10) / 10] += 1

        # 序参量动力学（放大模式下更多模拟步数）
        self.order_param = OrderParameterDynamics(self.coupling)
        if is_amplified:
            # 放大模式：步数随专家数非线性增长，确保大系统有足够统计采样
            self._metropolis_steps = max(30, int(self.n_experts * 0.5))
        else:
            self._metropolis_steps = max(15, int(self.n_experts * 0.3))

        # 自组织临界性（放大模式下更大网格）
        grid_sz = max(3, int(math.sqrt(self.n_experts)) + 2)
        if is_amplified:
            # 放大模式下，沙堆网格更大，更易触发沙崩
            grid_sz = max(5, int(math.sqrt(self.n_experts * 2)) + 3)
        self.soc = SelfOrganizedCriticality(grid_size=grid_sz)

        # 混沌边缘检测
        self.chaos = EdgeOfChaosDetector(self.coupling)

        # 量子叠加
        self.quantum = QuantumSuperposition(self.phase_vectors)

        # 缓存计算结果
        self._metrics_cache = None
        self._level_cache = None

    @staticmethod
    def _compute_diversity_index(vectors: list) -> float:
        """
        计算相空间覆盖多样性指数（0~1）。

        测量专家观点在相空间中的分布广度。
        高多样性 = 相空间覆盖充分，这是涌现综合的前提。
        """
        pvs = vectors
        if len(pvs) < 2:
            return 0.0

        arr = np.array([pv.vector for pv in pvs])
        # 每维度的方差
        variances = np.var(arr, axis=0)
        # 理论最大方差：均匀分布在 [0,1] 时为 1/12 ≈ 0.083
        max_var = 0.0833
        # 归一化
        dim_diversity = np.minimum(variances / max_var, 1.0)
        # 加权平均（分歧度和新颖性权重更高）
        weights = np.array([0.15, 0.25, 0.15, 0.25, 0.10, 0.10])
        score = float(np.average(dim_diversity, weights=weights))
        return min(1.0, score)

    def compute_emergence_level(self) -> int:
        """
        计算当前的涌现层级（0~4）。

        基于多维非线性判定，而非简单的加权平均。

        判定逻辑树：
        1. 基础条件不足 → Level 0
        2. 少量专家/精华 → Level 1
        3. 相变临界（谱半径+慢化）→ Level 2
        4. 自组织临界（沙堆）→ Level 3
        5. 混沌边缘 + 相变 + 叠加 → Level 4
        """
        # 如果已缓存，直接返回
        if self._level_cache is not None:
            return self._level_cache

        # 基本条件检查
        if self.n_experts < 2:
            self._metrics_cache = {
                "n_experts": self.n_experts, "n_phase_dimensions": 6,
                "spectral_radius": 0.0, "eigenvalue_gap": 0.0, "frustration_index": 0.0,
                "is_critical": False, "criticality_trend": "稳定", "dominant_clusters": "",
                "order_parameter": 0.0, "fluctuation": 0.0, "critical_slowing_down": 0.0,
                "soc_criticality": 0.0, "soc_avalanches": 0, "soc_is_at_criticality": False,
                "lyapunov_exponent": 0.0, "at_edge_of_chaos": False, "chaos_regime": "未知",
                "entanglement_entropy": 0.0, "superposition_depth": 0.0,
                "n_essences": 0, "avg_essence_score": 0.0, "round_count": self.round_count,
            }
            return 0

        # 精华池规模
        n_essences = len(self.essence_pool.items) if self.essence_pool and hasattr(self.essence_pool, 'items') else 0
        avg_score = 0.0
        if n_essences > 0:
            avg_score = sum(item.score for item in self.essence_pool.items) / n_essences

        # ====== 计算所有指标 ======

        # 1. 耦合矩阵谱半径
        spectral = self.coupling.spectral_radius

        # 2. 阻挫指数
        frustration = self.coupling.frustration_index()

        # 3. 序参量模拟（放大模式下步数更多，注入热噪声）
        order_result = self.order_param.simulate(steps=self._metropolis_steps, is_amplified=self.is_amplified)
        order_param = order_result["order_parameter"]
        slowing_down = order_result["critical_slowing_down"]

        # 涨落：大系统直接使用多样性指数，小系统使用模拟结果
        # 物理直觉：相空间多样性直接反映系统的"热涨落"程度
        diversity = self.diversity_index if self.diversity_index > 0 else PhaseTransitionEngine._compute_diversity_index(self.phase_vectors)
        if self.is_amplified or self.n_experts >= 20:
            # 大系统：序参量模拟因统计稳定性而低估涨落，
            # 改用相空间多样性作为涨落指标
            fluctuation = diversity * 0.3  # 多样性按比例映射到涨落
        else:
            fluctuation = order_result["fluctuation"]

        # 4. 沙堆模型（放大模式下投入更多沙粒）
        n_sand_grains = min(self.n_experts, 10)
        if self.is_amplified:
            # 放大模式：每个真实专家投入更多沙粒，模拟更多专家观点
            n_sand_grains = min(self.n_experts * 2, 30)
        for i in range(n_sand_grains):
            idx = min(i, len(self.phase_vectors) - 1)
            energy = float(np.linalg.norm(self.phase_vectors[idx].vector))
            self.soc.add_grain(energy=max(0.5, energy))
        criticality = self.soc.criticality_level
        soc_active = self.soc.is_at_criticality

        # 5. 混沌指数（放大模式下多次平均）
        lyapunov = self.chaos.estimate_lyapunov()
        at_edge = self.chaos.at_edge_of_chaos

        # 6. 量子叠加
        entanglement = self.quantum.entanglement_entropy()
        superposition = self.quantum.superposition_depth()

        # ====== 放大因子：虚拟专家带来的非线性增强 ======
        # 放大比例越高，各条件阈值越容易被满足
        amp_factor = min(3.0, self.amplification_ratio / 3.0)  # 3x 封顶

        # ====== 多维度综合判定（放大感知） ======

        # 条件 A: 相变临界条件（放大后阈值更宽松）
        # 放大模式：更多专家 → 更易进入临界态
        spectral_threshold = 0.12 / max(1.0, amp_factor * 0.5)
        slowing_threshold = 0.35 / max(1.0, amp_factor * 0.3)
        fluct_threshold = 0.04 / max(1.0, amp_factor * 0.3)
        phase_transition_ready = (
            spectral > spectral_threshold
            and slowing_down > slowing_threshold
            and fluctuation > fluct_threshold
        )

        # 条件 B: 自组织临界条件（放大后沙堆更易临界）
        soc_crit_threshold = max(0.15, 0.45 / max(1.0, amp_factor * 0.4))
        soc_ready = (criticality > soc_crit_threshold or soc_active) and n_essences >= 6

        # 条件 C: 混沌边缘条件（放大后纠缠+叠加要求降低）
        entangle_threshold = max(0.2, 0.4 / max(1.0, amp_factor * 0.3))
        superpos_threshold = max(0.15, 0.3 / max(1.0, amp_factor * 0.3))
        chaos_ready = at_edge and entanglement > entangle_threshold and superposition > superpos_threshold

        # 条件 D: 基本耦合条件
        basic_ready = self.n_experts >= 3 and n_essences >= 2

        # 条件 E: 深度综合条件（放大后阻挫要求降低）
        frust_threshold = max(0.08, 0.2 / max(1.0, amp_factor * 0.3))
        deep_ready = frustration > frust_threshold and self.round_count >= 4

        # 条件 F: 多样性条件（放大模式下，高多样性直接提升潜力）
        diversity_ready = diversity > 0.35

        # 条件 G: 人口规模条件（放大模式下，大量虚拟专家直接提升阶跃潜力）
        pop_ready = self.n_experts >= 20

        # ====== 层级判定（放大感知、多样性驱动） ======
        if not basic_ready:
            level = 0
        elif self.n_experts < 4 or n_essences < 4:
            level = 1
        # Level 4：混沌边缘 + 相变 + 多样性 + 深度
        elif chaos_ready and phase_transition_ready and deep_ready and (diversity_ready or pop_ready):
            level = 4
        # Level 3：自组织临界 + 相变 + 放大
        elif soc_ready and phase_transition_ready:
            level = 3
        # Level 2：相变临界 + 深度（或人口规模达到 20+）
        elif phase_transition_ready and (deep_ready or pop_ready):
            level = 2
        # Level 3（备选）：自组织临界单独触发
        elif soc_ready:
            level = 3
        # Level 2（备选）：深度综合 + 多样性
        elif deep_ready and diversity_ready:
            level = 2
        # Level 1：默认
        else:
            level = 1

        self._level_cache = level
        # 同时缓存度量指标，避免 emergence_metrics 重复计算
        self._metrics_cache = {
            "n_experts": self.n_experts,
            "n_phase_dimensions": 6,
            "spectral_radius": round(spectral, 3),
            "eigenvalue_gap": round(self.coupling.eigenvalue_gap, 3),
            "frustration_index": round(frustration, 3),
            "is_critical": self.coupling.is_critical,
            "criticality_trend": self.coupling.criticality_trend,
            "dominant_clusters": self._describe_clusters(),
            "order_parameter": round(order_param, 3),
            "fluctuation": round(fluctuation, 3),
            "critical_slowing_down": round(slowing_down, 3),
            "soc_criticality": round(criticality, 3),
            "soc_avalanches": len(self.soc.avalanche_sizes),
            "soc_is_at_criticality": soc_active,
            "lyapunov_exponent": round(lyapunov, 3),
            "at_edge_of_chaos": at_edge,
            "chaos_regime": self.chaos.chaos_regime,
            "entanglement_entropy": round(entanglement, 3),
            "superposition_depth": round(superposition, 3),
            "n_essences": n_essences,
            "avg_essence_score": round(avg_score, 3) if n_essences > 0 else 0.0,
            "round_count": self.round_count,
            "diversity_index": round(diversity, 3),
            "amplification_ratio": round(self.amplification_ratio, 1),
            "is_amplified": self.is_amplified,
        }
        return level

    def _describe_clusters(self) -> str:
        """生成聚类描述文本（用于 metrics）"""
        try:
            cluster_info = self.coupling.dominant_clusters()
            descs = []
            for ci, cl in enumerate(cluster_info[:3]):
                names = []
                for i in cl:
                    pv = self.phase_vectors[i]
                    names.append(pv.player_name[:4] if pv.player_name else str(i))
                descs.append(f"簇{ci+1}: [{', '.join(names)}]")
            return "; ".join(descs)
        except Exception:
            return ""

    def recursive_emergence(self, llm_client, model_name: str,
                              problem: str, real_discussions: list,
                              caller_tag: str = "递归涌现") -> str:
        """
        递归涌现综合：将引擎的输出重新注入，实现多层涌现。

        流程：
        1. 计算当前涌现层级
        2. 在当前层级进行综合
        3. 将综合结果作为"虚拟专家观点"加入讨论
        4. 重新计算涌现层级（通常更高）
        5. 重复直到层级不再提升

        这模拟了"认知的螺旋上升"——每个层次的涌现
        都成为下一层次涌现的输入。
        """
        from .prompts_b64 import _get_b64_prompt

        current_level = self.compute_emergence_level()
        max_level = current_level
        all_responses = []
        enhanced_discussions = list(real_discussions)

        for depth in range(3):  # 最多递归 3 层
            if current_level < 1:
                break

            # 在当前层级进行综合
            level_response = synthesize_with_emergence(
                problem=problem,
                round_discussions=enhanced_discussions,
                essence_pool=self.essence_pool,
                round_count=self.round_count + depth,
                llm_client=llm_client,
                model_name=model_name,
                caller_tag=f"{caller_tag}-递归{depth+1}",
            )

            if not level_response:
                break

            all_responses.append(level_response)

            # 将当前输出作为新的"虚拟专家观点"加入
            enhanced_discussions.append({
                "player_name": f"递归涌现-层{depth+1}",
                "speech": level_response,
                "key_insight": f"第{depth+1}层递归涌现输出",
            })

            # 用增强后的讨论重新计算涌现层级
            new_engine = PhaseTransitionEngine(
                enhanced_discussions, self.essence_pool,
                self.round_count + depth + 1,
                amplification_ratio=self.amplification_ratio,
                diversity_index=self.diversity_index,
                is_amplified=self.is_amplified,
            )
            new_level = new_engine.compute_emergence_level()

            if new_level <= current_level:
                # 层级不再提升，停止递归
                break

            current_level = new_level
            if current_level > max_level:
                max_level = current_level

        # 返回所有递归输出的综合
        if len(all_responses) > 1:
            combined = "\n\n".join(all_responses)
            final_prompt = (
                "你是一个统一的意识体。以下是对同一问题的多层次涌现综合结果：\n\n"
                f"{combined}\n\n"
                f"请将这些层次整合为一个统一的、深刻的回答。"
                f"直接输出你的最终回答，不要提及讨论过程："
            )
            try:
                final_response, _ = llm_client.chat(
                    [{"role": "user", "content": final_prompt}],
                    model=model_name,
                    thinking="disabled",
                    caller=f"{caller_tag}-递归整合",
                    show_reasoning=False, show_answer=False,
                )
                return final_response.strip() if final_response else all_responses[-1]
            except Exception:
                return all_responses[-1]

        return all_responses[-1] if all_responses else ""

    @property
    def emergence_metrics(self) -> dict:
        """
        返回完整的涌现度量指标，用于调试和可视化。

        如果 compute_emergence_level() 已缓存了结果，直接返回。
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        # 尚未缓存时，先计算层级（会同时缓存 metrics）
        self.compute_emergence_level()
        return self._metrics_cache or {}

    # ════════════════════════════════════════════════════════════
    # 神经认知反馈闭环（Neural Cognitive Feedback Loop）
    # ════════════════════════════════════════════════════════════

    def hebbian_update(self, real_vectors: np.ndarray,
                        virtual_vectors: np.ndarray,
                        learning_rate: float = 0.05) -> np.ndarray:
        """
        Hebbian 更新虚拟专家向量（神经认知反馈闭环 Step 3）。

        对每个真实发言向量 p_real，计算其与所有虚拟专家向量的距离，
        通过 sigmoid 激活，然后对激活强度 > 0.3 的虚拟专家进行 Hebbian 更新。

        参数:
            real_vectors: (N, 6) numpy array，真实专家相空间向量
            virtual_vectors: (M, 6) numpy array，虚拟专家相空间向量
            learning_rate: Hebbian 学习率（默认 0.05，防止漂移过快）

        返回:
            updated_vectors: (M, 6) 更新后的虚拟专家向量矩阵
        """
        if len(virtual_vectors) == 0 or len(real_vectors) == 0:
            return virtual_vectors

        D_max = 2.0
        updated = virtual_vectors.copy()

        # 对每个真实发言向量计算激活并更新
        for p_real in real_vectors:
            # Step 2: 计算距离与 sigmoid 激活
            distances = np.linalg.norm(updated - p_real, axis=1)  # (M,)
            alpha = 1.0 / (1.0 + np.exp(3.0 * (distances / D_max - 0.5)))  # (M,)

            # Step 3: Hebbian 更新（只对激活的虚拟专家）
            active_mask = alpha > 0.3
            if np.any(active_mask):
                # ΔW_j = learning_rate * alpha_j * (p_real - W_j)
                delta = learning_rate * alpha[active_mask, np.newaxis] * (p_real - updated[active_mask])
                updated[active_mask] += delta
                # 重新归一化到 [0, 1] 区间（保持相空间边界）
                updated[active_mask] = np.clip(updated[active_mask], 0.0, 1.0)

        # 存储到 temporal_memory
        if self.temporal_memory is not None:
            self.temporal_memory.update_virtual_weights(updated)

        return updated

    def compute_collective_response(self, real_vectors: np.ndarray,
                                      virtual_vectors: np.ndarray,
                                      mix_coefficient: float = 0.3) -> np.ndarray:
        """
        计算虚拟专家网络的集体认知响应（封装 Step 2-4）。

        流程:
          Step 2: 真实发言 → 激活虚拟专家（距离计算 + sigmoid 激活）
          Step 3: Hebbian 调参（更新虚拟专家在相空间中的位置）
          Step 4: 加权响应生成与汇总（注意力机制 → P_final）

        参数:
            real_vectors: (N, 6) numpy array，真实专家相空间向量
            virtual_vectors: (M, 6) numpy array，虚拟专家相空间向量
            mix_coefficient: 混合系数（默认 0.3）

        返回:
            P_final: (6,) 六维集体认知响应向量
        """
        if len(virtual_vectors) == 0 or len(real_vectors) == 0:
            return np.zeros(6)

        # Step 2+3: Hebbian 更新
        updated_virtual = self.hebbian_update(real_vectors, virtual_vectors)

        N = len(real_vectors)
        M = len(updated_virtual)
        D_max = 2.0

        # Step 2 (续): 对每个真实发言计算激活向量
        all_alphas = np.zeros((N, M))
        for i, p_real in enumerate(real_vectors):
            distances = np.linalg.norm(updated_virtual - p_real, axis=1)
            all_alphas[i] = 1.0 / (1.0 + np.exp(3.0 * (distances / D_max - 0.5)))

        # 多个真实发言时，按时序衰减加权（越近的发言权重越高）
        if N > 1:
            time_weights = np.array([0.5 + 0.5 * (i / (N - 1)) for i in range(N)])
            time_weights = time_weights / time_weights.sum()
            alpha = np.average(all_alphas, axis=0, weights=time_weights)
        else:
            alpha = all_alphas[0]

        # 真实发言的平均向量作为"集体认知中心"
        p_real_avg = np.mean(real_vectors, axis=0)

        # Step 4: 加权响应生成
        # R_j = W_j + alpha_j * (p_real_avg - W_j) * mix_coefficient
        R = np.zeros_like(updated_virtual)
        for j in range(M):
            R[j] = updated_virtual[j] + alpha[j] * (p_real_avg - updated_virtual[j]) * mix_coefficient

        # 注意力权重：beta_j = alpha_j * exp(similarity * 2) / sum
        p_norm = np.linalg.norm(p_real_avg)
        v_norms = np.linalg.norm(updated_virtual, axis=1)
        similarities = np.zeros(M)
        for j in range(M):
            if p_norm > 1e-10 and v_norms[j] > 1e-10:
                similarities[j] = np.dot(p_real_avg, updated_virtual[j]) / (p_norm * v_norms[j])

        beta_raw = alpha * np.exp(similarities * 2)
        beta_sum = np.sum(beta_raw)
        if beta_sum > 1e-10:
            beta = beta_raw / beta_sum
        else:
            beta = np.ones(M) / M

        # 最终汇总：P_final = Σ_j beta_j * R_j
        P_final = np.sum(beta[:, np.newaxis] * R, axis=0)
        P_final = np.clip(P_final, 0.0, 1.0)

        return P_final


# ═══════════════════════════════════════════════════════════════
# 8. 旧版兼容函数（保留原有 API 签名）
# ═══════════════════════════════════════════════════════════════

def _compress_speech(speech: str, max_chars: int = 80) -> str:
    """
    压缩发言文本到关键句，大幅降低 token 消耗。

    保留第一句（通常是核心论点），如果太长则截断到 max_chars。
    如果发言很短（< 20 字）则原样保留，避免信息丢失。
    """
    if not speech:
        return speech
    if len(speech) <= max_chars:
        return speech
    # 尝试按句号/问号/感叹号/分号切出第一句
    for sep in ('。', '？', '！', '；', '.\n', '?\n', '!\n'):
        idx = speech.find(sep)
        if 20 <= idx <= max_chars:
            return speech[:idx + 1]
    # 没找到合适的句末，直接截断
    return speech[:max_chars].rstrip('，, ') + '…'


def _truncate_response(response: str, max_chars: int = 200) -> str:
    """截断整合意识回复到 max_chars 字以内，超出时末尾加…"""
    if not response:
        return response
    text = response.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip('，, ') + '…'


def _emit_init_neuron_map(event_callback: callable, real_discussions: list,
                          all_vectors: list) -> None:
    """
    构建神经元点阵图的初始化事件并发送。

    所有节点（真实专家 + 虚拟专家代表）基于相空间余弦相似度
    建立拓扑连接，形成高维神经网络。
    """
    if not event_callback:
        return

    # 节点：真实专家
    nodes_vectors = []
    nodes_labels = []
    nodes_kinds = []
    n_real = len(real_discussions)
    for i, d in enumerate(real_discussions):
        vec = OpinionPhaseVector(d.get('speech', ''), d.get('player_name', '')).vector
        nodes_vectors.append(vec.tolist())
        nodes_labels.append(d.get('player_name', f'专家{i}'))
        nodes_kinds.append('real')

    # 采样虚拟专家代表（与真实专家统一拓扑连接）
    if all_vectors:
        import random as _rng
        _rng.seed(42)
        n_rep = min(20, len(all_vectors))
        sample_idx = _rng.sample(range(len(all_vectors)), n_rep)
        for idx in sample_idx:
            nodes_vectors.append(all_vectors[idx].vector.tolist())
            nodes_labels.append(f'V{idx}')
            nodes_kinds.append('rep')

    # ── 连线：真实专家之间 + 真实专家到采样神经元 ──
    # 真实专家之间全连接，每个真实专家再连到 2 个虚拟代表
    n_total = len(nodes_vectors)
    edges = []
    if n_real >= 2:
        for i in range(n_real):
            for j in range(i + 1, n_real):
                edges.append([i, j, 1.0])
    # 每个真实专家连到 2 个采样神经元
    n_rep = max(0, n_total - n_real)
    for i in range(min(n_real, 4)):
        for k in range(2):
            if n_rep > 0:
                j = n_real + (i * 2 + k) % n_rep
                edges.append([i, j, 0.7])

    # 云点下采样（UDP 数据报限 64KB，最多传 250 个背景点）
    _cloud = []
    if all_vectors:
        step = max(1, len(all_vectors) // 250)
        _cloud = [v.vector.tolist() for v in all_vectors[::step]][:250]

    event_callback({
        "type": "init",
        "all_vectors": _cloud,
        "nodes": {
            "vectors": nodes_vectors,
            "labels": nodes_labels,
            "kinds": nodes_kinds,
        },
        "edges": edges,
    })


def _calc_emergence_potential(essence_pool, expert_count: int, round_count: int) -> float:
    """
    （兼容）计算涌现势能。

    现在使用 PhaseTransitionEngine 代替简单的加权求和。
    """
    dummy_discussions = [{"player_name": f"专家{i}", "speech": "观点", "key_insight": ""}
                         for i in range(max(1, expert_count))]
    engine = PhaseTransitionEngine(dummy_discussions, essence_pool, round_count)
    level = engine.compute_emergence_level()
    # 将层级映射回 0~1 势能
    return min(1.0, (level + 1) * 0.2)


def get_emergence_level(essence_pool, expert_count: int, round_count: int) -> int:
    """
    （兼容）根据涌现势能决定综合层级。

    现在使用 PhaseTransitionEngine 的多维非线性判定。
    """
    dummy_discussions = [{"player_name": f"专家{i}", "speech": "观点", "key_insight": ""}
                         for i in range(max(1, expert_count))]
    engine = PhaseTransitionEngine(dummy_discussions, essence_pool, round_count)
    return engine.compute_emergence_level()


def _build_cross_critique_prompt(expert_opinions: list) -> str:
    """构建交叉审视 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n观点: {op['speech']}\n核心洞见: {op.get('key_insight', '无')}"
        for op in expert_opinions
    )
    base = _get_b64_prompt("emergence_cross_critique")
    return base.replace("{opinions_text}", opinions_text)


def _build_meta_synthesis_prompt(problem: str, expert_opinions: list,
                                  cross_critique: str, essence_summary: str) -> str:
    """构建元综合 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    base = _get_b64_prompt("emergence_meta_synthesis")
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    return base


def _build_emergence_synthesis_prompt(problem: str, expert_opinions: list,
                                       cross_critique: str, essence_summary: str) -> str:
    """构建涌现综合 prompt（从 prompts_b64.py 读取加密模板）"""
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    n = len(expert_opinions)
    base = _get_b64_prompt("emergence_emergence_synthesis")
    base = base.replace("{n}", str(n))
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    return base


def _build_soc_synthesis_prompt(problem: str, expert_opinions: list,
                                 cross_critique: str, essence_summary: str,
                                 metrics: dict) -> str:
    """
    构建自组织临界综合 prompt（Level 3）。

    利用沙堆模型的沙崩动力学信息来指导综合。
    """
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    n = len(expert_opinions)

    soc_info = (
        f"当前系统处于自组织临界状态。\n"
        f"沙崩规模序列: {metrics.get('soc_avalanches', 0)} 次沙崩\n"
        f"临界程度: {metrics.get('soc_criticality', 0):.2f}\n"
        f"谱半径: {metrics.get('spectral_radius', 0):.2f}\n"
        f"序参量涨落: {metrics.get('fluctuation', 0):.3f}\n"
        f"混沌状态: {metrics.get('chaos_regime', '未知')}\n"
    )

    base = _get_b64_prompt("emergence_soc_synthesis")
    base = base.replace("{n}", str(n))
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    base = base.replace("{soc_info}", soc_info)
    return base


def _build_quantum_synthesis_prompt(problem: str, expert_opinions: list,
                                     cross_critique: str, essence_summary: str,
                                     metrics: dict) -> str:
    """
    构建量子叠加综合 prompt（Level 4）。

    利用量子叠加态和混沌边缘的信息来指导最深层次的综合。
    """
    opinions_text = "\n\n".join(
        f"【{op['player_name']}】\n{op['speech']}"
        for op in expert_opinions
    )
    n = len(expert_opinions)

    quantum_info = (
        f"系统已达到混沌边缘，量子叠加态已形成。\n"
        f"纠缠熵: {metrics.get('entanglement_entropy', 0):.3f}\n"
        f"叠加深度: {metrics.get('superposition_depth', 0):.3f}\n"
        f"李雅普诺夫指数: {metrics.get('lyapunov_exponent', 0):.3f}\n"
        f"混沌状态: {metrics.get('chaos_regime', '未知')}\n"
        f"耦合谱半径: {metrics.get('spectral_radius', 0):.2f}\n"
        f"阻挫指数: {metrics.get('frustration_index', 0):.2f}\n"
    )

    base = _get_b64_prompt("emergence_quantum_synthesis")
    base = base.replace("{n}", str(n))
    base = base.replace("{problem}", problem)
    base = base.replace("{opinions_text}", opinions_text)
    base = base.replace("{cross_critique}", cross_critique)
    base = base.replace("{essence_summary}", essence_summary)
    base = base.replace("{quantum_info}", quantum_info)
    return base


# ═══════════════════════════════════════════════════════════════
# 9. 核心合成函数（保留原有 API 签名）
# ═══════════════════════════════════════════════════════════════

def synthesize_with_emergence(problem: str, round_discussions: list,
                                essence_pool, round_count: int,
                                llm_client, model_name: str,
                                caller_tag: str = "涌现综合",
                                target_experts: int = 2000,
                                event_callback: callable = None,
                                current_round_pairs: list = None,
                                temporal_memory: 'TemporalCouplingMemory' = None) -> str:
    """
    使用相变拓扑引擎进行综合的核心函数（超级相变引擎）。

    10人→100人效果的核心机制：
    1. VirtualExpertGenerator 从 N 个真实专家生成 100 个虚拟专家
    2. PhaseTransitionEngine 在 100 人规模的相空间上计算涌现层级
    3. 放大比例（amplification_ratio = 10）使阈值降低，更容易触发高阶涌现
    4. 多样性指数（diversity_index）确保相空间覆盖充分
    5. 递归涌现（recursive_emergence）将输出重新注入，实现认知螺旋上升

    参数：
      round_discussions: [{"player_name", "speech", "key_insight"}, ...]
      essence_pool: EssencePool 实例
      round_count: 当前轮次
      llm_client: LLMClient 实例
      model_name: 模型名
      target_experts: 目标专家数（含虚拟专家），默认 100
      event_callback: 可选，接收推理过程事件 dict（用于神经元点阵图实时显示）
      temporal_memory: 可选，时间维度耦合记忆。提供时引擎会融合历史累积的
          耦合信息，并在综合后自动更新记忆，实现跨轮次的知识累积和连接净化。

    返回：
      str: 综合后的统一回复
    """
    # 事件发射辅助（供神经元点阵图实时显示）
    def _emit(event: dict):
        if event_callback:
            try:
                event_callback(event)
            except Exception:
                pass

    # 使用虚拟专家生成器将 10 人放大为 100 人效果
    n_real = len(round_discussions)
    use_amplification = n_real >= 3 and n_real < target_experts

    if use_amplification:
        _emit({"type": "status", "text": f"虚拟专家生成中：{n_real} 个真实专家 → {target_experts} 个神经元..."})
        generator = VirtualExpertGenerator(round_discussions, target_experts=target_experts)
        amplified_discussions = generator.get_all_discussions()
        amp_ratio = generator.amplification_ratio
        # 计算多样性指数
        all_vectors = [
            OpinionPhaseVector(d.get('speech', ''), d.get('player_name', ''))
            for d in amplified_discussions
        ]
        div_index = PhaseTransitionEngine._compute_diversity_index(all_vectors)
        # 从虚拟专家中选出 20 个代表性"神经元"注入 LLM
        neuron_experts = generator.select_neuron_representatives(n=20)
        # 压缩神经元发言（只保留第一句/关键句），大幅降低 token 消耗
        for nd in neuron_experts:
            nd['speech'] = _compress_speech(nd.get('speech', ''))
    else:
        amplified_discussions = round_discussions
        amp_ratio = 1.0
        div_index = 0.0
        neuron_experts = round_discussions
        for nd in neuron_experts:
            nd['speech'] = _compress_speech(nd.get('speech', ''))
        # 非放大模式也构建相空间向量用于可视化
        all_vectors = [
            OpinionPhaseVector(d.get('speech', ''), d.get('player_name', ''))
            for d in round_discussions
        ]

    # ── 推送神经元点阵图初始化数据（始终发送，不限放大模式） ──
    try:
        _emit_init_neuron_map(event_callback, round_discussions, all_vectors)
    except Exception as e:
        import sys
        print(f"[神经图] 初始化推送失败: {e}", file=sys.stderr)

    # ── 发射本轮讨论信号（专家间的信息传递） ──
    if event_callback and current_round_pairs:
        for from_i, to_j, text in current_round_pairs:
            try:
                event_callback({"type": "signal", "from": from_i, "to": to_j, "text": text[:40]})
            except Exception:
                pass
        # 同时发送信号缓冲区，供神经元点阵图在合成期间持续重放
        try:
            event_callback({
                "type": "signal_buffer",
                "signals": [
                    {"from": s[0], "to": s[1], "text": s[2][:40]}
                    for s in current_round_pairs
                ],
            })
        except Exception:
            pass

    # 使用相变拓扑引擎（放大版）计算涌现层级
    engine = PhaseTransitionEngine(
        amplified_discussions, essence_pool, round_count,
        amplification_ratio=amp_ratio,
        diversity_index=div_index,
        is_amplified=use_amplification,
        temporal_memory=temporal_memory,
    )
    level = engine.compute_emergence_level()
    metrics = engine.emergence_metrics

    # ── 更新时间维度耦合记忆 ──
    # 将本轮真实专家的观点向量注入记忆，使引擎在时间中演化
    if temporal_memory is not None:
        try:
            temporal_memory.update(engine.phase_vectors, round_count)
        except Exception:
            pass

    # 推送涌现层级事件
    _emit({
        "type": "phase",
        "text": f"涌现层级判定: Level {level}",
        "level": level,
    })
    # 高亮代表性神经元
    _emit({
        "type": "highlight",
        "nodes": list(range(min(8, len(round_discussions)))),
    })

    # ── 神经认知反馈闭环：真实发言 → 相空间映射 → 虚拟网络调参 → 加权输出 ──
    # Step 1: 提取真实专家的六维向量
    real_vectors = np.array([
        OpinionPhaseVector(d.get('speech', ''), d.get('player_name', '')).vector
        for d in round_discussions
    ])
    # Step 2-4: 通过虚拟专家网络计算集体认知响应
    P_final = np.zeros(6)
    if use_amplification and hasattr(generator, 'virtual_vectors'):
        virt_vecs = generator.virtual_vectors
        if len(virt_vecs) > 0:
            try:
                P_final = engine.compute_collective_response(real_vectors, virt_vecs)
            except Exception:
                # fallback: 使用真实发言的平均向量
                P_final = np.mean(real_vectors, axis=0) if len(real_vectors) > 0 else np.zeros(6)
    else:
        P_final = np.mean(real_vectors, axis=0) if len(real_vectors) > 0 else np.zeros(6)

    # Step 5: 将六维向量转为自然语言认知方向描述
    dim_names = ['逻辑一致性', '新颖性', '认知深度', '分歧度', '具体性', '情感强度']
    dim_descriptions = [f"{dim_names[i]}: {P_final[i]:.2f}" for i in range(6)]
    cognitive_orientation = (
        f"\n\n【当前认知重心】\n"
        f"基于虚拟专家网络的集体调参，当前认知重心偏向：\n"
        + ", ".join(dim_descriptions) +
        f"\n请在上述认知方向上生成回复，确保回复与该方向一致。"
    )
    # 调试输出
    print(f"[认知反馈] P_final = [{', '.join(f'{v:.3f}' for v in P_final)}]")

    # ── 推送更新后的虚拟专家拓扑到神经元点阵图 ──
    if use_amplification and hasattr(generator, 'virtual_vectors'):
        try:
            virt_vecs = generator.virtual_vectors
            if len(virt_vecs) > 0:
                # 获取更新后的虚拟专家向量（从 temporal_memory 或 engine）
                updated_virt = virt_vecs.copy()
                if temporal_memory is not None:
                    stored = temporal_memory.get_virtual_weights(len(virt_vecs))
                    if np.any(stored):
                        updated_virt = stored
                # 构建虚拟专家云点（用于可视化更新）
                _cloud = [v.tolist() for v in updated_virt[::max(1, len(updated_virt)//250)]][:250]
                # 计算 P_final 在相空间中的位置作为"认知重心"节点
                _emit({
                    "type": "cognitive_center",
                    "vector": P_final.tolist(),
                    "all_vectors": _cloud,
                })
        except Exception:
            pass

    # 精华池摘要
    essence_summary = "（空）"
    if essence_pool and hasattr(essence_pool, 'items') and essence_pool.items:
        essence_summary = essence_pool.get_pool_summary(top_n=5)

    # Level 0: 直接综合（线性，保持原有行为）
    if level == 0:
        discussion_text = "\n\n".join(
            f"【{d['player_name']}】\n{d['speech']}"
            for d in neuron_experts
        )
        prompt = (
            f"你是一个统一的意识体。以下是对同一问题的内部讨论记录。\n\n"
            f"用户问: {problem}\n\n"
            f"内部讨论记录:\n{discussion_text}\n\n"
            f"请直接给出你的统一回复（一段话，不要分段太多，不要提及子模块或讨论过程，就是你自己在回答）。"
            f"\n\n请简短回答，控制在200字以内，精炼有力。"
            f"{cognitive_orientation}"
        )
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": prompt}],
                model=model_name,
                thinking="disabled",
                caller=caller_tag,
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 1: 交叉耦合综合（非线性耦合矩阵）
    if level == 1:
        _emit({"type": "phase", "text": "L1 交叉耦合综合：非线性耦合矩阵 + 交叉审视", "level": 1})
        _emit({"type": "signal", "from": 0, "to": 1, "text": "耦合矩阵构建"})
        # 第一步：交叉审视
        critique_prompt = _build_cross_critique_prompt(neuron_experts)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-交叉审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：基于交叉审视的元综合
        _emit({"type": "signal", "from": 1, "to": 0, "text": "元综合输出"})
        synth_prompt = _build_meta_synthesis_prompt(
            problem, neuron_experts, critique_result, essence_summary
        )
        synth_prompt += "\n\n请简短回答，控制在200字以内，精炼有力。"
        synth_prompt += cognitive_orientation
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-元综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 2: 序参量涌现（相变触发）
    if level == 2:
        _emit({"type": "phase", "text": "L2 序参量涌现：临界慢化检测 + 相变触发", "level": 2})
        _emit({"type": "signal", "from": 0, "to": 2, "text": "临界慢化检测"})
        # 第一步：深度交叉审视
        critique_prompt = _build_cross_critique_prompt(neuron_experts)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-深度交叉审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：涌现综合（相变级）
        _emit({"type": "signal", "from": 2, "to": 1, "text": "相变触发 → 涌现综合"})
        synth_prompt = _build_emergence_synthesis_prompt(
            problem, neuron_experts, critique_result, essence_summary
        )
        synth_prompt += "\n\n请简短回答，控制在200字以内，精炼有力。"
        synth_prompt += cognitive_orientation
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-涌现综合",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 3: 自组织临界综合（沙崩涌现）
    if level == 3:
        _emit({"type": "phase", "text": "L3 自组织临界：沙堆模型 + 沙崩涌现", "level": 3})
        _emit({"type": "signal", "from": 1, "to": 3, "text": "沙崩传播中..."})
        # 第一步：深度交叉审视
        critique_prompt = _build_cross_critique_prompt(neuron_experts)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-沙崩审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：自组织临界综合
        _emit({"type": "signal", "from": 3, "to": 0, "text": "临界涌现完成"})
        synth_prompt = _build_soc_synthesis_prompt(
            problem, neuron_experts, critique_result, essence_summary, metrics
        )
        synth_prompt += "\n\n请简短回答，控制在200字以内，精炼有力。"
        synth_prompt += cognitive_orientation
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-自组织临界涌现",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    # Level 4: 量子叠加与混沌边缘（深度质变）
    if level == 4:
        _emit({"type": "phase", "text": "L4 量子叠加：叠加态坍缩 + 混沌边缘", "level": 4})
        _emit({"type": "signal", "from": 2, "to": 3, "text": "量子干涉建立"})
        # 第一步：量子干涉态分析
        critique_prompt = _build_cross_critique_prompt(neuron_experts)
        try:
            critique_result, _ = llm_client.chat(
                [{"role": "user", "content": critique_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-量子干涉分析",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            critique_result = ""

        # 第二步：递归深度交叉审视（二次审视）
        _emit({"type": "signal", "from": 3, "to": 2, "text": "递归元审视"})
        second_critique = ""
        try:
            second_prompt = (
                f"以下是第一轮交叉审视的结果：\n\n{critique_result}\n\n"
                f"请进行第二轮元审视：\n"
                f"1. 第一轮审视自身是否存在盲点？\n"
                f"2. 哪些矛盾观点实际上可以量子叠加共存？\n"
                f"3. 在混沌边缘，哪些看似无关的洞见实际上高度相关？\n\n"
                f"请输出你的元审视分析："
            )
            second_critique, _ = llm_client.chat(
                [{"role": "user", "content": second_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-递归元审视",
                show_reasoning=False, show_answer=False,
            )
        except Exception:
            second_critique = ""

        # 合并两轮审视
        combined_critique = f"【第一轮交叉审视】\n{critique_result}\n\n【第二轮递归元审视】\n{second_critique}"

        # 第三步：量子叠加综合
        _emit({"type": "signal", "from": 0, "to": 3, "text": "叠加态坍缩 · 深度质变"})
        synth_prompt = _build_quantum_synthesis_prompt(
            problem, neuron_experts, combined_critique, essence_summary, metrics
        )
        synth_prompt += "\n\n请简短回答，控制在200字以内，精炼有力。"
        synth_prompt += cognitive_orientation
        try:
            response, _ = llm_client.chat(
                [{"role": "user", "content": synth_prompt}],
                model=model_name,
                thinking="disabled",
                caller=f"{caller_tag}-量子叠加深度涌现",
                show_reasoning=False, show_answer=False,
            )
            return response.strip() if response else ""
        except Exception:
            return ""

    return ""


# ═══════════════════════════════════════════════════════════════
# 10. 解决方案综合函数（保留原有 API 签名）
# ═══════════════════════════════════════════════════════════════

def synthesize_solution_with_emergence(problem: str, all_essences_text: str,
                                         evolution_history: str,
                                         discussion_mode: str,
                                         essence_pool, round_count: int,
                                         players: list,
                                         llm_client, model_name: str) -> dict:
    """
    使用相变拓扑引擎生成最终综合解决方案。

    现在使用 PhaseTransitionEngine 的多维非线性判定，
    支持 5 级涌现层级。

    改进：
    - 收集所有存活专家的方案
    - 使用相变拓扑引擎计算涌现层级
    - 多层次涌现综合
    """
    # 收集所有存活专家的方案
    all_solutions = []
    for player in players:
        if not player.alive:
            continue
        try:
            result, _ = player.synthesize_solution(
                problem=problem,
                all_essences=all_essences_text,
                evolution_history=evolution_history,
                discussion_mode=discussion_mode,
            )
            if result and result.get("solution_title"):
                all_solutions.append(result)
        except Exception:
            pass

    if not all_solutions:
        return {
            "solution_title": "综合解决方案",
            "summary": "基于多轮讨论的综合方案",
            "core_ideas": [],
            "key_insights": [],
            "divergence_points": [],
            "final_conclusion": "讨论结束，综合各方观点形成最终方案",
        }

    # 使用相变拓扑引擎计算涌现层级（超级相变引擎）
    n_alive = sum(1 for p in players if p.alive)
    # 构建虚拟讨论用于引擎计算
    dummy_discussions = [
        {"player_name": s.get("solution_title", f"方案{i}"), "speech": s.get("summary", ""), "key_insight": ""}
        for i, s in enumerate(all_solutions)
    ]

    # 10人→100人：虚拟专家放大
    n_real = len(dummy_discussions)
    target_experts = max(2000, n_real * 5)
    if n_real >= 2 and n_real < target_experts:
        generator = VirtualExpertGenerator(dummy_discussions, target_experts=target_experts)
        amp_discussions = generator.get_all_discussions()
        amp_ratio = generator.amplification_ratio
        all_vectors = [OpinionPhaseVector(d.get('speech', ''), d.get('player_name', '')) for d in amp_discussions]
        div_index = PhaseTransitionEngine._compute_diversity_index(all_vectors)
        is_amp = True
    else:
        amp_discussions = dummy_discussions
        amp_ratio = 1.0
        div_index = 0.0
        is_amp = False

    engine = PhaseTransitionEngine(
        amp_discussions, essence_pool, round_count,
        amplification_ratio=amp_ratio,
        diversity_index=div_index,
        is_amplified=is_amp,
    )
    level = engine.compute_emergence_level()
    metrics = engine.emergence_metrics

    # Level 0: 直接返回质量最高的方案
    if level == 0:
        best = max(all_solutions, key=lambda s: (
            len(s.get("core_ideas", [])) +
            len(s.get("key_insights", []))
        ))
        return best

    # 构建所有方案的文本
    solutions_text = "\n\n".join(
        f"【{s.get('solution_title', '未命名方案')}】\n"
        f"摘要: {s.get('summary', '')}\n"
        f"核心思想: {'; '.join(s.get('core_ideas', []))}\n"
        f"关键洞见: {'; '.join(s.get('key_insights', []))}\n"
        f"分歧点: {'; '.join(s.get('divergence_points', []))}\n"
        f"最终结论: {s.get('final_conclusion', '')}\n"
        for s in all_solutions
    )

    import json, re

    # Level 1: 交叉审视 + 元综合
    if level == 1:
        base = _get_b64_prompt("emergence_solution_level1")
        cross_prompt = base.replace("{solutions_text}", solutions_text)
        try:
            content, _ = llm_client.chat(
                [{"role": "user", "content": cross_prompt}],
                model=model_name,
                thinking="disabled",
                caller="涌现综合-元综合方案",
                show_reasoning=False, show_answer=False,
            )
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("solution_title"):
                    return result
        except Exception:
            pass
        return max(all_solutions, key=lambda s: len(s.get("core_ideas", [])) + len(s.get("key_insights", [])))

    # Level 2: 涌现综合（相变级）
    if level == 2:
        n_essences = len(essence_pool.items) if essence_pool and hasattr(essence_pool, 'items') else 0
        base = _get_b64_prompt("emergence_solution_level2")
        cross_prompt = base.replace("{solutions_text}", solutions_text)
        cross_prompt = cross_prompt.replace("{n_essences}", str(n_essences))
        cross_prompt = cross_prompt.replace("{round_count}", str(round_count))
        try:
            content, _ = llm_client.chat(
                [{"role": "user", "content": cross_prompt}],
                model=model_name,
                thinking="disabled",
                caller="涌现综合-相变级方案",
                show_reasoning=False, show_answer=False,
            )
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("solution_title"):
                    return result
        except Exception:
            pass
        return max(all_solutions, key=lambda s: len(s.get("core_ideas", [])) + len(s.get("key_insights", [])))

    # Level 3+: 深度涌现综合（沙崩/量子级）
    n_essences = len(essence_pool.items) if essence_pool and hasattr(essence_pool, 'items') else 0
    metrics_str = (
        f"系统度量:\n"
        f"- 谱半径: {metrics.get('spectral_radius', '?'):.2f}\n"
        f"- 序参量: {metrics.get('order_parameter', '?'):.2f}\n"
        f"- 临界慢化: {metrics.get('critical_slowing_down', '?'):.2f}\n"
        f"- 临界程度: {metrics.get('soc_criticality', '?'):.2f}\n"
        f"- 李雅普诺夫指数: {metrics.get('lyapunov_exponent', '?'):.2f}\n"
        f"- 纠缠熵: {metrics.get('entanglement_entropy', '?'):.2f}\n"
        f"- 混沌状态: {metrics.get('chaos_regime', '未知')}\n"
    )

    deep_prompt = (
        f"以下是多位专家对同一问题提出的综合方案：\n\n{solutions_text}\n\n"
        f"精华池中包含 {n_essences} 条精华，经过 {round_count} 轮讨论。\n\n"
        f"{metrics_str}\n\n"
        f"【涌现综合指令——深度涌现级】\n"
        f"系统已达到临界状态，多个方案之间存在量子叠加态。请执行：\n"
        f"1. 【相变识别】这些方案碰撞中，哪些是「量变」，哪些是「质变」？\n"
        f"2. 【涌现特性】提炼出任何单个方案都无法单独得出的涌现性洞见\n"
        f"3. 【辩证统一】将分歧视为更高层次统一的驱动力\n"
        f"4. 【混沌边缘洞见】在混沌边缘，哪些看似矛盾的方案实际上可以互补共存？\n\n"
        f"请以结构化 JSON 输出：\n"
        '{\n'
        '  "solution_title": "综合方案标题",\n'
        '  "summary": "方案摘要（200字以内）",\n'
        '  "core_ideas": ["核心思想列表"],\n'
        '  "key_insights": ["关键洞见列表"],\n'
        '  "divergence_points": ["融合后的分歧点"],\n'
        '  "consciousness_emergence": "涌现性认知——从量变到质变的关键洞见",\n'
        '  "quantum_insight": "混沌边缘的量子洞见——矛盾观点的统一",\n'
        '  "final_conclusion": "最终结论"\n'
        '}'
    )
    try:
        content, _ = llm_client.chat(
            [{"role": "user", "content": deep_prompt}],
            model=model_name,
            thinking="disabled",
            caller="涌现综合-深度涌现级方案",
            show_reasoning=False, show_answer=False,
        )
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("solution_title"):
                return result
    except Exception:
        pass

    return max(all_solutions, key=lambda s: len(s.get("core_ideas", [])) + len(s.get("key_insights", [])))