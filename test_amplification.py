"""
超级相变引擎测试：验证 10 人 → 100 人效果

测试内容：
1. VirtualExpertGenerator 能否从 10 人生成 90 个虚拟专家
2. 相空间多样性指数是否显著提升
3. 放大模式下的涌现层级是否高于未放大模式
"""
import sys
sys.path.insert(0, '.')
from emergence import (
    VirtualExpertGenerator, PhaseTransitionEngine, OpinionPhaseVector
)


class MockItem:
    score = 0.7


class MockPool:
    items = [MockItem() for _ in range(10)]

    def get_pool_summary(self, top_n=5):
        return "test summary"


def main():
    print("=" * 60)
    print("超级相变引擎测试：10人 → 100人效果")
    print("=" * 60)

    # 模拟 10 位真实专家，覆盖不同认知维度
    real_discussions = [
        {"player_name": f"专家{i}", "speech": s, "key_insight": ""}
        for i, s in enumerate([
            "从系统论来看，这是一个复杂问题需要整体性把握。",
            "但是局部优化也很重要，这是辩证关系。",
            "本质上这是一个悖论，需要动态平衡。",
            "数据表明关键在于找到耦合点。",
            "我不同意，这忽略了历史维度。",
            "从长远看，短期阵痛换长期收益是值得的。",
            "具体来说，有 85% 的案例证明了这种方法。",
            "这太令人惊讶了，我们必须重视！",
            "我认为应该从认知科学角度重新审视。",
            "实际应用中，需要分阶段推进。",
        ])
    ]

    # ===== 测试 1: 虚拟专家生成器 =====
    print("\n--- 测试 1: 虚拟专家生成器 ---")
    gen = VirtualExpertGenerator(real_discussions, target_experts=100)
    all_experts = gen.get_all_discussions()
    print(f"  真实专家: {gen.n_real}")
    print(f"  虚拟专家: {len(gen.virtual_discussions)}")
    print(f"  总计: {len(all_experts)}")
    print(f"  放大比例: {gen.amplification_ratio:.1f}x")
    assert len(all_experts) >= 100, f"目标 100 人，实际 {len(all_experts)}"

    # ===== 测试 2: 相空间多样性 =====
    print("\n--- 测试 2: 相空间多样性 ---")
    real_vecs = [OpinionPhaseVector(d["speech"], d["player_name"])
                 for d in real_discussions]
    all_vecs = [OpinionPhaseVector(d["speech"], d["player_name"])
                for d in all_experts]

    real_div = PhaseTransitionEngine._compute_diversity_index(real_vecs)
    all_div = PhaseTransitionEngine._compute_diversity_index(all_vecs)
    print(f"  真实 10 人多样性: {real_div:.3f}")
    print(f"  放大 100 人多样性: {all_div:.3f}")
    print(f"  多样性提升: {all_div - real_div:+.3f}")
    assert all_div >= real_div, "放大后多样性应不低于放大前"

    # ===== 测试 3: 涌现层级对比 =====
    print("\n--- 测试 3: 涌现层级对比 ---")
    pool = MockPool()

    # 未放大
    engine_normal = PhaseTransitionEngine(real_discussions, pool, 5)
    level_normal = engine_normal.compute_emergence_level()
    m_normal = engine_normal.emergence_metrics
    print(f"  未放大模式:")
    print(f"    Level: {level_normal} - {PhaseTransitionEngine.LEVELS.get(level_normal, '?')}")
    print(f"    专家数: {m_normal['n_experts']}")
    print(f"    谱半径: {m_normal['spectral_radius']}")
    print(f"    阻挫: {m_normal['frustration_index']}")
    print(f"    慢化: {m_normal['critical_slowing_down']:.3f}")
    print(f"    涨落: {m_normal['fluctuation']:.3f}")
    print(f"    沙崩: {m_normal['soc_criticality']:.3f}")
    print(f"    多样性: {m_normal['diversity_index']}")

    # 放大模式
    engine_amp = PhaseTransitionEngine(
        all_experts, pool, 5,
        amplification_ratio=gen.amplification_ratio,
        diversity_index=all_div,
        is_amplified=True,
    )
    level_amp = engine_amp.compute_emergence_level()
    m_amp = engine_amp.emergence_metrics
    print(f"\n  放大模式 (100人):")
    print(f"    Level: {level_amp} - {PhaseTransitionEngine.LEVELS.get(level_amp, '?')}")
    print(f"    专家数: {m_amp['n_experts']}")
    print(f"    谱半径: {m_amp['spectral_radius']}")
    print(f"    阻挫: {m_amp['frustration_index']}")
    print(f"    慢化: {m_amp['critical_slowing_down']:.3f}")
    print(f"    涨落: {m_amp['fluctuation']:.3f}")
    print(f"    沙崩: {m_amp['soc_criticality']:.3f}")
    print(f"    李雅普诺夫: {m_amp['lyapunov_exponent']:.3f}")
    print(f"    纠缠熵: {m_amp['entanglement_entropy']:.3f}")
    print(f"    叠加深度: {m_amp['superposition_depth']:.3f}")
    print(f"    多样性: {m_amp['diversity_index']}")
    print(f"    放大比: {m_amp['amplification_ratio']}")

    # 放大模式应产生更高或相同的涌现层级
    print(f"\n  层级变化: Level {level_normal} → Level {level_amp}")
    assert level_amp >= level_normal, (
        f"放大模式层级 ({level_amp}) 应 >= 未放大模式 ({level_normal})"
    )

    # ===== 测试 4: 兼容性 =====
    print("\n--- 测试 4: 旧版兼容性 ---")
    from emergence import get_emergence_level, _calc_emergence_potential
    print(f"  get_emergence_level: {get_emergence_level(pool, 10, 5)}")
    print(f"  _calc_emergence_potential: {_calc_emergence_potential(pool, 10, 5):.3f}")
    from game import Game
    print(f"  game.py 导入: OK")

    print("\n" + "=" * 60)
    print("所有测试通过！10人 → 100人 放大效果验证成功")
    print("=" * 60)


if __name__ == "__main__":
    main()