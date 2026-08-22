"""Quick test for the new Phase Transition Topology Engine"""
import sys
sys.path.insert(0, '.')
from emergence import PhaseTransitionEngine, get_emergence_level, _calc_emergence_potential


class MockItem:
    score = 0.7


class MockPool:
    items = [MockItem() for _ in range(10)]

    def get_pool_summary(self, top_n=5):
        return "test summary"


def main():
    print("=== Phase Transition Topology Engine v2.0 Test ===\n")

    # Test 1: Basic engine creation
    engine = PhaseTransitionEngine(
        [{"player_name": "A", "speech": "我认为应该从系统论角度思考这个问题，因为复杂系统需要整体性视角。", "key_insight": "sys"},
         {"player_name": "B", "speech": "但是局部优化也很重要，这是辩证关系。", "key_insight": "local"},
         {"player_name": "C", "speech": "本质上这是一个悖论，需要动态平衡而非静态取舍。", "key_insight": "balance"},
         {"player_name": "D", "speech": "数据表明在实际案例中，关键在于找到耦合点。", "key_insight": "coupling"}],
        MockPool(), 5
    )

    level = engine.compute_emergence_level()
    print(f"Level: {level} - {PhaseTransitionEngine.LEVELS[level]}")

    metrics = engine.emergence_metrics
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Test 2: Compatibility
    print("\n--- Compatibility ---")
    print(f"get_emergence_level: {get_emergence_level(MockPool(), 4, 5)}")
    print(f"potential: {_calc_emergence_potential(MockPool(), 4, 5):.3f}")

    # Test 3: Scenarios
    print("\n--- Scenarios ---")
    scenarios = [
        ("small", [{"player_name": "A", "speech": "x", "key_insight": ""}
                   for _ in range(2)], MockPool(), 1),
        ("medium", [{"player_name": f"E{i}", "speech": f"这是一个中等长度的观点包含逻辑推理{i}", "key_insight": ""}
                    for i in range(5)], MockPool(), 3),
        ("large", [{"player_name": f"E{i}", "speech": f"本质上这是复杂系统问题因为整体和局部之间存在辩证关系从数据来看我们需要找到耦合点{i}", "key_insight": ""}
                   for i in range(8)], MockPool(), 8),
    ]
    for name, disc, pool, rnd in scenarios:
        e = PhaseTransitionEngine(disc, pool, rnd)
        lvl = e.compute_emergence_level()
        m = e.emergence_metrics
        print(f"  [{name}] L={lvl} SR={m['spectral_radius']:.3f} "
              f"Fluct={m['fluctuation']:.3f} Slow={m['critical_slowing_down']:.3f} "
              f"SOC={m['soc_criticality']:.3f} Lyap={m['lyapunov_exponent']:.3f} "
              f"Regime={m['chaos_regime']}")

    # Test 4: Phase vectors
    print("\n--- Phase Vectors ---")
    for pv in engine.phase_vectors:
        vec_str = ", ".join(f"{v:.2f}" for v in pv.vector)
        print(f"  {pv.player_name}: [{vec_str}]")
        print(f"    energy={pv.energy:.2f}")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()