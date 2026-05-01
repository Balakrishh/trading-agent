"""Quick smoke test for RiskManager — exercises the new delta_aware_floor
branch on top of the existing static-floor path that test_risk_manager.py
already covers (without needing pytest)."""

import sys
import types

# Stubs identical to the chain-scanner runner
pmc = types.ModuleType("pandas_market_calendars")
class _C:
    def schedule(self, **k): pass
def _gc(name): return _C()
pmc.get_calendar = _gc
sys.modules["pandas_market_calendars"] = pmc
sm = types.ModuleType("scipy"); ss = types.ModuleType("scipy.stats")
ss.percentileofscore = lambda *a, **k: 50.0
sys.modules["scipy"] = sm; sys.modules["scipy.stats"] = ss

from trading_agent.risk_manager import RiskManager
from trading_agent.strategy import SpreadPlan, SpreadLeg


def make_plan(net_credit, width, max_loss, ratio, delta):
    return SpreadPlan(
        ticker="SPY", strategy_name="Bull Put Spread", regime="bullish",
        legs=[
            SpreadLeg("S1", 480.0, "sell", "put", delta, -0.05, 1.80, 2.00, 1.90),
            SpreadLeg("S2", 475.0, "buy",  "put", -0.10, -0.03, 0.50, 0.70, 0.60),
        ],
        spread_width=width, net_credit=net_credit, max_loss=max_loss,
        credit_to_width_ratio=ratio, expiration="2025-05-08", reasoning="t",
        valid=True,
    )


# Static floor (legacy path): min_credit_ratio gate
rm_static = RiskManager(max_risk_pct=0.02, min_credit_ratio=0.33, max_delta=0.25)

# Adaptive floor: |Δ|×(1+edge_buffer)
rm_adaptive = RiskManager(max_risk_pct=0.02, min_credit_ratio=0.33, max_delta=0.50,
                          delta_aware_floor=True, edge_buffer=0.10)

# 1) Plan with Δ-0.20 short, CW 0.23 (just above floor 0.22).
#    Static floor 0.33 → reject. Adaptive floor 0.20*1.10=0.22 → accept.
plan = make_plan(net_credit=1.15, width=5.0, max_loss=385.0, ratio=0.23, delta=-0.20)
v_s = rm_static.evaluate(plan, 100_000, "paper", True, False)
v_a = rm_adaptive.evaluate(plan, 100_000, "paper", True, False)
print(f"Static rm  on Δ-0.20/CW 0.23 → approved={v_s.approved} (failed: {v_s.checks_failed})")
print(f"Adaptive rm on Δ-0.20/CW 0.23 → approved={v_a.approved} (failed: {v_a.checks_failed})")

# 2) Below-floor adaptive — Δ-0.30 needs CW 0.33; offer 0.25 → reject.
plan2 = make_plan(net_credit=1.25, width=5.0, max_loss=375.0, ratio=0.25, delta=-0.30)
v_a2 = rm_adaptive.evaluate(plan2, 100_000, "paper", True, False)
print(f"Adaptive rm on Δ-0.30/CW 0.25 → approved={v_a2.approved}, failed reasons:")
for f in v_a2.checks_failed:
    print("  -", f)

# Pass criteria
fails = []
if v_s.approved:
    fails.append("static rm should reject CW 0.23 < 0.33 floor")
if not v_a.approved:
    # Could fail on max_delta etc; we only care that the C/W check itself passed.
    cw_failed = [f for f in v_a.checks_failed if "Credit/Width" in f]
    if cw_failed:
        fails.append(f"adaptive rm should accept CW 0.23 vs floor 0.22, got: {cw_failed}")
if v_a2.approved:
    fails.append("adaptive rm should reject CW 0.25 vs floor 0.33")

if fails:
    print("\nFAIL"); [print(" -", f) for f in fails]; sys.exit(1)
print("\nAll branches behave as designed.")
