"""
Run hedging experiments, risk metrics, American put comparison, and figures.

Usage (from repo root):
    python scripts/run_experiments.py

Writes PNGs to reports/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.binomial import crr_american_put, crr_put
from src.hedging import simulate_delta_hedge
from src.risk import expected_shortfall, var_abs

FIGURES = ROOT / "reports" / "figures"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def main() -> None:
    base = dict(s0=100.0, k=100.0, r=0.05, t=1.0, n_paths=5000, seed=42)

    print("=== A) Rehedge frequency (matched vol, no costs) ===")
    step_grid = sorted(set([12, 26, 52, 84, 126, 168, 210, 252] + list(range(12, 253, 12))))
    stds_a = []
    for n in step_grid:
        out = simulate_delta_hedge(
            **base,
            n_steps=n,
            sigma_real=0.2,
            sigma_hedge=0.2,
            trade_cost_bps=0.0,
        )
        stds_a.append(float(out["std_pnl"]))
        print(
            f"  n_steps={n:3d}  mean={out['mean_pnl']:+.4f}  "
            f"std={out['std_pnl']:.4f}  "
            f"VaR95={var_abs(out['pnl']):.4f}  ES95={expected_shortfall(out['pnl']):.4f}"
        )

    fig, ax = plt.subplots()
    ax.plot(step_grid, stds_a, marker="o")
    ax.set_xlabel("Rehedge steps per year")
    ax.set_ylabel("Std of hedging PnL")
    ax.set_title("More frequent delta-hedging shrinks residual PnL volatility")
    ax.grid(True, alpha=0.3)
    _save(fig, "std_vs_steps.png")

    print("=== B) Transaction costs (n_steps=100, matched vol) ===")
    cost_grid = [float(x) for x in range(0, 11)]
    means_b = []
    for cost in cost_grid:
        out = simulate_delta_hedge(
            **base,
            n_steps=100,
            sigma_real=0.2,
            sigma_hedge=0.2,
            trade_cost_bps=cost,
        )
        means_b.append(float(out["mean_pnl"]))
        print(
            f"  cost={cost:4.1f}bps  mean={out['mean_pnl']:+.4f}  "
            f"std={out['std_pnl']:.4f}  "
            f"VaR95={var_abs(out['pnl']):.4f}  ES95={expected_shortfall(out['pnl']):.4f}"
        )

    fig, ax = plt.subplots()
    ax.plot(cost_grid, means_b, marker="o", color="#b85c38")
    ax.set_xlabel("Proportional transaction cost (bps)")
    ax.set_ylabel("Mean hedging PnL")
    ax.set_title("Transaction costs drag mean hedge PnL downward")
    ax.grid(True, alpha=0.3)
    _save(fig, "mean_vs_costs.png")

    print("=== C) Hedge vol misspecification (n_steps=100, no costs) ===")
    matched = simulate_delta_hedge(
        **base, n_steps=100, sigma_real=0.2, sigma_hedge=0.2, trade_cost_bps=0.0
    )
    wrong = simulate_delta_hedge(
        **base, n_steps=100, sigma_real=0.2, sigma_hedge=0.4, trade_cost_bps=0.0
    )
    for label, out in [("hedge sigma=0.2", matched), ("hedge sigma=0.4", wrong)]:
        print(
            f"  {label}: mean={out['mean_pnl']:+.4f}  std={out['std_pnl']:.4f}  "
            f"VaR95={var_abs(out['pnl']):.4f}  ES95={expected_shortfall(out['pnl']):.4f}"
        )

    fig, ax = plt.subplots()
    ax.hist(matched["pnl"], bins=50, alpha=0.55, density=True, label="sigma_hedge=0.2")
    ax.hist(wrong["pnl"], bins=50, alpha=0.55, density=True, label="sigma_hedge=0.4")
    ax.set_xlabel("Terminal hedging PnL")
    ax.set_ylabel("Density")
    ax.set_title("Wrong hedge volatility shifts / worsens PnL distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "pnl_hist_vol_misspec.png")

    print("=== D) American vs European put (CRR) ===")
    n_tree = 300
    eu = crr_put(100, 100, 0.05, 0.2, 1.0, n_tree)
    am = crr_american_put(100, 100, 0.05, 0.2, 1.0, n_tree)
    print(f"  European put = {eu:.4f}")
    print(f"  American put = {am:.4f}")
    print(f"  Early-exercise premium = {am - eu:.4f}")

    fig, ax = plt.subplots()
    ax.bar(["European put", "American put"], [eu, am], color=["#4c6a92", "#2f9e6b"])
    ax.set_ylabel("Option price")
    ax.set_title("American put premium from early exercise (CRR tree)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "american_vs_european_put.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
