"""
Run hedging experiments, risk metrics, American put comparison, and figures.

Usage (from repo root):
    python scripts/run_experiments.py

Writes dark-theme PNGs to reports/figures/ (tuned for GitHub dark README).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.binomial import crr_american_put, crr_put
from src.hedging import simulate_delta_hedge
from src.risk import expected_shortfall, var_abs

FIGURES = ROOT / "reports" / "figures"

# GitHub-dark friendly palette
BG = "#0d1117"
AX_BG = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#ffa657"
ACCENT_GREEN = "#3fb950"
ACCENT_PURPLE = "#d2a8ff"
HIST_A = "#58a6ff"
HIST_B = "#ff7b72"


def _apply_dark_style(ax: Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, which="both")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.85, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def _new_fig(figsize: tuple[float, float] = (8.0, 4.8)) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    _apply_dark_style(ax)
    return fig, ax


def _save(fig: Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )

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

    fig, ax = _new_fig()
    ax.plot(
        step_grid,
        stds_a,
        marker="o",
        color=ACCENT_BLUE,
        markerfacecolor=ACCENT_BLUE,
        markeredgecolor=BG,
        linewidth=2.0,
        markersize=6,
    )
    ax.set_xlabel("Rehedge steps per year")
    ax.set_ylabel("Std of hedging PnL")
    ax.set_title("More frequent delta-hedging shrinks residual PnL volatility")
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

    fig, ax = _new_fig()
    ax.plot(
        cost_grid,
        means_b,
        marker="o",
        color=ACCENT_ORANGE,
        markerfacecolor=ACCENT_ORANGE,
        markeredgecolor=BG,
        linewidth=2.0,
        markersize=6,
    )
    ax.set_xlabel("Proportional transaction cost (bps)")
    ax.set_ylabel("Mean hedging PnL")
    ax.set_title("Transaction costs drag mean hedge PnL downward")
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

    fig, ax = _new_fig()
    ax.hist(
        matched["pnl"],
        bins=50,
        alpha=0.65,
        density=True,
        label="sigma_hedge=0.2",
        color=HIST_A,
        edgecolor=BG,
        linewidth=0.3,
    )
    ax.hist(
        wrong["pnl"],
        bins=50,
        alpha=0.55,
        density=True,
        label="sigma_hedge=0.4",
        color=HIST_B,
        edgecolor=BG,
        linewidth=0.3,
    )
    ax.set_xlabel("Terminal hedging PnL")
    ax.set_ylabel("Density")
    ax.set_title("Wrong hedge volatility shifts / worsens PnL distribution")
    legend = ax.legend(frameon=True, fancybox=False)
    legend.get_frame().set_facecolor(AX_BG)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)
    _save(fig, "pnl_hist_vol_misspec.png")

    print("=== D) American vs European put (CRR) ===")
    n_tree = 300
    eu = crr_put(100, 100, 0.05, 0.2, 1.0, n_tree)
    am = crr_american_put(100, 100, 0.05, 0.2, 1.0, n_tree)
    print(f"  European put = {eu:.4f}")
    print(f"  American put = {am:.4f}")
    print(f"  Early-exercise premium = {am - eu:.4f}")

    fig, ax = _new_fig(figsize=(7.0, 4.6))
    bars = ax.bar(
        ["European put", "American put"],
        [eu, am],
        color=[ACCENT_PURPLE, ACCENT_GREEN],
        edgecolor=BG,
        width=0.55,
    )
    ax.set_ylabel("Option price")
    ax.set_title("American put premium from early exercise (CRR tree)")
    ax.set_ylim(0, max(am, eu) * 1.18)
    for bar, val in zip(bars, [eu, am]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=11,
        )
    _save(fig, "american_vs_european_put.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
