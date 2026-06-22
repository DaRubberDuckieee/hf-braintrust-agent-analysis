"""
Restyle graphs 02_1a, 03_1b, 07_1f with Braintrust brand colors.
Run from repo root: python scripts/restyle_graphs.py [02_1a|03_1b|07_1f|all]
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

# ── Braintrust palette ────────────────────────────────────────────────────────
BT_BLUE        = "#2C1FEB"   # BT Blue — solid bars, bubble fill
BT_BLUE_ACCENT = "#3A77EB"   # BT Blue Accent — mid-gradient
BT_ICE_GREY    = "#D8DEEE"   # BT Ice Grey — thin bars, grid, axes
BT_PINK        = "#F5AFD1"   # BT Pink — diamond markers
BT_GREEN       = "#14382D"   # BT Green — RELIABLE label
BT_ROSE        = "#651D31"   # BT Rose — ERRATIC label
BT_CONCRETE    = "#504F4F"   # BT Concrete — secondary text
BT_BLACK       = "#000000"   # BT Black — primary text

# Gradient colormap: ice grey → blue accent → BT blue
BT_CMAP = LinearSegmentedColormap.from_list(
    "bt_blue", [BT_ICE_GREY, BT_BLUE_ACCENT, BT_BLUE]
)

# Heatmap colormap: ice grey → blue accent → BT blue
BT_HEAT = LinearSegmentedColormap.from_list(
    "bt_heat", [BT_ICE_GREY, BT_BLUE_ACCENT, BT_BLUE]
)

# ── Shared matplotlib theme ───────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D8DEEE",   # BT Ice Grey
    "axes.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.titlecolor": "#000000",  # BT Black
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#504F4F",  # BT Concrete
    "axes.labelpad": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#D8DEEE",       # BT Ice Grey
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#504F4F",      # BT Concrete
    "ytick.color": "#504F4F",      # BT Concrete
    "legend.fontsize": 9,
    "legend.frameon": False,
    "font.size": 10.5,
})


def titled(ax, main, sub=None, pad=22):
    ax.set_title(main, loc="left", pad=pad, fontweight="bold")
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 5), xycoords="axes fraction",
                    textcoords="offset points", fontsize=9, color="#9ca3af", va="bottom")


# ── Data loading (mirrors notebook setup) ────────────────────────────────────
def norm_model(m):
    if not m:
        return None
    base = m.split("/")[-1].lower()
    base = re.sub(r"-preview$", "", base)
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base)
    return base


def load_df():
    raw = json.loads(Path("data/full.json").read_text())
    rows = []
    for r in raw:
        dur = r.get("duration")
        rows.append({
            "harness":          r.get("harness"),
            "model":            norm_model(r.get("model")),
            "benchmark":        r.get("benchmark"),
            "success":          r.get("task_success"),
            "tool_error_count": r.get("tool_error_count") or 0,
            "total_tokens":     r.get("total_tokens") or 0,
        })
    df = pd.DataFrame(rows).dropna(subset=["success"]).reset_index(drop=True)
    df["success"] = df["success"].astype(int)
    df["config"]  = df["harness"] + " · " + df["model"]
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (center - margin, p, center + margin, margin)


def rate_table(frame, group):
    rows = []
    for key, g in frame.groupby(group):
        k, n = int(g.success.sum()), len(g)
        lo, p, hi, half = wilson(k, n)
        rows.append({**({group: key} if isinstance(key, str) else dict(zip(group, key))),
                     "n": n, "success_rate": p, "ci_lo": lo, "ci_hi": hi, "ci_half": half})
    return pd.DataFrame(rows)


def config_reliability(frame, min_cell=5, min_total=10):
    micro = rate_table(frame, "config").rename(columns={"success_rate": "micro"})
    cells = rate_table(frame, ["config", "benchmark"])
    cells = cells[cells.n >= min_cell]
    macro = (cells.groupby("config")
             .agg(macro=("success_rate", "mean"),
                  cross_task_std=("success_rate", "std"),
                  n_benchmarks=("benchmark", "nunique")).reset_index())
    out = micro.merge(macro, on="config", how="left")
    out["cross_task_std"] = out["cross_task_std"].fillna(0)
    out["mix_gap"] = out["micro"] - out["macro"]
    out["floor"]   = out["ci_lo"]
    return out[out.n >= min_total].reset_index(drop=True)


# ── Plot 02_1a: pooled bars vs benchmark-balanced diamonds ───────────────────
def plot_02_1a(df, out_dir):
    MIN_N = 10
    COMPARABLE_BM = 3

    relc = (config_reliability(df, min_cell=5, min_total=MIN_N)
            .dropna(subset=["macro"]).sort_values("macro").reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(relc) + 2))

    err    = np.vstack([relc.micro - relc.ci_lo, relc.ci_hi - relc.micro])
    colors = BT_CMAP(relc.macro)   # indigo gradient mapped to success rate

    for y, (_, r) in enumerate(relc.iterrows()):
        thin = r.n_benchmarks < COMPARABLE_BM
        bar_color = BT_ICE_GREY if thin else colors[y]
        ax.barh(y, r.micro, xerr=err[:, [y]],
                color=bar_color,
                error_kw=dict(ecolor=BT_BLUE, capsize=3.5, lw=1.3),
                height=0.72, zorder=3,
                edgecolor="none")

    ax.set_yticks(range(len(relc)))
    ax.set_yticklabels(relc.config, fontsize=10.5)

    # Diamonds — benchmark-balanced (macro) rate
    ax.scatter(relc.macro, range(len(relc)), marker="D", s=62,
               facecolors="none", edgecolors=BT_PINK, linewidth=1.5,
               zorder=5)

    for y, (_, r) in enumerate(relc.iterrows()):
        thin = r.n_benchmarks < COMPARABLE_BM
        ax.text(1.10, y, f"{r.micro:.0%}", va="center", ha="right",
                fontsize=10.5, weight="semibold", color=BT_BLACK)
        ax.text(1.235, y, f"n={r.n} · {int(r.n_benchmarks)}bm", va="center", ha="right",
                fontsize=9, color=BT_PINK if thin else BT_CONCRETE,
                weight="semibold" if thin else "normal")

    ax.set_xlim(0, 1.25)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("task success rate")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10)

    titled(ax, "Pooled bars vs benchmark-balanced diamonds",
           "bar = pooled rate (±Wilson 95%) · ♦ = macro-avg over suites · "
           "light bars = <3 suites, not cross-comparable", pad=28)

    ax.margins(y=0.01)
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)

    ax.legend(handles=[
        Patch(facecolor=BT_BLUE, label="pooled (micro) rate ±Wilson 95% CI"),
        Patch(facecolor=BT_ICE_GREY, label="<3 suites, not cross-compatible"),
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor="none", markeredgecolor=BT_PINK,
               markeredgewidth=1.5, markersize=9, label="macro avg over suites"),
    ], loc="lower right", frameon=True, framealpha=0.95,
       edgecolor="#e5e7eb", fontsize=9)

    plt.tight_layout()
    path = out_dir / "02_1a.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


# ── Plot 03_1b: reliability quadrant ─────────────────────────────────────────
def plot_03_1b(df, out_dir):
    try:
        from adjustText import adjust_text
        def repel(ax, xs, ys, texts):
            objs = [ax.text(x, y, t, fontsize=8.5, color=BT_CONCRETE)
                    for x, y, t in zip(xs, ys, texts)]
            adjust_text(objs, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#d1d5db", lw=0.6),
                        expand=(1.15, 1.4), force_text=(0.4, 0.6))
    except ImportError:
        def repel(ax, xs, ys, texts):
            for x, y, t in zip(xs, ys, texts):
                ax.text(x, y, t, fontsize=8.5, color=BT_CONCRETE)

    relc = (config_reliability(df, min_cell=5, min_total=10)
            .dropna(subset=["macro"]).sort_values("macro").reset_index(drop=True))
    rel = relc[relc.n_benchmarks >= 2].copy()

    fig, ax = plt.subplots(figsize=(12, 8.5))

    xm = rel.macro.median()
    ym = rel.cross_task_std.median()
    ax.axvline(xm, ls="--", c="#e5e7eb", lw=1, zorder=0)
    ax.axhline(ym, ls="--", c="#e5e7eb", lw=1, zorder=0)

    # Bubble size mapped to n; single color (BT_DARK) instead of gradient
    sizes = np.interp(rel.n, (rel.n.min(), rel.n.max()), (120, 800))
    ax.scatter(rel.macro, rel.cross_task_std, s=sizes,
               color=BT_BLUE, edgecolor="white", linewidth=1.2,
               zorder=3, alpha=0.85)

    repel(ax, rel.macro, rel.cross_task_std, rel.config)

    ax.text(0.985, 0.04, "RELIABLE\nhigh score · consistent",
            transform=ax.transAxes, ha="right", va="bottom",
            color=BT_GREEN, fontsize=11, weight="bold", linespacing=1.3)
    ax.text(0.015, 0.97, "ERRATIC\nlow score · inconsistent",
            transform=ax.transAxes, ha="left", va="top",
            color=BT_ROSE, fontsize=11, weight="bold", linespacing=1.3)

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("benchmark-balanced (macro) success  →  higher is better")
    ax.set_ylabel("cross-task std of success  →  lower is more predictable")
    titled(ax, "Reliability quadrant",
           "Dependable configs sit bottom right",
           pad=28)
    ax.margins(0.08)
    plt.tight_layout()

    path = out_dir / "03_1b.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


# ── Plot 07_1f: config × benchmark heatmap ───────────────────────────────────
def plot_07_1f(df, out_dir):
    cb      = rate_table(df, ["config", "benchmark"])
    cb_n    = cb.pivot(index="config", columns="benchmark", values="n")
    cb_rate = cb.pivot(index="config", columns="benchmark", values="success_rate")

    keep_rows       = ((cb_n >= 5).sum(axis=1) > 0)
    cb_rate, cb_n   = cb_rate[keep_rows], cb_n[keep_rows]
    order_idx       = cb_rate.mean(axis=1).sort_values(ascending=False).index
    cb_rate, cb_n   = cb_rate.loc[order_idx], cb_n.loc[order_idx]

    mask       = (cb_n < 5) | cb_rate.isna()
    annot_lab  = cb_rate.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(cb_rate) + 2))
    sns.heatmap(cb_rate, mask=mask, annot=annot_lab, fmt="",
                cmap=BT_HEAT, vmin=0, vmax=1,
                linewidths=2, linecolor="white",
                cbar_kws={"label": "success rate", "shrink": 0.7},
                ax=ax, annot_kws={"fontsize": 8.5, "color": "white"})

    # Override annotation text color: dark text on light cells, white on dark
    for text in ax.texts:
        try:
            val = float(text.get_text().replace("%", "")) / 100
            text.set_color("white" if val > 0.5 else "#1f2937")
        except ValueError:
            pass

    ax.set_facecolor("#f9fafb")  # masked cells
    titled(ax, "Config x benchmark success – spotting the gamblers",
           "grey = <5 tasks (too thin) · a red cell in an otherwise green row = a brittle suite")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    plt.tight_layout()

    path = out_dir / "07_1f.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    target  = sys.argv[1] if len(sys.argv) > 1 else "02_1a"
    out_dir = Path("out/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df()
    print(f"loaded {len(df)} rows")

    if target in ("02_1a", "all"):
        plot_02_1a(df, out_dir)
    if target in ("03_1b", "all"):
        plot_03_1b(df, out_dir)
    if target in ("07_1f", "all"):
        plot_07_1f(df, out_dir)
