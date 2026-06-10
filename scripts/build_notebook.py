"""Generate reliability_analysis.ipynb from cell definitions.

Keeps the notebook source under version control as plain Python so it is easy
to diff and regenerate. Run from the repo root: .venv/bin/python scripts/build_notebook.py
(add --export to also re-render the PNGs in out/plots/).
"""
from pathlib import Path

import nbformat as nbf

# This script lives in scripts/; everything it reads/writes is anchored to the repo
# root (its parent) so it works regardless of the directory it's launched from.
REPO = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- title
md(r"""# Agent-eval reliability, failure modes & benchmark bleed

A look at the **full** set of agentic-eval logs pulled from Braintrust via BTQL
(`data/full.json`, all 1,781 root task spans — the UI export caps at 1,000 rows,
so this is pulled through the API). Each row is one agent rollout scored by an
LLM judge with a binary `task_success`.

The run is described by three knobs:

| dimension | values |
|---|---|
| **harness** | how the agent is scaffolded (`claude_code`, `tool_calling`, `smolagents_code`, `openai_solo`, `tool_calling_with_shortlisting`) |
| **model** | the underlying LLM (`gpt-4.1`, `claude-opus-4-5`, `gemini-3-pro-preview`, `kimi-k2.5`, `deepseek-v3.2`, `gpt-5.2`) |
| **benchmark** | the task suite (`swebench`, `appworld`, `tau2_{retail,airline,telecom}`, `browsecompplus`) |

This notebook extends the basic "who scores highest" view with three questions:

1. **Reliability / uncertainty** — which *configurations* (harness × model) are
   both **successful** *and* **predictable**? Point estimates are shown with
   Wilson confidence intervals and cross-task standard deviations overlaid.
2. **Failure-mode indexing** — when a run fails, *how* does it fail, and do the
   harnesses fail **differently**?
3. **Benchmark bleed** — some suites (notably SWE-bench and the τ²-bench family)
   are public, static, and very likely sit in these models' training data. We
   flag the high-risk suites, exclude them, and check whether the reliability
   conclusions survive.
""")

# ---------------------------------------------------------------- setup
md("## 0 · Setup & tidy frame")

code(r"""import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.titlecolor": "#1a1a1a",
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#333333",
    "axes.labelpad": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.fontsize": 9,
    "legend.title_fontsize": 9.5,
    "legend.frameon": False,
    "font.size": 10.5,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})
ANNOT = dict(fontsize=8.5, color="#222222")  # shared style for point labels


def titled(ax, main, sub=None, pad=22):
    # Left-aligned bold title with an optional grey sub-line stacked beneath it.
    ax.set_title(main, loc="left", pad=pad, fontweight="semibold")
    if sub:
        ax.annotate(sub, xy=(0, 1), xytext=(0, 5), xycoords="axes fraction",
                    textcoords="offset points", fontsize=9, color="#888888", va="bottom")


def fig_title(fig, main, sub=None):
    # Figure title + grey subtitle placed with fig.text (NOT suptitle) so tight_layout
    # doesn't reserve an empty band beneath them. Spaced by *inches* so the gap is the
    # same at any figure height. Both sit just above the canvas; pair with a high
    # tight_layout top (e.g. rect=[...0.97]) so panels fill the space.
    h = fig.get_figheight()
    fig.text(0.012, 1 + 0.32 / h, main, ha="left", fontsize=14, fontweight="bold")
    if sub:
        fig.text(0.012, 1 + 0.10 / h, sub, ha="left", fontsize=9, color="#888888")


from adjustText import adjust_text  # smart label de-collision


def repel_labels(ax, xs, ys, texts, fontsize=8.5, color="#222222"):
    # Place point labels and nudge them apart so dense clusters stay legible.
    objs = [ax.text(x, y, t, fontsize=fontsize, color=color) for x, y, t in zip(xs, ys, texts)]
    adjust_text(objs, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6),
                expand=(1.15, 1.4), force_text=(0.4, 0.6))
    return objs


pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

# Pull straight from Braintrust, cached to disk. First run (or after you delete the
# file) runs the BTQL below for all root task spans and writes data/full.json; every
# run after reads that local snapshot — so the analysis stays offline, fast, and
# reproducible with no separate "export the JSON" step. bt_helpers is the helper in
# scripts/ (paginated, retrying BTQL + JSON cache); a refresh needs BRAINTRUST_API_KEY
# in .env. Run this notebook from the repo root so "scripts/" and "data/" resolve.
sys.path.insert(0, "scripts")
import bt_helpers as bt

DATA = Path("data/full.json")
BTQL = (
    "select: metadata.benchmark AS benchmark, metadata.harness AS harness, "
    "metadata.model AS model, metadata.session_id AS session_id, "
    "metadata.total_tokens AS total_tokens, metadata.num_llm_calls AS num_llm_calls, "
    "metadata.has_errors AS has_errors, metadata.tool_error_count AS tool_error_count, "
    "metadata.error_span_count AS error_span_count, metadata.judge_model AS judge_model, "
    "metadata.judge_confidence AS judge_confidence, metadata.judge_reasoning AS judge_reasoning, "
    "metadata.judge_reliability AS judge_reliability, scores.task_success AS task_success, "
    "span_attributes.name AS task, duration, id "
    "| from: project_logs('6da0ad7f-d092-4d04-95c5-a2ae182883ec') "
    "| filter: is_root = true"
)
raw = bt.cached_pull(BTQL, str(DATA))   # delete data/full.json to refresh from the API
print(f"{len(raw)} rows")
""")

code(r'''def norm_model(m: str | None) -> str | None:
    """Collapse provider/casing variants -> a clean model-family label.

    e.g. openai/Azure/gpt-4.1 and Azure/gpt-4.1 -> gpt-4.1;
    gcp/gemini-3-pro-preview -> gemini-3-pro. Strip the provider path, lowercase,
    drop a trailing -preview and date stamp so the ~10 logged strings collapse to
    the 6 underlying models.
    """
    if not m:
        return None
    base = m.split("/")[-1].lower()
    base = re.sub(r"-preview$", "", base)
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base)  # gpt-5.2-2025-12-11 -> gpt-5.2
    return base


def flatten(r: dict) -> dict:
    # data/full.json is already flat (BTQL select aliased each metadata.* field
    # to a top-level key); we just normalise types and the model label.
    dur = r.get("duration")  # BTQL `duration` is in SECONDS already
    return {
        "harness": r.get("harness"),
        "model": norm_model(r.get("model")),
        "benchmark": r.get("benchmark"),
        "success": r.get("task_success"),
        "has_errors": bool(r.get("has_errors")),
        "tool_error_count": r.get("tool_error_count") or 0,
        "error_span_count": r.get("error_span_count") or 0,
        "num_llm_calls": r.get("num_llm_calls") or 0,
        "total_tokens": r.get("total_tokens") or 0,
        "duration_s": dur if dur is not None else np.nan,
        "judge_model": r.get("judge_model"),
        "judge_confidence": r.get("judge_confidence"),
        "judge_reliability": r.get("judge_reliability"),
        "judge_reasoning": r.get("judge_reasoning") or "",
        "session_id": r.get("session_id"),
    }


df = pd.DataFrame(flatten(r) for r in raw)
# One row has a null task_success (judge never scored it) -> drop before casting.
df = df.dropna(subset=["success"]).reset_index(drop=True)
df["success"] = df["success"].astype(int)
df["config"] = df["harness"] + " · " + df["model"]
print(df.shape)
df.head(3)
''')

code(r"""# Sanity: coverage of the design grid
print("models   :", sorted(df.model.unique()))
print("harnesses:", sorted(df.harness.unique()))
print("benchmarks:", sorted(df.benchmark.unique()))
print("\noverall success rate: {:.1%}".format(df.success.mean()))
df.groupby("harness").success.agg(["mean", "count"]).sort_values("mean", ascending=False)
""")

# ---------------------------------------------------------------- helpers
code(r'''def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, p, hi, half)."""
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin, p, center + margin, margin)


def rate_table(frame, group):
    """Per-group success rate with Wilson 95% CI."""
    rows = []
    for key, g in frame.groupby(group):
        k, n = int(g.success.sum()), len(g)
        lo, p, hi, half = wilson(k, n)
        rows.append({**({group: key} if isinstance(key, str) else dict(zip(group, key))),
                     "n": n, "success_rate": p, "ci_lo": lo, "ci_hi": hi, "ci_half": half})
    return pd.DataFrame(rows)


def config_reliability(frame, min_cell=5, min_total=10):
    """Per-config success, computed the benchmark-fair way.

    Pooling all of a config's rollouts into one rate (a *micro*-average) is
    confounded: each config ran a different mix / number of tasks per suite, so a
    config that happened to run more of an easy benchmark scores higher for free
    (Simpson's paradox). We therefore also report the *macro*-average -- the
    unweighted mean of the config's per-benchmark rates -- which gives every
    suite equal weight regardless of task counts. ``mix_gap = micro - macro`` is
    how much the pooled view flatters (+) or understates (-) a config purely from
    its benchmark mix. ``cross_task_std`` and ``n_benchmarks`` come from the same
    per-(config, benchmark) cells, so the quadrant in 1b is internally consistent.
    """
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
    # Floor score = Wilson lower bound on the pooled rate: the conservative
    # "you can count on at least this much" number. It folds success AND our
    # certainty about it into one figure -- high mean but small n (wide CI)
    # is penalised, so a high floor means successful *and* predictable.
    out["floor"] = out["ci_lo"]
    return out[out.n >= min_total].reset_index(drop=True)
''')

# ---------------------------------------------------------------- stats key
md(r"""### 📐 Stats key — what every band, bar, and dot is computed from

Every plot below reuses the same handful of statistics.(`k` = successes, `n` = tasks, `z` = 1.96 for 95%).

- **Success rate** $p = k/n$ — the fraction of tasks the LLM judge scored 1.
- **Wilson 95% CI** (the error bars / `±` half-widths) — a binomial interval that,
  unlike the textbook normal one, stays inside [0, 1] and widens correctly for
  small `n`. Center and half-width:
  $$\text{center}=\frac{p + z^2/2n}{1+z^2/n},\qquad
    \text{half}=\frac{z}{1+z^2/n}\sqrt{\tfrac{p(1-p)}{n}+\tfrac{z^2}{4n^2}}$$
  Wide bars = *few samples*, not necessarily an erratic agent. `floor` = the CI's
  lower bound (`center − half`): "you can count on at least this much."
- **Micro vs macro rate.** *Micro* = pool every rollout into one $k/n$ (flattered by
  an easy task mix). *Macro* = mean of the per-benchmark rates,
  $\frac{1}{B}\sum_b p_b$, giving each suite equal weight — the Simpson's-paradox-safe
  number. `mix_gap = micro − macro` is how much pooling flatters a config.
- **Cross-task std** (the y-axis in 1b) — the spread of a config's per-benchmark
  rates, $\sqrt{\tfrac{1}{B-1}\sum_b (p_b-\bar p)^2}$. Low = consistent across suites.
- **η²** (1d) — share of total success variance explained by one factor:
  $SS_\text{between}/SS_\text{total}$. **Incremental R²** (1h) — the extra variance a
  factor adds *on top of the other two* in the regression, $R^2_\text{full}-R^2_\text{without}$.
- **Box plots** (2b) — box = inter-quartile range (25th–75th pct), line = median,
  whiskers = 1.5×IQR; outliers hidden so the bulk of the distribution is readable.""")

# ================================================================ SECTION 1
md(r"""## 1 · Reliability & uncertainty

A configuration is only useful in production if it is **both** high-scoring
**and** dependable. We separate three things that a single "average success"
number quietly blends together:

- **Statistical uncertainty** — how sure are we of the success *rate* given the
  sample size? Captured by the **Wilson 95% CI**.
- **Benchmark-mix confound** — a pooled average over all of a config's rollouts
  (a *micro*-average) is **not comparable across configs**, because each one ran a
  different mix and number of tasks per suite. A config that happened to run more
  of an easy benchmark scores higher for free — *Simpson's paradox*. We neutralise
  it by holding the benchmark constant: compute each config's rate **per
  benchmark**, then take the **unweighted mean across suites** (a *macro*-average),
  giving every suite equal weight regardless of task counts.
- **Behavioural consistency** — does the config hold up *across* task suites, or
  is a good average hiding a suite where it falls apart? Captured by the
  **standard deviation of the per-benchmark success rates** (the same cells the
  macro-average is built from).

The sweet spot is high **macro** mean, tight CI, low cross-task spread. Even the
macro-average is only strictly comparable between configs that ran the *same set*
of suites — so coverage (`n_benchmarks`) is shown alongside to flag when it isn't.
""")

md(r"""### 1·0 · Coverage — the design is unbalanced (read this first)

Before any ranking: **not every config ran every benchmark**, and the gaps are
large enough to invalidate naive cross-config comparisons. The matrix below is the
task count per (config, benchmark); blank = never run, grey = run but thin (<5
tasks, too few to score). Two facts to carry through the rest of §1:

- **Only a handful of configs span all 6 suites.** A config's macro-average over 1–2
  suites is *not* comparable to one computed over 6 — they are measuring different
  (and differently hard) things.
- **`browsecompplus` was run by `claude_code` only**, and
  `tool_calling_with_shortlisting` essentially only ran `appworld`. So any
  cross-*harness* number that pools over suites is partly comparing *which suites a
  harness happened to run*, not the harness itself.

The honest fixes follow: §1e and §1h restrict every comparison to a **shared
benchmark set** or adjust for benchmark as a covariate, and the rankings below are
read with the coverage count (`…bm`) firmly in mind.""")

code(r'''cov = (df.groupby(["config", "benchmark"]).size()
       .unstack("benchmark").reindex(columns=sorted(df.benchmark.unique())))
cov = cov.loc[cov.sum(axis=1).sort_values(ascending=False).index]  # busiest configs on top
n_suites = (cov >= 5).sum(axis=1)

fig, ax = plt.subplots(figsize=(9, 0.42 * len(cov) + 2))
mask = cov.isna() | (cov < 5)               # blank/thin cells -> masked
sns.heatmap(cov, mask=mask, annot=cov.fillna(0).astype(int), fmt="d",
            cmap="Blues", linewidths=2, linecolor="white", vmin=0,
            cbar_kws={"label": "tasks run", "shrink": .6}, ax=ax,
            annot_kws={"fontsize": 8.5})
ax.set_facecolor("#f0f0f0")                  # masked (blank/thin) cells show grey
# overlay thin-but-nonzero counts in grey so they're visible as "ran but <5"
for yi, cfg in enumerate(cov.index):
    for xi, bm in enumerate(cov.columns):
        v = cov.loc[cfg, bm]
        if pd.notna(v) and v < 5:
            ax.text(xi + 0.5, yi + 0.5, int(v), ha="center", va="center",
                    fontsize=7.5, color="#999999")
for yi, cfg in enumerate(cov.index):         # suite-coverage count on the right
    ax.text(len(cov.columns) + 0.15, yi + 0.5, f"{n_suites[cfg]}/6", va="center",
            ha="left", fontsize=8.5, weight="semibold",
            color="#2a8a4a" if n_suites[cfg] >= 5 else "#c0392b")
titled(ax, "Config × benchmark coverage", "cell = tasks run · grey = <5 (too thin) · right = suites covered")
ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(length=0)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
plt.tight_layout(); plt.show()
print(f"configs covering all 6 suites: {(n_suites == 6).sum()} of {len(n_suites)}")
''')

md("### 1a · Pooled vs benchmark-balanced success per configuration")

code(r'''MIN_N = 10  # configs with fewer rollouts are too noisy to rank
relc = (config_reliability(df, min_cell=5, min_total=MIN_N)
        .dropna(subset=["macro"]).sort_values("macro").reset_index(drop=True))

COMPARABLE_BM = 3  # configs spanning fewer suites aren't fairly cross-comparable
fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(relc) + 2))
# Bars = pooled (micro) rate with Wilson CI -> statistical uncertainty.
err = np.vstack([relc.micro - relc.ci_lo, relc.ci_hi - relc.micro])
colors = sns.color_palette("crest", as_cmap=True)(relc.macro)
for y, (_, r) in enumerate(relc.iterrows()):
    thin = r.n_benchmarks < COMPARABLE_BM
    ax.barh(y, r.micro, xerr=err[:, [y]], color=colors[y],
            error_kw=dict(ecolor="#555555", capsize=3.5, lw=1.3), height=0.72, zorder=3,
            alpha=0.4 if thin else 1.0, hatch="///" if thin else None,
            edgecolor="white" if thin else "none")
ax.set_yticks(range(len(relc))); ax.set_yticklabels(relc.config, fontsize=10.5)
# Diamonds = benchmark-balanced (macro) rate -> the mix-corrected estimate.
ax.scatter(relc.macro, range(len(relc)), marker="D", s=62, color="#c0392b",
           edgecolor="white", linewidth=1, zorder=5, label="benchmark-balanced (macro)")
for y, (_, r) in enumerate(relc.iterrows()):
    thin = r.n_benchmarks < COMPARABLE_BM
    ax.text(max(r.ci_hi, r.macro) + 0.014, y, f"{r.micro:.0%}", va="center", ha="left",
            fontsize=10.5, weight="semibold", color="#333333")
    ax.text(1.235, y, f"n={r.n} · {int(r.n_benchmarks)}bm", va="center", ha="right",
            fontsize=9, color="#c0392b" if thin else "#999999",
            weight="semibold" if thin else "normal")
ax.set_xlim(0, 1.25)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("task success rate")
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=10)
titled(ax, "Pooled bars vs benchmark-balanced diamonds",
       "bar = pooled rate (±Wilson 95%) · ♦ = macro-avg over suites · hatched/red = <3 suites, "
       "not cross-comparable", pad=24)
ax.margins(y=0.01)
sns.despine(ax=ax, left=True)
ax.tick_params(left=False)
plt.tight_layout()
plt.show()

relc.sort_values("floor", ascending=False)[
    ["config", "n", "n_benchmarks", "micro", "macro", "floor", "mix_gap", "cross_task_std"]].round(3)
''')

md(r"""**Bars** are the pooled (micro) rate with the Wilson 95% CI — wide bars are
*unreliable estimates* (small `n`), not necessarily unreliable agents.
**Diamonds** are the benchmark-balanced (macro) rate. Where a diamond sits well to
the **left** of its bar, the pooled headline is **flattered by an easy benchmark
mix** (`mix_gap > 0`); to the **right**, the pooled number *understates* the
config. Also read `n_benchmarks` (`…bm`): a macro-average over 2 suites is not
comparable to one over 6, and configs whose Wilson CIs overlap heavily are
statistically indistinguishable.

The table is **sorted by `floor`** — the Wilson *lower* bound on the pooled rate.
Read it as "whatever else is true, this config delivers at least this much,"
which is the single number that rewards being **both** high-scoring **and**
well-sampled/predictable: a flashy mean on thin data has a low floor and sinks in
the ranking.

**Coverage gate:** bars that are **hatched / have a red `…bm` count** ran fewer
than 3 suites, so their position on this axis is *not* comparable to the
full-coverage configs — they may simply have run an easier (or harder) subset.
Treat them as "insufficient evidence to rank," not as genuinely better/worse, and
rely on the other plots for cross-config comparison.""")

md("### 1b · Success vs. predictability — the reliability quadrant")

code(r'''# Mean (macro) and spread (cross-task std) come from the SAME per-(config,
# benchmark) cells, so the quadrant can't be gamed by benchmark mix.
rel = relc[relc.n_benchmarks >= 2].copy()  # need >=2 suites for a cross-task std

fig, ax = plt.subplots(figsize=(12, 8.5))
xm = rel.macro.median(); ym = rel.cross_task_std.median()
ax.axvline(xm, ls="--", c="#cccccc", lw=1, zorder=0)
ax.axhline(ym, ls="--", c="#cccccc", lw=1, zorder=0)
sns.scatterplot(data=rel, x="macro", y="cross_task_std", size="n",
                hue="macro", palette="crest", sizes=(120, 800),
                edgecolor="white", linewidth=1.2, legend=False, ax=ax, zorder=3)
# Smart de-collision so the dense mid-cluster stays legible.
repel_labels(ax, rel.macro, rel.cross_task_std, rel.config)
ax.text(0.985, 0.04, "RELIABLE\nhigh score · consistent", transform=ax.transAxes,
        ha="right", va="bottom", color="#2a8a4a", fontsize=11, weight="bold", linespacing=1.3)
ax.text(0.015, 0.97, "ERRATIC\nlow score · inconsistent", transform=ax.transAxes,
        ha="left", va="top", color="#c0392b", fontsize=11, weight="bold", linespacing=1.3)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("benchmark-balanced (macro) success  →  higher is better")
ax.set_ylabel("cross-task std of success  →  lower is more predictable")
titled(ax, "Reliability quadrant — dependable configs sit bottom-right",
       "x = macro rate (mean of per-suite rates) · y = std of those per-suite rates · "
       "bubble = sessions (see Stats key)")
ax.margins(0.08)
plt.tight_layout(); plt.show()
rel.sort_values(["macro", "cross_task_std"], ascending=[False, True])[
    ["config", "n", "n_benchmarks", "micro", "macro", "floor", "mix_gap", "cross_task_std", "ci_half"]].round(3)
''')

md(r"""**Read it as:** bottom-right = the dependable workhorses (score high,
behave the same on every suite). Top-left = avoid. A config high on the x-axis
but also high on the y-axis is a *gambler*: great on average, but with a suite
where it collapses. Because the x-axis is the macro (benchmark-balanced) rate and
the y-axis is the spread of the *same* per-suite cells, a config can't buy its way
rightward by running more easy tasks — the Simpson's-paradox loophole from §1a is
closed here.""")

md("### 1c · Harness × model success heatmap (mean ± Wilson half-width)")

code(r'''hm = rate_table(df, ["harness", "model"])
hm = hm[hm.n >= 5]
pivot = hm.pivot(index="harness", columns="model", values="success_rate")
annot = hm.assign(lab=hm.apply(
    lambda r: f"{r.success_rate:.0%}\n±{r.ci_half:.0%} (n={r.n})", axis=1)
).pivot(index="harness", columns="model", values="lab")

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=2, linecolor="white", cbar_kws={"label": "success rate", "shrink": .8},
            ax=ax, annot_kws={"fontsize": 8.5, "color": "#1a1a1a"})
titled(ax, "Success by harness × model", "cell shows mean ± Wilson 95% half-width (n)")
ax.set_xlabel(""); ax.set_ylabel("")
ax.tick_params(length=0)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout(); plt.show()
''')

md(r"""The `±` term is the statistical noise floor for each cell. Where the
half-width is large, the cell is under-sampled — treat colour differences with
caution. Each cell also still **pools across benchmarks**, so it inherits the same
mix confound as §1a's bars — use it to spot broad patterns, and lean on the
per-benchmark macro rates in §1a/§1b for fair ranking.""")

md(r"""### 1d · What moves the needle — harness vs model vs benchmark

The heatmap hints that *how* you scaffold the agent matters more than *which*
model you drop in. We make that quantitative with a one-way **η²** (eta-squared):
for each factor, the share of the total variance in `success` explained by knowing
only that factor's value. Higher = that knob swings the outcome more.

Caveat: the design is **unbalanced** (not every harness ran every model×benchmark),
so these one-way effects are partly confounded with each other and **do not sum to
100%**. They are a directional read on *which knob dominates*, not a clean variance
decomposition — read the size of the gap between bars, not their absolute values.""")

code(r'''def eta_sq(frame, factor):
    """One-way eta-squared: SS_between / SS_total for `success` grouped by factor."""
    grand = frame.success.mean()
    ss_tot = ((frame.success - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_bet = sum(len(g) * (g.success.mean() - grand) ** 2
                 for _, g in frame.groupby(factor))
    return ss_bet / ss_tot


factors = ["harness", "model", "benchmark", "config"]
eta = pd.DataFrame({"factor": factors,
                    "eta_sq": [eta_sq(df, f) for f in factors]}).sort_values("eta_sq")

fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#30638e" if f != "config" else "#9bb8d3" for f in eta.factor]
ax.barh(eta.factor, eta.eta_sq, color=colors, height=0.62, zorder=3)
for y, v in enumerate(eta.eta_sq):
    ax.text(v + 0.004, y, f"{v:.1%}", va="center", fontsize=10, weight="semibold",
            color="#333333")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlim(0, max(eta.eta_sq) * 1.18)
ax.set_xlabel("share of success variance explained (one-way η²)")
titled(ax, "Which knob moves task success most?",
       "config = harness × model jointly · unbalanced design, bars don't sum to 100%")
sns.despine(ax=ax, left=True); ax.tick_params(left=False)
plt.tight_layout(); plt.show()
eta.set_index("factor").round(4)
''')

md(r"""If **harness** outranks **model**, the scaffolding is the bigger lever —
swapping in a stronger LLM buys less than fixing how it is driven. `config`
(harness × model jointly) sits highest by construction; the gap between it and the
single factors is the slice explained by the *interaction* — i.e. the right model
needs the right harness.""")

md(r"""### 1e · Harness swing for a fixed model — on a *shared* benchmark set

Hold the **model** constant and watch success move as the harness changes. But per
§1·0, a model's harnesses ran *different* suites, so the earlier "macro over each
harness's own suites" version mixed the harness effect with coverage. Here we close
that hole: for each model we take only the benchmarks that **every** one of its
(qualifying) harnesses ran (≥5 tasks each), and score all harnesses **on that shared
set only**. Each panel below is one model; each bar is a harness's pooled success on
the shared suites (±Wilson 95%). The **Δ** in each title is the best−worst gap —
the apples-to-apples swing from scaffolding — and each panel **names the shared
suites underneath**, because *which* suites (and how many) back the swing is exactly
what tells you how far to trust it. Note that for Claude, Gemini, and DeepSeek the
only common ground is **appworld** — so their report card is really an appworld-only
read — whereas gpt-4.1 spans two τ²-bench suites.""")

code(r'''MIN_CELL, MIN_TOT = 5, 10
# Short, unambiguous suite labels for the panel titles (the tau2_* family drops its
# prefix — retail/airline/telecom are only ever tau2, so no information is lost).
BENCH_ABBR = {"swebench": "swebench", "appworld": "appworld",
              "browsecompplus": "browse", "tau2_airline": "airline",
              "tau2_retail": "retail", "tau2_telecom": "telecom"}
cells_mhb = rate_table(df, ["model", "harness", "benchmark"])
cells_mhb = cells_mhb[cells_mhb.n >= MIN_CELL]
tot = cells_mhb.groupby(["model", "harness"]).n.sum()
qual = {mh for mh, v in tot.items() if v >= MIN_TOT}  # harnesses with enough data

records, shared_bms = [], {}
for model in sorted(df.model.unique()):
    hs = sorted(h for (m, h) in qual if m == model)
    if len(hs) < 2:
        continue
    bm_sets = [set(cells_mhb[(cells_mhb.model == model) & (cells_mhb.harness == h)].benchmark)
               for h in hs]
    common = set.intersection(*bm_sets)            # suites EVERY harness of this model ran
    if not common:
        continue
    # Name the shared suites (not just count them): a swing measured on a single
    # suite is much weaker evidence than one measured across several, and which
    # suite it is matters (e.g. appworld is hard for Claude). Surface both.
    shared_bms[model] = ", ".join(BENCH_ABBR.get(b, b) for b in sorted(common))
    sub = df[(df.model == model) & (df.benchmark.isin(common))]
    for h in hs:
        gh = sub[sub.harness == h]
        lo, p, hi, _ = wilson(int(gh.success.sum()), len(gh))
        records.append({"model": model, "harness": h, "rate": p,
                        "ci_lo": lo, "ci_hi": hi, "n": len(gh)})
ce = pd.DataFrame(records)
swing = (ce.groupby("model").rate.agg(["min", "max"])
         .assign(swing=lambda d: d["max"] - d["min"],
                 shared=lambda d: d.index.map(shared_bms)).sort_values("swing"))

hh = sorted(df.harness.unique())
hpal = dict(zip(hh, sns.color_palette("Set2", len(hh))))
models_ord = swing.sort_values("swing", ascending=False).index.tolist()

ncol = 3
nrow = int(np.ceil(len(models_ord) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.5 * nrow + 1), squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)
for i, model in enumerate(models_ord):
    ax = axes[i // ncol][i % ncol]; ax.set_visible(True)
    sub = ce[ce.model == model].sort_values("rate").reset_index(drop=True)
    err = np.vstack([sub.rate - sub.ci_lo, sub.ci_hi - sub.rate])
    ax.barh(range(len(sub)), sub.rate, xerr=err, height=0.62,
            color=[hpal[h] for h in sub.harness],
            error_kw=dict(ecolor="#666666", capsize=3, lw=1.1), zorder=3)
    for y, s in sub.iterrows():
        ax.text(min(s.ci_hi + 0.025, 1.0), y, f"{s.rate:.0%}", va="center", ha="left",
                fontsize=9.5, weight="semibold", color="#333333")
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.harness, fontsize=9.5)
    ax.set_xlim(0, 1.12); ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # Title names the shared suite(s) the swing is measured on — so a Δ computed on
    # appworld-only reads differently from one spanning several suites.
    ax.set_title(f"{model}   Δ{swing.loc[model, 'swing']:.0%}", loc="left",
                 fontsize=11, fontweight="semibold", pad=22)
    ax.annotate(f"shared suites: {swing.loc[model, 'shared']}", xy=(0, 1), xytext=(0, 3),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=8.5, color="#888888", va="bottom")
    sns.despine(ax=ax, left=True); ax.tick_params(left=False)
fig_title(fig, "Harness report card — same model, same suites",
          "each panel fixes the model · bar = success on the shared suites named per panel "
          "(±Wilson 95%) · Δ = best−worst harness")
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
swing[["min", "max", "swing"]].round(3).assign(shared=swing["shared"])
''')

md(r"""Read each panel as a model's **report card**: how well does it do under each
harness, scored on the *same* tasks? The bigger the spread between the top and
bottom bar (the **Δ** in the title), the more that model's success is dictated by
its scaffolding rather than its weights. Even on the shared suites the spreads stay
large — so the harness effect is **not** a coverage artefact. Read the **shared-suites
label under each title** before trusting a swing: a panel whose only shared suite is
`appworld` is an appworld-only verdict (narrower evidence than gpt-4.1's two-suite
span), even though the Δ can look just as dramatic. Bars whose Wilson whiskers don't
overlap differ beyond sampling noise.""")

md(r"""### 1f · Where do the 'gamblers' collapse?

§1b flagged configs with a high average but a high **cross-task std** — strong on
some suites, brittle on others. The cross-task std is a single number; this heatmap
is the receipt. Each cell is a config's success on one benchmark (greyed where that
config ran <5 tasks on the suite, too thin to read). Scan a row: an all-green row is
a genuine generalist; a row with a red cell is a gambler whose headline average hid
a suite where it falls apart.""")

code(r'''cb = rate_table(df, ["config", "benchmark"])
cb_n = cb.pivot(index="config", columns="benchmark", values="n")
cb_rate = cb.pivot(index="config", columns="benchmark", values="success_rate")
# Drop configs with no cell that clears the n>=5 bar (nothing to show but an empty row).
keep_rows = ((cb_n >= 5).sum(axis=1) > 0)
cb_rate, cb_n = cb_rate[keep_rows], cb_n[keep_rows]
# Order rows by overall macro (mean of available per-benchmark rates), best on top.
order_idx = cb_rate.mean(axis=1).sort_values(ascending=False).index
cb_rate, cb_n = cb_rate.loc[order_idx], cb_n.loc[order_idx]
# Mask thin cells (n < 5) so they read as "not enough data", not "scored 0".
mask = (cb_n < 5) | cb_rate.isna()
annot = cb_rate.copy()
annot_lab = annot.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")

fig, ax = plt.subplots(figsize=(10, 0.5 * len(cb_rate) + 2))
sns.heatmap(cb_rate, mask=mask, annot=annot_lab, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=2, linecolor="white", cbar_kws={"label": "success rate", "shrink": .7},
            ax=ax, annot_kws={"fontsize": 8.5, "color": "#1a1a1a"})
ax.set_facecolor("#f3f3f3")  # masked (thin) cells show as grey
titled(ax, "Config × benchmark success — spotting the gamblers",
       "grey = <5 tasks (too thin) · a red cell in an otherwise green row = a brittle suite")
ax.set_xlabel(""); ax.set_ylabel("")
ax.tick_params(length=0)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
plt.tight_layout(); plt.show()
''')

md(r"""Read with §1b: a config can earn a high macro average yet still own a
single red cell — that suite is where it will burn you in production. Conversely an
all-green row with tight cells (e.g. the top `claude_code` configs) is what
"reliable" actually looks like: high *and* even across task types.""")

md(r"""### 1g · Where the harness actually changes the outcome

§1e held the model fixed but averaged over suites. This fixes **both the model and
the benchmark** — the cleanest cut at the harness effect, difficulty confound and all.

**The honest filter:** a harness comparison only *exists* where **≥2 harnesses ran the
same (model, benchmark)** — anywhere else there is literally nothing to compare, so we
drop it. That alone removes a lot of the grid (e.g. `browsecompplus` disappears
entirely — only `claude_code` ever ran it), leaving just the cells that carry a real
signal. Each surviving panel is one (benchmark, model); bars are the harnesses that
ran it (±Wilson 95%), and panels are **ranked by Δ** — the success gap between the
best and worst harness — so the most decisive cases come first.""")

code(r'''cbm = rate_table(df, ["benchmark", "model", "harness"])
cbm = cbm[cbm.n >= 5]
# A (benchmark, model) cell is informative ONLY if >=2 harnesses ran it -- otherwise
# there is no harness comparison to make. Keep just those; rank by the harness swing.
keep = cbm.groupby(["benchmark", "model"]).harness.nunique().loc[lambda s: s >= 2].index
cells = [cbm[(cbm.benchmark == b) & (cbm.model == m)].sort_values("success_rate")
         for (b, m) in keep]
cells.sort(key=lambda g: g.success_rate.max() - g.success_rate.min(), reverse=True)

hh = sorted(df.harness.unique())
hpal = dict(zip(hh, sns.color_palette("Set2", len(hh))))
ncol = 3
nrow = int(np.ceil(len(cells) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 1.45 * nrow + 1), squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)
for i, grp in enumerate(cells):
    ax = axes[i // ncol][i % ncol]; ax.set_visible(True)
    err = np.vstack([grp.success_rate - grp.ci_lo, grp.ci_hi - grp.success_rate])
    ax.barh(range(len(grp)), grp.success_rate, xerr=err, height=0.62,
            color=[hpal[h] for h in grp.harness],
            error_kw=dict(ecolor="#666666", capsize=2.5, lw=1), zorder=3)
    for y, (_, s) in enumerate(grp.iterrows()):
        ax.text(min(s.ci_hi + 0.03, 1.0), y, f"{s.success_rate:.0%}", va="center",
                ha="left", fontsize=9, weight="semibold", color="#333333")
    ax.set_yticks(range(len(grp))); ax.set_yticklabels(grp.harness, fontsize=8.5)
    ax.set_xlim(0, 1.15); ax.set_ylim(-0.6, len(grp) - 0.4)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    b, m = grp.iloc[0][["benchmark", "model"]]
    delta = grp.success_rate.max() - grp.success_rate.min()
    ax.set_title(f"{b} · {m}   Δ{delta:.0%}", loc="left", fontsize=10, fontweight="semibold")
    sns.despine(ax=ax, left=True); ax.tick_params(left=False)
fig_title(fig, "Where the harness changes the outcome — model & benchmark both fixed",
          "only cells where ≥2 harnesses ran the same (model, benchmark) · bar = success (±Wilson 95%) · "
          "panels ranked by Δ = best−worst harness")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print(f"{len(cells)} (benchmark, model) cells have a real harness comparison "
      f"(of {cbm.groupby(['benchmark','model']).ngroups} populated cells)")
''')

md(r"""Difficulty stripped out, the harness effect doesn't wash away — within a single
suite the *same model* swings by tens of points on scaffolding alone (top-left panels
clear **60 points**). These are the most defensible harness comparisons in the whole
analysis: the two biggest confounds, **model and task difficulty, are both held
fixed**, and only cells with an actual head-to-head are shown.""")

md(r"""### 1h · Confound-adjusted harness effect (two-way model)

§1e/§1g control the confounds by *slicing*; this controls them by *modelling*. We
fit a **linear probability model**

`success ~ C(benchmark) + C(model) + C(harness)`

with heteroskedasticity-robust (HC1) standard errors. Because benchmark and model
are in the model, each **harness coefficient is the effect of that harness in
percentage points, holding the suite mix and the model fixed** — exactly the
coverage-adjusted number the raw averages couldn't give. `tool_calling` (the
weakest) is the reference, so every coefficient reads as "points above
`tool_calling`, all else equal."

To settle *harness vs model* fairly, we also compute each factor's **incremental
R²** — how much variance it explains when *added on top of the other two*. That is
the adjusted analogue of §1d's one-way η², with the coverage confound removed.""")

code(r'''import statsmodels.formula.api as smf

dfm = df.dropna(subset=["harness", "model", "benchmark"]).copy()
full = smf.ols("success ~ C(benchmark) + C(model) + "
               "C(harness, Treatment(reference='tool_calling'))",
               data=dfm).fit(cov_type="HC1")

# --- harness coefficients (vs tool_calling reference) ---
hp = []
for term in full.params.index:
    if term.startswith("C(harness"):
        name = term.split("[T.")[1].rstrip("]")
        ci = full.conf_int().loc[term]
        hp.append({"harness": name, "coef": full.params[term],
                   "lo": ci[0], "hi": ci[1], "p": full.pvalues[term]})
hp.append({"harness": "tool_calling", "coef": 0.0, "lo": 0.0, "hi": 0.0, "p": np.nan})  # reference
hp = pd.DataFrame(hp).sort_values("coef")

# --- incremental R^2: variance each factor adds on top of the other two ---
def r2(formula):
    return smf.ols(formula, data=dfm).fit().rsquared
r2_full = r2("success ~ C(benchmark) + C(model) + C(harness)")
incr = pd.Series({
    "harness":   r2_full - r2("success ~ C(benchmark) + C(model)"),
    "model":     r2_full - r2("success ~ C(benchmark) + C(harness)"),
    "benchmark": r2_full - r2("success ~ C(model) + C(harness)"),
}).sort_values()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
# Left: harness coefficient forest plot
for y, (_, r) in enumerate(hp.iterrows()):
    is_ref = r.harness == "tool_calling"
    c = "#999999" if is_ref else ("#2a8a4a" if r.coef > 0 else "#c0392b")
    if not is_ref:
        axL.plot([r.lo, r.hi], [y, y], color=c, lw=2, alpha=0.5, zorder=2)
    axL.scatter(r.coef, y, s=130, color=c, edgecolor="white", linewidth=1.3, zorder=3)
    tag = "  (ref)" if is_ref else (f"  +{r.coef:.0%}" if r.coef >= 0 else f"  {r.coef:.0%}")
    axL.text(r.hi + 0.01 if not is_ref else 0.01, y, tag, va="center", fontsize=9,
             weight="semibold", color=c)
axL.axvline(0, ls="--", c="#bbbbbb", lw=1, zorder=0)
axL.set_yticks(range(len(hp))); axL.set_yticklabels(hp.harness)
axL.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axL.set_xlabel("effect on success vs. tool_calling (pp, adjusted for model + benchmark)")
titled(axL, "Adjusted harness effect", "linear prob. model · HC1 SEs · 95% CI", pad=20)
sns.despine(ax=axL, left=True); axL.tick_params(left=False)
# Right: incremental R^2 by factor
colors = {"harness": "#30638e", "model": "#9bb8d3", "benchmark": "#cfcfcf"}
axR.barh(incr.index, incr.values, color=[colors[f] for f in incr.index], height=0.6, zorder=3)
for y, v in enumerate(incr.values):
    axR.text(v + 0.003, y, f"{v:.1%}", va="center", fontsize=10, weight="semibold", color="#333")
axR.set_xlim(0, max(incr.values) * 1.2)
axR.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axR.set_xlabel("incremental R²  (added on top of the other two)")
titled(axR, "Which knob matters, adjusted?", "coverage confound removed", pad=20)
sns.despine(ax=axR, left=True); axR.tick_params(left=False)
plt.tight_layout(); plt.show()

print(f"full model R² = {r2_full:.3f}  (n={int(full.nobs)})")
hp.set_index("harness")[["coef", "lo", "hi", "p"]].round(3)
''')

md(r"""**This is the headline, properly adjusted.** The left panel ranks harnesses
by their effect on success *with model and benchmark held fixed* — `tool_calling`
is the floor and the code/agentic harnesses sit well above it, with CIs that clear
zero. The right panel is the clean version of §1d: even after removing the coverage
confound, **harness adds more incremental R² than model does**, confirming the
scaffolding is the larger lever. (LPM is used for interpretable percentage-point
effects; a logit fit gives the same ordering and significance.)""")

# ================================================================ SECTION 2
md(r"""## 2 · Failure-mode indexing — do harnesses fail differently?

We have several orthogonal error signals per run:

- `tool_error_count` — failed tool/function calls
- `error_span_count` — spans the tracer marked as errored
- `total_tokens` / `num_llm_calls` — proxy for non-convergence / looping
- `judge_reasoning` — free-text explanation from the judge

Among the **failed** runs we bucket each into a dominant failure mode, then ask
whether the mix of modes differs by harness.

> **Coverage caveat (per §1·0):** harnesses ran *different* benchmark mixes, and
> different suites fail in different ways (a `swebench` failure looks nothing like a
> `tau2` one). So a harness's *pooled* failure fingerprint partly reflects **which
> suites it happened to run**, not how the harness itself fails. The pooled views
> (§2a/§2b/§2c) are kept as an overview but flagged as confounded; the honest
> read is the **benchmark-faceted** version beside each, which holds the suite
> fixed and only compares harnesses that actually ran it.
""")

code(r'''fails = df[df.success == 0].copy()
print(f"{len(fails)} failed runs ({len(fails)/len(df):.0%} of all)")

tok_hi = df.total_tokens.quantile(0.90)  # global "blow-up" threshold

def classify(r):
    if r.tool_error_count >= 5:
        return "tool-call errors"
    if r.total_tokens >= tok_hi or r.num_llm_calls >= df.num_llm_calls.quantile(0.90):
        return "non-convergence / loop"
    if r.error_span_count > 0:
        return "runtime / span error"
    return "silent wrong answer"

fails["mode"] = fails.apply(classify, axis=1)
fails["mode"].value_counts()
''')

md("### 2a · Failure-mode mix by harness (composition, not volume)")

code(r'''order = fails.harness.value_counts().index.tolist()
modes = ["tool-call errors", "non-convergence / loop", "runtime / span error", "silent wrong answer"]
counts = (fails.groupby(["harness", "mode"]).size()
          .unstack("mode").reindex(order)[modes].fillna(0))
mp = counts.div(counts.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(mp))
colors = sns.color_palette("Set2", len(modes))
n_lab = fails.harness.value_counts().reindex(order)
xpos = np.arange(len(mp))
for c, m in zip(colors, modes):
    ax.bar(xpos, mp[m], bottom=bottom, label=m, color=c, width=0.68, zorder=3)
    for x, (frac, b) in enumerate(zip(mp[m].values, bottom)):
        if frac >= 0.07:  # only label visible segments
            ax.text(x, b + frac / 2, f"{frac:.0%}", ha="center", va="center",
                    fontsize=8.5, color="#333333", weight="medium")
    bottom += mp[m].values
ax.set_ylabel("share of harness's failures")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xticks(xpos)
ax.set_xticklabels([f"{h}\n(n={n_lab[h]})" for h in mp.index], rotation=12, ha="center")
ax.legend(title="failure mode", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
titled(ax, "How each harness fails", "normalised within harness — bars sum to 100%")
sns.despine(ax=ax)
plt.tight_layout(); plt.show()
mp.round(2)
''')

md(r"""If harnesses failed the same way, every bar would have identical
stripes. Differences here are the harness-specific fingerprints — e.g. a
code-execution harness leaning on `tool-call errors` vs. a solo prompt leaning
on `silent wrong answer`. **But** (see the caveat above) this pools each harness
over its own suite mix — so part of any difference is just *which benchmarks* the
harness ran. The next chart removes that.""")

md("### 2a-2 · Failure-mode mix by harness, **within each benchmark**")

code(r'''MIN_FAIL = 15  # need this many failures in a (benchmark, harness) cell to show it
fc = fails.groupby(["benchmark", "harness", "mode"]).size().rename("k").reset_index()
cell_tot = fails.groupby(["benchmark", "harness"]).size().rename("tot")
fc = fc.join(cell_tot, on=["benchmark", "harness"])
fc["frac"] = fc.k / fc.tot
# keep (benchmark, harness) cells with enough failures, and benchmarks with >=2 such harnesses
ok_cells = cell_tot[cell_tot >= MIN_FAIL]
ok = ok_cells.reset_index()
bench_keep = ok.groupby("benchmark").harness.nunique()
bench_keep = bench_keep[bench_keep >= 2].index.tolist()

ncol = 2
nrow = int(np.ceil(len(bench_keep) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow + 0.5), squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)
mode_colors = dict(zip(modes, sns.color_palette("Set2", len(modes))))
for bi, bench in enumerate(sorted(bench_keep)):
    ax = axes[bi // ncol][bi % ncol]; ax.set_visible(True)
    hs = sorted(ok[ok.benchmark == bench].harness)
    sub = fc[(fc.benchmark == bench) & (fc.harness.isin(hs))]
    piv = sub.pivot_table(index="harness", columns="mode", values="frac", fill_value=0).reindex(hs)
    piv = piv.reindex(columns=modes, fill_value=0)
    bottom = np.zeros(len(piv))
    xpos = np.arange(len(piv))
    for m in modes:
        ax.bar(xpos, piv[m], bottom=bottom, color=mode_colors[m], width=0.66, zorder=3)
        for x, (frac, b) in enumerate(zip(piv[m].values, bottom)):
            if frac >= 0.10:
                ax.text(x, b + frac / 2, f"{frac:.0%}", ha="center", va="center",
                        fontsize=8, color="#333333")
        bottom += piv[m].values
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{h}\n(n={int(cell_tot[(bench, h)])})" for h in piv.index],
                       fontsize=8, rotation=0)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(bench, loc="left", fontsize=11, fontweight="semibold")
    sns.despine(ax=ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=mode_colors[m]) for m in modes]
fig.legend(handles, modes, title="failure mode", bbox_to_anchor=(0.5, -0.01),
           loc="upper center", ncol=len(modes), frameon=False)
fig_title(fig, "How each harness fails — within a fixed benchmark",
          f"only (benchmark, harness) cells with ≥{MIN_FAIL} failures · bars sum to 100%")
plt.tight_layout(rect=[0, 0.04, 1, 0.97]); plt.show()
''')

md(r"""Now a harness's fingerprint can be compared *like-for-like*: each panel fixes
the benchmark, so any difference in stripe pattern is the harness, not the task mix.
Where a harness keeps the same fingerprint across panels, that's a genuine
harness-level failure signature; where its stripes change suite-to-suite, the
"fingerprint" from the pooled chart was partly a coverage artefact.""")

md("### 2b · Error-signal distributions by harness (failed runs)")

code(r'''fig, axes = plt.subplots(1, 2, figsize=(14, 6))
short = {h: h.replace("_", "\n", 1).replace("tool_calling_with_shortlisting", "tool_calling\n+shortlist")
         for h in order}
sns.boxplot(data=fails, x="harness", y="tool_error_count", order=order,
            hue="harness", palette="Set2", legend=False, ax=axes[0],
            showfliers=False, linewidth=1.2, width=0.6)
axes[0].set_title("Tool-call errors per failed run", loc="left")
sns.boxplot(data=fails, x="harness", y="total_tokens", order=order,
            hue="harness", palette="Set2", legend=False, ax=axes[1],
            showfliers=False, linewidth=1.2, width=0.6)
axes[1].set_title("Token usage per failed run", loc="left")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
fig_title(fig, "Error-signal distributions by harness (failed runs)",
          "box = IQR (25th–75th pct) · line = median · whiskers = 1.5×IQR · outliers hidden")
for a in axes:
    a.set_xlabel("")
    a.set_xticklabels([short[t.get_text()] for t in a.get_xticklabels()], fontsize=8.5)
    sns.despine(ax=a)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

# Success vs failure on the discriminating signals
df.groupby("success")[["tool_error_count", "error_span_count", "total_tokens", "num_llm_calls"]].mean().round(1)
''')

md(r"""Token usage is **dominated by the benchmark** (`swebench` runs are far
longer than `tau2`), so the pooled boxplot above mostly ranks harnesses by the
suites they ran. (`tool_error_count`, by contrast, is near-zero everywhere except
`appworld` — too sparse to facet usefully.) Holding the benchmark fixed below
isolates each harness's own **token-burn / non-convergence** profile — does a
harness run hot (looping, re-trying) on the *same* tasks?""")

md("### 2b-2 · Token usage per failed run, **within each benchmark**")

code(r'''fb = fails.groupby(["benchmark", "harness"]).size()
keep = fb[fb >= MIN_FAIL].reset_index()
bench_keep2 = sorted(keep.groupby("benchmark").harness.nunique().loc[lambda s: s >= 2].index)

ncol = 2
nrow = int(np.ceil(len(bench_keep2) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow + 0.5), squeeze=False)
for ax in axes.flat:
    ax.set_visible(False)
hpal2 = dict(zip(sorted(fails.harness.unique()), sns.color_palette("Set2", fails.harness.nunique())))
for bi, bench in enumerate(bench_keep2):
    ax = axes[bi // ncol][bi % ncol]; ax.set_visible(True)
    hs = sorted(keep[keep.benchmark == bench].harness)
    sub = fails[(fails.benchmark == bench) & (fails.harness.isin(hs))]
    sns.boxplot(data=sub, x="harness", y="total_tokens", order=hs,
                hue="harness", palette=hpal2, legend=False, ax=ax,
                showfliers=False, linewidth=1.1, width=0.6)
    ax.set_title(bench, loc="left", fontsize=11, fontweight="semibold")
    ax.set_xlabel(""); ax.set_ylabel("tokens / run")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.set_xticklabels([f"{h}\n(n={int(fb[(bench, h)])})" for h in hs], fontsize=8)
    sns.despine(ax=ax)
fig_title(fig, "Token usage per failed run — benchmark held fixed",
          f"box = IQR · line = median · whiskers = 1.5×IQR · outliers hidden · "
          f"(benchmark, harness) cells with ≥{MIN_FAIL} failures")
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()
''')

md(r"""### 2c · What the judge says — discriminating terms, **on one shared benchmark**

Comparing judge-reasoning vocabulary across harnesses pooled over all suites has the
same confound: each suite has its own vocabulary (`patch`/`test` for swebench,
`flight`/`refund` for tau2). To compare *harnesses* fairly we fix the benchmark to
the one with the broadest harness coverage among failures, and compute each
harness's term frequency relative to the **within-that-benchmark** base — so an
enriched term reflects how that harness fails *on the same tasks*, not which tasks
it ran.""")

code(r'''STOP = set("the a an and or of to in is was for with on at by it that this be as not no "
           "user agent did does provide provided required answer task is were are has have "
           "been which from any clear correct incorrect found evidence based information".split())

def keywords(texts):
    cnt = {}
    for t in texts:
        for w in set(re.findall(r"[a-z]{4,}", t.lower())):
            if w not in STOP:
                cnt[w] = cnt.get(w, 0) + 1
    n = len(texts) or 1
    return {w: c / n for w, c in cnt.items()}

# Pick the benchmark with the most harnesses having >=15 failures (ties -> most failures).
fcount = fails.groupby(["benchmark", "harness"]).size()
cand = (fcount[fcount >= MIN_FAIL].reset_index()
        .groupby("benchmark").agg(nh=("harness", "nunique")))
cand["total"] = fails.groupby("benchmark").size()
FOCUS_BM = cand.sort_values(["nh", "total"], ascending=False).index[0]
fbm = fails[fails.benchmark == FOCUS_BM]
focus_h = sorted(h for h in fbm.harness.unique()
                 if (fcount.get((FOCUS_BM, h), 0) >= MIN_FAIL))

base = keywords(fbm.judge_reasoning)
rows = []
for h in focus_h:
    kw = keywords(fbm[fbm.harness == h].judge_reasoning)
    for w, f in kw.items():
        if f >= 0.10:  # appears in >=10% of this harness's failures on FOCUS_BM
            rows.append({"harness": h, "term": w, "lift": f / (base.get(w, 1e-9))})
lift = pd.DataFrame(rows)
top = (lift[lift.lift > 1].sort_values("lift", ascending=False).groupby("harness").head(6))

ncol = min(3, len(focus_h))
g = sns.catplot(data=top, kind="bar", y="term", x="lift", col="harness", col_order=focus_h,
                col_wrap=ncol, sharey=False, sharex=False, height=3.0, aspect=1.5,
                hue="term", palette="rocket_r", legend=False)
g.set_titles("{col_name}", size=11, weight="semibold")
g.set_axis_labels("lift vs. all failures on this benchmark", "")
for ax in g.axes.flat:
    ax.tick_params(labelsize=9)
    ax.axvline(1, ls="--", c="#bbbbbb", lw=1, zorder=0)  # 1 = no enrichment
g.fig.suptitle(f"Distinctive judge-reasoning terms per harness — on {FOCUS_BM}", y=1.07,
               fontsize=14, weight="bold")
g.fig.text(0.5, 1.015, f"failed runs on {FOCUS_BM} only · harnesses with ≥{MIN_FAIL} failures · "
           "lift vs. this benchmark's failure base",
           ha="center", fontsize=9, color="#888888")
g.fig.subplots_adjust(top=0.90, hspace=0.45)
plt.show()
print(f"focus benchmark: {FOCUS_BM} · harnesses: {focus_h}")
''')

# ================================================================ SECTION 3
md(r"""## 3 · Benchmark bleed (contamination) check

Several suites here are **public, static, and pre-date these models' training
cutoffs** — so solutions plausibly leaked into pre-training. Treating a
contaminated score as a measure of *capability* is a mistake; it partly measures
*memorisation*. Risk tiers used below (and easily edited):

| risk | benchmarks | rationale |
|---|---|---|
| **high** | `swebench` | gold patches + issues sit on public GitHub, heavily mirrored & studied |
| **medium** | `tau2_retail`, `tau2_airline`, `tau2_telecom`, `appworld` | public static benchmarks with released task data |
| **low** | `browsecompplus` | live-web retrieval; answers shift, harder to memorise |

The test: **exclude the high-risk suite and see whether the model/harness
ranking and the reliability conclusions from §1 still hold.** If they do, the
findings are robust to bleed; if they move, the leaderboard was partly
measuring recall.

> **Coverage caveat (per §1·0):** the only low-bleed suite, `browsecompplus`, was
> run by **`claude_code` alone**. So the high-vs-clean gap below would silently
> compare SWE-bench-across-many-harnesses against BrowseComp+-on-claude_code — a
> harness confound dressed up as contamination. We therefore compute the gap
> **within `claude_code` only** (the single harness present in both tiers). It
> also means the "clean suites" reliability quadrant in §3c is effectively a
> claude_code view, not a cross-harness one — read it that way.
""")

code(r'''RISK = {
    "swebench": "high",
    "tau2_retail": "medium", "tau2_airline": "medium",
    "tau2_telecom": "medium", "appworld": "medium",
    "browsecompplus": "low",
}
df["bleed_risk"] = df.benchmark.map(RISK)
clean = df[df.bleed_risk == "low"]            # contamination-resistant only
no_high = df[df.bleed_risk != "high"]         # drop the worst offender

print("rows by risk tier:")
print(df.bleed_risk.value_counts(), "\n")

# Per-model high-vs-clean gap, restricted to claude_code so the comparison isn't
# confounded by harness (browsecompplus, the only clean suite, is claude_code-only).
cc = df[df.harness == "claude_code"]
gap = []
for m, g in cc.groupby("model"):
    hi = g[g.bleed_risk == "high"]; lo = g[g.bleed_risk == "low"]
    if len(hi) >= 8 and len(lo) >= 8:
        gap.append({"model": m, "high_risk_success": hi.success.mean(),
                    "clean_success": lo.success.mean(),
                    "gap": hi.success.mean() - lo.success.mean(),
                    "n_high": len(hi), "n_low": len(lo)})
gap = pd.DataFrame(gap).sort_values("gap", ascending=False)
print("(gap computed within claude_code only — see coverage caveat above)")
gap.round(3)
''')

md("### 3a · Contaminated-vs-clean success gap per model")

code(r'''if len(gap):
    fig, ax = plt.subplots(figsize=(10, 6))
    gm = gap.melt(id_vars="model", value_vars=["high_risk_success", "clean_success"],
                  var_name="suite", value_name="rate")
    gm["suite"] = gm["suite"].map({"high_risk_success": "SWE-bench (high bleed)",
                                   "clean_success": "BrowseComp+ (clean)"})
    sns.barplot(data=gm, x="model", y="rate", hue="suite",
                palette=["#d1495b", "#30638e"], ax=ax, width=0.7, zorder=3)
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{v*100:.0f}%" for v in c.datavalues],
                     fontsize=8.5, color="#444444", padding=2)
    titled(ax, "High-bleed vs. clean success per model",
           "large gap → score may be inflated by memorisation")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate"); ax.set_xlabel("")
    ax.legend(title="", loc="upper right")
    sns.despine(ax=ax)
    plt.tight_layout(); plt.show()
else:
    print("Not enough overlapping samples to compute per-model bleed gap.")
''')

md(r"""### 3b · Does the ranking survive excluding the high-risk suite?

> **Coverage gate (per §1·0):** "exclude SWE-bench and see if ranks move" is only
> meaningful for models that *ran* SWE-bench **and** still have ≥2 other suites left
> to rank on afterwards. A model that never ran it can't move, and one that ran
> *only* it has nothing left — including either would fake stability. We restrict to
> the eligible set and **recompute ranks within it**, so the bump chart is a
> like-for-like contamination test, not a coverage artefact.""")

code(r'''def macro_rate(frame, label, min_cell=8):
    cells = rate_table(frame, ["model", "benchmark"])
    cells = cells[cells.n >= min_cell]
    return (cells.groupby("model")
            .agg(**{"rate_" + label: ("success_rate", "mean"),
                    "nbm_" + label: ("benchmark", "nunique")}))

# Eligibility: ran swebench (>=8 tasks) AND >=2 non-swebench suites (>=8) so that
# dropping swebench leaves a comparable, non-empty macro for every model shown.
cmb = rate_table(df, ["model", "benchmark"]); cmb = cmb[cmb.n >= 8]
ran_sb = set(cmb[cmb.benchmark == "swebench"].model)
other_n = cmb[cmb.benchmark != "swebench"].groupby("model").benchmark.nunique()
eligible = sorted(m for m in ran_sb if other_n.get(m, 0) >= 2)

comp = macro_rate(df, "all").join(macro_rate(no_high, "no_high"), how="outer")
comp = comp.loc[[m for m in comp.index if m in eligible]]
# Recompute ranks *within* the eligible set so they're comparable across the two views.
comp = comp.sort_values("rate_all", ascending=False); comp["rank_all"] = range(1, len(comp) + 1)
comp = comp.sort_values("rate_no_high", ascending=False); comp["rank_no_high"] = range(1, len(comp) + 1)
comp["rank_shift"] = comp["rank_no_high"] - comp["rank_all"]
comp = comp.sort_values("rate_all", ascending=False)
print(f"eligible models (ran swebench + >=2 other suites): {eligible}")
display(comp.round(3))

# Slope/bump chart of rank change
fig, ax = plt.subplots(figsize=(9, 6))
bump = comp.dropna(subset=["rank_all", "rank_no_high"])
pal = dict(zip(bump.index, sns.color_palette("crest", len(bump))))
for m, r in bump.iterrows():
    moved = r.rank_no_high != r.rank_all
    ax.plot([0, 1], [r.rank_all, r.rank_no_high], "-o", lw=2.4 if moved else 1.6,
            color=pal[m], markersize=8, alpha=1 if moved else 0.55, zorder=3)
    ax.text(-0.04, r.rank_all, f"{m}  ({r.rate_all:.0%})", ha="right", va="center",
            fontsize=9.5, color="#333333")
    tag = m if not moved else f"{m}  ▲" if r.rank_no_high < r.rank_all else f"{m}  ▼"
    ax.text(1.04, r.rank_no_high, f"{tag}  ({r.rate_no_high:.0%})", ha="left", va="center",
            fontsize=9.5, color="#333333")
ax.set_xlim(-0.75, 1.75); ax.invert_yaxis()
ax.set_xticks([0, 1]); ax.set_xticklabels(["all benchmarks", "SWE-bench excluded"], fontsize=10)
ax.set_yticks([]); ax.set_ylabel("rank  (1 = best)")
ax.set_title("Model leaderboard — does excluding the high-bleed suite move ranks?", loc="left")
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout(); plt.show()
''')

md("### 3c · Reliability quadrant recomputed on clean suites only")

code(r'''def quadrant(frame, ax, title):
    # Benchmark-balanced (macro) x-axis, same as §1b — so the "does it survive?"
    # comparison isn't itself reintroducing the mix confound.
    r = config_reliability(frame, min_cell=4, min_total=8).dropna(subset=["macro"])
    r = r[r.n_benchmarks >= 2]  # need >=2 benchmarks for a meaningful cross-task std
    sns.scatterplot(data=r, x="macro", y="cross_task_std", size="n", sizes=(70, 420),
                    hue="macro", palette="crest", edgecolor="white",
                    linewidth=1, legend=False, ax=ax, zorder=3)
    repel_labels(ax, r.macro, r.cross_task_std, r.config, fontsize=7.5)
    ax.set_title(title, loc="left"); ax.set_xlabel("benchmark-balanced success rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.margins(0.1)
    return r

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
quadrant(df, axes[0], "All benchmarks")
quadrant(no_high, axes[1], "SWE-bench excluded")
axes[0].set_ylabel("cross-task std of success")
fig.suptitle("Reliability conclusions, with & without the high-bleed suite", y=1.0)
plt.tight_layout(); plt.show()
''')

# ================================================================ takeaways
md(r"""## 4 · How to use this

**The coverage caveat runs through everything.** Only 2 of 22 configs ran all six
suites (§1·0); `browsecompplus` is `claude_code`-only. So *any* number pooled across
benchmarks is partly a statement about which suites a config happened to run. Every
cross-config / cross-harness claim below is the version that equalises coverage —
either by restricting to a shared suite set, faceting by benchmark, or adjusting for
benchmark as a covariate.

- **§1 — reliability.** Rank by `floor` (Wilson lower bound), not raw average, and
  ignore the hatched (<3-suite) bars for cross-config comparison. The harness-effect
  headline survives the coverage fix: in the adjusted model (§1h) `claude_code` is
  **+28 pp over `tool_calling` holding model and benchmark fixed**, and harness still
  adds more incremental R² than model (≈5% vs ≈1%) — but the *raw* §1d gap was
  inflated by coverage, and some per-model swings (e.g. claude-opus in §1e) shrink to
  near-noise once you restrict to shared suites. Believe §1g/§1h over §1a/§1d.
- **§2 — failure modes.** Read the **benchmark-faceted** panels (§2a-2, §2b-2, §2c),
  not the pooled ones: a harness "fingerprint" that changes suite-to-suite was a
  coverage artefact; one that holds within every benchmark is a real harness
  signature worth designing mitigations around.
- **§3 — bleed.** The contamination tests are gated to comparable coverage: the
  high-vs-clean gap is computed within `claude_code` only, and the rank-survival test
  only to models that ran SWE-bench plus ≥2 other suites. Re-run with your own `RISK`
  mapping to match your training-cutoff knowledge.

Knobs to tweak: `MIN_N`, `COMPARABLE_BM`, `MIN_FAIL`, the `classify()` thresholds,
the keyword `STOP` set, and the `RISK` dictionary.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, str(REPO / "reliability_analysis.ipynb"))
print("wrote reliability_analysis.ipynb with", len(cells), "cells")


# --------------------------------------------------------------- figure export
# Run `python scripts/build_notebook.py --export` to (re)generate the PNGs in
# out/plots/ that ANALYSIS-SUMMARY.md embeds. We exec each code cell in one shared
# namespace (so later cells see earlier dataframes) and, whenever a cell leaves an
# open figure, save it under the next filename in PLOT_FILES. The order of figure-
# producing cells is stable, so the sequential mapping below stays correct.
PLOT_FILES = [
    "01_1.png", "02_1a.png", "03_1b.png", "04_1c.png", "05_1d.png",
    "06_1e.png", "07_1f.png", "08_1g.png", "09_1h.png", "10_2a.png",
    "11_2a_2.png", "12_2b.png", "13_2b_2.png", "14_2c.png",
    "15_3a.png", "16_3b.png", "17_3c.png",
]


def export_figures(cells):
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")  # headless: render to file, never to screen
    import matplotlib.pyplot as plt

    # Run cells as if launched from the repo root: relative paths in the cells
    # ("data/full.json", "scripts/") and `import bt_helpers` all resolve there.
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "scripts"))
    out = REPO / "out/plots"; out.mkdir(parents=True, exist_ok=True)
    ns = {"display": lambda *a, **k: None, "__name__": "__nb__"}
    saved = []
    plt.show = lambda *a, **k: None  # cells call plt.show(); we save instead
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        exec(cell["source"], ns)
        for num in plt.get_fignums():           # save every figure this cell opened
            if len(saved) >= len(PLOT_FILES):
                raise SystemExit(f"more figures than names ({len(PLOT_FILES)})")
            fig = plt.figure(num)
            fig.savefig(out / PLOT_FILES[len(saved)], dpi=120, bbox_inches="tight")
            saved.append(PLOT_FILES[len(saved)])
            plt.close(fig)
    if len(saved) != len(PLOT_FILES):
        raise SystemExit(f"expected {len(PLOT_FILES)} figures, saved {len(saved)}: {saved}")
    print(f"exported {len(saved)} figures to {out}/")


if "--export" in __import__("sys").argv:
    export_figures(cells)
