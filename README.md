# HuggingFace × Braintrust — Agent Trace Analysis

An analysis of the [`Exgentic/agent-llm-traces`](https://huggingface.co/datasets/Exgentic/agent-llm-traces)
dataset (~1,800 agent sessions across 6 benchmarks and 6 models), explored in
[Braintrust](https://www.braintrust.dev/). We look at agent **efficiency**
(tokens, latency, LLM calls) and **task success**, sliced by model × benchmark ×
agent harness.

## 📄 Start here

**[ANALYSIS-SUMMARY.md](./ANALYSIS-SUMMARY.md)** — the full write-up: dataset,
how it was ported into Braintrust, the benchmarks and harnesses, the LLM-as-judge
success measure (incl. the prompt and a worked example), the combined
efficiency × success table, and the main learnings.

## Highlights

- **Open weights are frontier-class at coding** — on SWEBench (harness held
  constant), DeepSeek (96%) and Kimi (94%) basically tie the best closed models.
- **The harness matters as much as the model** — the same model can swing ~30% →
  ~94% success just by changing the agent scaffold.
- **"Cheap" ≠ good** — GPT-4.1 looked token-efficient but was often giving up early.
- **Failure looks opposite by task type** — coding tasks fail by *thrashing* (more
  tokens, still fails); conversational tasks fail by *giving up* (bails early).
- **Control for benchmark AND harness** before comparing models, or results get
  confounded.

> Task success is an LLM-as-judge proxy (the agent's own verification), not the
> official benchmark grader — see the caveats in the summary.

We also highlight **[Braintrust Topics](https://www.braintrust.dev/blog/topics)** —
AI-powered clustering that auto-organized the traces into named **Task** and
**Issues** groups, recovering the benchmark structure and the failure taxonomy with
no manual tagging (see §1c and §6 of the summary).

## Repo layout

| Path | What it is |
|---|---|
| [`ANALYSIS-SUMMARY.md`](./ANALYSIS-SUMMARY.md) | The full write-up (start here). |
| [`reliability_analysis.ipynb`](./reliability_analysis.ipynb) | The analysis notebook — all stats and figures. |
| [`scripts/`](./scripts/) | Analysis and data processing scripts (see below). |
| [`out/plots/`](./out/plots/) | The 17 figures embedded in the summary. |
| [`bt_screencaptures/`](./bt_screencaptures/) | Braintrust UI screenshots (Logs view + Topics facets). |
| [`data/`](./data/) | `full.json` (all 1,781 root spans, full API pull) + `reliability.json`. |

## Scripts

| Script | What it does |
|---|---|
| [`scripts/build_notebook.py`](./scripts/build_notebook.py) | Generates the notebook from `data/full.json`; `--export` also re-renders every plot in `out/plots/`. |
| [`scripts/bt_helpers.py`](./scripts/bt_helpers.py) | Braintrust connection + BTQL query helpers (pulls logs into pandas). |
| [`scripts/convert_hf_traces.py`](./scripts/convert_hf_traces.py) | Ports the HuggingFace dataset → Braintrust logs (one root span/session + child spans/LLM call). |
| [`scripts/score_and_push.py`](./scripts/score_and_push.py) | LLM-as-judge scorer — judges each session and writes `scores.task_success` back via `bt sync push`. |
| [`scripts/explore_trace.py`](./scripts/explore_trace.py) | Helper to dump a single trace's spans for inspection. |
| [`scripts/verify_pull.py`](./scripts/verify_pull.py) | Reproducibility check — re-pulls via BTQL and diffs row-for-row against `data/full.json`. |

## Setup

```bash
cp .env.example .env   # add your BRAINTRUST_API_KEY
pip install -r requirements.txt   # or: pandas requests python-dotenv matplotlib seaborn statsmodels
python scripts/build_notebook.py            # regenerates reliability_analysis.ipynb (run from repo root)
python scripts/build_notebook.py --export   # also re-renders every PNG in out/plots/
```
