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

## Scripts

| Script | What it does |
|---|---|
| [`scripts/convert_hf_traces.py`](./scripts/convert_hf_traces.py) | Ports the HuggingFace dataset → Braintrust logs (one root span/session + child spans/LLM call). |
| [`scripts/score_and_push.py`](./scripts/score_and_push.py) | LLM-as-judge scorer — judges each session and writes `scores.task_success` back via `bt sync push`. |
| [`scripts/explore_trace.py`](./scripts/explore_trace.py) | Helper to dump a single trace's spans for inspection. |
