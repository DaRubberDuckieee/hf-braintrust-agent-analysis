# hf-to-braintrust

You found a dataset on [HuggingFace](https://huggingface.co/datasets) and you want it in
[Braintrust](https://www.braintrust.dev/) — a benchmark to evaluate against, or a pile of agent
traces to dig through. Either way you need a small converter: `load_dataset`, look at the columns,
map them into Braintrust's shape, push.

This folder is that converter as a **cookbook** — two short scripts you read, copy, and edit. The
column mapping lives in plain Python at the top of each one, so what happens to your data is right
in front of you, not hidden behind config or inference.

> **What this is (and isn't).** Worked examples you adapt — **not** an officially supported Braintrust
> product, and not covered by support or SLAs. No promise they handle every dataset. Preview before
> you push, and expect to edit the mapping. HuggingFace and Braintrust both move fast, so a mapping
> that ports cleanly today may need a tweak later.

## Which script?

| You have… | You want… | Use | It builds |
|---|---|---|---|
| A benchmark with golden answers | To **run and score** evals | [`import_dataset.py`](import_dataset.py) | a Braintrust **Dataset** — rows of `{input, expected, metadata}` |
| Already-run agent traces | To **explore and cluster** them | [`import_logs.py`](import_logs.py) | Braintrust **Logs** — a span tree per row |

Rule of thumb: **run → Dataset, analyze → Logs.** Have traces but want a gradable benchmark out of
them? That's the *traces-as-expected* mapping noted inside `import_dataset.py` — lift each trace's
task into `input` and its recorded answer into `expected`.

The fiddly parts — normalizing chat / OpenTelemetry-GenAI messages, and building the span tree with
deterministic IDs — live in two helpers the scripts import, [`normalize.py`](normalize.py) and
[`braintrust_logs.py`](braintrust_logs.py). You shouldn't need to touch those.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install datasets braintrust requests python-dotenv   # + the `bt` CLI for the logs push
cp ../.env.example .env                                   # then add BRAINTRUST_API_KEY

python hf_bt_cookbook/import_dataset.py            # preview the first few records — no writes
python hf_bt_cookbook/import_dataset.py --push     # insert into Braintrust
```

Both scripts have one **EDIT ME** block at the top, and the same loop: **edit the mapping → run to
preview → flip `--push` to write.** Nothing leaves your machine until you pass `--push` (or set
`PUSH = True`).

| Variable | Required | Purpose |
|---|---|---|
| `BRAINTRUST_API_KEY` | to write | Dataset insert / project resolution / `bt sync push`. |
| `HF_TOKEN` | optional | Higher HF rate limits; required for gated or private datasets. |
| `BRAINTRUST_API_URL` | optional | Self-hosted only; defaults to `https://api.braintrust.dev`. |

Keys are read from the environment by the SDK/CLI — the scripts never print them or put them in the
payload they upload.

## Dataset: edit the mapping, then push

Open [`import_dataset.py`](import_dataset.py), set the repo/split, and rewrite `to_record` — that
function *is* the mapping:

```python
HF_REPO, HF_SUBSET, HF_SPLIT = "openai/gsm8k", "main", "test"
BT_PROJECT, BT_DATASET = "hf-imports", "gsm8k"

def to_record(row, i):
    return {
        "id": str(row.get("task_id") or i),
        "input": row["question"],
        "expected": row["answer"],          # optional
        "metadata": {"hf_dataset": HF_REPO, "hf_subset": HF_SUBSET, "hf_split": HF_SPLIT},
    }
```

Run without `--push` to print the first few records and see the exact shape that would land in
Braintrust. When it looks right, `--push`. The comments in the file show the common variations:

- **Chat-formatted prompt** (a column that's a list of `{role, content}`):
  `"input": normalize_messages(parse_messages(row["messages"]))`
- **Several columns into metadata:** `"metadata": {"category": row["category"], …}`
- **Traces-as-expected:** lift a recorded trace's task into `input` and its final answer into
  `expected` (uses the trace helpers from `import_logs.py`).

> A recorded answer is a *reference*, not necessarily ground truth. Grade traces-as-expected with an
> LLM-judge or similarity scorer rather than exact match — unless the answers are genuinely canonical.

## Logs: point it at the trace columns

Open [`import_logs.py`](import_logs.py) and say which columns hold the session id, the trace, the
metadata, and any scores:

```python
HF_REPO    = "Exgentic/agent-llm-traces"
BT_PROJECT = "Hugging Face topics"   # must already exist
ID_COL     = "session_id"            # drives the deterministic root-span id
TRACE_COL  = "spans"                 # list of OTel spans per session
METADATA_COLS = ["benchmark", "harness", "models", "total_tokens"]
SCORE_COLS = {}                      # {"task_success": "judge_task_success"}
KEYS = TraceKeys()                   # override only if your spans aren't OTel-GenAI
```

```bash
python hf_bt_cookbook/import_logs.py            # writes the bt-sync JSONL to out/logs
python hf_bt_cookbook/import_logs.py --push     # also runs `bt sync push` for you
```

Each row becomes a span tree:

```
root  task span   (one per session)
│       input  = first user message
│       output = final answer  (+ tool errors surfaced here, so Topics/Issues can cluster on them)
├── llm span      call 1   — token metrics, model
├── llm span      call 2
└── …             one child per LLM call
```

A row with no trace column still imports as a single root span carrying just its metadata.

## Re-runs are safe — and that's also how you write scores back

Root span IDs are deterministic: `uuid5("root:" + session_id)`. So re-importing the same sessions
**upserts** them instead of creating duplicates. Two things fall out of that:

- **A run that fails partway is fine.** Hit an HF or Braintrust rate cap, lose your connection — just
  re-run. Already-imported sessions overwrite themselves and you converge on the full set, no dupes.
  (Re-runs restart from the first row; there's no incremental checkpoint.)
- **Scores write back the same way.** Spans are immutable, so you don't patch them — you rebuild the
  row and re-push. Join your judge's output onto the source rows as a column, add it to `SCORE_COLS`,
  and re-run. Same ids → the scores land on the existing spans.

## What keeps imports safe

The guardrails live in the helpers, applied explicitly so you can see and adjust them:

- **Credentials stay out of the payload.** Keys are read from the environment only. On top of that,
  string fields are scanned for common secret shapes (`sk-…`, `hf_…`, AWS/GitHub/Slack/bearer tokens)
  and redacted before upload — pattern-based and best-effort, a backstop not a guarantee.
- **Payloads stay bounded.** Long strings are truncated (`truncate_str`) and metadata is held under a
  byte cap (`cap_metadata`), dropping the largest keys first and recording them under `_dropped_keys`
  so the cut is visible, not silent.
- **You preview before you write.** Both scripts default to no network writes; an upload happens only
  with `--push` (or `PUSH = True`).
- **Bad input behaves predictably.** Malformed JSON in a message column isn't dropped — it's kept as a
  single raw-text message so the row survives and you can see what happened.

## Layout

```
hf_bt_cookbook/
  import_dataset.py    # HF dataset → Braintrust Dataset   (edit the mapping here)
  import_logs.py       # HF traces  → Braintrust Logs tree  (edit the columns here)
  normalize.py         # chat / OTel message + trace normalization, plus redact/truncate/cap
  braintrust_logs.py   # build span tree, resolve project, write JSONL, bt sync push
```

Loading a HuggingFace dataset and mapping it into Braintrust's shape recurs across the
[Braintrust cookbook](https://github.com/braintrustdata/braintrust-cookbook) — see
[ClassifyingNewsArticles](https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/ClassifyingNewsArticles/ClassifyingNewsArticles.ipynb),
[PrecisionRecall](https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/PrecisionRecall/PrecisionRecall.ipynb),
and
[WebAgent](https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/WebAgent/WebAgent.ipynb).
