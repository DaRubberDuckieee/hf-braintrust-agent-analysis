# HuggingFace × Braintrust — Agent Trace Analysis (Summary)

A full walkthrough of what we explored, how we did it, and what we learned from a
complex agent-trace dataset using Braintrust. Built for a HuggingFace × Braintrust
collaboration: take a messy real-world dataset and surface genuinely interesting,
defensible insights.

Project: **Hugging Face topics** (`6da0ad7f-d092-4d04-95c5-a2ae182883ec`)

---

## 1. What the dataset is

The data comes from the HuggingFace dataset **`Exgentic/agent-llm-traces`** — a
collection of execution traces from AI agents attempting tasks across several
agentic benchmarks. We ported it into Braintrust logs (one root span per agent
session, with child spans for each LLM call).

**Structure:** each session = one agent's full attempt at one benchmark task.
- **Root span** = the session (the task as input, the agent's final answer as output).
- **Child spans** = the individual LLM calls within that session (≈49k child spans total).
- **1,781 root sessions** in total.

**Metadata that lives on each root session:**

| Field | Meaning |
|---|---|
| `metadata.model` | the LLM being evaluated (e.g. `aws/claude-opus-4-5`) |
| `metadata.benchmark` | which eval suite the task came from |
| `metadata.harness` | the agent scaffold running the model (see §3) |
| `metadata.num_llm_calls` | how many LLM calls the session took |
| `metadata.total_tokens` | total tokens consumed in the session |
| `metadata.session_id` | unique session id |
| `metrics.start` / `metrics.end` | timestamps → duration = end − start (seconds) |
| `error` | a tool-error *summary string* (see note) |
| `scores.task_success` | **added by us** — LLM-as-judge success (0/1), see §6 |

> **Note on the `error` field:** it is NOT a crash flag. It's a diagnostic string
> like `"[N tool error(s) detected. Examples: ...]"` that's present even when N=0.
> So naive "error rate" is meaningless — which is exactly why we built a real
> success measure (§6). What the original dataset did **not** include: any
> ground-truth pass/fail/reward. We confirmed the grading verdicts were never
> exported — only the traces.

---

## 1b. How we ported the logs from HuggingFace into Braintrust

The dataset ships as 39 parquet shards on HuggingFace (`train-00000-of-00039` …).
A converter script (`scripts/convert_hf_traces.py`) turns each raw session into
Braintrust-shaped JSONL spans, then we upload with the Braintrust CLI. Steps:

1. **Download** each parquet shard via `huggingface_hub.hf_hub_download`.
2. **One session → many spans.** For each session row we emit:
   - a **root span** (`type=task`): input = the initial user task, output = the
     agent's final response, plus all session metadata (model, benchmark, harness,
     total_tokens, num_llm_calls, session_id).
   - one **child span** per LLM call (`type=llm`): the input/output messages and
     per-call token metrics.
3. **Deterministic span ids** (this mattered later for recovery):
   `root_id = uuid5(NAMESPACE_DNS, "root:" + session_id)` and
   `child_id = uuid5(NAMESPACE_DNS, "child:" + session_id + ":" + i + ":" + span_id)`.
   Stable ids mean re-imports upsert the same rows instead of duplicating.
4. **Message normalization.** The source uses OpenTelemetry "parts" format; we
   convert to OpenAI-style messages (text, tool_calls, tool results) so Braintrust
   renders them as readable chat turns.
5. **Tool-error summary.** We scan child spans for tool errors and write a summary
   string into the root `error` field (`"[N tool error(s) detected. Examples: ...]"`)
   so Topics/Issues can cluster on them. (This is the field we later learned NOT to
   use as a success signal — it's present even when N=0.)
6. **Upload** with the Braintrust CLI:
   `bt sync push project_logs:"Hugging Face topics" --in hf-traces-jsonl/`

Output: 1,781 sessions → ~50k spans in the `Hugging Face topics` project.

> The same `bt sync push` mechanism (upserting COMPLETE rows by id) is what we later
> used to write success scores back safely — see §6 / the scripts.

---

## 2. The benchmarks (what each one actually tests)

| Benchmark | What it tests | Coding? |
|---|---|---|
| **swebench** | Fix a real GitHub issue inside a large EXISTING repo — read many unfamiliar files, edit code, run tests iteratively. The only pure software-engineering benchmark. | ✅ pure SWE |
| **appworld** | Complete a personal-assistant task by orchestrating app APIs (Venmo, Gmail, Spotify, Splitwise, Todoist, etc.). In code harnesses the agent writes Python to call the APIs. | ⚠️ often code-heavy |
| **browsecompplus** | Answer a hard question by browsing / searching the web. | ❌ |
| **tau2_airline** | Customer-service agent handling an airline support request via tools (tau-bench). | ❌ |
| **tau2_retail** | Customer-service agent handling a retail support request via tools. | ❌ |
| **tau2_telecom** | Customer-service agent handling a telecom support request via tools. | ❌ |

The tau2 family is conversational tool-use (lightest token footprint); swebench and
appworld are heavy, long-horizon agentic tasks.

---

## 3. What a "harness" is, and which ones are in these logs

A **harness** (agent scaffold) is the code wrapper around the model that turns a raw
LLM into an agent. The model only predicts text; the harness does everything else:
formats the task + available tools into the prompt, parses the model's output into
real actions, executes them and feeds results back, manages the loop / retries /
stop conditions.

**Same model + different harness = very different agent.** This turned out to be one
of the most important variables in the whole dataset (§7).

**Harnesses present in these logs:**

| Harness | How the model invokes tools |
|---|---|
| `claude_code` | Anthropic's agent scaffold. The most common harness here. |
| `tool_calling` | Native structured JSON function-calling (one discrete call at a time). |
| `tool_calling_with_shortlisting` | Like `tool_calling` but with a tool shortlisting step. |
| `smolagents_code` | HuggingFace's `smolagents`: the model WRITES PYTHON CODE to call tools; results return as "Observation: Execution logs:". Lets it chain logic in one step. |
| `openai_solo` | A solo/minimal OpenAI-style scaffold. |

---

## 4. Which models ran on which benchmarks

Session counts. This matters because comparisons are only fair where coverage
overlaps. **Not all models ran all benchmarks** — so global model rankings are
confounded (the first lesson of the project).

| Model | appworld | browsecompplus | swebench | tau2_airline | tau2_retail | tau2_telecom |
|---|---|---|---|---|---|---|
| claude-opus-4-5 🔒 | 119 | 49 | 138 | 34 | 126 | 43 |
| gemini-3-pro 🔒 | 170 | 77 | 30 | 38 | 34 | 12 |
| Kimi-K2.5 🔓 | 49 | 7 | 75 | — | — | — |
| DeepSeek-V3.2 🔓 | 62 | — | 55 | — | — | — |
| gpt-4.1 🔒 | — | — | — | 124 | 295/14 | 131 |
| gpt-5.2 🔒 | — | — | 37/15 | — | — | — |

🔓 = open weights (Kimi, DeepSeek) · 🔒 = closed (Claude, Gemini, GPT). Cells with
two numbers reflect two model-name variants in the raw data. The `openai/azure/...`
variants (small n) are folded into their parents.

**Key consequence:** only **claude-opus-4-5** and **gemini-3-pro** ran all 6
benchmarks — they are the only pair we can compare across the whole suite. GPT-4.1
ran only tau2; GPT-5.2 only swebench.

---

## 5. Per-benchmark efficiency: Claude Opus 4.5 vs Gemini 3 Pro

The only fair full-suite comparison. Done **per benchmark** (holding the benchmark
constant) because a single pooled average would be confounded by each model running
a different mix/number of tasks (Simpson's paradox).

| Benchmark | Metric | Claude Opus 4.5 | Gemini 3 Pro | Winner |
|---|---|---|---|---|
| appworld | avg tokens | 2,485,379 | 595,332 | **Gemini (4.2× fewer)** |
| | avg duration (s) | 232 | 314 | Claude |
| browsecompplus | avg tokens | 2,408,136 | 1,288,647 | **Gemini (1.9× fewer)** |
| | avg duration (s) | 168 | 231 | Claude |
| swebench | avg tokens | 819,283 | 1,874,468 | **Claude (2.3× fewer)** |
| | avg duration (s) | 213 | 469 | Claude |
| tau2_airline | avg tokens | 323,256 | 237,633 | Gemini |
| | avg duration (s) | 74 | 124 | Claude |
| tau2_retail | avg tokens | 354,912 | 270,031 | Gemini |
| | avg duration (s) | 69 | 112 | Claude |
| tau2_telecom | avg tokens | 453,399 | 194,960 | **Gemini (2.3× fewer)** |
| | avg duration (s) | 117 | 144 | Claude |

**Takeaways (efficiency only):**
1. **Gemini is more token-efficient on 5 of 6 benchmarks** — sometimes dramatically
   (4.2× fewer on appworld). The exception is **swebench**.
2. **Claude is faster (lower wall-clock) on every single benchmark**, even when it
   uses more tokens. "Efficiency" splits two ways: Gemini wins on token cost, Claude
   wins on latency.
3. **The swebench reversal** is the standout — the one benchmark where Gemini blows
   up (52 calls, 1.87M tokens) while Claude stays lean.

> But this is EFFICIENCY ONLY — it says nothing about whether the agent actually
> succeeded. That's what sent us deeper.

---

## 6. Digging deeper: there's no success signal, so we built one (LLM-as-judge)

Efficiency is hollow without knowing whether the task *succeeded* — a model can be
cheap and fast simply because it gives up. But the dataset has **no ground-truth
success**: `scores` and `expected` are null on all 1,781 sessions, and the original
HuggingFace dataset never exported the benchmarks' grading verdicts (swebench hidden
tests, tau2 DB-state checks). So we generated a **success proxy** with LLM-as-judge.

**How we did it:**
- For each session, a judge model reads (1) the task and (2) the agent's full final
  conversation — including real tool outputs, test runs, and the final diff — and
  returns `{success: 0/1, confidence, reasoning}`.
- **Judge input** = the last chat span's INPUT (the whole running conversation) PLUS
  its OUTPUT (the agent's final message/action). *(We caught a bug where omitting the
  final output made conversational tau2 runs look like "no response.")*
- **Judge model routing:** GPT-4.1 for everyone, except gpt-4.1's OWN runs are judged
  by GPT-4o (avoid self-grading). Only OpenAI models are available on the Braintrust
  proxy, so a fully neutral non-contestant judge wasn't possible.
- **Stored on each root span:** `scores.task_success` plus `metadata.judge_model`,
  `judge_confidence`, `judge_reasoning`, `judge_reliability` — all queryable in SQL /
  the UI. The reasoning makes every verdict auditable next to the trace.
- **Coverage:** 1,780/1,781 scored. Overall ~64% success.

**What to question about it (important):** the judge sees the agent's OWN
verification, NOT the benchmark's hidden official grader. So `task_success` means
"did the agent produce a complete, self-verified result" — which correlates with
true success but is not ground truth. Reliability varies by benchmark:
`swebench` STRONG (diff + tests visible), `appworld`/`tau2` MEDIUM, `browsecompplus`
WEAK (no gold answer).

### The judge prompt

System message (the `{task type}` and `{grading rule}` are filled per benchmark):

```
You grade whether an AI agent SUCCEEDED at a task, from its execution trace.
Task type: {benchmark description}
Grading rule: {benchmark-specific rubric}
You see the task and the agent's full final conversation incl real tool outputs.
Judge from visible verification; you may NOT have the official grader.
Be strict: ambiguous/unverified = 0.
Respond ONLY with JSON: {"success":0 or 1,"confidence":"low"|"medium"|"high","reasoning":"<=35 words"}
```

User message: `=== TASK ===\n{task}\n\n=== AGENT TRACE ===\n{final conversation}`

Per-benchmark grading rules (abbreviated):
- **swebench:** real diff addressing the issue AND verified (tests/checks pass, no
  unresolved errors) = 1.
- **appworld:** ALL requested actions done with correct params and finished cleanly,
  no unresolved errors blocking the goal = 1.
- **tau2_*:** fulfilled the user's actual request via correct tool actions and
  confirmed completion = 1.
- **browsecompplus:** clear, specific final answer well-supported by gathered
  evidence = 1.

### Worked example — gut-checking one verdict

The most instructive log we hand-checked: the agent **finished cleanly with ZERO
tool errors**, yet was correctly scored a **failure** — exactly the case a naive
metric would miss.

- **Log:** `86c1014d-321f-522b-ae63-5a0b2da977f9` · benchmark `appworld` · model
  `claude-opus-4-5` · `error = "[0 tool error(s) detected...]"` · **task_success = 0**
- **Task:** *"Reset friends on venmo to be the same as my friends in my phone.
  Befriend and unfriend as needed."*
- **Judge reasoning:** *"The agent only attempted to remove friends from Venmo but
  did not add any missing friends from the phone contacts. The task required both
  befriending and unfriending as needed."*
- **What the agent actually did** (tool calls from the trace): login →
  `search_friends` ×3 → `show_profile` ×2 → `remove_friend` ×~15 → `finish(success)`.
  **Zero `add_friend` calls.**
- **Our conclusion:** judge is **correct**. The task required both add and remove;
  the agent did only removals then declared success with 0 errors — a textbook
  "false success confirmation." An error-rate or finish-status metric would have
  called this a pass. (Honest caveat: the judge *assumed* additions were needed; it
  had the `search_friends` results in context, so it most likely saw the gap — but
  this is the inherent soft edge of a medium-reliability proxy.)

---

## 7. Indexing on harness too — the ultimate table

Two confounds had to be controlled together: **benchmark** (§5) and **harness** (§3).
A single model+benchmark cell often mixes harnesses (e.g. Claude's 119 appworld
sessions = `tool_calling` 83 + `claude_code` 27 + `openai_solo` 9). So the truly fair
unit is **benchmark × harness × model**. This table combines everything —
efficiency AND judged success — for cells with n ≥ 10.

Columns: n = sessions · calls = avg LLM calls · Mtok = avg tokens (millions) ·
dur = avg duration (sec) · **succ% = LLM-judge success rate**.

```
APPWORLD                                n  calls   Mtok   dur  succ%
claude_code      DeepSeek-V3.2  🔓     25   19.9   2.25   387    80
claude_code      Kimi-K2.5      🔓     18   17.7   1.17   398    78
claude_code      claude-opus-4-5      27   31.7   4.19   316    26
smolagents_code  Kimi-K2.5      🔓     12   12.9   0.53   429    92
openai_solo      gemini-3-pro         20   19.2   1.10   339    10
tool_calling     DeepSeek-V3.2  🔓     11   17.7   1.64   891    36
tool_calling     gemini-3-pro         77   15.6   0.59   263    16
tool_calling     claude-opus-4-5      83   25.7   2.09   221    14
tool_calling     Kimi-K2.5      🔓     17   17.5   0.91   657    12
tcw_short        DeepSeek-V3.2  🔓     17   19.3   0.21   531    18
tcw_short        gemini-3-pro         64   28.8   0.27   370     9

BROWSECOMPPLUS                          n  calls   Mtok   dur  succ%
claude_code      claude-opus-4-5      49   30.1   2.41   168    69
claude_code      gemini-3-pro         77   20.5   1.29   231    55

SWEBENCH                                n  calls   Mtok   dur  succ%
claude_code      claude-opus-4-5     138   27.8   0.82   213   100
claude_code      DeepSeek-V3.2  🔓     26   58.8   2.07  1220    96
claude_code      Kimi-K2.5      🔓     35   46.0   1.08   477    94
claude_code      gpt-5.2              15   29.9   0.74   106    93
claude_code      gemini-3-pro         30   52.2   1.87   469    87
claude_code      Az/gpt-5.2           37   24.1   0.52   144    76
smolagents_code  DeepSeek-V3.2  🔓     24   63.8   1.62  1503    88
smolagents_code  o/Kimi-K2.5    🔓     28   87.9   2.58   700    75
smolagents_code  o/DeepSeek-V3.2 🔓    13   84.7   2.36   647    69
openai_solo      Kimi-K2.5      🔓     12   36.8   0.33   227    33
tool_calling     Kimi-K2.5      🔓     21   43.3   0.55   273    29

TAU2_AIRLINE                            n  calls   Mtok   dur  succ%
claude_code      gemini-3-pro         38   11.1   0.24   124   100
claude_code      claude-opus-4-5      34   12.1   0.32    74    65
tool_calling     gpt-4.1              59    5.8   0.01    24    47
smolagents_code  gpt-4.1              47    6.3   0.01    54    40
openai_solo      gpt-4.1              16    4.8   0.01    31    12

TAU2_RETAIL                             n  calls   Mtok   dur  succ%
claude_code      claude-opus-4-5     126   13.6   0.35    69    95
claude_code      gpt-4.1              30    7.4   0.01   144    93
claude_code      Az/gpt-4.1           14    7.4   0.01   204    93
smolagents_code  gpt-4.1             104    6.3   0.01    97    90
tool_calling     gpt-4.1             127    7.4   0.01    92    90
claude_code      gemini-3-pro         34   12.4   0.27   112    82
openai_solo      gpt-4.1              34    5.9   0.01    76    65

TAU2_TELECOM                            n  calls   Mtok   dur  succ%
claude_code      claude-opus-4-5      39   16.8   0.48   116    82
smolagents_code  gpt-4.1              68   20.3   0.07   170    51
tool_calling     gpt-4.1              24   17.0   0.07   151    46
claude_code      gemini-3-pro         10    9.0   0.21   145    33
claude_code      gpt-4.1              22   24.7   0.09   224    18
openai_solo      gpt-4.1              17    3.1   0.01    18     6
```

`tcw_short` = tool_calling_with_shortlisting.

---

## 8. Main learnings

**1. On coding (SWEBench), open weights are frontier-class.** Holding harness
constant (claude_code), DeepSeek (96%) and Kimi (94%) essentially tie the best
closed models, beating Gemini (87%). For models you can self-host, that's the
strongest, most defensible result in the dataset.

**2. The harness matters as much as the model — and buys success, not cost.**
Switching a model's harness barely changes token cost but swings success hugely
(Kimi on swebench: 94% under claude_code vs 29% under tool_calling). `claude_code`
is the strongest harness overall — but it's NOT universal: it's tuned for Claude and
actively *hurts* gpt-4.1 (telecom: 51% smolagents → 18% claude_code). Pick the
harness for the model+task; it's the highest-leverage, near-free knob.

**3. The "open crushes closed on AppWorld" headline was mostly a harness artifact.**
Raw appworld looked like open ≫ closed, but closed models ran appworld mostly under
the worst harness (tool_calling). Same-harness, a narrower truth survives: open
weights still beat **Claude** specifically (Claude is weak on appworld — 26% even in
its own harness), but Gemini does fine (67%).

**4. GPT-4.1's "amazing efficiency" was failure in disguise.** It's 10–100× cheaper
on tokens, but cheap only pays off on tau2_retail (90–93%). On airline (40–47%) and
telecom (best 51%) it's mediocre — cheap because it does less, not because it's
better. Cost without success is meaningless.

**5. Cost-per-success flips rankings.** On swebench, Claude and DeepSeek both hit
~100/96% — but Claude does it at 0.82M tokens / 213s while DeepSeek burns 2.07M /
1220s (20 min) for the same outcome. gpt-5.2 is the sleeper: 93% at the lowest cost
and fastest. The best cost/outcome cell anywhere is Kimi+smolagents on appworld (92%
at 0.53M tokens); the worst is Claude+claude_code on appworld (4.19M tokens to fail
74%).

**6. Failure looks OPPOSITE on the two task families.**
- On hard agentic/coding tasks (swebench, appworld, browse) → failure = **THRASHING**:
  failed runs make *more* calls, burn *more* tokens, run *longer*, and still fail
  (browse failures use 2.3× the tokens of successes).
- On conversational tau2 tasks → failure = **GIVING UP**: failed runs make *fewer*
  calls, burn *fewer* tokens, finish *faster*.
- **Ops implication:** an "abnormal token usage" guardrail needs opposite thresholds
  per task type — cap the thrashers on coding, but a suspiciously cheap/short run is
  the danger sign on conversational tasks. A single "cap tokens" rule would help one
  and hurt the other.

**7. No universal winner — different models own different jobs.** Coding →
Claude/DeepSeek/Kimi; multi-app orchestration (appworld) → open weights; tau2_airline
→ Gemini (100%); tau2_retail/telecom + browse → Claude. "Pick the model (and harness)
for the job" is the honest framing.

**8. Tool errors ≠ task failure; clean finish ≠ success.** The `error` field counts
tool errors even when zero, and tool errors are often just exploration. A run can
finish cleanly with 0 errors and still fail (the venmo example). This is why a real
success measure was necessary.

---

## Caveats to publish with any of this

- `task_success` is an **LLM-judge proxy** of the agent's own verification, not the
  official hidden grader. swebench/appworld solid-ish, browsecompplus weak, tau2 medium.
- `claude_code` dominates the data, so most clean comparisons funnel through it.
- Within a benchmark × harness cell, models ran **different specific task instances**
  (not matched task-for-task), so small gaps are noise.
- n ≥ 10 filter applied, but several cells are only 10–15 (directional).
- swebench Claude 100% is suspiciously perfect — likely some judge leniency reading
  Claude's thorough self-verification. Flag it.

## Operational note — writing scores back safely (learned the hard way)

When adding `scores.task_success` to existing imported rows, do **NOT** use the SDK
`logger.log(id=..., scores=...)` — it REPLACES the row with only the fields passed,
wiping input/output/metadata. (This silently corrupted 185 rows mid-project before
we caught it; we fully recovered by re-pushing the originals from the source file,
since the ids are deterministic — `uuid5(NAMESPACE_DNS, "root:"+session_id)`.)

**Correct method:** build COMPLETE rows (the original row + the new score) and upload
with `bt sync push project_logs:"..." --in <dir>/`. Because every pushed row is
complete, nothing gets wiped. `bt sync` caches per-spec, so re-pushing a changed file
needs the `--fresh` flag. This is what `scripts/score_and_push.py` does.

## Where things live
- Scores in Braintrust: `scores.task_success` (0/1) + `metadata.judge_*` on root
  spans. Filter `scores.task_success IS NOT NULL`.
- **Scripts (`scripts/` in this folder):**
  - `convert_hf_traces.py` — HuggingFace → Braintrust importer (defines the
    deterministic id scheme; see §1b).
  - `score_and_push.py` — safe LLM-as-judge scorer (judges each session, writes
    COMPLETE rows + scores, uploads via `bt sync push`).
  - `explore_trace.py` — helper to dump a single trace's spans for inspection.
- Source of truth (full export): `hf-traces-jsonl/part-000001.jsonl` (39,833 rows),
  in the working dir `Coding Projects/Work (Braintrust)/`.
