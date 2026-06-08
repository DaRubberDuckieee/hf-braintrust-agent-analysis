# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
SAFE task-success scoring: judge from the LOCAL source file, write COMPLETE
root rows (original content + scores.task_success) to a JSONL dir, then push
with `bt sync push` (which sets full rows by id — no field wiping).

  uv run score_and_push.py --limit 5        # build 5 -> validate/  (then bt sync push)
  uv run score_and_push.py --all            # build 1781 -> scored/

This script NEVER calls logger.log(). It only reads Braintrust-free from the
local source and emits JSONL for bt sync push.
"""
import os, json, argparse, time, requests

PROXY = "https://api.braintrust.dev/v1/proxy/chat/completions"
KEY = os.environ["BRAINTRUST_API_KEY"]
SRC = "/Users/jess/Documents/Coding Projects/Work (Braintrust)/hf-traces-jsonl/part-000001.jsonl"
DEFAULT_JUDGE = "gpt-4.1"
ALT_JUDGE = "gpt-4o"
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

RUBRICS = {
    "swebench": ("Resolving a GitHub issue by editing an existing repo.",
        "SUCCESS(1): a real code diff addressing the issue AND the agent verified it (tests/checks pass, no unresolved errors). FAILURE(0): no diff, gave up, failing checks, incomplete/off-target, or ran out of steps.", "strong"),
    "appworld": ("Completing a personal-assistant task by orchestrating app APIs.",
        "SUCCESS(1): carried out ALL requested actions with correct params and finished cleanly with no unresolved tool errors blocking the goal. FAILURE(0): missed/incorrect actions, unresolved errors, gave up, or never finished.", "medium"),
    "tau2_airline": ("A customer-service agent handling an airline request via tools.",
        "SUCCESS(1): fulfilled the user's actual request via correct tool actions and confirmed completion. FAILURE(0): wrong/incomplete actions, unresolved errors, or request not satisfied.", "medium"),
    "tau2_retail": ("A customer-service agent handling a retail request via tools.",
        "SUCCESS(1): fulfilled the user's actual request via correct tool actions and confirmed completion. FAILURE(0): wrong/incomplete actions, unresolved errors, or request not satisfied.", "medium"),
    "tau2_telecom": ("A customer-service agent handling a telecom request via tools.",
        "SUCCESS(1): fulfilled the user's actual request via correct tool actions and confirmed completion. FAILURE(0): wrong/incomplete actions, unresolved errors, or request not satisfied.", "medium"),
    "browsecompplus": ("A web-research task: answer a hard question by browsing/searching.",
        "SUCCESS(1): produced a clear, specific final answer well-supported by gathered evidence. FAILURE(0): no answer, hedged/non-answer, or unsupported/contradicted by evidence.", "weak"),
}

def _post(url, body, tries=6):
    delay = 2.0
    for _ in range(tries):
        r = requests.post(url, headers=HDRS, json=body, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay); delay = min(delay*2, 30); continue
        r.raise_for_status(); return r
    r.raise_for_status(); return r

def load_source():
    roots = {}; children = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            if d.get("is_root"):
                roots[d["id"]] = d
            else:
                children.setdefault(d["root_span_id"], []).append(d)
    return roots, children

def text_of(msg):
    bits = []
    c = msg.get("content")
    if isinstance(c, str) and c not in ("", "(no content)"):
        bits.append(c)
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        bits.append(f"[tool_call {fn.get('name')}] {str(fn.get('arguments'))[:800]}")
    return "\n".join(bits) if bits else (c if isinstance(c, str) else json.dumps(c)[:1500])

def serialize(msgs, budget=45000):
    s = "\n\n".join(f"{m.get('role','?').upper()}: {text_of(m)}" for m in msgs if isinstance(m, dict))
    return ("...[truncated]...\n\n" + s[-budget:]) if len(s) > budget else s

def final_conversation(child_spans):
    if not child_spans:
        return []
    last = max(child_spans, key=lambda s: (s.get("span_attributes") or {}).get("exec_counter", 0))
    inp = last.get("input") or []
    out = last.get("output") or []
    inp = inp if isinstance(inp, list) else [inp]
    out = out if isinstance(out, list) else [out]
    return inp + out

def task_text(root):
    inp = root.get("input")
    if isinstance(inp, dict): return str(inp.get("content", inp))[:6000]
    return str(inp)[:6000]

def judge_for(model):
    return ALT_JUDGE if "gpt-4.1" in (model or "").lower() else DEFAULT_JUDGE

def judge(benchmark, task, conv, jm):
    desc, rubric, _ = RUBRICS.get(benchmark, ("An agent task.", "SUCCESS(1) if clearly completed; else FAILURE(0).", "weak"))
    sysmsg = f"""You grade whether an AI agent SUCCEEDED at a task, from its execution trace.
Task type: {desc}
Grading rule: {rubric}
You see the task and the agent's full final conversation incl real tool outputs. Judge from visible verification; you may not have the official grader. Be strict: ambiguous/unverified = 0.
Respond ONLY with JSON: {{"success":0 or 1,"confidence":"low"|"medium"|"high","reasoning":"<=35 words"}}"""
    user = f"=== TASK ===\n{task}\n\n=== AGENT TRACE ===\n{conv}"
    body = {"model": jm, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role":"system","content":sysmsg},{"role":"user","content":user}]}
    return json.loads(_post(PROXY, body).json()["choices"][0]["message"]["content"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    roots, children = load_source()
    ids = sorted(roots.keys())
    outdir = "scored" if args.all else "validate"
    if not args.all:
        # spread sample across benchmarks
        by_b = {}; sample = []
        for rid in ids:
            b = roots[rid]["metadata"]["benchmark"]; by_b.setdefault(b, 0)
            if by_b[b] < 1: sample.append(rid); by_b[b] += 1
            if len(sample) >= args.limit: break
        ids = sample

    os.makedirs(outdir, exist_ok=True)
    n = 0
    with open(f"{outdir}/part-000001.jsonl", "w") as out:
        for rid in ids:
            root = roots[rid]; model = root["metadata"]["model"]; bench = root["metadata"]["benchmark"]
            jm = judge_for(model)
            conv = serialize(final_conversation(children.get(rid, [])))
            try:
                v = judge(bench, task_text(root), conv, jm); s = int(v["success"])
            except Exception as e:
                print(f"ERR {bench} {model} {rid[:8]} {str(e)[:80]}"); continue
            row = dict(root)  # COMPLETE original row
            row["scores"] = {"task_success": float(s)}
            md = dict(root.get("metadata") or {})
            md.update({"judge_model": jm, "judge_confidence": v.get("confidence"),
                       "judge_reasoning": v.get("reasoning"),
                       "judge_reliability": RUBRICS.get(bench,(None,None,'weak'))[2]})
            row["metadata"] = md
            out.write(json.dumps(row) + "\n"); n += 1
            if not args.all:
                print(f"  {bench:14} {model:28} judge={jm:7} -> {s} ({v.get('confidence')})")
                print(f"       reason: {v.get('reasoning')}")
            elif n % 50 == 0:
                print(f"  ...{n}/{len(ids)}", flush=True)
    print(f"\nWrote {n} complete rows to {outdir}/part-000001.jsonl")
    print(f"Next: bt sync push project_logs:\"Hugging Face topics\" --in {outdir}/ --no-input")

if __name__ == "__main__":
    main()
