"""Verify the reconstructed BTQL pull reproduces data/full.json exactly.

Run:  .venv/bin/python scripts/verify_pull.py
Needs BRAINTRUST_API_KEY in .env. Pulls fresh, diffs row-for-row (keyed by id)
against the committed snapshot, and reports any mismatch. Does NOT overwrite
data/full.json — it only compares.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import bt_helpers as bt  # noqa: E402

QUERY = """select: metadata.benchmark AS benchmark, metadata.harness AS harness, metadata.model AS model,
        metadata.session_id AS session_id, metadata.total_tokens AS total_tokens,
        metadata.num_llm_calls AS num_llm_calls, metadata.has_errors AS has_errors,
        metadata.tool_error_count AS tool_error_count, metadata.error_span_count AS error_span_count,
        metadata.judge_model AS judge_model, metadata.judge_confidence AS judge_confidence,
        metadata.judge_reasoning AS judge_reasoning, metadata.judge_reliability AS judge_reliability,
        scores.task_success AS task_success, span_attributes.name AS task, duration, id
| from: project_logs('6da0ad7f-d092-4d04-95c5-a2ae182883ec')
| filter: is_root = true"""

FIELDS = ["benchmark", "harness", "model", "session_id", "total_tokens", "num_llm_calls",
          "has_errors", "tool_error_count", "error_span_count", "judge_model",
          "judge_confidence", "judge_reasoning", "judge_reliability", "task_success",
          "task", "duration", "id"]


def main():
    snap = json.loads((REPO / "data/full.json").read_text())
    fresh = bt.btql_all(QUERY)
    print(f"snapshot rows: {len(snap)}   fresh rows: {len(fresh)}")

    snap_by_id = {r["id"]: r for r in snap}
    fresh_by_id = {r["id"]: r for r in fresh}

    only_snap = snap_by_id.keys() - fresh_by_id.keys()
    only_fresh = fresh_by_id.keys() - snap_by_id.keys()
    if only_snap or only_fresh:
        print(f"  id mismatch: {len(only_snap)} only in snapshot, {len(only_fresh)} only in fresh")

    extra_keys = set().union(*(r.keys() for r in fresh)) - set(FIELDS)
    if extra_keys:
        print(f"  fresh rows carry unexpected keys: {sorted(extra_keys)}")

    mismatches = 0
    for rid in snap_by_id.keys() & fresh_by_id.keys():
        a, b = snap_by_id[rid], fresh_by_id[rid]
        for f in FIELDS:
            if a.get(f) != b.get(f):
                if mismatches < 10:
                    print(f"  [{rid}] {f}: snapshot={a.get(f)!r} fresh={b.get(f)!r}")
                mismatches += 1

    ok = not only_snap and not only_fresh and not extra_keys and mismatches == 0
    print("RESULT:", "✅ exact match — query reproduces full.json" if ok
          else f"❌ {mismatches} field mismatches (+id/key diffs above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
