#!/usr/bin/env python3
"""
Convert Exgentic/agent-llm-traces HuggingFace dataset → Braintrust JSONL
for import via: bt sync push project_logs:"Hugging Face topics" --in output/

One root span per session. Input = initial user task, output = final LLM response.
Child spans = one per LLM call within the session.

Streaming variant of scripts/convert_hf_traces.py — pulls rows via the
`datasets` library (streaming=True) instead of downloading parquet files.
"""

import json
import uuid
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID = "6da0ad7f-d092-4d04-95c5-a2ae182883ec"  # "Hugging Face topics"
ORG_ID     = "86cb0c6f-03c7-4225-b418-4c48af9e5543"
OUTPUT_DIR = Path("hf-traces-jsonl")
REPO_ID    = "Exgentic/agent-llm-traces"

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_messages(raw) -> list:
    """Parse gen_ai.input/output.messages — may be a JSON string or already a list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [{"role": "unknown", "content": raw}]
    if isinstance(raw, list):
        return raw
    return []


def to_unix(ts) -> float:
    """Convert ISO timestamp string or datetime to Unix float."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    # String
    ts = str(ts)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    return 0.0


def normalize_message(msg: dict) -> list:
    """Convert OTel parts-format message to OpenAI-compatible message(s).

    OTel format:  {"role": "user", "parts": [{"type": "text", "content": "..."}]}
    OpenAI format: {"role": "user", "content": "..."}

    Returns a list because one OTel msg may yield multiple OpenAI messages
    (e.g. tool_result parts become separate tool-role messages).
    """
    role    = msg.get("role", "user")
    parts   = msg.get("parts") or []
    content = msg.get("content")

    # Already simple format — return as-is
    if not parts and isinstance(content, str):
        return [{"role": role, "content": content[:8000]}]

    text_parts   = []
    tool_calls   = []
    tool_results = []

    for part in parts:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        ptype = part.get("type", "text")
        if ptype == "text":
            text_parts.append(part.get("content") or part.get("text") or "")
        elif ptype == "tool_call":
            args = part.get("arguments", {})
            tool_calls.append({
                "id":   part.get("id", ""),
                "type": "function",
                "function": {
                    "name":      part.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
        elif ptype in ("tool_result", "tool_call_response"):
            # tool_result:          content is a plain string or dict
            # tool_call_response:   content is in `result` list of {type, text} blocks
            raw_content = part.get("content") or part.get("result") or ""
            if isinstance(raw_content, list):
                # List of content blocks: [{"type": "text", "text": "..."}]
                texts = [
                    r.get("text") or r.get("content") or ""
                    for r in raw_content if isinstance(r, dict)
                ]
                text_content = " ".join(texts)
            else:
                text_content = str(raw_content)
            tool_results.append({
                "role":         "tool",
                "tool_call_id": part.get("id", ""),
                "content":      text_content[:1000],
            })

    result = []

    if role in ("user", "human"):
        if tool_results:
            # Keep as user role so Braintrust renders them as visible chat bubbles
            combined = "\n---\n".join(
                f"[tool result] {tr['content']}" for tr in tool_results
            )
            result.append({"role": "user", "content": combined[:4000]})
        else:
            text = " ".join(text_parts).strip() or (str(content) if content else "")
            if text:
                result.append({"role": "user", "content": text[:4000]})

    elif role == "assistant":
        text = " ".join(text_parts).strip() or (str(content) if isinstance(content, str) else None)
        entry = {"role": "assistant", "content": text[:4000] if text else None}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if entry.get("content") is not None or entry.get("tool_calls"):
            result.append(entry)

    elif role == "system":
        text = " ".join(text_parts).strip() or str(content or "")
        result.append({"role": "system", "content": text[:8000]})

    return result or [{"role": role, "content": ""}]


def normalize_messages(msgs: list) -> list:
    """Normalize a list of OTel messages to OpenAI format."""
    out = []
    for msg in msgs:
        if isinstance(msg, dict):
            out.extend(normalize_message(msg))
    return out


def detect_session_failures(spans_raw: list) -> dict:
    """Scan child spans for OTel error status codes and tool error messages.

    Returns a dict with:
      has_errors        - bool
      error_span_count  - OTel spans whose status.code == 2 (ERROR)
      tool_error_count  - tool_call_response messages that started with "Error"
      sample_errors     - up to 3 deduplicated error strings
    """
    error_spans = 0
    tool_errors = []

    for span in spans_raw:
        # OTel status code 2 = ERROR
        status = span.get("status") or {}
        if isinstance(status, dict) and status.get("code") == 2:
            error_spans += 1

        # Scan input messages for tool_call_response errors
        attrs = span.get("attributes") or {}
        raw = attrs.get("gen_ai.input.messages")
        msgs = parse_messages(raw)
        for msg in msgs:
            parts = msg.get("parts") or []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in ("tool_call_response", "tool_result"):
                    continue
                # Extract text from result list or content string
                raw_result = part.get("result") or part.get("content") or ""
                if isinstance(raw_result, list):
                    texts = [r.get("text") or r.get("content") or "" for r in raw_result if isinstance(r, dict)]
                    text = " ".join(texts)
                else:
                    text = str(raw_result)
                if text.lower().startswith("error"):
                    tool_errors.append(text[:200])

    # Deduplicate while preserving order
    seen = set()
    unique_errors = []
    for e in tool_errors:
        if e not in seen:
            seen.add(e)
            unique_errors.append(e)

    return {
        "has_errors":       error_spans > 0 or len(tool_errors) > 0,
        "error_span_count": error_spans,
        "tool_error_count": len(tool_errors),
        "sample_errors":    unique_errors[:5],
    }


def extract_task_input(first_span_attrs: dict) -> dict:
    """Pull the first meaningful user message from the first LLM call's input."""
    messages = parse_messages(first_span_attrs.get("gen_ai.input.messages"))
    # Find first non-system message (typically the task description)
    task_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    if task_msgs:
        msg = task_msgs[0]
        # Content may be a list of parts or a plain string
        content = msg.get("content") or msg.get("parts", "")
        if isinstance(content, list):
            # Extract text from parts
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("content") or part.get("text") or "")
                else:
                    text_parts.append(str(part))
            content = " ".join(text_parts)
        return {"role": "user", "content": str(content)[:4000]}  # cap length
    return {"role": "user", "content": "(no task content found)"}


def extract_final_output(last_span_attrs: dict) -> dict:
    """Pull the assistant's final response from the last LLM call's output.

    Falls back through: text parts → direct content → tool call name → empty.
    Also searches earlier spans if the last one is tool-call-only.
    """
    messages = parse_messages(last_span_attrs.get("gen_ai.output.messages"))
    assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]

    def extract_from_msg(msg: dict) -> str:
        parts = msg.get("parts") or []
        # 1. Text parts
        text_parts = [
            p.get("content") or p.get("text") or ""
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        if text_parts:
            return " ".join(text_parts)
        # 2. Direct content field
        direct = msg.get("content")
        if direct and isinstance(direct, str) and direct.strip():
            return direct
        # 3. Tool call summary (better than nothing for Topics)
        tool_parts = [
            p for p in parts
            if isinstance(p, dict) and p.get("type") == "tool_call"
        ]
        if tool_parts:
            tools = ", ".join(p.get("name", "tool") for p in tool_parts)
            return f"[tool calls: {tools}]"
        return ""

    for msg in reversed(assistant_msgs):
        text = extract_from_msg(msg)
        if text:
            return {"role": "assistant", "content": str(text)[:4000]}

    return {"role": "assistant", "content": "(no output found)"}


def make_span(
    span_id: str,
    root_span_id: str,
    span_parents,
    is_root: bool,
    name: str,
    span_type: str,
    input_data,
    output_data,
    metadata,
    metrics: dict,
    created: str,
    exec_counter: int,
) -> dict:
    return {
        "_async_scoring_state": None,
        "_pagination_key": None,
        "_xact_id": None,
        "audit_data": [],
        "classifications": None,
        "comments": None,
        "context": None,
        "created": created,
        "error": None,
        "expected": None,
        "facets": None,
        "id": span_id,
        "input": input_data,
        "is_root": is_root,
        "log_id": "g",
        "metadata": metadata,
        "metrics": metrics,
        "org_id": ORG_ID,
        "origin": None,
        "output": output_data,
        "project_id": PROJECT_ID,
        "root_span_id": root_span_id,
        "scores": None,
        "span_attributes": {
            "exec_counter": exec_counter,
            "name": name,
            "type": span_type,
        },
        "span_id": span_id,
        "span_parents": span_parents,
        "tags": None,
    }


def session_to_spans(session: dict) -> list[dict]:
    """Convert one session row to a list of Braintrust JSONL span dicts."""
    spans_raw = session.get("spans") or []
    if not spans_raw:
        return []

    session_id = session.get("session_id") or str(uuid.uuid4())
    harness    = session.get("harness") or ""
    benchmark  = session.get("benchmark") or ""
    models     = session.get("models") or []
    model_str  = models[0] if models else ""
    total_tok  = session.get("total_tokens") or 0
    collected  = session.get("collected_at") or datetime.now(timezone.utc).isoformat()

    if hasattr(collected, "isoformat"):
        collected = collected.isoformat()

    # Sort spans chronologically
    spans_raw = sorted(spans_raw, key=lambda s: to_unix(s.get("start_time")))

    first_attrs = spans_raw[0].get("attributes") or {}

    session_start = to_unix(spans_raw[0].get("start_time"))
    session_end   = to_unix(spans_raw[-1].get("end_time"))

    # Stable root span ID derived from session_id
    root_span_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"root:{session_id}"))

    task_input = extract_task_input(first_attrs)

    # Search backwards for the last span that has meaningful text output
    final_out = {"role": "assistant", "content": "(no output found)"}
    for span in reversed(spans_raw):
        attrs = span.get("attributes") or {}
        candidate = extract_final_output(attrs)
        if candidate["content"] != "(no output found)":
            final_out = candidate
            break

    # Detect failures across all child spans
    failures = detect_session_failures(spans_raw)

    # Surface errors into root span output so Topics/Issues can cluster on them
    if failures["has_errors"]:
        error_summary = (
            f"[{failures['tool_error_count']} tool error(s) detected. "
            f"Examples: {'; '.join(failures['sample_errors'])}]"
        )
        root_output = {
            "role":    "assistant",
            "content": final_out["content"],
            "issues":  error_summary,
        }
        root_error = error_summary
    else:
        root_output = final_out
        root_error  = None

    root_metadata = {
        "benchmark":        benchmark,
        "harness":          harness,
        "model":            model_str,
        "session_id":       session_id,
        "total_tokens":     total_tok,
        "num_llm_calls":    len(spans_raw),
        "has_errors":       failures["has_errors"],
        "tool_error_count": failures["tool_error_count"],
        "error_span_count": failures["error_span_count"],
    }

    root_metrics = {
        "start": session_start,
        "end":   session_end,
    }

    root_span = make_span(
        span_id=root_span_id,
        root_span_id=root_span_id,
        span_parents=None,
        is_root=True,
        name=f"{benchmark}/{session_id}",
        span_type="task",
        input_data=task_input,
        output_data=root_output,
        metadata=root_metadata,
        metrics=root_metrics,
        created=str(collected),
        exec_counter=1,
    )
    root_span["error"] = root_error
    result = [root_span]

    # Child spans — one per LLM call
    for i, span in enumerate(spans_raw, start=2):
        attrs = span.get("attributes") or {}
        child_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"child:{session_id}:{i}:{span.get('span_id','')}",
            )
        )

        child_input  = normalize_messages(parse_messages(attrs.get("gen_ai.input.messages")))
        child_output = normalize_messages(parse_messages(attrs.get("gen_ai.output.messages")))

        # Truncate messages to keep JSONL size reasonable (keep recent context)
        if len(child_input) > 20:
            child_input = child_input[-20:]
        if len(child_output) > 5:
            child_output = child_output[-5:]

        in_tok  = attrs.get("gen_ai.usage.input_tokens") or 0
        out_tok = attrs.get("gen_ai.usage.output_tokens") or 0

        child_metrics = {
            "start": to_unix(span.get("start_time")),
            "end":   to_unix(span.get("end_time")),
            "prompt_tokens":     in_tok,
            "completion_tokens": out_tok,
            "tokens":            in_tok + out_tok,
        }

        result.append(
            make_span(
                span_id=child_id,
                root_span_id=root_span_id,
                span_parents=[root_span_id],
                is_root=False,
                name=span.get("name") or "llm-call",
                span_type="llm",
                input_data=child_input,
                output_data=child_output,
                metadata={"model": attrs.get("gen_ai.request.model") or model_str},
                metrics=child_metrics,
                created=str(collected),
                exec_counter=i,
            )
        )

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "part-000001.jsonl"

    total_sessions = 0
    total_spans    = 0
    errors         = 0

    # Use streaming so we do not need to materialize the whole dataset in memory
    ds = load_dataset(REPO_ID, split="train", streaming=True)

    with open(out_path, "w") as fout:
        for row in ds:
            try:
                spans = session_to_spans(row)
                for span in spans:
                    fout.write(json.dumps(span) + "\n")
                total_sessions += 1
                total_spans    += len(spans)
            except Exception as e:
                print(
                    f"ERROR processing session {row.get('session_id')}: {e}",
                    file=sys.stderr,
                )
                errors += 1

    print("\nDone!")
    print(f"  Sessions: {total_sessions}")
    print(f"  Spans:    {total_spans}")
    print(f"  Errors:   {errors}")
    print(f"  Output:   {out_path}")
    print("\nNext step:")
    print(f'  bt sync push project_logs:"Hugging Face topics" --in {OUTPUT_DIR}')


if __name__ == "__main__":
    main()
