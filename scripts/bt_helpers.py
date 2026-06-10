"""Braintrust connection + query helpers.

Loads BRAINTRUST_API_KEY from .env, exposes thin wrappers over the REST API
and BTQL for pulling project logs into pandas.
"""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("BRAINTRUST_API_KEY")
API_URL = os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev").rstrip("/")


def _headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "BRAINTRUST_API_KEY not set. Add it to .env (see .env.example)."
        )
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def list_projects(limit: int = 100) -> pd.DataFrame:
    """Return projects visible to this API key."""
    r = requests.get(
        f"{API_URL}/v1/project",
        headers=_headers(),
        params={"limit": limit},
        timeout=60,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json().get("objects", []))


def get_project(name: str) -> dict[str, Any]:
    """Resolve a project by exact name."""
    r = requests.get(
        f"{API_URL}/v1/project",
        headers=_headers(),
        params={"project_name": name},
        timeout=60,
    )
    r.raise_for_status()
    objs = r.json().get("objects", [])
    if not objs:
        raise ValueError(f"No project named {name!r}")
    return objs[0]


def btql(query: str, fmt: str = "json") -> Any:
    """Run a BTQL query. Returns parsed JSON (list of row dicts under 'data')."""
    r = requests.post(
        f"{API_URL}/btql",
        headers=_headers(),
        json={"query": query, "fmt": fmt},
        timeout=300,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"BTQL {r.status_code}: {r.text[:2000]}")
    return r.json()


def btql_df(query: str) -> pd.DataFrame:
    """Run a BTQL query and return a DataFrame."""
    out = btql(query)
    rows = out.get("data", out) if isinstance(out, dict) else out
    return pd.json_normalize(rows)


def _btql_post(query: str, cursor: str | None, timeout: float = 45.0) -> dict:
    """POST one BTQL page, retrying on 429, transient 5xx, and network errors.

    Uses a short per-request timeout so a hung connection fails fast and retries
    instead of stalling for minutes.
    """
    body: dict[str, Any] = {"query": query, "fmt": "json"}
    if cursor:
        body["cursor"] = cursor
    for attempt in range(8):
        try:
            r = requests.post(
                f"{API_URL}/btql", headers=_headers(), json=body, timeout=timeout
            )
        except requests.exceptions.RequestException as e:
            # DNS failure, connection hang, read timeout -> back off and retry.
            if attempt == 7:
                raise RuntimeError(f"BTQL network error after retries: {e}") from e
            time.sleep(min(2 ** attempt, 20))
            continue
        if r.status_code == 429:
            # Parse "Retry after N seconds" from the message; fall back to 60s.
            wait = 60.0
            try:
                msg = r.json().get("Message", "")
                import re

                m = re.search(r"Retry after ([\d.]+)", msg)
                if m:
                    wait = float(m.group(1)) + 1
            except Exception:
                pass
            time.sleep(wait)
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"BTQL {r.status_code}: {r.text[:2000]}")
        return r.json()
    raise RuntimeError("BTQL: exhausted retries")


def btql_all(
    query: str,
    page: int = 1000,
    max_rows: int | None = None,
    throttle: float = 3.2,
) -> list[dict]:
    """Run a BTQL query, paginating via cursor until exhausted.

    `query` should NOT include a `limit:` clause; this adds one per page.
    `throttle` sleeps between pages to stay under 20 req/min.
    """
    rows: list[dict] = []
    seen: set = set()
    cursor = None
    first = True
    while True:
        if not first:
            time.sleep(throttle)
        first = False
        # BTQL cursor is a *query clause*, not a request-body field. Passing it
        # in the body silently loops the same page; appending it advances.
        q = f"{query} | limit: {page}"
        if cursor:
            q += f" | cursor: '{cursor}'"
        out = _btql_post(q, None)
        batch = [r for r in out.get("data", []) if r.get("id") not in seen]
        for r in batch:
            seen.add(r.get("id"))
        rows.extend(batch)
        cursor = out.get("cursor")
        if max_rows and len(rows) >= max_rows:
            return rows[:max_rows]
        if not cursor or not batch:
            return rows


def cached_pull(query: str, cache_path: str, **kw) -> list[dict]:
    """btql_all with a JSON disk cache. Delete the file to force a refresh."""
    import json
    import os

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    rows = btql_all(query, **kw)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(rows, f)
    return rows



if __name__ == "__main__":
    print(f"API_URL = {API_URL}")
    print(f"API_KEY set = {bool(API_KEY)}")
    if API_KEY:
        projs = list_projects()
        cols = [c for c in ["id", "name", "created"] if c in projs.columns]
        print(f"\n{len(projs)} projects:")
        print(projs[cols].to_string(index=False) if cols else projs.to_string())
