# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
import os, json, requests

API = "https://api.braintrust.dev/btql"
KEY = os.environ["BRAINTRUST_API_KEY"]
PROJECT = "6da0ad7f-d092-4d04-95c5-a2ae182883ec"
ROOT = "2072d3ef-f55e-5f48-aab3-a3c7e24087eb"

def q(query):
    r = requests.post(API, headers={"Authorization": f"Bearer {KEY}",
                                     "Content-Type": "application/json"},
                      json={"query": query, "fmt": "json"})
    r.raise_for_status()
    return r.json()["data"]

# Get all chat spans for this trace, with full input/output
rows = q(f"""select: span_attributes.exec_counter AS ec, input, output
| from: project_logs('{PROJECT}')
| filter: root_span_id = '{ROOT}' AND span_attributes.name LIKE 'chat%' AND created > now() - interval 30 day
| sort: span_attributes.exec_counter DESC
| limit: 1""")

print("Num rows:", len(rows))
row = rows[0]
print("exec_counter:", row.get("ec"))
inp = row.get("input")
print("input type:", type(inp), "len:" , len(inp) if isinstance(inp, list) else "n/a")
# input is a list of messages; print roles and a snippet of each
if isinstance(inp, list):
    for i, m in enumerate(inp):
        role = m.get("role")
        parts = m.get("parts") or m.get("content")
        snippet = json.dumps(parts)[:300]
        print(f"\n[{i}] role={role}\n{snippet}")
