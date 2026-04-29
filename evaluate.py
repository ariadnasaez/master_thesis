"""
Jaccard evaluation for the local Qwen agent.

Reads the questions from `golden_dataset.json`, runs each through the MCP agent,
and computes Jaccard similarity at two levels:

    - Query level:  agent's generated SQL vs golden_query
    - Output level: agent's returned rows vs expected_rows

Optionally runs each question N times to also report determinism
(pairwise Jaccard between the N runs of the same question).

Usage:
    python evaluate.py                # 1 run per question (accuracy only)
    python evaluate.py --runs 3       # 3 runs per question (accuracy + determinism)

Output:
    Console summary + jaccard_results.json with per-question details.
"""

import argparse
import asyncio
import json
import re
from itertools import combinations
from pathlib import Path
from statistics import mean

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent import (
    build_ollama_tools,
    build_system_prompt,
    parse_schema_cache,
    run_agent_loop_async,
    warmup_model,
)


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
OUTPUT_PATH = HERE / "jaccard_results.json"


# ---------------------------------------------------------------------------
# Jaccard helpers
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def pairwise_jaccard(items: list[set]) -> float:
    pairs = list(combinations(items, 2))
    if not pairs:
        return 1.0
    return mean(jaccard(a, b) for a, b in pairs)


# ---------------------------------------------------------------------------
# Normalization for SQL and rows
# ---------------------------------------------------------------------------

def sql_to_tokens(sql: str) -> set:
    """Normalize SQL → set of tokens. Case-insensitive, whitespace/quote-insensitive."""
    if not sql:
        return set()
    s = sql.lower().replace("`", " ").replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    tokens = re.findall(r"[\w_.%]+|[<>=!]+|[(),;]", s)
    return set(tokens)


def rows_to_set(rows) -> set:
    """Convert a list of rows (dicts or tuples) into a set of value-tuples for comparison.
    Cell values are stringified so int/str/Decimal are comparable.
    """
    if rows is None:
        return set()
    out = set()
    for row in rows:
        if isinstance(row, dict):
            values = [str(v) for _, v in sorted(row.items())]
        elif isinstance(row, (list, tuple)):
            values = [str(v) for v in row]
        else:
            values = [str(row)]
        out.add(tuple(values))
    return out


def parse_agent_rows(json_text: str) -> list[dict]:
    """Parse the JSON string returned by execute_query into a list of dict rows."""
    if not json_text:
        return []
    try:
        data = json.loads(json_text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Run a single question once via the MCP agent, capturing the last SQL + rows
# ---------------------------------------------------------------------------

async def run_once(question, session, system_prompt, schema_cache, ollama_tools):
    """Run one question through the agent. Returns (sql, rows) — both possibly empty on failure."""
    captured = {"sql": "", "rows": []}

    async def call_tool(name, args):
        result = await session.call_tool(name, args)
        text = result.content[0].text
        if name == "execute_query":
            captured["sql"] = args.get("query", "")
            captured["rows"] = parse_agent_rows(text)
        return text

    try:
        await run_agent_loop_async(
            question, system_prompt, schema_cache, ollama_tools, call_tool
        )
    except Exception as e:
        print(f"    ⚠️  agent error: {e}")
    return captured["sql"], captured["rows"]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(runs_per_question: int):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    print(f"Loaded {len(golden)} entries from {GOLDEN_PATH.name}")

    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            schema_resource = await session.read_resource("schema://full")
            schema_text = schema_resource.contents[0].text
            schema_cache = parse_schema_cache(schema_text)
            mcp_tools = await session.list_tools()
            ollama_tools = build_ollama_tools(mcp_tools)
            system_prompt = build_system_prompt(schema_text)

            print("⏳ Warming up the model...")
            warmup_model(system_prompt, ollama_tools)
            print("✅ Warmed up. Starting evaluation.\n")

            per_question = []
            for i, item in enumerate(golden, 1):
                question = item.get("question", "")
                golden_sql = item.get("golden_query", "")
                expected_rows = item.get("expected_rows", [])
                qid = item.get("id", i)

                if not question:
                    continue

                print(f"[{i}/{len(golden)}] (id={qid}) {question[:80]}")

                # Run N times
                runs = []
                for r in range(runs_per_question):
                    sql, rows = await run_once(
                        question, session, system_prompt, schema_cache, ollama_tools
                    )
                    runs.append((sql, rows))
                    print(f"    run {r+1}/{runs_per_question}: {len(rows)} rows")

                # Accuracy: mean Jaccard of each run vs the golden
                golden_sql_tokens = sql_to_tokens(golden_sql)
                expected_rows_set = rows_to_set(expected_rows)

                acc_sql_per_run = [jaccard(sql_to_tokens(sql), golden_sql_tokens) for sql, _ in runs]
                acc_rows_per_run = [jaccard(rows_to_set(rows), expected_rows_set) for _, rows in runs]
                accuracy_sql = mean(acc_sql_per_run) if acc_sql_per_run else 0.0
                accuracy_rows = mean(acc_rows_per_run) if acc_rows_per_run else 0.0

                # Determinism: pairwise Jaccard within the N runs
                if runs_per_question >= 2:
                    determinism_sql = pairwise_jaccard([sql_to_tokens(sql) for sql, _ in runs])
                    determinism_rows = pairwise_jaccard([rows_to_set(rows) for _, rows in runs])
                else:
                    determinism_sql = None
                    determinism_rows = None

                per_question.append({
                    "id": qid,
                    "question": question,
                    "runs": runs_per_question,
                    "accuracy_sql_jaccard": accuracy_sql,
                    "accuracy_rows_jaccard": accuracy_rows,
                    "determinism_sql_jaccard": determinism_sql,
                    "determinism_rows_jaccard": determinism_rows,
                    "agent_sqls": [sql for sql, _ in runs],
                    "agent_row_counts": [len(rows) for _, rows in runs],
                })

                line = f"  → accuracy SQL={accuracy_sql:.3f}  rows={accuracy_rows:.3f}"
                if determinism_sql is not None:
                    line += f"  | determinism SQL={determinism_sql:.3f}  rows={determinism_rows:.3f}"
                print(line + "\n")

    # ---------- summary ----------
    if not per_question:
        print("No questions evaluated.")
        return

    mean_acc_sql = mean(q["accuracy_sql_jaccard"] for q in per_question)
    mean_acc_rows = mean(q["accuracy_rows_jaccard"] for q in per_question)

    print("=" * 70)
    print(f"SUMMARY — {len(per_question)} questions × {runs_per_question} run(s)")
    print("=" * 70)
    print(f"Accuracy (vs golden):   SQL={mean_acc_sql:.3f}   rows={mean_acc_rows:.3f}")

    summary = {
        "questions_evaluated": len(per_question),
        "runs_per_question": runs_per_question,
        "mean_accuracy_sql_jaccard": mean_acc_sql,
        "mean_accuracy_rows_jaccard": mean_acc_rows,
    }

    if runs_per_question >= 2:
        det_sql_vals = [q["determinism_sql_jaccard"] for q in per_question if q["determinism_sql_jaccard"] is not None]
        det_rows_vals = [q["determinism_rows_jaccard"] for q in per_question if q["determinism_rows_jaccard"] is not None]
        mean_det_sql = mean(det_sql_vals) if det_sql_vals else 0.0
        mean_det_rows = mean(det_rows_vals) if det_rows_vals else 0.0
        print(f"Determinism (run-vs-run): SQL={mean_det_sql:.3f}   rows={mean_det_rows:.3f}")
        summary["mean_determinism_sql_jaccard"] = mean_det_sql
        summary["mean_determinism_rows_jaccard"] = mean_det_rows

    with open(OUTPUT_PATH, "w") as f:
        json.dump({**summary, "per_question": per_question}, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results → {OUTPUT_PATH.name}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to run each question (>=2 enables determinism metrics).")
    args = parser.parse_args()
    asyncio.run(evaluate(args.runs))
