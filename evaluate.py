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
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import mean

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent import (
    DEFAULT_MODEL,
    build_ollama_tools,
    build_system_prompt,
    warmup_model,
)


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
OUTPUT_PATH = HERE / "jaccard_results.json"
WORKER_PATH = HERE / "_worker.py"

QUESTION_TIMEOUT = 10 * 60  # seconds — questions that exceed this are skipped
COOLDOWN_AFTER_TIMEOUT = 60  # seconds to let CPU/GPU recover after a timeout


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


def rows_to_value_set(rows) -> set:
    """Flatten ALL cell values across rows and columns into a single set.
    Used for a 'loose' Jaccard that ignores schema/column-count differences —
    rewards the agent for surfacing the right values even when it returns
    extra explanatory columns.
    """
    if rows is None:
        return set()
    out = set()
    for row in rows:
        if isinstance(row, dict):
            cells = row.values()
        elif isinstance(row, (list, tuple)):
            cells = row
        else:
            cells = [row]
        for v in cells:
            out.add(str(v))
    return out


def count_ratio(agent_rows, golden_rows) -> float:
    """How close the agent's row count is to the golden's. 1.0 = exact match,
    0.0 = one is empty and the other isn't. Useful for list-shaped questions.
    """
    a = len(agent_rows or [])
    g = len(golden_rows or [])
    if a == 0 and g == 0:
        return 1.0
    if a == 0 or g == 0:
        return 0.0
    return min(a, g) / max(a, g)


# ---------------------------------------------------------------------------
# Subprocess-based question runner with hard timeout + post-timeout recovery
# ---------------------------------------------------------------------------

def _run_question_sync(question: str, schema_file: str, output_file: str):
    """Spawn _worker.py as a subprocess to evaluate one question.
    Raises subprocess.TimeoutExpired if it exceeds QUESTION_TIMEOUT.
    Returns (sql, rows, elapsed_seconds) on success.
    """
    subprocess.run(
        [sys.executable, str(WORKER_PATH), question, schema_file, output_file],
        timeout=QUESTION_TIMEOUT,
        capture_output=True,
        cwd=str(HERE),
        check=False,
    )
    try:
        data = json.loads(Path(output_file).read_text())
        return data.get("sql", ""), data.get("rows", []), float(data.get("elapsed", 0.0))
    except (FileNotFoundError, json.JSONDecodeError):
        return "", [], 0.0


def _recover_after_timeout(system_prompt: str, ollama_tools: list) -> None:
    """Force-abort any in-progress Ollama generation, cool down, then re-warmup.
    Called after a question times out, to prevent the orphaned generation from
    cascading into the next question.
    """
    subprocess.run(["ollama", "stop", DEFAULT_MODEL], capture_output=True, check=False)
    time.sleep(COOLDOWN_AFTER_TIMEOUT)
    warmup_model(system_prompt, ollama_tools)


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
            mcp_tools = await session.list_tools()
            ollama_tools = build_ollama_tools(mcp_tools)
            system_prompt = build_system_prompt(schema_text)

            print("⏳ Warming up the model...")
            warmup_model(system_prompt, ollama_tools)
            print("✅ Warmed up. Starting evaluation.\n")

    # MCP session above is only used for schema fetch + warmup. Each question
    # runs in its own subprocess (with its own MCP session) so it can be killed
    # cleanly when QUESTION_TIMEOUT fires.
    schema_file = HERE / "_eval_schema.txt"
    schema_file.write_text(schema_text)

    per_question = []
    loop = asyncio.get_running_loop()
    try:
        for i, item in enumerate(golden, 1):
            question = item.get("question", "")
            golden_sql = item.get("golden_query", "")
            expected_rows = item.get("expected_rows", [])
            qid = item.get("id", i)

            if not question:
                continue

            print(f"[{i}/{len(golden)}] (id={qid}) {question[:80]}")

            runs = []  # list of (sql, rows, elapsed_seconds)
            timed_out = False
            for r in range(runs_per_question):
                output_file = HERE / f"_eval_q{i}_r{r}.json"
                try:
                    try:
                        sql, rows, elapsed = await loop.run_in_executor(
                            None,
                            _run_question_sync,
                            question, str(schema_file), str(output_file),
                        )
                    except subprocess.TimeoutExpired:
                        print(f"    ⏰ TIMEOUT ({QUESTION_TIMEOUT // 60}min) — skipping question {qid}")
                        timed_out = True
                        print(f"    🔄 Aborting Ollama and cooling down ({COOLDOWN_AFTER_TIMEOUT}s)...")
                        await loop.run_in_executor(
                            None, _recover_after_timeout, system_prompt, ollama_tools
                        )
                        print("    ✅ Recovered. Continuing.")
                        break
                    runs.append((sql, rows, elapsed))
                    print(f"    run {r+1}/{runs_per_question}: {len(rows)} rows  ({elapsed:.1f}s)")
                finally:
                    output_file.unlink(missing_ok=True)

            if timed_out or not runs:
                continue

            # Accuracy: mean Jaccard of each run vs the golden
            golden_sql_tokens = sql_to_tokens(golden_sql)
            expected_rows_set = rows_to_set(expected_rows)
            expected_values_set = rows_to_value_set(expected_rows)

            acc_sql_per_run = [jaccard(sql_to_tokens(sql), golden_sql_tokens) for sql, _, _ in runs]
            acc_rows_per_run = [jaccard(rows_to_set(rows), expected_rows_set) for _, rows, _ in runs]
            acc_values_per_run = [jaccard(rows_to_value_set(rows), expected_values_set) for _, rows, _ in runs]
            count_ratio_per_run = [count_ratio(rows, expected_rows) for _, rows, _ in runs]
            accuracy_sql = mean(acc_sql_per_run) if acc_sql_per_run else 0.0
            accuracy_rows = mean(acc_rows_per_run) if acc_rows_per_run else 0.0
            accuracy_values = mean(acc_values_per_run) if acc_values_per_run else 0.0
            accuracy_count = mean(count_ratio_per_run) if count_ratio_per_run else 0.0

            # Determinism: pairwise Jaccard within the N runs
            if runs_per_question >= 2:
                determinism_sql = pairwise_jaccard([sql_to_tokens(sql) for sql, _, _ in runs])
                determinism_rows = pairwise_jaccard([rows_to_set(rows) for _, rows, _ in runs])
            else:
                determinism_sql = None
                determinism_rows = None

            elapsed_per_run = [round(elapsed, 2) for _, _, elapsed in runs]
            mean_elapsed = round(mean(elapsed_per_run), 2) if elapsed_per_run else 0.0

            per_question.append({
                "id": qid,
                "question": question,
                "runs": runs_per_question,
                "accuracy_sql_jaccard": accuracy_sql,
                "accuracy_rows_jaccard": accuracy_rows,
                "accuracy_values_jaccard": accuracy_values,
                "accuracy_count_ratio": accuracy_count,
                "determinism_sql_jaccard": determinism_sql,
                "determinism_rows_jaccard": determinism_rows,
                "elapsed_seconds_per_run": elapsed_per_run,
                "mean_elapsed_seconds": mean_elapsed,
                "agent_sqls": [sql for sql, _, _ in runs],
                "agent_row_counts": [len(rows) for _, rows, _ in runs],
            })

            line = (
                f"  → accuracy SQL={accuracy_sql:.3f}  rows={accuracy_rows:.3f}"
                f"  values={accuracy_values:.3f}  count={accuracy_count:.3f}"
            )
            if determinism_sql is not None:
                line += f"  | determinism SQL={determinism_sql:.3f}  rows={determinism_rows:.3f}"
            latency_label = "mean" if runs_per_question > 1 else "elapsed"
            line += f"  | {latency_label} {mean_elapsed:.1f}s"
            print(line + "\n")
    finally:
        schema_file.unlink(missing_ok=True)

    # ---------- summary ----------
    if not per_question:
        print("No questions evaluated.")
        return

    mean_acc_sql = mean(q["accuracy_sql_jaccard"] for q in per_question)
    mean_acc_rows = mean(q["accuracy_rows_jaccard"] for q in per_question)
    mean_acc_values = mean(q["accuracy_values_jaccard"] for q in per_question)
    mean_acc_count = mean(q["accuracy_count_ratio"] for q in per_question)
    mean_elapsed_all = mean(q["mean_elapsed_seconds"] for q in per_question)
    total_elapsed_all = sum(
        elapsed for q in per_question for elapsed in q["elapsed_seconds_per_run"]
    )

    print("=" * 70)
    print(f"SUMMARY — {len(per_question)} questions × {runs_per_question} run(s)")
    print("=" * 70)
    print(f"Accuracy (vs golden):     SQL={mean_acc_sql:.3f}   rows={mean_acc_rows:.3f}   "
          f"values={mean_acc_values:.3f}   count={mean_acc_count:.3f}")
    print(f"Latency:                  mean per question {mean_elapsed_all:.1f}s   "
          f"total wall clock {total_elapsed_all:.0f}s")

    summary = {
        "questions_evaluated": len(per_question),
        "runs_per_question": runs_per_question,
        "mean_accuracy_sql_jaccard": mean_acc_sql,
        "mean_accuracy_rows_jaccard": mean_acc_rows,
        "mean_accuracy_values_jaccard": mean_acc_values,
        "mean_accuracy_count_ratio": mean_acc_count,
        "mean_elapsed_seconds_per_question": round(mean_elapsed_all, 2),
        "total_elapsed_seconds": round(total_elapsed_all, 2),
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
