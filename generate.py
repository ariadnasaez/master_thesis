"""
Generation phase — runs the agent on each question in the golden dataset and
saves the raw outputs (SQL, rows, elapsed) to generation_results.json.

This is the EXPENSIVE phase (LLM inference). It is decoupled from scoring so
that you can re-run the metric computation many times (with new metrics, bug
fixes, etc.) without re-running the LLM.

Usage:
    python generate.py                # 1 run per question
    python generate.py --runs 3       # 3 runs per question (for determinism)

Output:
    generation_results.json — per-question raw outputs to feed into evaluate.py
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Backend selection (set BEFORE importing agent module so subprocess inherits it).
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "ollama")
if AGENT_BACKEND == "bedrock":
    from agent_bedrock import (
        DEFAULT_MODEL,
        build_ollama_tools,
        build_system_prompt,
        warmup_model,
    )
else:
    from agent import (
        DEFAULT_MODEL,
        build_ollama_tools,
        build_system_prompt,
        warmup_model,
    )


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
DEFAULT_OUTPUT_PATH = HERE / "generation_result_sonnet.json"
WORKER_PATH = HERE / "_worker.py"

QUESTION_TIMEOUT = 10 * 60  # seconds — questions that exceed this are skipped
COOLDOWN_AFTER_TIMEOUT = 60  # seconds to let CPU/GPU recover after a timeout


def _run_question_sync(question: str, schema_file: str, output_file: str):
    """Spawn _worker.py as a subprocess to run one question through the agent.
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
        return data.get("sql", ""), data.get("rows", []), float(data.get("elapsed", 0.0)), int(data.get("num_calls", 0))
    except (FileNotFoundError, json.JSONDecodeError):
        return "", [], 0.0, 0


def _recover_after_timeout(system_prompt: str, ollama_tools: list) -> None:
    """Force-abort any in-progress Ollama generation, cool down, then re-warmup."""
    subprocess.run(["ollama", "stop", DEFAULT_MODEL], capture_output=True, check=False)
    time.sleep(COOLDOWN_AFTER_TIMEOUT)
    warmup_model(system_prompt, ollama_tools)


async def generate(runs_per_question: int, output_path: Path):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    print(f"Loaded {len(golden)} entries from {GOLDEN_PATH.name}")
    print(f"Backend: {AGENT_BACKEND}  •  Model: {DEFAULT_MODEL}  •  Output → {output_path.name}\n")

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
            print("✅ Warmed up. Starting generation.\n")

    schema_file = HERE / "_eval_schema.txt"
    schema_file.write_text(schema_text)

    per_question = []
    timed_out_ids = []
    loop = asyncio.get_running_loop()
    try:
        for i, item in enumerate(golden, 1):
            question = item.get("question", "")
            qid = item.get("id", i)

            if not question:
                continue

            print(f"[{i}/{len(golden)}] (id={qid}) {question[:80]}")

            runs = []
            timed_out = False
            for r in range(runs_per_question):
                output_file = HERE / f"_eval_q{i}_r{r}.json"
                try:
                    try:
                        sql, rows, elapsed, num_calls = await loop.run_in_executor(
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
                    runs.append({"sql": sql, "rows": rows, "elapsed_seconds": round(elapsed, 2), "num_calls": num_calls})
                    print(f"    run {r+1}/{runs_per_question}: {len(rows)} rows  ({elapsed:.1f}s)  {num_calls} tool calls")
                finally:
                    output_file.unlink(missing_ok=True)

            if timed_out or not runs:
                if timed_out:
                    timed_out_ids.append(qid)
                continue

            per_question.append({
                "id": qid,
                "question": question,
                "runs": runs,
            })
            print()
    finally:
        schema_file.unlink(missing_ok=True)

    output = {
        "model": DEFAULT_MODEL,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_per_question": runs_per_question,
        "questions_total": len(golden),
        "questions_completed": len(per_question),
        "timed_out_ids": timed_out_ids,
        "per_question": per_question,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Generation complete: {len(per_question)}/{len(golden)} questions, "
          f"{len(timed_out_ids)} timed out")
    print(f"Raw outputs → {output_path.name}")
    print("Run `python evaluate.py` to compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to run each question (>=2 enables determinism metrics).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename. Defaults to generation_result_sonnet.json (ollama) "
                             "or generation_result_deepseek.json (bedrock).")
    args = parser.parse_args()

    if args.output:
        output_path = HERE / args.output
    elif AGENT_BACKEND == "bedrock":
        # Sanitize model ID into a filename slug:
        #   us.anthropic.claude-sonnet-4-6  → claude-sonnet-4-6
        #   deepseek.v3.2                   → deepseek.v3.2
        model_slug = DEFAULT_MODEL.removeprefix("us.anthropic.").replace(".", "_")
        runs_suffix = f"_{args.runs}runs" if args.runs > 1 else ""
        output_path = HERE / f"generation_result_{model_slug}{runs_suffix}.json"
    else:
        output_path = DEFAULT_OUTPUT_PATH

    asyncio.run(generate(args.runs, output_path))
