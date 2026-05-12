"""Single-question worker. Spawned as a subprocess by evaluate.py for each question.

Runs one question through the agent and writes the result (sql, rows, elapsed) as JSON
to the output file. Designed to be killed cleanly via subprocess.run(timeout=...).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Choose backend via env var: AGENT_BACKEND=bedrock or AGENT_BACKEND=ollama (default).
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "ollama")
if AGENT_BACKEND == "bedrock":
    from agent_bedrock import build_ollama_tools, build_system_prompt, parse_schema_cache, run_agent_loop_async
else:
    from agent import build_ollama_tools, build_system_prompt, parse_schema_cache, run_agent_loop_async


async def run(question: str, schema_text: str, output_path: str) -> None:
    schema_cache = parse_schema_cache(schema_text)
    system_prompt = build_system_prompt(schema_text)

    server_params = StdioServerParameters(command=sys.executable, args=[str(HERE / "server.py")])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ollama_tools = build_ollama_tools(await session.list_tools())

            captured = {"sql": "", "rows": []}

            async def call_tool(name, args):
                result = await session.call_tool(name, args)
                text = result.content[0].text
                if name == "execute_query":
                    captured["sql"] = args.get("query", "")
                    try:
                        captured["rows"] = json.loads(text) if text else []
                    except Exception:
                        captured["rows"] = []
                return text

            t0 = time.monotonic()
            try:
                await run_agent_loop_async(question, system_prompt, schema_cache, ollama_tools, call_tool)
            except Exception as e:
                sys.stderr.write(f"agent error: {e}\n")
            elapsed = time.monotonic() - t0

    Path(output_path).write_text(json.dumps({
        "sql": captured["sql"],
        "rows": captured["rows"],
        "elapsed": round(elapsed, 2),
    }))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: _worker.py <question> <schema_file> <output_file>\n")
        sys.exit(1)
    schema_text = Path(sys.argv[2]).read_text()
    asyncio.run(run(sys.argv[1], schema_text, sys.argv[3]))
