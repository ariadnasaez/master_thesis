"""
Measure the actual token counts Bedrock reports for the system prompt
used in the evaluation, for both Claude Sonnet 4.6 and DeepSeek V3.2.

Run from the master_thesis/ directory:
    python count_tokens.py

Requires the MCP server (server.py) to be startable — it fetches the real
schema text the same way generate.py does.
"""

import asyncio
import json
import os

import boto3
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent import build_system_prompt, build_ollama_tools

REGION = os.getenv("BEDROCK_REGION", "us-east-2")

MODELS = {
    "Claude Sonnet 4.6": os.getenv(
        "BEDROCK_SONNET_ID", "us.anthropic.claude-sonnet-4-6"
    ),
    "DeepSeek V3.2": os.getenv(
        "BEDROCK_DEEPSEEK_ID", "deepseek.v3.2"
    ),
}

client = boto3.client("bedrock-runtime", region_name=REGION)


def _build_system(system_prompt: str, model: str) -> list:
    """Mirror agent_bedrock._build_system: add cachePoint for Anthropic only."""
    system = [{"text": system_prompt}]
    if "anthropic" in model.lower():
        system.append({"cachePoint": {"type": "default"}})
    return system


def _bedrock_tool_config(mcp_tools_response) -> dict:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.inputSchema},
                }
            }
            for t in mcp_tools_response.tools
        ]
    }


async def measure():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch schema (same as generate.py / app.py)
            schema_resource = await session.read_resource("schema://full")
            schema_text = schema_resource.contents[0].text

            mcp_tools = await session.list_tools()
            tool_config = _bedrock_tool_config(mcp_tools)

            system_prompt = build_system_prompt(schema_text)

            print(f"System prompt length : {len(system_prompt):,} chars")
            print(f"Number of tools      : {len(mcp_tools.tools)}")
            print()

            # One trivial user message — we only care about usage, not the answer
            messages = [{"role": "user", "content": [{"text": "ping"}]}]

            for label, model_id in MODELS.items():
                system = _build_system(system_prompt, model_id)
                try:
                    response = client.converse(
                        modelId=model_id,
                        system=system,
                        messages=messages,
                        toolConfig=tool_config,
                        inferenceConfig={"temperature": 0, "maxTokens": 10},
                    )
                    usage = response["usage"]
                    print(f"=== {label} ({model_id}) ===")
                    print(json.dumps(usage, indent=2))
                    # cacheReadInputTokens / cacheWriteInputTokens appear for Claude
                    total_in  = usage.get("inputTokens", 0)
                    total_out = usage.get("outputTokens", 0)
                    cached_r  = usage.get("cacheReadInputTokens", 0)
                    cached_w  = usage.get("cacheWriteInputTokens", 0)
                    print(f"  → input: {total_in:,}  output: {total_out}")
                    if cached_r or cached_w:
                        print(f"  → cache write: {cached_w:,}  cache read: {cached_r:,}")
                    print()
                except Exception as e:
                    print(f"=== {label} — ERROR: {e} ===\n")


if __name__ == "__main__":
    asyncio.run(measure())
