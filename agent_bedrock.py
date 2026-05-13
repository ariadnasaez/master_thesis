"""
Bedrock-backed agent — drop-in replacement for agent.py using AWS Bedrock
(DeepSeek V3.2, Claude, etc.) via the boto3 Converse API instead of local Ollama.

Public interface mirrors agent.py so generate.py and _worker.py can swap which
module they import without any further code changes:
    - DEFAULT_MODEL
    - build_ollama_tools(mcp_tools_response)
    - build_system_prompt(schema_text)     [re-exported from agent]
    - parse_schema_cache(schema_text)      [re-exported from agent]
    - warmup_model(system_prompt, tools)   [no-op for cloud]
    - run_agent_loop_async(...)            [async, same signature as agent.py]

The "build_ollama_tools" name is kept for parity even though the format we
return here is Bedrock's toolConfig, not Ollama's.
"""

import asyncio
import json
import os

import boto3

from agent import (
    fix_sql,
    parse_schema_cache,
    build_system_prompt,
    CORRECTION_MARKER,
    MAX_AGENT_STEPS,
    MAX_CORRECTION_STEPS,
)


# Default model can be overridden via BEDROCK_MODEL_ID env var so you can swap
# between DeepSeek V3.2, Claude Sonnet, etc. without touching code.
DEFAULT_MODEL = os.getenv("BEDROCK_MODEL_ID", "deepseek.v3.2")
REGION = os.getenv("BEDROCK_REGION", "us-east-2")
INFERENCE_CONFIG = {"temperature": 0, "maxTokens": 8192}


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


# ---------------------------------------------------------------------------
# Tool / prompt helpers
# ---------------------------------------------------------------------------

def build_ollama_tools(mcp_tools_response):
    """Convert MCP tools to Bedrock toolConfig format.
    Named build_ollama_tools (not build_bedrock_tools) so generate.py and
    _worker.py don't need to change their import line — the returned shape
    is fed to the Bedrock client.
    """
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


def warmup_model(system_prompt, tools, model=DEFAULT_MODEL):
    """No-op for Bedrock — cloud-hosted models have no cold-start to pay for."""
    return


# ---------------------------------------------------------------------------
# Bedrock conversation helpers
# ---------------------------------------------------------------------------

async def _converse_async(model, system, messages, tool_config=None):
    """Run client.converse in a thread executor so the event loop stays free."""
    loop = asyncio.get_event_loop()
    kwargs = {
        "modelId": model,
        "system": system,
        "messages": messages,
        "inferenceConfig": INFERENCE_CONFIG,
    }
    if tool_config is not None:
        kwargs["toolConfig"] = tool_config
    return await loop.run_in_executor(None, lambda: _get_client().converse(**kwargs))


def _extract_text(content):
    return "".join(b.get("text", "") for b in content if "text" in b)


def _extract_tool_uses(content):
    return [b["toolUse"] for b in content if "toolUse" in b]


def _last_assistant_text(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            text = _extract_text(m["content"])
            if text:
                return text
    return None


def _maybe_fix_empty_result(tool_name, args, result_text, schema_cache):
    """If execute_query returned an empty list, try fix_sql to LIKE-rewrite text filters.
    Returns (maybe-new-result-text, fixed-sql-or-None). Mirrors agent.py behaviour.
    """
    if tool_name != "execute_query":
        return result_text, None
    sql_query = args.get("query", "")
    try:
        parsed = json.loads(result_text)
    except Exception:
        parsed = []
    if isinstance(parsed, list) and len(parsed) == 0:
        fixed = fix_sql(sql_query, schema_cache)
        if fixed != sql_query:
            return None, fixed
    return result_text, None


VERIFY_PROMPT = (
    "VERIFY YOUR ANSWER (this is a verification pass — do NOT call any more tools).\n"
    "Check carefully:\n"
    "1. Does your answer address the user's ORIGINAL question precisely "
    "(not a related but different one)?\n"
    "2. Did you apply EVERY filter mentioned in the question "
    "(sex, condition, year, specialty, location, etc.)?\n"
    "3. Are the codes/values you used the correct ones from the lookup results?\n"
    "4. If the question asks for a count, average, or list, is that exactly what you returned?\n\n"
    "If correct, RESTATE the answer concisely with no preamble.\n"
    "If you spot an error, reply: 'CORRECTION NEEDED: <one sentence describing what is wrong>'."
)


# ---------------------------------------------------------------------------
# Main agent loop (Bedrock variant)
# ---------------------------------------------------------------------------

async def run_agent_loop_async(user_input,
                               system_prompt,
                               schema_cache,
                               tools,
                               call_tool,
                               model=DEFAULT_MODEL,
                               max_steps=MAX_AGENT_STEPS,
                               verify=True):
    """Bedrock-backed agent loop. Mirrors agent.run_agent_loop_async semantics:
    same discover → answer → verify → correction pipeline, same nudge logic if
    the model fails to call a tool, same fix_sql retry for empty execute_query
    results.

    call_tool: async callable (tool_name: str, args: dict) -> result_text: str
    Returns: (answer: str, tool_log: list[dict])
    """
    tool_log = []
    system = [{"text": system_prompt}]
    messages = [{"role": "user", "content": [{"text": user_input}]}]
    tool_was_called = False
    malformed_count = 0

    for _ in range(max_steps):
        response = await _converse_async(model, system, messages, tools)
        msg = response["output"]["message"]
        messages.append(msg)

        tool_uses = _extract_tool_uses(msg["content"])

        if not tool_uses:
            if not tool_was_called:
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid tool call. Try rephrasing.", tool_log
                messages.append({
                    "role": "user",
                    "content": [{"text": (
                        "You must call one of the available tools (lookup_codes, list_distinct_values, "
                        "or execute_query) before answering. Do not respond from memory."
                    )}],
                })
                malformed_count += 1
                continue
            break  # done — final text answer produced after tool use

        # Execute every tool call in this turn and collect results into one user message
        tool_results = []
        for tu in tool_uses:
            tool_name = tu["name"]
            tool_use_id = tu["toolUseId"]
            args = tu.get("input", {}) or {}

            if tool_name == "execute_query" and not str(args.get("query", "")).strip():
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid SQL query. Try rephrasing.", tool_log
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": "ERROR: query argument is empty. Provide a valid SQL SELECT statement."}],
                        "status": "error",
                    }
                })
                malformed_count += 1
                continue

            tool_log.append({"name": tool_name, "args": args})
            result_text = await call_tool(tool_name, args)

            new_text, fixed = _maybe_fix_empty_result(tool_name, args, result_text, schema_cache)
            if fixed is not None:
                result_text = await call_tool(tool_name, {"query": fixed})
                tool_log.append({"name": tool_name, "args": {"query": fixed}, "retry": True})
            else:
                result_text = new_text

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": result_text}],
                }
            })
            tool_was_called = True

        messages.append({"role": "user", "content": tool_results})

    answer = _last_assistant_text(messages)

    if not answer and tool_was_called:
        messages.append({
            "role": "user",
            "content": [{"text": (
                "Summarize the results you have so far and answer the original question. "
                "Do not call any more tools."
            )}],
        })
        try:
            response = await _converse_async(model, system, messages)
            answer = _extract_text(response["output"]["message"]["content"])
        except Exception:
            answer = None

    if verify and answer and tool_was_called:
        verify_messages = messages + [{"role": "user", "content": [{"text": VERIFY_PROMPT}]}]
        try:
            verify_response = await _converse_async(model, system, verify_messages)
            verify_text = _extract_text(verify_response["output"]["message"]["content"]).strip()
        except Exception:
            verify_text = ""

        if verify_text and verify_text.upper().startswith(CORRECTION_MARKER):
            messages.append({
                "role": "user",
                "content": [{"text": (
                    f"Verification flagged your answer: {verify_text}\n"
                    "Use additional tool calls if needed (lookup_codes, list_distinct_values, execute_query) "
                    "to fix this, then provide the corrected answer. Do not run verify again."
                )}],
            })

            for _ in range(MAX_CORRECTION_STEPS):
                response = await _converse_async(model, system, messages, tools)
                msg = response["output"]["message"]
                messages.append(msg)

                tool_uses = _extract_tool_uses(msg["content"])
                if not tool_uses:
                    break

                tool_results = []
                for tu in tool_uses:
                    tool_name = tu["name"]
                    tool_use_id = tu["toolUseId"]
                    args = tu.get("input", {}) or {}
                    tool_log.append({"name": tool_name, "args": args, "correction": True})
                    result_text = await call_tool(tool_name, args)
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": result_text}],
                        }
                    })
                messages.append({"role": "user", "content": tool_results})

            answer = _last_assistant_text(messages) or verify_text
        else:
            answer = verify_text or answer

    return answer or "⚠️ No answer returned by model.", tool_log
