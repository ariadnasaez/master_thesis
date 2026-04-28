# Web UI for the Grounded MySQL Agent using Gradio.
# Launches a chat interface where users can ask questions about the database.
import asyncio
import threading
import traceback

import gradio as gr
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent import (
    build_ollama_tools,
    build_system_prompt,
    parse_schema_cache,
    run_agent_loop_sync,
    warmup_model,
)


# Global state — initialized once when the server starts
agent_state = {"ready": False}

_loop = None
_session = None
_ollama_tools = None
_system_prompt = None
_schema_cache = None


def _call_tool_sync(tool_name, args):
    """Schedule an MCP tool call on the MCP event loop and wait for the result text."""
    future = asyncio.run_coroutine_threadsafe(
        _session.call_tool(tool_name, args), _loop
    )
    result = future.result(timeout=30)
    return result.content[0].text


async def _init_mcp():
    """Initialize the MCP session and keep it alive."""
    global _session, _ollama_tools, _system_prompt, _schema_cache

    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            schema_resource = await session.read_resource("schema://full")
            schema_text = schema_resource.contents[0].text
            schema_cache = parse_schema_cache(schema_text)

            mcp_tools = await session.list_tools()
            ollama_tools = build_ollama_tools(mcp_tools)

            _session = session
            _ollama_tools = ollama_tools
            _system_prompt = build_system_prompt(schema_text)
            _schema_cache = schema_cache

            print("⏳ Warming up the model (loading + caching prompt prefix)...")
            warmup_model(_system_prompt, _ollama_tools)
            agent_state["ready"] = True

            print("✅ MCP session ready (model warm)")
            while True:
                await asyncio.sleep(1)


def _run_mcp_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_until_complete(_init_mcp())


def ask_question(message, history):
    """Gradio chat handler. Runs the agent and returns the response."""
    if not agent_state["ready"]:
        return "⏳ Agent is still initializing, please wait..."

    try:
        answer, tool_log = run_agent_loop_sync(
            message,
            _system_prompt,
            _schema_cache,
            _ollama_tools,
            _call_tool_sync,
        )
    except Exception:
        tb = traceback.format_exc()
        print(f"Error processing question:\n{tb}")
        return f"❌ Error: {tb}"

    response = ""
    if tool_log:
        last = tool_log[-1]
        tag = " (retry)" if last.get("retry") else ""
        response += "**🔍 Final SQL query:**\n\n"
        response += f"🔧 **{last['name']}**{tag}\n```sql\n{last['args'].get('query', '')}\n```\n\n---\n\n"
    response += answer
    return response


# Start MCP in background thread
mcp_thread = threading.Thread(target=_run_mcp_loop, daemon=True)
mcp_thread.start()

# Build Gradio UI
demo = gr.ChatInterface(
    fn=ask_question,
    title="🏥 DataNex Clinical Database Agent",
    description="Ask questions about the hospital database in natural language.",
    examples=[
        "Cuantos ingresos por ictus isquémico hubo en 2024?",
        "Qué valor medio tienen de HbA1c los diabéticos tipo 2 atendidos por endocrinología?",
        "Cuántas mujeres tienen diagnóstico de fibrilación auricular?",
        "Qué microorganismos se aislaron en hemocultivos positivos de pacientes hospitalizados?",
    ],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
