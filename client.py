# MCP client: terminal interface that orchestrates the conversation between
# the user and the LLM (Qwen via Ollama). Shares all agent logic with app.py
# via the `agent` module.
import asyncio

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from agent_ollama import (
    build_ollama_tools,
    build_system_prompt,
    parse_schema_cache,
    run_agent_loop_async,
    warmup_model,
)


async def main():
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

            async def call_tool(name, args):
                result = await session.call_tool(name, args)
                return result.content[0].text

            print("⏳ Warming up the model (loading + caching prompt prefix)...")
            warmup_model(system_prompt, ollama_tools)

            print("\n" + "=" * 60)
            print("  GROUNDED MYSQL AGENT")
            print("=" * 60)

            while True:
                user_input = input("\nAsk: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue

                print("⏳ Thinking...", end="\r", flush=True)
                answer, tool_log = await run_agent_loop_async(
                    user_input, system_prompt, schema_cache, ollama_tools, call_tool
                )
                print("             ", end="\r", flush=True)

                for entry in tool_log:
                    tag = " (retry)" if entry.get("retry") else ""
                    print(f"\n🔧 Tool: {entry['name']}{tag}")
                    print(f"📥 Args: {entry['args']}")

                if answer:
                    print(f"\n✅ Answer:\n{answer}")
                else:
                    print("\n⚠️  No answer returned by model.")


if __name__ == "__main__":
    asyncio.run(main())
