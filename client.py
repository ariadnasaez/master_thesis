import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama
import json
import re

TEXT_TYPES = {"char", "varchar", "text", "mediumtext", "longtext"}


def is_text_column(data_type: str) -> bool:
    return any(t in data_type.lower() for t in TEXT_TYPES)


def fix_sql(query: str, schema: dict) -> str:
    """Convert exact-match conditions to LIKE for text columns."""
    if not query or not schema:
        return query
    text_columns = {
        col["column"].lower()
        for entry in schema.values()
        for col in (entry["columns"] if isinstance(entry, dict) and "columns" in entry else entry)
        if is_text_column(col.get("type", ""))
    }

    def replacer(match):
        column, value = match.group(1), match.group(2)
        if column.lower() in text_columns:
            return f"LOWER({column}) LIKE '%{value.lower()}%'"
        return match.group(0)

    return re.sub(r"(\w+)\s*=\s*'([^']*)'", replacer, query, flags=re.IGNORECASE)


async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch live schema from server resource
            # schema_resource calls the get_full_schema resource on the server, which connects to MySQL, runs DESCRIBE on every table,
            # and returns the result as a string. That string is then stored in schema_text and injected into the system prompt.
            schema_resource = await session.read_resource("schema://full")
            schema_text = schema_resource.contents[0].text

            # Parse compact format "table(col type, ...)" into schema_cache for fix_sql
            schema_cache = {}
            for line in schema_text.splitlines():
                if "(" not in line:
                    continue
                table_name = line[:line.index("(")]
                cols_part = line[line.index("(")+1 : line.index(")")]
                schema_cache[table_name] = {
                    "columns": [
                        {"column": c.strip().split()[0], "type": c.strip().split()[1] if len(c.strip().split()) > 1 else ""}
                        for c in cols_part.split(",") if c.strip()
                    ]
                }

            # Only expose execute_query to the LLM — schema is already in context
            mcp_tools = await session.list_tools()
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools.tools
                if t.name == "execute_query"
            ]

            system_prompt = (
                "You are a MySQL expert. You MUST always call the execute_query tool to answer questions. "
                "Never answer from memory. Always run SQL first, then explain the result.\n"
                "Rules:\n"
                "- Only SELECT queries are allowed.\n"
                "- Use LIKE for text/varchar column searches, not =.\n"
                "- Never invent table or column names not listed in the schema below.\n"
                "- To count distinct patients, use COUNT(DISTINCT patient_ref) from the table that contains patient_ref.\n\n"
                f"Database schema:\n{schema_text}"
            )

            print("\n" + "=" * 60)
            print("  GROUNDED MYSQL AGENT")
            print("=" * 60)

            while True:
                user_input = input("\nAsk: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ]

                tool_was_called = False
                malformed_count = 0
                for _ in range(6):  # max steps
                    print("⏳ Thinking...", end="\r", flush=True)
                    response = ollama.chat(model="llama3.2", messages=messages, tools=ollama_tools)
                    print("             ", end="\r", flush=True)  # clear line
                    msg = response["message"]
                    messages.append(msg)

                    if not msg.get("tool_calls"):
                        if not tool_was_called:
                            if malformed_count >= 2:
                                print("⚠️  Model failed to produce a valid tool call. Try rephrasing.")
                                break
                            messages.append({
                                "role": "user",
                                "content": "You must call execute_query with a SQL SELECT statement. Do not answer without querying the database."
                            })
                            malformed_count += 1
                            continue
                        break

                    for tool_call in msg["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]

                        # llama3.2 sometimes wraps string args as {'type': 'string', 'value': '...'}
                        # or sends {'type': 'string'} with no value — extract string or skip
                        def unwrap(v):
                            if isinstance(v, dict):
                                return v.get("value", "") if isinstance(v.get("value"), str) else ""
                            return v
                        args = {k: unwrap(v) for k, v in args.items()}

                        # If the query arg is empty or missing after unwrapping, skip and nudge
                        if tool_name == "execute_query" and not args.get("query", "").strip():
                            if malformed_count >= 2:
                                print("⚠️  Model failed to produce a valid SQL query. Try rephrasing.")
                                break
                            messages.append({
                                "role": "user",
                                "content": "You must provide a valid SQL SELECT statement as the 'query' argument to execute_query."
                            })
                            malformed_count += 1
                            continue

                        print(f"\n🔧 Tool: {tool_name}")
                        print(f"📥 Args: {args}")

                        result = await session.call_tool(tool_name, args)
                        result_text = result.content[0].text

                        # Self-healing: if empty result, retry with LIKE fix
                        if tool_name == "execute_query":
                            sql_query = args.get("query", "")
                            try:
                                parsed = json.loads(result_text)
                            except Exception:
                                parsed = []
                            if isinstance(parsed, list) and len(parsed) == 0:
                                fixed = fix_sql(sql_query, schema_cache)
                                if fixed != sql_query:
                                    print(f"⚠️  Empty result → retrying with LIKE fix")
                                    print(f"🔁 Fixed SQL: {fixed}")
                                    retry = await session.call_tool(tool_name, {"query": fixed})
                                    result_text = retry.content[0].text

                        tool_was_called = True
                        messages.append({"role": "tool", "name": tool_name, "content": result_text})

                # Find the last assistant text message as the answer
                answer = next(
                    (m["content"] for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")),
                    None
                )
                print(f"\n✅ Answer:\n{answer}" if answer else "\n⚠️  No answer returned by model.")


if __name__ == "__main__":
    asyncio.run(main())
