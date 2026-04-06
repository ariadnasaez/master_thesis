# MCP client: orchestrates the conversation between the user and the LLM (qwen2.5:9b via ollama).
# Fetches the DB schema once at startup, builds the system prompt, and manages the tool-calling loop.
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama
import json
import re

TEXT_TYPES = {"char", "varchar", "text", "mediumtext", "longtext"}


def is_text_column(data_type: str) -> bool:
    return any(t in data_type.lower() for t in TEXT_TYPES)


CODE_COLUMNS = {"ou_med_ref", "episode_type_ref", "care_level_ref", "sex", "natio_ref",
                "diag_ref", "lab_sap_ref", "lab_ref", "ou_loc_ref", "care_level_type_ref",
                "facility_ref", "rc_sap_ref", "rc_ref", "catalog", "code"}

def fix_sql(query: str, schema: dict) -> str:
    """Convert exact-match conditions to LIKE for text columns, excluding controlled code columns."""
    if not query or not schema:
        return query
    text_columns = {
        col["column"].lower()
        for entry in schema.values()
        for col in (entry["columns"] if isinstance(entry, dict) and "columns" in entry else entry)
        if is_text_column(col.get("type", ""))
    } - CODE_COLUMNS

    def replacer(match):
        table_prefix, column, value = match.group(1), match.group(2), match.group(3)
        if column.lower() in text_columns:
            qualified = f"{table_prefix}{column}" if table_prefix else column
            return f"LOWER({qualified}) LIKE '%{value.lower()}%'"
        return match.group(0)

    # Match optional table prefix: col = 'val' or table.col = 'val' (single and double quotes)
    query = re.sub(r"(\w+\.)?"r"(\w+)\s*=\s*'([^']*)'", replacer, query, flags=re.IGNORECASE)
    query = re.sub(r"(\w+\.)?"r'(\w+)\s*=\s*"([^"]*)"', replacer, query, flags=re.IGNORECASE)
    return query


async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch live schema from server resource ONCE at startup.
            # The schema is stored in memory and reused in every question via the system prompt.
            # The DB is only queried again if the client is restarted.
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
                "- To count distinct patients, use COUNT(DISTINCT patient_ref) from the table that contains patient_ref.\n"
                "- When grouping by a description column (e.g. snomed_descr), always include it in the SELECT clause so names appear in results.\n"
                "- MEDICAL SPECIALTY FILTER — read carefully:\n"
                "  * 'Diagnosed by' a specialty → filter g_health_issues.ou_med_ref\n"
                "  * 'Attended by' / 'seen by' / 'atendidos por' a specialty → filter g_labs.ou_med_ref (for lab questions) or g_episodes joined with g_movements (for admission questions). Do NOT filter g_health_issues.ou_med_ref — diagnoses are often recorded by a different unit than the one treating the patient.\n"
                "  * First query g_movements to find the ou_med_ref code for the specialty, then use it as an exact filter on the appropriate table.\n"
                "- For procedure-based questions: do exactly ONE lookup query using BOTH the procedure type keyword AND the anatomy keyword (e.g. WHERE descr LIKE '%sustitucion%' AND descr LIKE '%cadera%'). Use ALL codes returned by the lookup in code IN (...) in the final query — do not add or remove any codes.\n"
                "- When a question asks about ingresos or hospitalizations after an event: filter g_episodes with episode_type_ref='HOSP' AND admission_date > event_date (strictly greater — do not include the same day).\n"
                "- When a query returns non-empty results, use them immediately to answer. Do not keep refining or retrying.\n"
                "- Only return columns that are directly relevant to the question asked. Do not add extra columns like sex, age, or nationality unless explicitly requested.\n"
                "- When filtering lab tests: first do a lookup query on g_labs to find the exact lab_descr and lab_sap_ref values (e.g. SELECT DISTINCT lab_sap_ref, lab_descr FROM g_labs WHERE lab_descr LIKE '%keyword%'). Then use lab_sap_ref IN (...) in the main query for reliable filtering. Lab names are in Spanish (e.g. 'Hemoglobina glicada' not 'HbA1c', 'Glucosa' not 'Glucose'). Use LIKE 'term%' (starts with) not LIKE '%term%' when filtering lab_descr directly.\n"
                "- Always add ORDER BY to sort results alphabetically by the main description column.\n"
                "- To get average lab values per diagnosis for patients attended by a specialty: start from g_health_issues, "
                "LEFT JOIN g_labs on patient_ref, filter g_labs.ou_med_ref for the unit (not g_health_issues.ou_med_ref), "
                "use lab_sap_ref IN (...) for the test, GROUP BY g_health_issues.snomed_descr.\n"
                "- FIRST/LAST LAB VALUE BEFORE/AFTER A PROCEDURE (per episode):\n"
                "  Use a CTE pattern with ROW_NUMBER(). MANDATORY rules:\n"
                "  1) Join g_procedures to g_labs on BOTH episode_ref AND patient_ref — never patient_ref alone.\n"
                "  2) Use extrac_date (blood draw time), NOT result_date (report time), when comparing to procedure start_date.\n"
                "  3) For 'after': filter extrac_date >= start_date. For 'before': filter extrac_date < start_date.\n"
                "  4) Use ROW_NUMBER() OVER (PARTITION BY episode_ref ORDER BY extrac_date ASC) for first, DESC for last.\n"
                "  5) Filter WHERE rn = 1 in the outer query.\n"
                "  Template:\n"
                "    WITH proc AS (SELECT episode_ref, patient_ref, code, descr, start_date FROM g_procedures WHERE code IN (...)),\n"
                "         lab AS (SELECT episode_ref, patient_ref, extrac_date, lab_descr, result_num FROM g_labs WHERE lab_descr LIKE 'Term%' AND result_num IS NOT NULL),\n"
                "         joined AS (SELECT proc.*, lab.extrac_date, lab.lab_descr, lab.result_num,\n"
                "                    ROW_NUMBER() OVER (PARTITION BY proc.episode_ref ORDER BY lab.extrac_date ASC) AS rn\n"
                "                    FROM proc JOIN lab ON lab.episode_ref = proc.episode_ref AND lab.patient_ref = proc.patient_ref\n"
                "                    AND lab.extrac_date >= proc.start_date)\n"
                "    SELECT episode_ref, code, descr, start_date, lab_descr, result_num, extrac_date FROM joined WHERE rn = 1;\n\n"
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
                for _ in range(10):  # max steps
                    print("⏳ Thinking...", end="\r", flush=True)
                    response = ollama.chat(model="qwen3.5:9b", messages=messages, tools=ollama_tools)
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

                # If the loop exhausted and the last message is a tool result,
                # make one final call WITHOUT tools to force a text answer.
                answer = next(
                    (m["content"] for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")),
                    None
                )
                if not answer and tool_was_called:
                    messages.append({
                        "role": "user",
                        "content": "Summarize the results you have so far and answer the original question. Do not call any more tools."
                    })
                    print("⏳ Generating final answer...", end="\r", flush=True)
                    final = ollama.chat(model="qwen3.5:9b", messages=messages)
                    final_msg = final["message"]
                    messages.append(final_msg)
                    answer = final_msg.get("content")
                    print("                              ", end="\r", flush=True)

                print(f"\n✅ Answer:\n{answer}" if answer else "\n⚠️  No answer returned by model.")


if __name__ == "__main__":
    asyncio.run(main())
