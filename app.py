# Web UI for the Grounded MySQL Agent using Gradio.
# Launches a chat interface where users can ask questions about the database.
import asyncio
import threading
import traceback
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama
import json
import re
import gradio as gr

TEXT_TYPES = {"char", "varchar", "text", "mediumtext", "longtext"}

def is_text_column(data_type: str) -> bool:
    return any(t in data_type.lower() for t in TEXT_TYPES)

CODE_COLUMNS = {"ou_med_ref", "episode_type_ref", "care_level_ref", "sex", "natio_ref",
                "diag_ref", "lab_sap_ref", "lab_ref", "ou_loc_ref", "care_level_type_ref",
                "facility_ref", "rc_sap_ref", "rc_ref", "catalog", "code"}

def fix_sql(query: str, schema: dict) -> str:
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

    query = re.sub(r"(\w+\.)?"r"(\w+)\s*=\s*'([^']*)'", replacer, query, flags=re.IGNORECASE)
    query = re.sub(r"(\w+\.)?"r'(\w+)\s*=\s*"([^"]*)"', replacer, query, flags=re.IGNORECASE)
    return query


# Global state — initialized once when the server starts
agent_state = {
    "session": None,
    "ollama_tools": None,
    "system_prompt": None,
    "schema_cache": None,
    "ready": False,
}


def build_system_prompt(schema_text):
    return (
        "You are a MySQL expert. You MUST always call the execute_query tool to answer questions. "
        "Never answer from memory. Always run SQL first, then explain the result.\n\n"

        "=== CRITICAL RULES (read first) ===\n"
        "1. DIAGNOSIS FILTERING — ALWAYS use g_diagnostics (ICD codes), NOT g_health_issues, as your FIRST choice:\n"
        "   - Step 1: Lookup ICD codes: SELECT DISTINCT code, diag_descr FROM g_diagnostics WHERE diag_descr LIKE '%keyword1%' OR diag_descr LIKE '%keyword2%'\n"
        "   - Step 2: From the lookup results, identify the common ICD prefix (e.g. I21.3, I21.4 → prefix is 'I21'). In the main query, ALWAYS use code LIKE 'prefix%' — NEVER list individual codes with IN (...). Examples: code LIKE 'I63%' for stroke, 'I21%' for MI, 'E11%' for diabetes tipo 2, 'J18%' for pneumonia. This catches ALL subtypes including ones not yet in the data.\n"
        "   - NEVER add diag_descr LIKE '%keyword%' as a fallback alongside the code filter. Description matching catches unrelated conditions (e.g. '%miocardio%' matches miocardiopatía and infarto antiguo, not just acute MI). The ICD prefix alone is sufficient and more precise.\n"
        "   - Step 3: Join g_diagnostics to g_episodes on BOTH patient_ref AND episode_ref.\n"
        "   - NEVER use g_health_issues for counting admissions — its episode_ref is DOUBLE and incompatible with g_episodes.\n"
        "   - Only use g_health_issues (SNOMED) when the question specifically requires SNOMED codes or when ICD lookup returns nothing.\n"
        "2. LOOKUP EFFICIENCY: Run at most 2 lookup queries total. Combine keywords with OR. If the first lookup returns results, proceed immediately.\n"
        "3. COUNTING ADMISSIONS BY DIAGNOSIS — use this exact pattern:\n"
        "   SELECT COUNT(DISTINCT e.episode_ref) FROM g_episodes e JOIN g_diagnostics gd ON e.patient_ref = gd.patient_ref AND e.episode_ref = gd.episode_ref WHERE e.episode_type_ref = 'HOSP' AND e.start_date >= 'YYYY-01-01' AND e.start_date < 'YYYY+1-01-01' AND gd.code LIKE 'prefix%'\n\n"

        "=== GENERAL RULES ===\n"
        "- Only SELECT queries are allowed.\n"
        "- Use LIKE for text/varchar searches, not =. NEVER use * in LIKE — use % only.\n"
        "- Never invent table or column names not in the schema below.\n"
        "- To count distinct patients, use COUNT(DISTINCT patient_ref).\n"
        "- When grouping by a description column, always include it in SELECT.\n"
        "- When a query returns non-empty results, use them immediately. Do not keep refining.\n"
        "- Only return columns directly relevant to the question.\n"
        "- Do NOT add episode_type_ref = 'HOSP' or g_episodes joins unless the question specifically asks about hospitalizations/ingresos.\n"
        "- Always add ORDER BY to sort results.\n\n"

        "=== MEDICAL SPECIALTY FILTER ===\n"
        "- 'Problemas de [specialty]' / 'pacientes de [specialty]' / 'diagnosed by' → use g_health_issues filtered by ou_med_ref. This gives all health issues recorded by that specialty. Do NOT search for ICD codes manually — just filter ou_med_ref.\n"
        "  Example: 'problemas de nefrología' → g_health_issues WHERE ou_med_ref = 'NEF'.\n"
        "  ALWAYS GROUP BY hi.snomed_descr and include it in SELECT — this shows results broken down per condition, which is always more informative than a single aggregate.\n"
        "- 'Attended by' / 'atendidos por' → filter g_labs.ou_med_ref (lab questions) or g_episodes+g_movements (admission questions).\n"
        "- To find the ou_med_ref code: SELECT DISTINCT ou_med_ref, ou_med_descr FROM g_movements WHERE ou_med_descr LIKE '%keyword%'.\n"
        "- When joining g_health_issues with g_labs: join on patient_ref ONLY (episode_ref types incompatible). No g_episodes needed unless the question asks about hospitalizations.\n\n"

        "=== AGE FILTER ===\n"
        "- Use g_demographics for age: JOIN g_demographics d ON d.patient_ref = [table].patient_ref\n"
        "- Calculate age: TIMESTAMPDIFF(YEAR, d.birth_date, CURDATE())\n"
        "- ALWAYS explicitly JOIN g_demographics — never reference it without joining.\n\n"

        "=== DRUGS / MEDICATIONS ===\n"
        "- For drug administration questions, use g_administrations. For prescription questions, use g_prescriptions.\n"
        "- ALWAYS filter by atc_descr (standardized drug name like 'Apixaban'), NEVER by drug_descr (which includes dosage forms like 'APIXABAN, 2,5 MG COMP' and would split one drug into multiple rows).\n"
        "- Lookup first: SELECT DISTINCT atc_descr FROM g_administrations WHERE atc_descr LIKE '%keyword%'. Then use atc_descr IN (...) in the main query.\n"
        "- given = 'X' means the drug was actually administered. Filter given = 'X' when the question asks about drugs actually given.\n\n"

        "=== LAB TESTS ===\n"
        "- Lookup first: SELECT DISTINCT lab_sap_ref, lab_descr, units FROM g_labs WHERE lab_descr LIKE '%keyword%'. Then filter by lab_sap_ref IN (...). Lab names are in Spanish.\n"
        "- UNIT CONVERSION: When different lab codes have different units, convert with CASE in AVG. HbA1c IFCC→NGSP: (result_num / 10.929) + 2.15. GROUP BY lab_descr.\n\n"

        "=== PROCEDURES ===\n"
        "- Lookup codes first: SELECT DISTINCT code, descr FROM g_procedures WHERE descr LIKE '%type%' AND descr LIKE '%anatomy%'. Use ALL returned codes.\n"
        "- Join g_procedures to g_labs/g_episodes on BOTH patient_ref AND episode_ref — NEVER patient_ref alone.\n\n"

        "=== FIRST/LAST LAB VALUE BEFORE/AFTER A PROCEDURE ===\n"
        "Use CTE + ROW_NUMBER(). Rules:\n"
        "1) Join on BOTH episode_ref AND patient_ref.\n"
        "2) Use extrac_date (blood draw), NOT result_date.\n"
        "3) 'after': extrac_date >= start_date. 'before': extrac_date < start_date.\n"
        "4) ROW_NUMBER() OVER (PARTITION BY episode_ref ORDER BY extrac_date ASC) for first.\n"
        "5) WHERE rn = 1 in outer query.\n"
        "Template:\n"
        "  WITH proc AS (SELECT episode_ref, patient_ref, code, descr, start_date FROM g_procedures WHERE code IN (...)),\n"
        "       lab AS (SELECT episode_ref, patient_ref, extrac_date, lab_descr, result_num FROM g_labs WHERE lab_descr LIKE 'Term%' AND result_num IS NOT NULL),\n"
        "       joined AS (SELECT proc.*, lab.extrac_date, lab.lab_descr, lab.result_num,\n"
        "                  ROW_NUMBER() OVER (PARTITION BY proc.episode_ref ORDER BY lab.extrac_date ASC) AS rn\n"
        "                  FROM proc JOIN lab ON lab.episode_ref = proc.episode_ref AND lab.patient_ref = proc.patient_ref\n"
        "                  AND lab.extrac_date >= proc.start_date)\n"
        "  SELECT * FROM joined WHERE rn = 1;\n\n"

        "=== LOCATION-BASED QUESTIONS ===\n"
        "- Use g_movements for ward/location questions. Lookup: SELECT DISTINCT ou_loc_ref, ou_loc_descr FROM g_movements WHERE ou_loc_descr LIKE '%keyword%'. Filter by ou_loc_ref.\n\n"

        "=== HOSPITALIZATION RULES ===\n"
        "- 'ingresos' or 'hospitalizations' → filter g_episodes.episode_type_ref = 'HOSP'.\n"
        "- After an event → e.start_date > event_date (strictly greater).\n\n"

        f"Database schema:\n{schema_text}"
    )


def _call_tool(tool_name, args):
    """Schedule an MCP tool call on the MCP event loop and wait for the result."""
    future = asyncio.run_coroutine_threadsafe(
        _session.call_tool(tool_name, args), _loop
    )
    return future.result(timeout=30)


def process_question(user_input, ollama_tools, system_prompt, schema_cache):
    """Process a single question through the agent loop. Returns (answer, tool_log)."""
    tool_log = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tool_was_called = False
    malformed_count = 0
    for _ in range(10):
        response = ollama.chat(model="qwen3.5:9b", messages=messages, tools=ollama_tools)
        msg = response["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            if not tool_was_called:
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid tool call. Try rephrasing.", tool_log
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

            def unwrap(v):
                if isinstance(v, dict):
                    return v.get("value", "") if isinstance(v.get("value"), str) else ""
                return v
            args = {k: unwrap(v) for k, v in args.items()}

            if tool_name == "execute_query" and not args.get("query", "").strip():
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid SQL query. Try rephrasing.", tool_log
                messages.append({
                    "role": "user",
                    "content": "You must provide a valid SQL SELECT statement as the 'query' argument to execute_query."
                })
                malformed_count += 1
                continue

            tool_log.append(f"🔧 **{tool_name}**\n```sql\n{args.get('query', '')}\n```")

            result = _call_tool(tool_name, args)
            result_text = result.content[0].text

            if tool_name == "execute_query":
                sql_query = args.get("query", "")
                try:
                    parsed = json.loads(result_text)
                except Exception:
                    parsed = []
                if isinstance(parsed, list) and len(parsed) == 0:
                    fixed = fix_sql(sql_query, schema_cache)
                    if fixed != sql_query:
                        tool_log.append(f"⚠️ Empty result → retrying with LIKE fix")
                        retry = _call_tool(tool_name, {"query": fixed})
                        result_text = retry.content[0].text

            tool_was_called = True
            messages.append({"role": "tool", "name": tool_name, "content": result_text})

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
        final = ollama.chat(model="qwen3.5:9b", messages=messages)
        final_msg = final["message"]
        answer = final_msg.get("content")

    return answer or "⚠️ No answer returned by model.", tool_log


# MCP session runs in its own event loop in a background thread
_loop = None
_session = None
_ollama_tools = None
_system_prompt = None
_schema_cache = None


async def _init_mcp():
    """Initialize the MCP session and keep it alive."""
    global _session, _ollama_tools, _system_prompt, _schema_cache

    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            schema_resource = await session.read_resource("schema://full")
            schema_text = schema_resource.contents[0].text

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

            _session = session
            _ollama_tools = ollama_tools
            _system_prompt = build_system_prompt(schema_text)
            _schema_cache = schema_cache
            agent_state["ready"] = True

            print("✅ MCP session ready")

            # Keep the session alive until the app exits
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
        answer, tool_log = process_question(message, _ollama_tools, _system_prompt, _schema_cache)
    except Exception:
        tb = traceback.format_exc()
        print(f"Error processing question:\n{tb}")
        return f"❌ Error: {tb}"

    # Build response with tool calls shown
    response = ""
    if tool_log:
        response += "**🔍 SQL queries executed (" + str(len(tool_log)) + "):**\n\n"
        response += "\n\n".join(tool_log)
        response += "\n\n---\n\n"
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
