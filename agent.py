"""
Shared agent module for the DataNex Clinical Database Agent.

Used by both:
- client.py (terminal interface)
- app.py (Gradio web UI)

Single source of truth for:
- The system prompt builder
- SQL fix-up helpers (LIKE conversion for text columns)
- Schema parsing
- The LLM tool-calling loop (sync and async variants)
"""

import json
import re
import ollama


TEXT_TYPES = {"char", "varchar", "text", "mediumtext", "longtext"}

CODE_COLUMNS = {"ou_med_ref", "episode_type_ref", "care_level_ref", "sex", "natio_ref",
                "diag_ref", "lab_sap_ref", "lab_ref", "ou_loc_ref", "care_level_type_ref",
                "facility_ref", "rc_sap_ref", "rc_ref", "catalog", "code"}

DEFAULT_MODEL = "qwen3.5:9b"
MAX_AGENT_STEPS = 10

# Ollama runtime options shared by every chat call.
# - temperature=0 makes generation deterministic (required for Jaccard determinism eval).
# - keep_alive keeps the model in GPU memory between calls so we don't pay cold-start
#   prompt-eval costs (the slow 30s+ first call).
LLM_OPTIONS = {"temperature": 0}
LLM_KEEP_ALIVE = "30m"


def is_text_column(data_type: str) -> bool:
    return any(t in data_type.lower() for t in TEXT_TYPES)


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

    query = re.sub(r"(\w+\.)?"r"(\w+)\s*=\s*'([^']*)'", replacer, query, flags=re.IGNORECASE)
    query = re.sub(r"(\w+\.)?"r'(\w+)\s*=\s*"([^"]*)"', replacer, query, flags=re.IGNORECASE)
    return query


def parse_schema_cache(schema_text: str) -> dict:
    """Parse compact schema 'table(col type, ...)' into a dict for fix_sql."""
    schema_cache = {}
    for line in schema_text.splitlines():
        if "(" not in line:
            continue
        table_name = line[:line.index("(")]
        cols_part = line[line.index("(") + 1: line.index(")")]
        schema_cache[table_name] = {
            "columns": [
                {
                    "column": c.strip().split()[0],
                    "type": c.strip().split()[1] if len(c.strip().split()) > 1 else "",
                }
                for c in cols_part.split(",") if c.strip()
            ]
        }
    return schema_cache


def build_ollama_tools(mcp_tools_response):
    """Convert MCP tools response to Ollama tool format. Only exposes execute_query."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools_response.tools
        if t.name == "execute_query"
    ]


def build_system_prompt(schema_text: str) -> str:
    """Build the LLM system prompt with the database schema injected at the end."""
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
        "3. COUNTING ADMISSIONS BY DIAGNOSIS — use this exact pattern (note: episode_ref join is correct ONLY for admission/episode counts):\n"
        "   SELECT COUNT(DISTINCT e.episode_ref) FROM g_episodes e JOIN g_diagnostics gd ON e.patient_ref = gd.patient_ref AND e.episode_ref = gd.episode_ref WHERE e.episode_type_ref = 'HOSP' AND e.start_date >= 'YYYY-01-01' AND e.start_date < 'YYYY+1-01-01' AND gd.code LIKE 'prefix%'\n"
        "4. LAB QUERIES FILTERED BY A DIAGNOSIS — DO NOT use a JOIN with g_diagnostics. "
        "Use a subquery instead: WHERE l.patient_ref IN (SELECT DISTINCT patient_ref FROM g_diagnostics WHERE code LIKE 'X%'). "
        "Reason: joining on episode_ref restricts labs to the same encounter as the diagnosis, which is wrong for chronic "
        "conditions (diabetes, hypertension, CKD) where labs are monitored across many encounters. The subquery pattern "
        "also avoids row multiplication when patients have multiple diagnosis codes. This OVERRIDES the join pattern in rule 1 "
        "for lab × diagnosis questions.\n\n"

        "=== GENERAL RULES ===\n"
        "- Only SELECT queries are allowed.\n"
        "- Use LIKE for text/varchar searches, not =. NEVER use * in LIKE — use % only.\n"
        "- Never invent table or column names not in the schema below.\n"
        "- To count distinct patients, use COUNT(DISTINCT patient_ref).\n"
        "- When grouping by a description column, always include it in SELECT.\n"
        "- When a query returns non-empty results, use them immediately. Do not keep refining.\n"
        "- Only return columns directly relevant to the question.\n"
        "- Do NOT add episode_type_ref = 'HOSP' or g_episodes joins unless the question specifically asks about hospitalizations/ingresos.\n"
        "- Always add ORDER BY to sort results.\n"
        "- PARENTHESES: when mixing OR and AND in a WHERE clause, ALWAYS wrap the OR group in parentheses. "
        "AND binds tighter than OR, so missing parens silently change the query meaning. "
        "Example: WHERE (code LIKE 'E10%' OR code LIKE 'E11%') AND lab_sap_ref IN (...).\n\n"

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
        "- UNIT CONVERSION: When different lab codes have different units, convert with CASE inside AVG to unify "
        "(e.g. HbA1c IFCC→NGSP: (result_num / 10.929) + 2.15). After unifying units, return a SINGLE AVG WITHOUT "
        "GROUP BY — one number, not a per-method breakdown. Only use GROUP BY lab_descr if you genuinely cannot "
        "unify variants to the same unit.\n"
        "- LAB + CHRONIC CONDITION (diabetes, hipertensión, CKD, etc.): filter patients via subquery, NOT a JOIN. "
        "Use: WHERE l.patient_ref IN (SELECT DISTINCT patient_ref FROM g_diagnostics WHERE code LIKE 'X%'). "
        "Joining on episode_ref restricts to labs in the same encounter as the diagnosis, which is wrong for "
        "chronic conditions monitored across many encounters.\n\n"

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


def warmup_model(system_prompt, ollama_tools, model=DEFAULT_MODEL):
    """Pre-load the model and prime the KV cache with the system prompt.

    Call this once at app startup so the first user question doesn't pay the
    full prompt-eval cost (~30s for a 3000-token prompt on Qwen 9B).
    """
    try:
        ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "ready"},
            ],
            tools=ollama_tools,
            options=LLM_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
        )
    except Exception as e:
        # Non-fatal — the first real query will just pay the cold-start cost itself.
        print(f"⚠️  Warmup failed (non-fatal): {e}")


def _unwrap_arg(v):
    """Some models wrap string args as {'type': 'string', 'value': '...'}. Extract the raw string."""
    if isinstance(v, dict):
        return v.get("value", "") if isinstance(v.get("value"), str) else ""
    return v


def _next_step(messages, ollama_tools, model, tool_was_called, malformed_count):
    """Run one Ollama turn. Returns (msg, action) where action is one of:
    ('done',)                   — agent finished, return last assistant message
    ('nudge_no_tool',)          — model didn't call a tool when it should have
    ('tool_calls', tool_calls)  — model wants to call tools
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=ollama_tools,
        options=LLM_OPTIONS,
        keep_alive=LLM_KEEP_ALIVE,
    )
    msg = response["message"]
    if not msg.get("tool_calls"):
        if not tool_was_called:
            return msg, ("nudge_no_tool",)
        return msg, ("done",)
    return msg, ("tool_calls", msg["tool_calls"])


def _validate_tool_args(tool_call):
    """Unwrap and validate tool call args. Returns (tool_name, args, ok). If not ok, args is the failure reason."""
    tool_name = tool_call["function"]["name"]
    raw_args = tool_call["function"]["arguments"]
    args = {k: _unwrap_arg(v) for k, v in raw_args.items()}
    if tool_name == "execute_query" and not args.get("query", "").strip():
        return tool_name, args, False
    return tool_name, args, True


def _maybe_fix_empty_result(tool_name, args, result_text, schema_cache):
    """If the result is an empty list, try fix_sql to convert = to LIKE on text columns.
    Returns (maybe-new-result-text, fixed-sql-or-None).
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
            return None, fixed  # signal caller to retry with fixed
    return result_text, None


def run_agent_loop_sync(user_input,
                        system_prompt,
                        schema_cache,
                        ollama_tools,
                        call_tool,
                        model=DEFAULT_MODEL,
                        max_steps=MAX_AGENT_STEPS):
    """LLM tool-calling loop with a SYNC tool caller.

    call_tool: callable (tool_name: str, args: dict) -> result_text: str
    Returns: (answer: str, tool_log: list[dict])
    Each tool_log entry is {"name": str, "args": dict}.
    """
    tool_log = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    tool_was_called = False
    malformed_count = 0

    for _ in range(max_steps):
        msg, action = _next_step(messages, ollama_tools, model, tool_was_called, malformed_count)
        messages.append(msg)

        if action[0] == "done":
            break

        if action[0] == "nudge_no_tool":
            if malformed_count >= 2:
                return "⚠️ Model failed to produce a valid tool call. Try rephrasing.", tool_log
            messages.append({"role": "user",
                             "content": "You must call execute_query with a SQL SELECT statement. Do not answer without querying the database."})
            malformed_count += 1
            continue

        # action[0] == "tool_calls"
        for tool_call in action[1]:
            tool_name, args, ok = _validate_tool_args(tool_call)
            if not ok:
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid SQL query. Try rephrasing.", tool_log
                messages.append({"role": "user",
                                 "content": "You must provide a valid SQL SELECT statement as the 'query' argument to execute_query."})
                malformed_count += 1
                continue

            tool_log.append({"name": tool_name, "args": args})
            result_text = call_tool(tool_name, args)

            new_text, fixed = _maybe_fix_empty_result(tool_name, args, result_text, schema_cache)
            if fixed is not None:
                result_text = call_tool(tool_name, {"query": fixed})
                tool_log.append({"name": tool_name, "args": {"query": fixed}, "retry": True})
            else:
                result_text = new_text

            tool_was_called = True
            messages.append({"role": "tool", "name": tool_name, "content": result_text})

    answer = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
        None,
    )
    if not answer and tool_was_called:
        messages.append({"role": "user",
                         "content": "Summarize the results you have so far and answer the original question. Do not call any more tools."})
        final = ollama.chat(
            model=model,
            messages=messages,
            options=LLM_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
        )
        answer = final["message"].get("content")

    return answer or "⚠️ No answer returned by model.", tool_log


async def run_agent_loop_async(user_input,
                               system_prompt,
                               schema_cache,
                               ollama_tools,
                               call_tool,
                               model=DEFAULT_MODEL,
                               max_steps=MAX_AGENT_STEPS):
    """LLM tool-calling loop with an ASYNC tool caller.

    call_tool: async callable (tool_name: str, args: dict) -> result_text: str
    Returns: (answer: str, tool_log: list[dict])
    """
    tool_log = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    tool_was_called = False
    malformed_count = 0

    for _ in range(max_steps):
        msg, action = _next_step(messages, ollama_tools, model, tool_was_called, malformed_count)
        messages.append(msg)

        if action[0] == "done":
            break

        if action[0] == "nudge_no_tool":
            if malformed_count >= 2:
                return "⚠️ Model failed to produce a valid tool call. Try rephrasing.", tool_log
            messages.append({"role": "user",
                             "content": "You must call execute_query with a SQL SELECT statement. Do not answer without querying the database."})
            malformed_count += 1
            continue

        for tool_call in action[1]:
            tool_name, args, ok = _validate_tool_args(tool_call)
            if not ok:
                if malformed_count >= 2:
                    return "⚠️ Model failed to produce a valid SQL query. Try rephrasing.", tool_log
                messages.append({"role": "user",
                                 "content": "You must provide a valid SQL SELECT statement as the 'query' argument to execute_query."})
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

            tool_was_called = True
            messages.append({"role": "tool", "name": tool_name, "content": result_text})

    answer = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
        None,
    )
    if not answer and tool_was_called:
        messages.append({"role": "user",
                         "content": "Summarize the results you have so far and answer the original question. Do not call any more tools."})
        final = ollama.chat(
            model=model,
            messages=messages,
            options=LLM_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
        )
        answer = final["message"].get("content")

    return answer or "⚠️ No answer returned by model.", tool_log
