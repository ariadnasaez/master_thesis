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

import asyncio
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
    """Convert MCP tools response to Ollama tool format. Exposes ALL tools the server publishes
    (e.g. lookup_codes, list_distinct_values, execute_query) so the LLM can pick the right one."""
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
    ]


def build_system_prompt(schema_text: str) -> str:
    """Build the LLM system prompt with the database schema injected at the end."""
    return (
        "You are a MySQL expert with four tools. Always run a tool before answering — never answer from memory.\n\n"

        "=== TOOLS ===\n"
        "1. lookup_ontology(keyword) — FIRST, when a Spanish-language column or coded reference field is "
        "ambiguous (e.g. freq_ref, mot_type, encounter_type, place_ref). Returns ontology classes with their "
        "parent table and a Spanish description, so you know which DB table/column the concept belongs to. "
        "Skip this when the schema already makes the mapping obvious.\n"
        "2. lookup_codes(table, descr_column, keyword) — BEFORE filtering on any coded column (ICD codes, "
        "lab_sap_ref, ou_med_ref, snomed_ref, atc_descr) when the user mentions a clinical concept by name. "
        "Returns DISTINCT (code, description) pairs. Use plain ASCII keywords (e.g. 'diabetes' not 'diabéticos').\n"
        "3. list_distinct_values(table, column) — BEFORE filtering on any varchar column whose stored values "
        "you don't know (avoids guessing 'España' vs 'Espana', 'Catalunya' vs 'Cataluña'). ALWAYS use for "
        "natio_descr, ou_med_descr, ou_loc_descr, episode_type_ref, mue_descr, sensitivity, given, etc.\n"
        "4. execute_query(query) — the MAIN SELECT/WITH query. Call LAST, never guess literals here — every "
        "code/string filter must be backed by a previous lookup.\n"
        "Workflow: clarify (tool 1, optional) → discover (tools 2, 3) → answer (tool 4). Most questions need "
        "1-2 lookups. Combine keywords with OR where possible; if the first lookup returns the codes you need, "
        "proceed immediately.\n\n"

        "=== JOIN INVARIANTS ===\n"
        "- g_diagnostics, g_procedures, g_labs, g_movements, g_administrations, g_prescriptions, g_micro: "
        "join to g_episodes on BOTH patient_ref AND episode_ref. NEVER on patient_ref alone.\n"
        "- g_demographics: join on patient_ref only (one row per patient).\n"
        "- g_health_issues: episode_ref is DOUBLE and incompatible with g_episodes — join on patient_ref ONLY, "
        "and never use it for counting admissions.\n\n"

        "=== DIAGNOSIS FILTERING ===\n"
        "ALWAYS use g_diagnostics (ICD codes) as first choice. Use g_health_issues (SNOMED) only when the "
        "question explicitly requires SNOMED or when ICD lookup returns nothing.\n"
        "- Step 1 (lookup): SELECT DISTINCT code, diag_descr FROM g_diagnostics WHERE diag_descr LIKE '%kw1%' OR diag_descr LIKE '%kw2%'\n"
        "- Step 2 (filter): identify the common ICD prefix from the lookup (e.g. I21.3, I21.4 → 'I21') and "
        "filter with code LIKE 'prefix%'. NEVER list individual codes with IN (...) — prefix LIKE catches all "
        "subtypes including future ones. Examples: 'I63%' stroke, 'I21%' MI, 'E11%' diabetes tipo 2, 'J18%' pneumonia.\n"
        "- General vs specific: broad conditions ('diabéticos', 'cardiopatía', 'enfermedad renal') typically map "
        "to MULTIPLE prefix families (across ICD chapters and legacy 3-digit ICD-9 families). Use ALL prefixes "
        "the lookup returned, joined with OR. Narrow to one prefix only when the question names a subtype "
        "('tipo 2', 'isquémico', 'fase 5').\n"
        "- NEVER add diag_descr LIKE '%kw%' alongside the code filter — description matching catches unrelated "
        "conditions (e.g. '%miocardio%' matches miocardiopatía AND infarto antiguo). The ICD prefix is sufficient.\n\n"

        "=== MEDICAL SPECIALTY ===\n"
        "- 'Problemas de [specialty]' / 'pacientes de [specialty]' / 'diagnosed by' → g_health_issues filtered "
        "by ou_med_ref. Do NOT look up ICD codes — just filter ou_med_ref. Example: 'problemas de nefrología' "
        "→ g_health_issues WHERE ou_med_ref = 'NEF'. ALWAYS GROUP BY hi.snomed_descr (and include in SELECT) "
        "to break results down per condition.\n"
        "- 'Attended by' / 'atendidos por' → filter g_labs.ou_med_ref (lab questions) or g_episodes+g_movements "
        "(admission questions).\n"
        "- Find the ou_med_ref code: SELECT DISTINCT ou_med_ref, ou_med_descr FROM g_movements WHERE ou_med_descr LIKE '%kw%'.\n\n"

        "=== LAB TESTS ===\n"
        "- MANDATORY lookup when the question names a specific measurement (potasio, creatinina, HbA1c, "
        "troponina, PCR, etc.): SELECT DISTINCT lab_sap_ref, lab_descr, units FROM g_labs WHERE lab_descr LIKE "
        "'%kw%'. The main query MUST then filter by lab_sap_ref IN (...). Querying g_labs without a lab_sap_ref "
        "filter returns EVERY lab in the system for those patients (hundreds of unrelated tests).\n"
        "- UNIT CONVERSION: the lookup returns the units column so you can spot heterogeneous units. If two or "
        "more rows have different units (e.g. % vs mmol/mol, mg/dL vs mmol/L), you MUST convert with a CASE "
        "inside AVG to unify BEFORE averaging — branch the CASE on lab_sap_ref values, not on the units column. "
        "Known conversion: HbA1c IFCC (mmol/mol) → NGSP (%): (result_num / 10.929) + 2.15. After unifying, "
        "return a SINGLE AVG WITHOUT GROUP BY.\n"
        "- LAB + CHRONIC CONDITION (diabetes, hipertensión, CKD, etc.): filter patients via SUBQUERY, not a "
        "JOIN: WHERE l.patient_ref IN (SELECT DISTINCT patient_ref FROM g_diagnostics WHERE code LIKE 'X%'). "
        "Joining on episode_ref restricts labs to the same encounter as the diagnosis, which is wrong for "
        "chronic conditions monitored across many encounters. This OVERRIDES the standard join pattern.\n"
        "  EXCEPTION — if the question asks for labs *during a specific type of admission* ('ingresados por X', "
        "'durante una hospitalización por X'): join labs to episodes by BOTH patient_ref AND episode_ref, and "
        "filter episode_type_ref = 'HOSP' AND diagnosis code in that same episode. Do NOT use the subquery "
        "pattern here — you need labs from that specific encounter, not all-time labs.\n\n"

        "=== PROCEDURES ===\n"
        "- Lookup first: SELECT DISTINCT code, descr FROM g_procedures WHERE descr LIKE '%type%' AND descr LIKE "
        "'%anatomy%'. Use ALL returned codes in the main query.\n"
        "- CROSS-DOMAIN PATIENT QUERIES (diagnosis + procedure): when a question asks which patients satisfy "
        "two independent clinical criteria (e.g. 'patients with diagnosis X who also had procedure Y'), do NOT "
        "join g_diagnostics to g_procedures directly — that produces one row per (diagnosis × procedure) "
        "combination and explodes into thousands of rows. Instead, intersect via subqueries:\n"
        "    SELECT DISTINCT patient_ref FROM g_diagnostics WHERE code LIKE 'X%'\n"
        "    AND patient_ref IN (SELECT DISTINCT patient_ref FROM g_procedures WHERE code IN (...))\n"
        "  Or use a CTE per domain and then JOIN on patient_ref with SELECT DISTINCT.\n"
        "- FIRST/LAST LAB VALUE BEFORE/AFTER A PROCEDURE — use CTE + ROW_NUMBER(). Use extrac_date (blood draw), "
        "NOT result_date. 'after': extrac_date >= start_date. 'before': extrac_date < start_date. WHERE rn = 1 "
        "in outer query.\n"
        "  Template:\n"
        "    WITH proc AS (SELECT episode_ref, patient_ref, code, descr, start_date FROM g_procedures WHERE code IN (...)),\n"
        "         lab  AS (SELECT episode_ref, patient_ref, extrac_date, lab_descr, result_num FROM g_labs WHERE lab_descr LIKE 'Term%' AND result_num IS NOT NULL),\n"
        "         joined AS (SELECT proc.*, lab.extrac_date, lab.lab_descr, lab.result_num,\n"
        "                    ROW_NUMBER() OVER (PARTITION BY proc.episode_ref ORDER BY lab.extrac_date ASC) AS rn\n"
        "                    FROM proc JOIN lab ON lab.episode_ref = proc.episode_ref AND lab.patient_ref = proc.patient_ref\n"
        "                    AND lab.extrac_date >= proc.start_date)\n"
        "    SELECT * FROM joined WHERE rn = 1;\n\n"

        "=== DRUGS / MEDICATIONS ===\n"
        "- g_administrations for administered drugs, g_prescriptions for prescribed drugs.\n"
        "- Prefer atc_descr for grouping/filtering (standardized name like 'Apixaban'). drug_descr includes dosage "
        "forms ('APIXABAN, 2,5 MG COMP') and splits one drug into multiple rows when used alone.\n"
        "- When atc_descr may be NULL for some records, use COALESCE(atc_descr, drug_descr) as the display/group "
        "column and add drug_descr LIKE '%KW%' fallback conditions in WHERE so those records are not missed.\n"
        "- Lookup: SELECT DISTINCT atc_descr FROM g_administrations WHERE atc_descr LIKE '%kw%'. Then atc_descr IN (...).\n"
        "- given = 'X' means actually administered. Filter given = 'X' when the question asks about drugs actually given.\n\n"

        "=== DEMOGRAPHICS / AGE ===\n"
        "- For age or sex filters, JOIN g_demographics d ON d.patient_ref = [table].patient_ref.\n"
        "- Age: TIMESTAMPDIFF(YEAR, d.birth_date, CURDATE()).\n"
        "- ALWAYS explicitly join g_demographics — never reference d.* without the join.\n\n"

        "=== LOCATION ===\n"
        "- Use g_movements for ward/location. Lookup: SELECT DISTINCT ou_loc_ref, ou_loc_descr FROM g_movements "
        "WHERE ou_loc_descr LIKE '%kw%'. Filter by ou_loc_ref.\n\n"

        "=== EPISODE TYPES & TIME ===\n"
        "All encounter/visit/episode questions go through g_episodes — there is NO g_encounters table. "
        "Filter by g_episodes.episode_type_ref:\n"
        "  • 'ingresos' / 'hospitalizations' / 'hospitalizados' → 'HOSP'\n"
        "  • 'ambulatorios' / 'consulta externa' / 'visita externa' → 'AM'\n"
        "  • 'urgencias' / 'emergencias' → 'EM'\n"
        "If the question doesn't specify an encounter type, do NOT filter episode_type_ref at all.\n"
        "- After an event: e.start_date > event_date (strictly greater).\n"
        "- DURATION (length of stay, 'how many days'): use DATEDIFF(end, start) or TIMESTAMPDIFF(DAY, start, end). "
        "NEVER write (end_date - start_date) — in MySQL that returns a numeric expression, NOT days, and silently "
        "produces wrong results. Example: DATEDIFF(e.end_date, e.start_date) > 7.\n"
        "- ELAPSED HOURS (integer): use TIMESTAMPDIFF(HOUR, start, end) which returns a whole-number integer "
        "(truncates fractional hours). Do NOT use TIMESTAMPDIFF(MINUTE, start, end) / 60 — that produces "
        "decimal strings like '21.57' instead of integers like 21.\n\n"

        "=== SQL HYGIENE ===\n"
        "- Only SELECT queries. Use LIKE for text/varchar (with % only, never *). Never invent table or column names.\n"
        "- COUNT(DISTINCT patient_ref) for distinct patient counts; COUNT(DISTINCT episode_ref) for admissions.\n"
        "- When grouping by a description column, include it in SELECT. Add ORDER BY to sort.\n"
        "- Return only columns directly relevant to the question. Use first non-empty result — don't keep refining.\n"
        "- PARENTHESES: when mixing OR and AND in WHERE, ALWAYS wrap the OR group in parens. AND binds tighter "
        "than OR, so missing parens silently change meaning. Example: WHERE (code LIKE 'E10%' OR code LIKE 'E11%') "
        "AND lab_sap_ref IN (...).\n"
        "- RESULT SHAPE: match the query shape to the question:\n"
        "  • 'cuál es el valor medio/máximo/mínimo de X' / 'cuántos X' (singular) → ONE row, NO GROUP BY.\n"
        "  • '… para cada Y' / '… por cada Y' / '… por unidad/mes/especialidad' → one row per Y, GROUP BY Y.\n"
        "  • 'qué pacientes …' / 'cuáles son …' / 'muestra los …' → list of rows, NO aggregate.\n"
        "  If your query has GROUP BY but the question never says 'para cada' / 'por cada' / 'por X', remove "
        "the GROUP BY (and unify variants via CASE if units differ).\n"
        "- SCALAR AGGREGATE — EXACTLY ONE COLUMN: when the question asks for a single statistic ('cuál es el "
        "promedio de X', 'qué media tiene X'), return ONLY that aggregate column. Do NOT add companion columns "
        "like num_measurements, min_value, max_value, std_dev, or any other statistics the question did not "
        "ask for. Extra columns make the result look like a table rather than a scalar and will not match the "
        "expected output.\n\n"

        f"Database schema:\n{schema_text}"
    )


def _verify_answer(messages, answer, model):
    """Explicit verify pass. Asks the model to re-check whether its answer correctly addresses
    the user's question, given the tool calls and results so far. Returns the (possibly revised)
    answer.

    Adds one extra LLM call per question. No tool calls allowed in the verify pass — the model
    can only restate, refine, or flag the answer.
    """
    verify_messages = messages + [
        {"role": "user", "content": (
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
        )}
    ]
    try:
        response = ollama.chat(
            model=model,
            messages=verify_messages,
            options=LLM_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
        )
        verified = (response["message"].get("content") or "").strip()
        return verified or answer
    except Exception:
        return answer


CORRECTION_MARKER = "CORRECTION NEEDED"
MAX_CORRECTION_STEPS = 3


def _needs_correction(verify_result: str) -> bool:
    return bool(verify_result) and verify_result.strip().upper().startswith(CORRECTION_MARKER)


def _correction_user_message(verify_result: str) -> dict:
    return {
        "role": "user",
        "content": (
            f"Verification flagged your answer: {verify_result}\n"
            "Use additional tool calls if needed (lookup_codes, list_distinct_values, execute_query) "
            "to fix this, then provide the corrected answer. Do not run verify again."
        ),
    }


def _run_correction_pass_sync(messages, ollama_tools, call_tool, model, tool_log,
                              max_steps=MAX_CORRECTION_STEPS):
    """After a failed verify, give the agent a bounded chance to fix its answer.
    Mirrors the main loop body but caps iterations and tags tool_log entries as 'correction'.
    """
    for _ in range(max_steps):
        response = ollama.chat(
            model=model, messages=messages, tools=ollama_tools,
            options=LLM_OPTIONS, keep_alive=LLM_KEEP_ALIVE,
        )
        msg = response["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return  # revised text answer produced, exit
        for tool_call in msg["tool_calls"]:
            tool_name, args, ok = _validate_tool_args(tool_call)
            if not ok:
                continue
            tool_log.append({"name": tool_name, "args": args, "correction": True})
            result_text = call_tool(tool_name, args)
            messages.append({"role": "tool", "name": tool_name, "content": result_text})


async def _run_correction_pass_async(messages, ollama_tools, call_tool, model, tool_log,
                                     max_steps=MAX_CORRECTION_STEPS):
    """Async twin of _run_correction_pass_sync."""
    for _ in range(max_steps):
        response = await _ollama_chat_async(
            model=model, messages=messages, tools=ollama_tools,
            options=LLM_OPTIONS, keep_alive=LLM_KEEP_ALIVE,
        )
        msg = response["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            return
        for tool_call in msg["tool_calls"]:
            tool_name, args, ok = _validate_tool_args(tool_call)
            if not ok:
                continue
            tool_log.append({"name": tool_name, "args": args, "correction": True})
            result_text = await call_tool(tool_name, args)
            messages.append({"role": "tool", "name": tool_name, "content": result_text})


def _extract_last_assistant_answer(messages):
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("content")),
        None,
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


async def _ollama_chat_async(model, messages, tools=None, options=None, keep_alive=None):
    """Run ollama.chat in a thread executor so the event loop stays free and asyncio.wait_for can fire."""
    loop = asyncio.get_event_loop()
    kwargs = {"model": model, "messages": messages, "options": options, "keep_alive": keep_alive}
    if tools is not None:
        kwargs["tools"] = tools
    return await loop.run_in_executor(None, lambda: ollama.chat(**kwargs))


async def _next_step_async(messages, ollama_tools, model, tool_was_called, malformed_count):
    """Async twin of _next_step — ollama.chat runs in a thread so the event loop stays responsive."""
    response = await _ollama_chat_async(
        model=model, messages=messages, tools=ollama_tools,
        options=LLM_OPTIONS, keep_alive=LLM_KEEP_ALIVE,
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
                        max_steps=MAX_AGENT_STEPS,
                        verify=True):
    """LLM tool-calling loop with a SYNC tool caller.

    call_tool: callable (tool_name: str, args: dict) -> result_text: str
    verify: if True, run an explicit verify pass after the answer is produced (+1 LLM call).
            Disable to measure baseline cost without verification.
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
                             "content": "You must call one of the available tools (lookup_codes, list_distinct_values, or execute_query) before answering. Do not respond from memory."})
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

    if verify and answer and tool_was_called:
        verify_result = _verify_answer(messages, answer, model)
        if _needs_correction(verify_result):
            messages.append(_correction_user_message(verify_result))
            _run_correction_pass_sync(messages, ollama_tools, call_tool, model, tool_log)
            answer = _extract_last_assistant_answer(messages) or verify_result
        else:
            answer = verify_result or answer

    return answer or "⚠️ No answer returned by model.", tool_log


async def run_agent_loop_async(user_input,
                               system_prompt,
                               schema_cache,
                               ollama_tools,
                               call_tool,
                               model=DEFAULT_MODEL,
                               max_steps=MAX_AGENT_STEPS,
                               verify=True):
    """LLM tool-calling loop with an ASYNC tool caller.

    call_tool: async callable (tool_name: str, args: dict) -> result_text: str
    verify: if True, run an explicit verify pass after the answer is produced (+1 LLM call).
            Disable to measure baseline cost without verification.
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
        msg, action = await _next_step_async(messages, ollama_tools, model, tool_was_called, malformed_count)
        messages.append(msg)

        if action[0] == "done":
            break

        if action[0] == "nudge_no_tool":
            if malformed_count >= 2:
                return "⚠️ Model failed to produce a valid tool call. Try rephrasing.", tool_log
            messages.append({"role": "user",
                             "content": "You must call one of the available tools (lookup_codes, list_distinct_values, or execute_query) before answering. Do not respond from memory."})
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
        final = await _ollama_chat_async(
            model=model,
            messages=messages,
            options=LLM_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
        )
        answer = final["message"].get("content")

    if verify and answer and tool_was_called:
        loop = asyncio.get_event_loop()
        verify_result = await loop.run_in_executor(None, lambda: _verify_answer(messages, answer, model))
        if _needs_correction(verify_result):
            messages.append(_correction_user_message(verify_result))
            await _run_correction_pass_async(messages, ollama_tools, call_tool, model, tool_log)
            answer = _extract_last_assistant_answer(messages) or verify_result
        else:
            answer = verify_result or answer

    return answer or "⚠️ No answer returned by model.", tool_log
