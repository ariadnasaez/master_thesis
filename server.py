# MCP server: connects to the MySQL database and exposes it to the client.
# Provides a schema resource (fetched once at startup) and an execute_query tool (called per question).
from fastmcp import FastMCP
import mysql.connector
import json

mcp = FastMCP("MySQL_Server")

db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tfm_datanex",
    "port": 3306
}

# ----------------------------
# Domain descriptions for tables that need clarification.
# Add or edit entries here to guide the LLM toward the right table.
# ----------------------------
TABLE_DESCRIPTIONS = {
    "g_demographics": "Primary patient table. One row per patient. Use this to count patients or get demographic info. sex: 1=male/hombre, 2=female/mujer. natio_descr sample values: Espana, Europa, América, África, Marruecos, Pakistan, Andorra, Resto del mundo. birth_date is stored as TEXT in 'YYYY-MM-DD' format — use TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) to calculate age.",
    "g_administrations": "Drug administration records. One row per drug administration event. Multiple rows per patient.",
    "g_health_issues": "Patient diagnoses and health conditions. Use this for any question about diseases, conditions, or diagnoses. Column snomed_descr contains the condition name in Spanish. snomed_ref is the numeric SNOMED code — use it for reliable filtering when known. Always SELECT and GROUP BY snomed_descr to include condition names in results. ou_med_ref is the medical unit that RECORDED the diagnosis — this is NOT necessarily the unit that treats the patient. Only filter ou_med_ref here when the question specifically asks 'diagnosed by' a specialty. For 'attended by' or 'seen by' a specialty, filter ou_med_ref on g_labs or g_episodes instead. episode_ref is stored as double — join with g_labs, g_procedures and g_episodes on patient_ref only (episode_ref type incompatible with those tables).",
    "g_labs": "Lab test results. Use this for any question about lab values or analytical results (e.g. PCR, glucose, hemoglobin). result_num is the numeric result. lab_descr is the test name in Spanish (e.g. 'Hemoglobina glicada' for HbA1c, 'Glucosa' for glucose). lab_sap_ref is the stable lab code — always do a lookup first (SELECT DISTINCT lab_sap_ref, lab_descr FROM g_labs WHERE lab_descr LIKE '%keyword%') and then filter by lab_sap_ref IN (...) in the main query. extrac_date is when the blood sample was drawn; result_date is when the result was reported — use extrac_date when comparing timing to procedures. ou_med_ref is the unit that ORDERED the lab — filter here when the question asks about patients 'attended by' or 'seen by' a specialty. Join with g_health_issues on patient_ref only (episode_ref types are incompatible). Join with g_procedures and g_episodes on BOTH patient_ref AND episode_ref to stay within the same episode.",
    "g_micro": "Microbiology results (cultures, microorganisms). Use this for questions about infections or microorganisms, NOT for lab values like PCR.",
    "g_procedures": "Clinical procedures performed on patients. Use this for ANY question about procedures or interventions — prefer this over g_surgery. Column descr contains the procedure name in Spanish (plain ASCII, no accents). Column code contains procedure codes. ALWAYS look up codes first using BOTH the procedure type keyword AND the anatomy keyword: SELECT DISTINCT code, descr FROM g_procedures WHERE descr LIKE '%procedure_type%' AND descr LIKE '%anatomy%'. Then use ALL returned codes in code IN (...) in the main query. Join with g_labs and g_episodes on BOTH patient_ref AND episode_ref — NEVER join on patient_ref alone, as that cross-matches across episodes.",
    "g_surgery": "Surgical planning and scheduling data. Use only for questions specifically about surgical scheduling, waiting lists, or surgical teams — NOT for querying what procedures were performed (use g_procedures for that).",
    "g_episodes": "Hospital episodes/admissions. episode_type_ref='HOSP' means hospitalization (ingreso hospitalario) — ALWAYS filter episode_type_ref='HOSP' when the question asks about ingresos/hospitalizations. start_date is the admission date. Can join with g_procedures and g_labs on both patient_ref AND episode_ref.",
    "g_movements": "Patient movements between hospital units/locations. ou_med_ref and ou_med_descr map medical unit codes to names (e.g. ou_med_ref='CAR' → ou_med_descr='CARDIOLOGIA'). ou_loc_ref and ou_loc_descr identify the physical location/ward (e.g. ou_loc_ref='I064' → a specific ward). Use this table to find patients who were physically in a specific unit or ward. JOIN on patient_ref and episode_ref. To find a location code, query: SELECT DISTINCT ou_loc_ref, ou_loc_descr FROM g_movements WHERE ou_loc_descr LIKE '%keyword%'.",
    "g_diagnostics": "ICD-coded diagnoses (separate from g_health_issues which uses SNOMED). Column code contains ICD-9/ICD-10 codes (e.g. 'I63.9' for cerebral infarction). diag_descr is the diagnosis name. Use LIKE 'code%' to match all subcodes in a family (e.g. code LIKE 'I63%' for all cerebral infarction subtypes). Prefer g_diagnostics with ICD code ranges when the question mentions ICD codes or when broader diagnostic coverage is needed. episode_ref is bigint — can join with g_episodes, g_procedures, g_labs on both patient_ref AND episode_ref.",
}

# ----------------------------
# RESOURCE: Full live schema
# ----------------------------
@mcp.resource("schema://full")
def get_full_schema() -> str:
    """Return all tables with their columns, types, and descriptions, fetched live from the database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]

        lines = []
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`;")
            columns = cursor.fetchall()
            cols_str = ", ".join(f"{col[0]} {col[1]}" for col in columns)
            desc = TABLE_DESCRIPTIONS.get(table, "")
            prefix = f"  # {desc}" if desc else ""
            lines.append(f"{table}({cols_str}){prefix}")

        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# TOOL: Execute SQL SELECT query
# ----------------------------
@mcp.tool()
def execute_query(query: str) -> str:
    """Execute SELECT query and return results in JSON."""
    first_word = query.strip().split()[0].upper() if query.strip() else ""
    if first_word not in ("SELECT", "WITH"):
        return "Error: Only SELECT queries allowed."

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query)
        column_names = [desc[0] for desc in cursor.description]

        results = [{column_names[i]: row[i] for i in range(len(row))} for row in cursor.fetchall()]
        conn.close()
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Database Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()