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
    "g_demographics": "Primary patient table. One row per patient. Use this to count patients or get demographic info. sex: 1=male/hombre, 2=female/mujer. natio_descr sample values: Espana, Europa, América, África, Marruecos, Pakistan, Andorra, Resto del mundo.",
    "g_administrations": "Drug administration records. One row per drug administration event. Multiple rows per patient.",
    "g_health_issues": "Patient diagnoses and health conditions. Use this for any question about diseases, conditions, or diagnoses. Column snomed_descr contains the condition name in Spanish. Always GROUP BY snomed_descr, never by snomed_ref. ou_med_ref is the medical unit that recorded the diagnosis (e.g. RMT=rheumatology, HEM=hematology, PSI=psychiatry, CAR=cardiology, NRL=neurology). Join with g_labs on patient_ref only (episode_ref types are incompatible).",
    "g_labs": "Lab test results. Use this for any question about lab values or analytical results (e.g. PCR, glucose, hemoglobin). result_num is the numeric result. lab_descr is the test name — always filter with LIKE 'pcr%' not exact match. ou_med_ref is the requesting unit but filter on g_health_issues.ou_med_ref to scope by medical specialty. Join with g_health_issues on patient_ref only (episode_ref types are incompatible).",
    "g_micro": "Microbiology results (cultures, microorganisms). Use this for questions about infections or microorganisms, NOT for lab values like PCR.",
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
# TOOL 1: Dynamic table descriptions
# ----------------------------
@mcp.tool()
def get_table_descriptions() -> str:
    """Return all tables and optional comments dynamically."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]

        descriptions = {}
        for table in tables:
            # Try to get table comment
            cursor.execute(f"SHOW TABLE STATUS LIKE '{table}';")
            status = cursor.fetchone()
            comment = status[17] if status and len(status) > 17 else ""
            descriptions[table] = comment or "No description available"

        conn.close()
        return json.dumps(descriptions)

    except Exception as e:
        return f"Database Error: {str(e)}"

# ----------------------------
# TOOL 2: Schema for selected tables
# ----------------------------
@mcp.tool()
def get_schema_for_tables(tables: list[str]) -> str:
    """Retrieve schema only for selected tables."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        schema = {}
        for table in tables:
            cursor.execute(f"DESCRIBE {table};")
            columns = cursor.fetchall()
            schema[table] = [{"column_name": col[0], "data_type": col[1]} for col in columns]
        conn.close()
        return json.dumps(schema, indent=2)
    except Exception as e:
        return f"Database Error: {str(e)}"

# ----------------------------
# TOOL 3: Execute SQL SELECT query
# ----------------------------
@mcp.tool()
def execute_query(query: str) -> str:
    """Execute SELECT query and return results in JSON."""
    if not query.strip().upper().startswith("SELECT"):
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