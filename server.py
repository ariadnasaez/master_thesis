# server.py
from fastmcp import FastMCP
import mysql.connector
import json

# Initialize MCP server
mcp = FastMCP("MySQL_Server")

# Database configuration
db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tfm_datanex",
    "port": 3306
}

# ----------------------------
# TOOL 1: Get all tables
# ----------------------------
@mcp.tool()
def get_tables() -> str:
    """Retrieve all tables in the database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return json.dumps(tables)
    except Exception as e:
        return f"Database Error: {str(e)}"


# ----------------------------
# TOOL 2: Get columns of a table
# ----------------------------
@mcp.tool()
def get_columns(table_name: str) -> str:
    """Retrieve column names and types for a specific table."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(f"DESCRIBE {table_name};")
        columns = cursor.fetchall()
        conn.close()

        formatted_columns = [{"column_name": col[0], "data_type": col[1]} for col in columns]
        return json.dumps(formatted_columns)
    except Exception as e:
        return f"Database Error: {str(e)}"


# ----------------------------
# TOOL 3: Get full schema
# ----------------------------
@mcp.tool()
def get_schema() -> str:
    """Retrieve the full database schema (tables + columns)."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]

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
# TOOL 4: Execute query
# ----------------------------
@mcp.tool()
def execute_query(query: str) -> str:
    """Execute a SQL SELECT query dynamically, returning results in JSON."""
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are permitted."

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query)

        # Extract column names dynamically
        column_names = [desc[0] for desc in cursor.description]

        # Build JSON-friendly result
        results = []
        for row in cursor.fetchall():
            results.append({column_names[i]: row[i] for i in range(len(row))})

        conn.close()
        return json.dumps(results, default=str)

    except Exception as e:
        return f"Database Error: {str(e)}"


# ----------------------------
# Run server (stdio mode)
# ----------------------------
if __name__ == "__main__":
    mcp.run()