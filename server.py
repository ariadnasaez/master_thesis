# server.py
from fastmcp import FastMCP
import mysql.connector
import json

# Initialize the FastMCP server
mcp = FastMCP("MySQL_Server")

@mcp.tool()
def get_patients() -> str:
    """Get distinct patients in the database."""
    query = '''SELECT COUNT(DISTINCT patient_ref)
FROM tfm_datanex.g_demographics LIMIT 10;'''
    # Safety check to prevent destructive operations
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are permitted."
        
    db_config = {
        "host": "127.0.0.1",
        "user": "root",
        "password": "",      # Replace with your MySQL 8.0 password
        "database": "tfm_datanex",      # Replace with your database name
        "port": 3306
    }
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Database Error: {str(e)}"

if __name__ == "__main__":
    # Start the server to listen for standard input/output (stdio) requests
    mcp.run()
