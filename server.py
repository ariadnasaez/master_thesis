# server.py
from fastmcp import FastMCP
import mysql.connector
import json

# Initialize the FastMCP server
mcp = FastMCP("MySQL_Server")

@mcp.tool()
def execute_query(query: str) -> str:
    """
    Execute a MySQL SELECT query against the tfm_datanex database.
    
    Table: g_demographics
    Columns:
    - patient_ref (INT, PK): Unique identifier
    - birth_date (DATE): Date of birth
    - sex (INT): -1=Unknown, 1=Male, 2=Female, 3=Other
    - natio_ref (CHAR): Nationality reference code
    - natio_descr (CHAR): Country description
    - health_area (CHAR): Health area
    - postcode (CHAR): Postal code
    - load_date (DATETIME): Update timestamp
    
    Example: To find female patients, use 'WHERE sex = 2'.
    Only SELECT queries are permitted.
    """
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are permitted."
        
    db_config = {
        "host": "127.0.0.1",
        "user": "root",
        "password": "", 
        "database": "tfm_datanex",
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