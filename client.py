import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama
import json

async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch available tools
            mcp_tools = await session.list_tools()
            ollama_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            # Dynamically fetch the schema from the server
            schema_result = await session.call_tool("get_schema")
            schema_json = schema_result.content[0].text
            schema = json.dumps(json.loads(schema_json), indent=2)

            print("\n" + "="*50)
            print("  MySQL NATURAL LANGUAGE INTERFACE ACTIVE")
            print("  Database: tfm_datanex")
            print("  (Type 'exit' to quit)")
            print("="*50)

            while True:
                user_input = input("\nAsk about your patients: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Shutting down...")
                    break

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a MySQL expert for the 'tfm_datanex' database. "
                            "Use the 'execute_query' tool to answer questions.\n\n"

                            "TABLE SELECTION RULES:\n"
                            "- Use 'g_demographics' when the question is about patient attributes or population "
                            "(e.g., total patients, sex, nationality).\n"
                            "- Use 'g_administrations' when the question is about treatments, drugs, or administrations.\n\n"

                            "IMPORTANT:\n"
                            "- If the question is about patients who received a drug, use g_administrations.\n"
                            "- If the question is about general patient counts, use g_demographics.\n"
                            "- Always use COUNT(DISTINCT patient_ref) when counting patients.\n"
                            "- If the user provides a partial drug name, use LIKE with wildcards.\n"
                            "- Drug descriptions in the database may contain dosage and extra text.\n"
                            "- Always wrap string values in single quotes (e.g., 'Ibuprofen', 'Europa').\n\n"

                            f"Database schema:\n{schema}"
                        )
                    },
                    {"role": "user", "content": user_input}
                ]

                print("Thinking...")
                response = ollama.chat(model="llama3.2", messages=messages, tools=ollama_tools)

                if response.get("message", {}).get("tool_calls"):
                    messages.append(response["message"])

                    for tool_call in response["message"]["tool_calls"]:
                        sql_query = tool_call['function']['arguments'].get('query')
                        print(f"🔍 Generated SQL: {sql_query}")

                        # Execute query on MCP server
                        result = await session.call_tool(
                            tool_call["function"]["name"],
                            tool_call["function"]["arguments"]
                        )

                        messages.append({
                            "role": "tool",
                            "content": result.content[0].text,
                            "name": tool_call["function"]["name"]
                        })

                    final_response = ollama.chat(model="llama3.2", messages=messages)
                    print(f"\nAnswer: {final_response['message']['content']}")
                else:
                    print(f"\nAnswer: {response['message']['content']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass