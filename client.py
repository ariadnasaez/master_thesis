import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama

async def main():
    # 1. Configuration for the FastMCP server
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # 2. Fetch available tools (execute_query)
            mcp_tools = await session.list_tools()
            ollama_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            print("\n" + "="*50)
            print("  MySQL NATURAL LANGUAGE INTERFACE ACTIVE")
            print("  Table: g_demographics | Database: tfm_datanex")
            print("  (Type 'exit' to quit)")
            print("="*50)

            while True:
                # 3. Interactive Terminal Input
                user_input = input("\nAsk about your patients: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Shutting down...")
                    break

                # 4. System Prompt with the Schema Knowledge
                messages = [
                    {
                        "role": "system", 
                        "content": (
                            "You are a MySQL expert for the 'tfm_datanex' database. "
                            "Use the 'execute_query' tool to answer questions about the 'g_demographics' table. "
                            "DATA MAPPING SCHEMA:\n"
                            "- sex: 1=Male, 2=Female, 3=Other, -1=Unknown\n"
                            "- table name: tfm_datanex.g_demographics\n"
                            "Return clear, conversational answers based on the query results."
                            "IMPORTANT: Always wrap string values in single quotes (e.g., 'Europa'). "
                        )
                    },
                    {"role": "user", "content": user_input}
                ]
                
                print("Thinking...")
                
                # 5. First LLM call to decide if a tool is needed
                response = ollama.chat(model="llama3.2", messages=messages, tools=ollama_tools)
                
                # 6. Check if the LLM wants to execute SQL
                if response.get("message", {}).get("tool_calls"):
                    messages.append(response["message"])
                    
                    for tool_call in response["message"]["tool_calls"]:
                        sql_query = tool_call['function']['arguments'].get('query')
                        print(f"🔍 Generated SQL: {sql_query}")
                        
                        # Execute against the MCP server
                        result = await session.call_tool(
                            tool_call["function"]["name"], 
                            tool_call["function"]["arguments"]
                        )
                        
                        # Feed the database results back to the LLM
                        messages.append({
                            "role": "tool",
                            "content": result.content[0].text,
                            "name": tool_call["function"]["name"]
                        })
                    
                    # 7. Final NL response generation
                    final_response = ollama.chat(model="llama3.2", messages=messages)
                    print(f"\nAnswer: {final_response['message']['content']}")
                else:
                    # If the model answered without needing the DB
                    print(f"\nAnswer: {response['message']['content']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass