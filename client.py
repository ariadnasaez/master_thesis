# client.py
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama

async def main():
    # 1. Connect to the FastMCP server via standard input/output
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2. Fetch available tools from the MCP server
            mcp_tools = await session.list_tools()
            
            # Convert MCP tool schemas into Ollama's expected JSON function format
            ollama_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]
            
            # 3. Send prompt and tools to Llama 3.2
            messages = [{"role": "user", "content": "How many patients are there in the database? Use the mcp tool available"}]
            print("Thinking...")
            response = ollama.chat(model="llama3.2", messages=messages, tools=ollama_tools)
            
            # 4. Intercept tool calls and execute them against the MCP server
            if response.get("message", {}).get("tool_calls"):
                messages.append(response["message"])
                
                for tool_call in response["message"]["tool_calls"]:
                    print(f"Executing database query via tool: {tool_call['function']['name']}")
                    
                    # Execute the database query via MCP
                    result = await session.call_tool(
                        tool_call["function"]["name"], 
                        tool_call["function"]["arguments"]
                    )
                    
                    # Append the MySQL result to the chat history
                    messages.append({
                        "role": "tool",
                        "content": result.content[0].text,
                        "name": tool_call["function"]["name"]
                    })
                
                # 5. Generate the final natural language response using the database rows
                final_response = ollama.chat(model="llama3.2", messages=messages)
                print("\nFinal Answer:\n", final_response["message"]["content"])
            else:
                print("hola")
                # The model answered without needing the database
                print("\nFinal Answer:\n", response["message"]["content"])

if __name__ == "__main__":
    asyncio.run(main())
