import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ollama

async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Fetch tools once at the start
            mcp_tools = await session.list_tools()
            ollama_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in mcp_tools.tools]

            print("\n--- MySQL MCP Client Active (Type 'exit' or 'quit' to stop) ---")

            while True:
                # Get question from terminal
                user_input = input("\nAsk a question: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    break

                messages = [{"role": "user", "content": user_input}]
                print("Thinking...")
                
                response = ollama.chat(model="llama3.2", messages=messages, tools=ollama_tools)
                
                if response.get("message", {}).get("tool_calls"):
                    messages.append(response["message"])
                    
                    for tool_call in response["message"]["tool_calls"]:
                        print(f"Executing: {tool_call['function']['name']}")
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
                    print("\nAnswer:", final_response["message"]["content"])
                else:
                    print("\nAnswer:", response["message"]["content"])

if __name__ == "__main__":
    asyncio.run(main())