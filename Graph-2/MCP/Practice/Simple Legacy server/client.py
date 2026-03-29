import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

async def main():       
        client = MultiServerMCPClient(
                {
                    "math": {
                        "command": "python",
                        "args": ["math_server.py"],
                        "transport": "stdio",
                    }}
            )
        tools =await client.get_tools()
        llm=ChatGroq(model="llama-3.1-8b-instant")
        agent = create_agent(llm, tools)
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(math_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())  