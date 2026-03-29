import asyncio
from urllib import response
from langchain_community import memory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,  START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages,AnyMessage
from typing import Annotated, TypedDict, List
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph():       
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
        llm_with_tools = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use tools when necessary."),
        MessagesPlaceholder("messages"),
        ])

        chain = prompt | llm_with_tools

        # nodes

        async def call_model(state: State) -> State:
            response = await chain.ainvoke({
            "messages": state["messages"]
            })
            return {"messages": [response]}
        
        tool_node = ToolNode(tools)

        # routing

        def should_continue(state: State):
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return END
        
        # Graph Workflow

        builder = StateGraph(State)

        builder.add_node("agent", call_model)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "agent")

        builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        builder.add_edge("tools", "agent")

        memory = MemorySaver()

        return builder.compile(checkpointer=memory)

async def main():
    graph = await create_graph()

    config = {"configurable": {"thread_id": "1"}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is 25 * 16?"}]},
        config=config
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())