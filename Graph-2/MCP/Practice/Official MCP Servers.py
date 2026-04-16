import asyncio
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.ERROR)

# ── Workspace ────────────────────────────────────────────────────────────────
WORKSPACE = os.path.abspath("./agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

# ── Server configs ────────────────────────────────────────────────────────────
server_configs = {
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "transport": "stdio",
    },
    "sequential-thinking": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE],
        "transport": "stdio",
    },
}

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# ── Graph factory ─────────────────────────────────────────────────────────────
def create_graph(tools: list):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant with three capabilities:\n\n"
         "1. Knowledge Graph Memory — persist and recall facts, entities, and\n"
         "   relationships across the conversation. Use this to remember user\n"
         "   preferences, important names, decisions, and any info worth keeping.\n"
         "   Key tools: create_entities, create_relations, add_observations,\n"
         "   search_nodes, read_graph, delete_entities.\n\n"
         "2. Sequential Thinking — break down complex problems into clear,\n"
         "   numbered reasoning steps before giving a final answer. Use this\n"
         "   whenever the task involves planning, analysis, debugging, or any\n"
         "   multi-step logic.\n"
         "   Key tool: sequentialthinking.\n\n"
         "3. File System — read and write files inside the agent workspace.\n"
         f"   WORKSPACE: {WORKSPACE}\n"
         "   Always use the FULL path when saving files, e.g.:\n"
         f"   '{WORKSPACE}/notes.txt'\n"
         "   NEVER write outside the workspace.\n\n"
         "Guidelines:\n"
         "- Proactively store important facts in the knowledge graph so you can\n"
         "  recall them later in the same or future sessions.\n"
         "- Use sequential thinking for any request that benefits from step-by-step\n"
         "  reasoning before answering.\n"
         "- Combine tools freely: think → remember → write to file, etc."),
        MessagesPlaceholder("messages"),
    ])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END,
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
        client = MultiServerMCPClient(server_configs)
        all_tools = await client.get_tools()

        print("Tools loaded:")
        for t in all_tools:
            print(f"  • {t.name}")

        agent = create_graph(all_tools)

        print("\nMCP Agent ready — memory, sequential thinking, and filesystem connected.")
        print(f"Workspace: {WORKSPACE}\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Bye!")
                break
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", user_input)]},
                    config={"configurable": {"thread_id": "main-session"}},
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())