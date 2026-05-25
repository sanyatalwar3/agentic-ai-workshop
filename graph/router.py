import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import AgentState
from tools.mcp_tools import get_mcp_tools

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2,
    max_tokens=1000,
)

SYSTEM_PROMPT = """
You are a helpful AI agent.

You have access to 3 tools:
1. rag_search -> for uploaded document questions
2. search_web -> for current/general web information
3. calculator -> for math calculations

Rules:
- For greetings/casual chat → answer normally
- For document questions → use rag_search
- For current events/general info → use search_web
- For calculations/math → use calculator
- Never mention tool names in the final answer
"""


async def build_graph():
    tools = await get_mcp_tools()

    model_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState):
        messages = state["messages"]

        response = await model_with_tools.ainvoke(messages)

        return {"messages": [response]}

    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")

    builder.add_conditional_edges("agent", tools_condition)

    builder.add_edge("tools", "agent")

    return builder.compile()