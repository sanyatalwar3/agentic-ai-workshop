import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = Path(__file__).resolve().parent.parent

MCP_SERVER_CONFIG = {
    "rag": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(BASE_DIR / "tools" / "rag_tool.py")],
    },
    "web": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(BASE_DIR / "tools" / "web_tool.py")],
    },
    "calculator": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(BASE_DIR / "tools" / "calculator_tool.py")],
    },
}


async def get_mcp_tools():
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)

    tools = await client.get_tools()

    return tools