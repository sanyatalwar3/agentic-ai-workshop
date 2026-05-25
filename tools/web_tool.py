import os
from pathlib import Path
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

mcp = FastMCP("web_search_server")

web_search = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


@mcp.tool()
def search_web(query: str) -> str:
    """
    Use this tool for:
    - current events
    - general web information
    - latest news
    - public information
    """

    results = web_search.invoke({"query": query})

    if not results:
        return "No results found."

    if isinstance(results, dict) and "results" in results:
        items = results["results"]
    else:
        items = results

    output = []

    for item in items:
        title = item.get("title", "")
        content = item.get("content", "")

        output.append(
            f"Title: {title}\nContent: {content}"
        )

    return "\n\n".join(output)


if __name__ == "__main__":
    mcp.run(transport="stdio")