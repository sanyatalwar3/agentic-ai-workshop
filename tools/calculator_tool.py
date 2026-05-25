from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator_server")


@mcp.tool()
def calculator(expression: str) -> str:
    """
    Use this tool for math calculations.
    Example:
    - 25 + 30
    - 100 / 5
    - (5 * 4) + 10
    """

    try:
        result = eval(expression)
        return f"The answer is {result}"
    except Exception:
        return "Invalid mathematical expression."


if __name__ == "__main__":
    mcp.run(transport="stdio")