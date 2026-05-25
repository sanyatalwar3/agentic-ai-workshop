import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from embed_client import get_embeddings
from vector_store import query_vectorstore

mcp = FastMCP("rag_server")


@mcp.tool()
def rag_search(question: str, session_id: str) -> str:
    """
    Use this tool when the user asks questions about uploaded documents.
    """

    query_embedding = get_embeddings(
        [question],
        input_type="query"
    )[0]

    docs = query_vectorstore(
        session_id,
        query_embedding,
        top_k=4,
    )

    return "\n\n".join(docs)


if __name__ == "__main__":
    mcp.run(transport="stdio")