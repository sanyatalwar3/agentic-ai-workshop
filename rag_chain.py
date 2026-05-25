from embed_client import get_embeddings
from vector_store import query_vectorstore


def get_retrieved_context(session_id: str, query: str, top_k: int = 4) -> str:
    query_embedding = get_embeddings(
        [query],
        input_type="query"
    )[0]

    docs = query_vectorstore(
        session_id,
        query_embedding,
        top_k=top_k,
    )

    if not docs:
        return "No relevant content was found in the uploaded document."

    return "\n\n".join(docs)


def build_rag_prompt(session_id: str, query: str) -> str:
    context = get_retrieved_context(session_id, query, top_k=4)

    return f"""
You are a friendly AI assistant answering questions using the uploaded document.

Formatting instructions:
- Follow the user's requested format strictly.
- If the user asks for detailed points, bullet points, list, headings, or step-by-step explanation, ALWAYS format the response that way.
- Separate the answer into clear categories/topics whenever possible.
- Never return one large paragraph when the user asks for detailed explanation.
- Keep the answer clean, readable, and natural.

Answering rules:
- Use the uploaded document context as the main source.
- Do not make up facts that are not supported by the context.
- If the answer is not found in the document, clearly say it is not mentioned in the uploaded document.
- Do not mention embeddings, vector database, chunks, or retrieval unless the user asks technically.

Uploaded document context:
{context}

User question:
{query}

Final answer:
""".strip()