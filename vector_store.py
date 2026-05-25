from pathlib import Path
import os
import uuid

import chromadb

DB_PATH = Path(__file__).resolve().parent / "data" / "chroma"

os.makedirs(DB_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=str(DB_PATH))


def get_collection(session_id: str):
    return client.get_or_create_collection(
        name=f"docs_{session_id}"
    )


def add_to_vectorstore(session_id, chunks, embeddings):
    collection = get_collection(session_id)

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )


def query_vectorstore(session_id, query_embedding, top_k=4):
    collection = get_collection(session_id)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results["documents"][0]


def clear_vectorstore(session_id: str) -> None:
    collection_name = f"docs_{session_id}"

    try:
        client.delete_collection(collection_name)
        print(f"Deleted vector store: {collection_name}")
    except Exception as e:
        print(f"Could not delete vector store: {e}")