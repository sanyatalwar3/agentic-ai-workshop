import io
import time
import asyncio
import uuid

import streamlit as st
import PyPDF2

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from graph.router import build_graph
from utils import chunk_text
from embed_client import get_embeddings
from vector_store import add_to_vectorstore, clear_vectorstore


# -----------------------------
# Load graph once
# -----------------------------
@st.cache_resource
def load_graph():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(build_graph())


graph = load_graph()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("NVIDIA Agentic Chat")
st.caption("Powered by Streamlit, LangGraph, MCP, and NVIDIA NIM")


# -----------------------------
# Session ID
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

SESSION_ID = st.session_state.session_id
st.sidebar.caption(f"Chat SID: {SESSION_ID[:8]}...")


# -----------------------------
# Track indexed files
# -----------------------------
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


# -----------------------------
# Sidebar upload
# -----------------------------
st.sidebar.title("Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Choose one or more files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_files_added = False

    for uploaded_file in uploaded_files:
        file_key = f"{SESSION_ID}_{uploaded_file.name}_{uploaded_file.size}"

        if file_key in st.session_state.indexed_files:
            continue

        filename = uploaded_file.name.lower()

        if filename.endswith(".txt"):
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            raw_text = ""

            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"

        else:
            continue

        if raw_text.strip():
            chunks = chunk_text(raw_text)
            embeddings = get_embeddings(chunks, input_type="passage")

            add_to_vectorstore(
                SESSION_ID,
                chunks,
                embeddings,
            )

            st.session_state.indexed_files.add(file_key)
            new_files_added = True

    if new_files_added:
        st.sidebar.success("New files uploaded and added to vector store.")


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
query = st.chat_input("Ask about your docs, the web, or calculations...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    system_prompt = f"""
You are a friendly conversational AI assistant.

You should respond naturally like ChatGPT.
Do not sound robotic or overly formal.
For summaries, explain clearly in simple language.

You have access to three MCP tools:
1. rag_search -> for uploaded document questions
2. search_web -> for current/general web information
3. calculator -> for math calculations

Rules:
- If the user is greeting or chatting casually, do not call any tool. Reply naturally.
- Use rag_search when the user asks about uploaded files, PDFs, notes, or document content.
- Use search_web when the user asks for current, external, or general web information.
- Use calculator when the user asks for math calculations.
- When calling rag_search, always pass this exact session_id: {SESSION_ID}
- When summarizing documents, give concise human-style summaries.
- Avoid generic phrases like "the system can analyze".
- Write directly about the document content.
- After using a tool, give a clear final answer.
- Never show tool call JSON to the user.
""".strip()

    messages = [SystemMessage(content=system_prompt)]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    start = time.time()

    with st.spinner("Agent is thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            graph.ainvoke({"messages": messages})
        )

        loop.close()

    end = time.time()
    st.write(f"_Agent response time: {end - start:.2f} seconds_")

    final_answer = response["messages"][-1].content

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": final_answer}
    )


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.divider()

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.indexed_files = set()

    clear_vectorstore(SESSION_ID)

    st.sidebar.success("Chat and uploaded documents cleared.")
    st.rerun()