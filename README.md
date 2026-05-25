# Agentic AI in Action: Exploring AI Agents & Tool Calling  
Organized by: ACM-W  

This repository contains the complete code used in the ACM-W workshop **“Agentic AI in Action: Exploring AI Agents & Tool Calling”**.  

In this workshop, students learn the basics of:
- AI Agents  
- Tool Calling  
- Retrieval-Augmented Generation (RAG)  
- MCP (Model Context Protocol)  
- LangChain & LangGraph  

Students also build a working AI assistant that can:
- answer questions from uploaded documents  
- access external web information  
- perform calculations using tools  

---

## Basics of Agentic AI

Traditional AI chatbots only generate answers from what they already know.

An Agentic AI system can:
- decide which tool to use  
- retrieve information dynamically  
- access external sources  
- perform actions/calculations  
- generate more accurate responses  

This makes the AI assistant:
- More intelligent  
- More flexible  
- More interactive  
- More accurate for real-world tasks  

---

## What This AI Assistant Does

- Allows users to upload TXT or PDF documents  
- Uses RAG to answer questions from uploaded files  
- Uses a Web Search Tool for external/live information  
- Uses a Calculator Tool for calculations  
- Uses MCP for tool communication 
- Uses LangGraph for workflow orchestration  
- Routes queries intelligently to the correct tool  
- Generates grounded responses  
- Maintains conversational interaction during the session  

---

## Tech Stack

- Python  
- Streamlit  
- LangChain  
- LangGraph  
- MCP  
- ChromaDB  
- NVIDIA NIM  
- Requests  
- PyPDF2  
- python-dotenv  

---

## How to Run the Project

**Clone the github repository**
```bash
git clone <PASTE YOUR GITHUB REPO LINK HERE>
```
**Windows
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
**Install all required python packages
```bash
pip install -r requirements.txt
```
**Create a file to store your api keys (.env) & add your api keys
```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx

TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```
**Run the streamlit application
```bash
streamlit run main.py
```

## Important Notes:
The vector database is created locally on your system
Users can upload multiple files
The AI assistant dynamically selects tools depending on the query
Web queries require internet access
Do NOT upload your .env file or API keys to GitHub
To reset everything, delete the chroma folder or use the clear option if available
