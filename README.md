# Document Intelligence Agent

An MCP server that lets Claude Desktop analyze any PDF using RAG. Load a document, ask questions — answers come from the document content, not Claude's general knowledge.

## Architecture
Claude Desktop → MCP Server → Gemini API (embeddings + LLM) → Pinecone (cloud vector storage)

## Tools
- `load_document(path)` — reads PDF, chunks it, embeds it, stores in Pinecone
- `ask_document(question)` — finds relevant chunks via vector search, answers using Gemini

## Stack
- PyMuPDF, Google Gemini API, Pinecone, MCP

## Setup

Install dependencies:
pip install -r requirements.txt

Set environment variables:
GOOGLE_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key

Add to Claude Desktop config (claude_desktop_config.json):
{
  "mcpServers": {
    "assistant": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "GOOGLE_API_KEY": "your_key",
        "PINECONE_API_KEY": "your_key"
      }
    }
  }
}

Restart Claude Desktop and start asking questions about any PDF.
