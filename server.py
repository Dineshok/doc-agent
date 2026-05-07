import fitz
from pinecone import Pinecone
from google import genai
from mcp.server.fastmcp import FastMCP

# MCP server + clients
mcp = FastMCP("document-agent")
genai_client = genai.Client(api_key="GOOGLE_API_KEY")
pc = Pinecone(api_key="PINECONE_API_KEY")
index = pc.Index("documents")

# Global state
document_loaded = False

# Helper functions
def get_embedding(text: str):
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def split_into_chunks(text: str, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Tool 1
@mcp.tool()
def load_document(path: str) -> str:
    """Load a PDF document for analysis. Must be called before ask_document,
    summarize_document or extract_entities. Takes the full file path as input."""
    
    global collection, document_loaded

    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    chunks = split_into_chunks(full_text)

    import time

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk))
        time.sleep(0.7)  # stay under 100 requests/minute

    try:
        index.delete(delete_all=True)
    except:
        pass  # index is empty, nothing to delete
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors)

    document_loaded = True
    return f"Document loaded successfully. {len(chunks)} chunks stored."




@mcp.tool()
def ask_document(question: str) -> str:
    """Answer a question using the loaded document.
    Use this when the user asks anything about the document content."""

    if not document_loaded:
        return "No document loaded. Please call load_document first."

    # Embed the question
    query_emb = get_embedding(question)

    # Find relevant chunks
    results = index.query(
        vector=query_emb,
        top_k=3,
        include_metadata=True
    )
    relevant_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(relevant_chunks)

    # Ask Gemini
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    response = genai_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text





if __name__ == "__main__":
    mcp.run()