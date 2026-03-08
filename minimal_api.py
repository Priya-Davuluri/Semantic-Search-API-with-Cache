# minimal_api.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups
import uvicorn
from threading import Thread
import time

# Create FastAPI app first (so /docs works immediately)
app = FastAPI(title="20 Newsgroups Semantic Search")

# Global variables
model = None
index = None
documents = None
is_ready = False

@app.get("/")
def root():
    return {
        "name": "20 Newsgroups Semantic Search",
        "status": "ready" if is_ready else "loading",
        "message": "System is loading the dataset. Please wait..."
    }

@app.get("/health")
def health():
    return {"status": "ready" if is_ready else "loading"}

@app.get("/docs")
def docs():
    return {"message": "Interactive docs available at /docs"}

@app.get("/search")
def search(q: str, top_k: int = 5):
    if not is_ready:
        return {
            "error": "System still loading",
            "message": "Please wait while the dataset loads (about 1-2 minutes)"
        }
    
    # Encode query
    query_vec = model.encode([q]).astype('float32')
    
    # Search
    distances, indices = index.search(query_vec, min(top_k, len(documents)))
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "preview": documents[idx][:200] + "...",
            "doc_id": int(idx)
        })
    
    return {"query": q, "results": results}

def load_data():
    global model, index, documents, is_ready
    
    print("Loading in background thread...")
    
    # Load model
    print("1. Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load just 100 documents for quick startup
    print("2. Loading dataset (small sample)...")
    dataset = fetch_20newsgroups(subset="all")
    documents = dataset.data[:100]  # Just 100 documents for testing
    print(f"   Loaded {len(documents)} documents")
    
    # Create embeddings
    print("3. Creating embeddings...")
    embeddings = model.encode(documents)
    
    # Build index
    print("4. Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    is_ready = True
    print("✅ System ready! Try /docs or /search?q=your query")

# Start loading in background
print("Starting background loading thread...")
thread = Thread(target=load_data)
thread.daemon = True
thread.start()

print("\n" + "="*50)
print("🚀 Server starting! Access:")
print("📚 Docs: http://localhost:8000/docs")
print("🔍 Search: http://localhost:8000/search?q=artificial intelligence")
print("❤️ Health: http://localhost:8000/health")
print("="*50 + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)