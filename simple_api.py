# simple_api.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups
import uvicorn

# Load model once
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load dataset
print("Loading 20 Newsgroups dataset...")
dataset = fetch_20newsgroups(subset="all")
documents = dataset.data
print(f"Loaded {len(documents)} documents")

# Create embeddings (you might want to save these to disk for faster loading)
print("Creating embeddings (this will take a few minutes)...")
embeddings = model.encode(documents[:1000])  # Start with 1000 for testing
print(f"Created {len(embeddings)} embeddings")

# Build FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
print(f"FAISS index built with {index.ntotal} vectors")

# Create FastAPI app
app = FastAPI(title="20 Newsgroups Semantic Search")

@app.get("/")
def root():
    return {
        "message": "20 Newsgroups Semantic Search API",
        "total_documents": len(documents[:1000]),
        "endpoints": {
            "/search?q=your query": "Search for similar documents"
        }
    }

@app.get("/search")
def search(q: str, top_k: int = 5):
    # Encode query
    query_embedding = model.encode([q])
    
    # Search
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Format results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "document_preview": documents[idx][:300] + "...",
            "similarity_score": float(1 / (1 + distances[0][i])),  # Convert distance to similarity
            "doc_id": int(idx)
        })
    
    return {
        "query": q,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)