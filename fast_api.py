# fast_api.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import uvicorn

# Load pre-saved data
print("Loading saved embeddings and documents...")
embeddings = np.load('newsgroups_embeddings.npy')
with open('newsgroups_documents.pkl', 'rb') as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} documents with embeddings shape {embeddings.shape}")

# Load model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Build FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
print(f"FAISS index ready with {index.ntotal} vectors")

# Create app
app = FastAPI(title="20 Newsgroups Semantic Search")

@app.get("/")
def root():
    return {
        "name": "20 Newsgroups Semantic Search",
        "total_documents": len(documents),
        "status": "ready"
    }

@app.get("/search")
def search(q: str, top_k: int = 5):
    # Encode query
    query_vec = model.encode([q]).astype('float32')
    
    # Search
    distances, indices = index.search(query_vec, top_k)
    
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)