# # main.py

# from fastapi import FastAPI
# from embeddings import get_embeddings, model
# from clustering import fuzzy_cluster
# from cache import search_with_cache
# import numpy as np
# import faiss
# from sklearn.datasets import fetch_20newsgroups

# # ------------------------------
# # Optional preprocessing function
# # ------------------------------
# def clean_text(text):
#     """Simple text cleaning: lowercase, strip whitespace."""
#     return text.lower().strip()

# # ------------------------------
# # FastAPI app
# # ------------------------------
# app = FastAPI(title="Semantic Search API")

# # ------------------------------
# # 1. Load 20 Newsgroups dataset
# # ------------------------------
# dataset = fetch_20newsgroups(subset="all")
# raw_documents = dataset.data

# # Optional preprocessing
# documents = [clean_text(doc) for doc in raw_documents]

# print("Total documents loaded:", len(documents))
# print("Sample document:", documents[0][:500])

# # ------------------------------
# # 2. Compute embeddings
# # ------------------------------
# embeddings = get_embeddings(documents)

# print("Number of embeddings:", len(embeddings))
# print("Shape of first embedding:", embeddings[0].shape)
# print("First embedding (first 5 values):", embeddings[0][:5])

# # ------------------------------
# # 3. Run fuzzy clustering
# # ------------------------------
# n_clusters = 3
# labels, membership = fuzzy_cluster(embeddings, n_clusters=n_clusters)

# print("\nCluster labels assigned to each document:")
# for i, label in enumerate(labels[:10]):  # first 10 for brevity
#     print(f"Doc{i+1} -> Cluster {label}")

# print("\nFuzzy membership probabilities (rows=clusters, columns=documents):")
# print(np.round(membership[:, :10], 2))  # first 10 columns for brevity

# # ------------------------------
# # 4. Build FAISS index for semantic search
# # ------------------------------
# embedding_matrix = np.array(embeddings).astype('float32')  # FAISS needs float32
# index = faiss.IndexFlatL2(embedding_matrix.shape[1])       # dimension = embedding size
# index.add(embedding_matrix)

# print("\nFAISS index built successfully.")

# # ------------------------------
# # 5. Cache dictionary for search
# # ------------------------------
# cache = {}

# # ------------------------------
# # 6. FastAPI endpoint
# # ------------------------------
# @app.get("/search")
# def search(query: str):
#     try:
#         results = search_with_cache(query, model, index, documents, cache)
#         return {"query": query, "results": results}
#     except Exception as e:
#         print("Error in /search endpoint:", e)
#         return {"error": str(e)}

# # ------------------------------
# # 7. Optional: test search when running script directly
# # ------------------------------
# if __name__ == "__main__":
#     test_query = "Artificial intelligence"
#     results = search_with_cache(test_query, model, index, documents, cache)  # ✅ cache added here
#     print(f"\nSearch results for query '{test_query}':")
#     for i, r in enumerate(results):
#         print(f"{i+1}. {r}")

# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups
import uvicorn
from typing import Optional, List, Dict, Any
from cache_manager import CacheManager
import time

# Initialize FastAPI app
app = FastAPI(title="Semantic Search API with Cache")

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Dict[str, Any]
    dominant_cluster: Optional[int] = None

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

# Global variables
model = None
index = None
documents = None
cluster_labels = None
cache_manager = None
is_ready = False

# ============================================================
# Initialize everything
# ============================================================
def initialize_system():
    global model, index, documents, cluster_labels, cache_manager, is_ready
    
    print("\n" + "="*60)
    print("Initializing Semantic Search System")
    print("="*60)
    
    # Load model
    print("\n1. Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ Model loaded")
    
    # Load dataset (using a subset for faster startup)
    print("\n2. Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset="all")
    documents = dataset.data[:500]  # Using 500 documents for demo
    print(f"   ✅ Loaded {len(documents)} documents")
    
    # Create embeddings
    print("\n3. Creating embeddings (this may take a moment)...")
    embeddings = model.encode(documents)
    print(f"   ✅ Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Build FAISS index
    print("\n4. Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f"   ✅ FAISS index built with {index.ntotal} vectors")
    
    # Simple clustering (using K-means approximation)
    print("\n5. Performing clustering...")
    n_clusters = 5
    # Use faiss for k-means clustering
    kmeans = faiss.Kmeans(embeddings.shape[1], n_clusters, niter=20)
    kmeans.train(embeddings.astype('float32'))
    _, cluster_labels = kmeans.index.search(embeddings.astype('float32'), 1)
    cluster_labels = cluster_labels.flatten()
    
    # Count documents per cluster
    cluster_counts = {i: int(np.sum(cluster_labels == i)) for i in range(n_clusters)}
    print(f"   ✅ Clustering complete: {cluster_counts}")
    
    # Initialize cache manager
    print("\n6. Initializing cache manager...")
    cache_manager = CacheManager(similarity_threshold=0.85)
    print("   ✅ Cache manager ready")
    
    is_ready = True
    print("\n" + "="*60)
    print("✅ SYSTEM READY!")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  POST /query    - Send a search query")
    print("  GET  /cache/stats - View cache statistics")
    print("  DELETE /cache  - Clear the cache")
    print("\nTry it with:")
    print('  curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d \'{"query": "computer graphics"}\'')
    print("="*60 + "\n")

# ============================================================
# FastAPI Endpoints
# ============================================================

# @app.get("/")
# def root():
#     return {
#         "name": "Semantic Search API with Cache",
#         "status": "ready" if is_ready else "initializing",
#         "endpoints": {
#             "POST /query": "Submit a search query",
#             "GET /cache/stats": "Get cache statistics",
#             "DELETE /cache": "Clear the cache"
#         }
#     }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Submit a natural language query and get semantically similar documents.
    
    Returns:
    - query: Original query
    - cache_hit: Whether result came from cache
    - matched_query: If cache hit, the similar query that matched
    - similarity_score: Similarity score with matched query (if cache hit)
    - result: The search results (top matching documents)
    - dominant_cluster: The cluster ID most relevant to this query
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="System still initializing. Please wait a moment.")
    
    query = request.query
    
    # Compute query embedding
    query_embedding = model.encode([query])[0]
    
    # Check cache first
    cached_result = cache_manager.get(query, query_embedding)
    
    if cached_result:
        # Cache hit
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=cached_result["matched_query"],
            similarity_score=cached_result["similarity_score"],
            result=cached_result["result"],
            dominant_cluster=cached_result["result"].get("dominant_cluster")
        )
    else:
        # Cache miss - compute results
        # Search FAISS index
        k = min(5, len(documents))
        distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        
        # Prepare results
        results_list = []
        cluster_votes = {}
        
        for i, idx in enumerate(indices[0]):
            doc_cluster = int(cluster_labels[idx])
            cluster_votes[doc_cluster] = cluster_votes.get(doc_cluster, 0) + 1
            
            results_list.append({
                "rank": i + 1,
                "doc_id": int(idx),
                "score": float(distances[0][i]),
                "cluster": doc_cluster,
                "preview": documents[idx][:200] + "..."
            })
        
        # Determine dominant cluster
        dominant_cluster = max(cluster_votes, key=cluster_votes.get) if cluster_votes else 0
        
        result_data = {
            "results": results_list,
            "dominant_cluster": dominant_cluster
        }
        
        # Store in cache
        cache_manager.set(query, query_embedding, result_data)
        
        return QueryResponse(
            query=query,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result_data,
            dominant_cluster=dominant_cluster
        )

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Get current cache statistics"""
    if not is_ready:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    return cache_manager.get_stats()

@app.delete("/cache")
async def clear_cache():
    """Clear the entire cache and reset statistics"""
    if not is_ready:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    cache_manager.clear()
    return {"message": "Cache cleared successfully", "status": "cache empty"}

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    import threading
    thread = threading.Thread(target=initialize_system)
    thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)