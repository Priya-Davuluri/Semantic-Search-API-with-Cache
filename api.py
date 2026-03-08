# api.py

from fastapi import FastAPI
from embeddings import get_embeddings, model
from clustering import fuzzy_cluster
from cache import search_with_cache
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups

app = FastAPI(title="Semantic Search API")

def clean_text(text):
    return text.lower().strip()

# ============================================================
# STEP 1: Dataset loading
# ============================================================
print("=" * 60)
print("[STEP 1] Loading 20 Newsgroups dataset...")
print("=" * 60)

dataset = fetch_20newsgroups(subset="all")
raw_documents = dataset.data
documents = [clean_text(doc) for doc in raw_documents]

print(f"[DEBUG] ✅ Total documents loaded      : {len(documents)}")
print(f"[DEBUG] ✅ Categories ({len(dataset.target_names)})          : {dataset.target_names}")
print(f"[DEBUG] ✅ Sample doc[0] preview       :\n{documents[0][:200]}\n")

# ============================================================
# STEP 2: Embeddings
# ============================================================
print("=" * 60)
print("[STEP 2] Computing embeddings (may take a few minutes)...")
print("=" * 60)

embeddings = get_embeddings(documents)

print(f"[DEBUG] ✅ Total embeddings computed    : {len(embeddings)}")
print(f"[DEBUG] ✅ Single embedding shape       : {embeddings[0].shape}")
print(f"[DEBUG] ✅ First 5 values of embedding  : {embeddings[0][:5]}")

# ============================================================
# STEP 3: FAISS index
# ============================================================
print("=" * 60)
print("[STEP 3] Building FAISS index...")
print("=" * 60)

embedding_matrix = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

print(f"[DEBUG] ✅ Embedding matrix shape       : {embedding_matrix.shape}  (docs x dims)")
print(f"[DEBUG] ✅ Total vectors in FAISS index : {index.ntotal}")

# ============================================================
# STEP 4: Fuzzy clustering
# ============================================================
print("=" * 60)
print("[STEP 4] Running fuzzy clustering...")
print("=" * 60)

n_clusters = 5
labels, membership = fuzzy_cluster(embeddings, n_clusters=n_clusters)

print(f"[DEBUG] ✅ Clusters created             : {n_clusters}")
print(f"[DEBUG] ✅ Docs per cluster             : { {i: int(np.sum(labels == i)) for i in range(n_clusters)} }")

# ============================================================
cache = {}

print("\n" + "=" * 60)
print(f"✅ API READY — serving {len(documents)} real documents from 20 Newsgroups")
print("=" * 60 + "\n")

# ============================================================
# SEARCH ENDPOINT
# ============================================================
@app.get("/search")
def search(query: str, top_k: int = 5):
    """
    Search the 20 Newsgroups dataset semantically.
    - **query**: the search string
    - **top_k**: number of results to return (default 5)
    """
    print(f"\n[REQUEST] 🔍 Query='{query}' | top_k={top_k} | cache_size={len(cache)}")

    results = search_with_cache(query, model, index, documents, cache, top_k=top_k)

    print(f"[REQUEST] ✅ {len(results)} results returned")
    for i, r in enumerate(results):
        print(f"  [{i+1}] doc_id={r.get('doc_id')} | score={r.get('score', 0):.4f} | {r.get('text','')[:80]}...")

    return {
        "query": query,
        "total_documents_searched": len(documents),
        "results": results
    }