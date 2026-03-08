# cache.py
import numpy as np
from embeddings import get_embeddings

def search_with_cache(query, model, index, documents, cache, top_k=5):
    """
    Perform semantic search with caching.

    Parameters:
    - query: str, the search query
    - model: SentenceTransformer model
    - index: FAISS index
    - documents: list of document strings
    - cache: dict, stores previous search results
    - top_k: int, number of top results to return (default: 5)

    Returns:
    - list of top matching documents
    """
    cache_key = f"{query}__top{top_k}"

    # Return cached results if available
    if cache_key in cache:
        print("Cache hit")
        return cache[cache_key]

    try:
        # Compute embedding for the query
        query_embedding = get_embeddings([query])

        # Convert to NumPy array safely
        if isinstance(query_embedding, list):
            query_vector = np.array(query_embedding[0], dtype="float32").reshape(1, -1)
        elif isinstance(query_embedding, dict):
            query_vector = np.array(list(query_embedding.values()), dtype="float32").reshape(1, -1)
        else:
            query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)

        # Search FAISS index
        k = min(top_k, len(documents))
        D, I = index.search(query_vector, k=k)

        # Retrieve matching documents with their index and distance
        results = [
            {
                "doc_id": int(I[0][rank]),
                "score": float(D[0][rank]),
                "text": documents[I[0][rank]][:300]  # preview first 300 chars
            }
            for rank in range(len(I[0]))
        ]

        # Save to cache
        cache[cache_key] = results

        return results

    except Exception as e:
        print("Error in search_with_cache:", e)
        return [{"error": str(e)}]