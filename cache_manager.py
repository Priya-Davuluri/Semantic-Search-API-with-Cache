# cache_manager.py
import time
import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict, Any
import json

class CacheManager:
    def __init__(self, similarity_threshold=0.85):
        self.cache = OrderedDict()  # Using OrderedDict for potential LRU implementation
        self.similarity_threshold = similarity_threshold
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        
        # For semantic matching
        self.query_embeddings = {}  # Store embeddings of cached queries
        
    def _generate_key(self, query: str) -> str:
        """Generate a cache key from query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def find_semantic_match(self, query_embedding: np.ndarray) -> Optional[tuple]:
        """
        Find if a semantically similar query exists in cache
        Returns (matched_query, similarity_score, result) if found
        """
        if not self.query_embeddings:
            return None
        
        best_match = None
        best_similarity = -1
        
        for cached_query, cached_embedding in self.query_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = cached_query
        
        if best_match:
            return (best_match, float(best_similarity), self.cache[best_match])
        return None
    
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get result from cache with semantic matching"""
        self.total_requests += 1
        
        # Check for exact match first
        exact_key = self._generate_key(query)
        if exact_key in self.cache:
            self.hit_count += 1
            return {
                "cache_hit": True,
                "matched_query": query,
                "similarity_score": 1.0,
                "result": self.cache[exact_key]
            }
        
        # Check for semantic match
        semantic_match = self.find_semantic_match(query_embedding)
        if semantic_match:
            matched_query, similarity, result = semantic_match
            self.hit_count += 1
            return {
                "cache_hit": True,
                "matched_query": matched_query,
                "similarity_score": similarity,
                "result": result
            }
        
        # Miss
        self.miss_count += 1
        return None
    
    def set(self, query: str, query_embedding: np.ndarray, result: Any):
        """Store result in cache"""
        key = self._generate_key(query)
        self.cache[key] = result
        self.query_embeddings[query] = query_embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.hit_count / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": self.total_requests,
            "hit_rate": round(hit_rate, 3)
        }
    
    def clear(self):
        """Clear entire cache and reset stats"""
        self.cache.clear()
        self.query_embeddings.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0