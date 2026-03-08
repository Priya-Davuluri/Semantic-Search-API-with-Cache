# save_embeddings.py
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

def save_all_embeddings():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading dataset...")
    dataset = fetch_20newsgroups(subset="all")
    documents = dataset.data
    
    print(f"Creating embeddings for {len(documents)} documents...")
    # Process in batches to avoid memory issues
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)//batch_size)+1}...")
        batch_embeddings = model.encode(batch)
        all_embeddings.append(batch_embeddings)
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings and documents
    print("Saving to disk...")
    np.save('newsgroups_embeddings.npy', embeddings)
    with open('newsgroups_documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    
    print("Done! Files saved:")
    print("  - newsgroups_embeddings.npy")
    print("  - newsgroups_documents.pkl")

if __name__ == "__main__":
    save_all_embeddings()