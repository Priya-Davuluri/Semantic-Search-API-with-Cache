# # embeddings.py
# from sentence_transformers import SentenceTransformer

# # Load the model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embeddings(texts):
#     """
#     Returns embeddings for a list of texts.
#     Output: list of numpy arrays
#     """
#     embeddings = model.encode(texts, convert_to_numpy=True)  # <-- important
#     return embeddings

# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    """
    Convert a list of texts into embeddings.
    Returns a list of np.array vectors.
    """
    # If model returns tensors/dicts, convert to numpy array
    embeddings = model.encode(texts, convert_to_numpy=True)
    # Ensure each embedding is a separate np.array
    return [np.array(vec, dtype="float32") for vec in embeddings]