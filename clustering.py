import numpy as np
import skfuzzy as fuzz

def fuzzy_cluster(embeddings, n_clusters=5):
    """
    Perform Fuzzy C-Means clustering.
    embeddings: list of embeddings
    returns: cluster labels, fuzzy membership matrix
    """
    data = np.array(embeddings).T  # scikit-fuzzy expects shape (features, samples)
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data,
        c=n_clusters,
        m=2,
        error=0.005,
        maxiter=1000
    )
    
    cluster_labels = np.argmax(u, axis=0)
    
    return cluster_labels, u