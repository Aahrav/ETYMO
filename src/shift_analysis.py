import numpy as np
import os
try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: gensim not available. Semantic shift analysis will be limited.")
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes

# Placeholder paths for embeddings
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings')

def load_embeddings(path):
    if not GENSIM_AVAILABLE:
        print("Gensim is not installed. Cannot load embeddings.")
        return None
    # Depending on format (word2vec, glove, etc.)
    # Here assuming gensim KeyedVectors format or similar
    if not os.path.exists(path):
        print(f"Embedding file not found: {path}")
        return None
    return KeyedVectors.load_word2vec_format(path, binary=False) # Change binary=True if needed

def align_embeddings(base_model, target_model):
    # Get common vocabulary
    # Note: Using KeyedVectors, vocab is in .key_to_index
    common_vocab = list(set(base_model.key_to_index) & set(target_model.key_to_index))
    
    if not common_vocab:
        return None, None
    
    # Sort for consistency
    common_vocab.sort()
    
    # Extract matrices
    base_vecs = np.array([base_model[w] for w in common_vocab])
    target_vecs = np.array([target_model[w] for w in common_vocab])
    
    # Procrustes Alignment: Find orthogonal matrix R that maps base to target
    # We want R such that base_vecs @ R ~ target_vecs
    # Scipy orthogonal_procrustes returns R, scale
    R, _ = orthogonal_procrustes(base_vecs, target_vecs)
    
    # Align base to target (or vice versa? Usually align to latest time period)
    # Let's align target to base (T2 aligned to T1 space)
    # Or align both to a common space using Compass method (more complex).
    # Standard: Align T2 to T1.
    # T2_aligned = T2 @ R
    # Wait, procrustes(A, B) minimizes ||A @ R - B||.
    # So if we want to map T1 to T2, we need R such that T1@R ~ T2.
    # Then aligned_T1 = T1 @ R.
    # But usually we align everything to the *latest* period.
    # So T_old aligned to T_new.
    
    aligned_vecs = base_vecs.dot(R)
    
    return common_vocab, base_vecs, target_vecs, aligned_vecs

def compute_drift(vocab, vec_t1, vec_t2):
    shifts = {}
    for i, word in enumerate(vocab):
        u = vec_t1[i]
        v = vec_t2[i]
        # Cosine distance = 1 - cosine_similarity
        dist = cosine(u, v)
        shifts[word] = dist
    return shifts

if __name__ == '__main__':
    print("This script provides functions for loading and aligning embeddings.")
    print("You need to download diachronic embeddings (e.g. HistWords) to 'data/embeddings'.")
