import faiss
import numpy as np
from embeddings import get_model


def create_vector_store(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_similar_chunks(query, index, chunks, top_k=3):
    q = get_model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(q, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({**chunks[idx], "score": float(score)})
    return results