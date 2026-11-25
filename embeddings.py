from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def create_embeddings(chunks, batch_size=16):
    texts = [c["text"] for c in chunks]
    emb = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype("float32")