import os
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectors.npz")


def _cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denom = vector_norms * query_norm
    dots = vectors @ query
    return np.divide(dots, denom, where=denom > 0, out=np.zeros_like(dots))


def store_in_pinecone(chunks: List[str], embeddings: List[List[float]], namespace: str = ""):
    path = Path(VECTOR_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.array(embeddings, dtype=float)
    texts = np.array(chunks, dtype=object)
    chunk_indices = np.arange(len(chunks))

    np.savez(path, vectors=vectors, texts=texts, chunk_indices=chunk_indices)
    print(f"Stored {len(chunks)} vectors to {path.resolve()}")


def search_in_pinecone(query_vector: List[float], top_k: int = 4, namespace: str = ""):
    path = Path(VECTOR_STORE_PATH)
    if not path.exists():
        print(f"No vector store found at {path.resolve()}.")
        return []

    data = np.load(path, allow_pickle=True)
    vectors = data["vectors"]
    texts = data["texts"]

    query = np.array(query_vector, dtype=float)
    scores = _cosine_similarity(query, vectors)
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    matched = [str(texts[i]) for i in ranked_idx]

    print(f"Found {len(matched)} matches in local store.")
    return matched