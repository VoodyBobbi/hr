import json
import os
from typing import List, Tuple, Any

import faiss
import numpy as np


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python -m backend.build_index` first to build the RAG index."
        )

    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return index, metadata


def search_similar(
    index: faiss.IndexFlatL2,
    metadata: np.ndarray,
    query_vec: np.ndarray,
    k: int = 3,
    max_distance: float = 1.0,
) -> List[Any]:
    """
    max_distance — порог L2-расстояния для эмбеддингов paraphrase-multilingual-MiniLM-L12-v2
    (нормализованные векторы, L2 в диапазоне ~0..2). Значение 1.0 — стартовая отсечка,
    НЕ откалибрована на реальных данных проекта — нужно проверить на живых вопросах
    ("привет", "кто ты?", односложные ответы) и подобрать точнее по логам.
    Без порога поиск всегда возвращал top_k ближайших даже при нерелевантном запросе,
    что подмешивало случайный FAQ-контекст в промпт модели.
    """
    distances, indices = index.search(query_vec, k)
    results = []
    for dist, i in zip(distances[0], indices[0]):
        if 0 <= i < len(metadata) and dist <= max_distance:
            results.append(metadata[i])
    return results


def load_faq_data(path: str):
    """Загружает FAQ данные из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)