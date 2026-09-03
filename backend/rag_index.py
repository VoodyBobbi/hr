import json
import os
from typing import List, Tuple, Any

import faiss
import numpy as np


def write_index(index: faiss.IndexFlatL2, path: str) -> None:
    """Сохраняет FAISS-индекс на диск.

    Пишет НЕ через faiss.write_index(index, path) напрямую, а через
    сериализацию в память (faiss.serialize_index) + обычную запись байт
    (open(path, "wb")). Причина: faiss.write_index/read_index на Windows
    используют C++-уровневый fopen(const char*), который не поддерживает
    пути с не-ASCII символами (кириллица и т.п.) — падает с "could not open
    ... for writing: No such file or directory", даже если папка реально
    существует. Подтверждено в официальном баг-трекере faiss
    (github.com/facebookresearch/faiss/issues/3073), статус "wontfix" — на
    уровне библиотеки чинить не планируют, обход нужен на уровне вызывающего
    кода. serialize_index/deserialize_index — официальный API faiss для
    этого (см. faiss wiki "Index IO, cloning and hyper parameter tuning");
    запись самих байт через open() использует Unicode-путь-совместимый
    Python I/O и не подвержена этой проблеме."""
    chunk = faiss.serialize_index(index)
    with open(path, "wb") as f:
        f.write(chunk.tobytes())


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python -m backend.build_index` first to build the RAG index."
        )

    # Через open()+deserialize_index, а не faiss.read_index() напрямую —
    # см. подробное объяснение в write_index() выше (та же проблема Unicode-путей
    # на Windows затрагивает и чтение).
    with open(index_path, "rb") as f:
        index_bytes = f.read()
    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
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