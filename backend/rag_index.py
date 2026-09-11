import json
import os
from typing import List, Tuple, Any

import faiss
import numpy as np


def write_atomic(path: str, write_fn) -> None:
    """Записывает файл атомарно: сначала во временный, потом переименование.

    os.replace на уровне файловой системы неделим — файл по пути path в
    любой момент либо старый целиком, либо новый целиком, промежуточного
    состояния не существует. Без этого при пересборке индекса возникало
    короткое окно, когда файл уже создан, но ещё не дописан: попавший в него
    запрос кандидата падал бы с технической ошибкой 500 вместо ответа."""
    tmp_path = path + ".tmp"
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def write_index(index: faiss.Index, path: str) -> None:
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

    def _write(target):
        with open(target, "wb") as f:
            f.write(chunk.tobytes())

    write_atomic(path, _write)


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.Index, np.ndarray]:
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
    index: faiss.Index,
    metadata: np.ndarray,
    query_vec: np.ndarray,
    k: int = 3,
    min_similarity: float = 0.45,
) -> List[Any]:
    """Ближайшие к запросу элементы, отсечённые по порогу близости.

    Индекс — faiss.IndexFlatIP (скалярное произведение). Все векторы
    нормализованы к единичной длине (normalize_embeddings=True в
    build_index.embed_texts и assistant.embed_text), а для единичных векторов
    скалярное произведение РАВНО косинусной близости. Поэтому index.search
    возвращает сразу косинусную близость в диапазоне от -1 до 1, где больше —
    похожее.

    Раньше использовался IndexFlatL2, который возвращает КВАДРАТ евклидова
    расстояния (меньше — похожее, диапазон 0..4). Порог в таких единицах
    невозможно осмыслить на глаз, и именно из-за этого прежнее значение 1.0
    было выставлено неверно. Косинусная близость читается напрямую: 0.45 —
    «примерно об одном», 0.75 — «почти то же самое».

    Реальные пороги задаёт вызывающий код из .env: FAQ_MIN_SIMILARITY
    (строже — готовый ответ уходит кандидату напрямую, без GigaChat) и
    KB_MIN_SIMILARITY (мягче — факты только подмешиваются в промпт).
    См. backend/assistant.py и README, раздел «Калибровка порогов поиска».

    Без порога поиск возвращал бы k ближайших даже при полностью
    нерелевантном запросе, подмешивая случайный контекст в промпт модели.
    """
    similarities, indices = index.search(query_vec, k)
    results = []
    for score, i in zip(similarities[0], indices[0]):
        if 0 <= i < len(metadata) and score >= min_similarity:
            results.append(metadata[i])
    return results


def load_faq_data(path: str):
    """Загружает FAQ данные из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)