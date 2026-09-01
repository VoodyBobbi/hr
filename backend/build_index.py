import hashlib
import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_index import load_faq_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "faqs.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
HASH_PATH = os.path.join(DATA_DIR, "data_hash.txt")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Файлы в data/, которые НЕ являются источником вопросов-ответов для этого
# индекса и поэтому исключены из отслеживания ниже:
#   - faiss_index.bin, faqs_metadata.npy, data_hash.txt, faqs_hash.txt —
#     сами артефакты сборки. Если их не исключить, пересборка меняла бы их
#     же, и при следующем запуске хеш "изменился" бы сам по себе — вечная
#     пересборка на каждом старте.
#   - system_prompt.md, knowledge_base.json — у них УЖЕ ЕСТЬ отдельный
#     механизм отслеживания: assistant.py:_ensure_fresh() подхватывает их
#     на лету по mtime, БЕЗ пересборки векторного индекса (см. README,
#     раздел "Обновление базы знаний"). Если включить их сюда тоже, каждое
#     изменение системного промпта запускало бы дорогую пересборку
#     эмбеддингов, хотя в этом нет необходимости — эти два файла не
#     используются для векторного поиска, только как текст, который
#     целиком идёт в системный промпт.
_IGNORED_NAMES = {
    "faiss_index.bin",
    "faqs_metadata.npy",
    "data_hash.txt",
    "faqs_hash.txt",
    "system_prompt.md",
    "knowledge_base.json",
}

# Расширения, которые понимаются как "сырой текстовый документ с
# инструкциями" — весь файл целиком идёт в базу как один Q&A-элемент (см.
# _load_text_documents). faqs.json обрабатывается отдельно, по-другому (см.
# main) — у него есть структура question/answer, которую нет смысла терять.
_TEXT_EXTENSIONS = {".txt", ".md"}

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.astype("float32")


def _iter_watched_files():
    """Все файлы в data/, которые реально участвуют в сборке индекса —
    faqs.json плюс любые новые текстовые файлы с инструкциями, если их
    туда положили (в любом порядке, с любым именем). Отсортировано по
    имени файла — чтобы порядок обхода (и, соответственно, порядок
    элементов в итоговом индексе) был одинаковым от запуска к запуску,
    а не зависел от порядка выдачи файлов файловой системой."""
    if not os.path.isdir(DATA_DIR):
        return []
    names = sorted(os.listdir(DATA_DIR))
    paths = []
    for name in names:
        if name in _IGNORED_NAMES:
            continue
        full_path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(full_path):
            continue
        paths.append(full_path)
    return paths


def _combined_hash(file_paths: List[str]) -> str:
    """Один хеш по содержимому ВСЕХ отслеживаемых файлов сразу, а не по
    одному faqs.json — так пересборка триггерится и при изменении
    существующего файла, и при добавлении нового, и при удалении старого
    (список путей тоже участвует в хеше, не только содержимое)."""
    digest = hashlib.sha256()
    for path in file_paths:
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def _load_saved_hash() -> str:
    if not os.path.exists(HASH_PATH):
        return ""
    with open(HASH_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def _save_hash(value: str):
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        f.write(value)


def _load_text_document(path: str) -> dict:
    """Читает произвольный текстовый файл (.txt/.md, кроме faqs.json и
    файлов из _IGNORED_NAMES) как ОДИН Q&A-элемент: "question" — имя файла
    (чтобы в контексте/логах было видно, откуда взялся ответ), "answer" —
    весь текст файла целиком, без изменений.

    Осознанное ограничение: файл не разбивается на несколько более мелких
    вопросов-ответов и не переформулируется — HR добавил файл с
    инструкциями в свободной форме, а не готовую пару вопрос/ответ, и
    придумывать структуру, которой в файле нет, значит домысливать за HR.
    Если нужны отдельные вопросы с отдельными ответами — их место в
    faqs.json, в штатном формате [{"question": ..., "answer": ...}, ...].
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    name = os.path.basename(path)
    return {"question": name, "answer": text}


def _load_all_items(file_paths: List[str]) -> List[dict]:
    items = []
    for path in file_paths:
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()

        if path == DATA_PATH:
            faq_items = load_faq_data(path)
            print(f"Loaded {len(faq_items)} FAQ items from {name}")
            items.extend(faq_items)
        elif ext == ".json":
            # Любой ДРУГОЙ .json в data/ (не faqs.json) — тот же формат
            # [{"question": ..., "answer": ...}, ...], что и faqs.json. Так
            # HR может добавить, например, faqs_vahta.json отдельным
            # файлом, не редактируя существующий faqs.json.
            faq_items = load_faq_data(path)
            print(f"Loaded {len(faq_items)} FAQ items from {name}")
            items.extend(faq_items)
        elif ext in _TEXT_EXTENSIONS:
            doc = _load_text_document(path)
            if doc["answer"]:
                print(f"Loaded 1 text document from {name}")
                items.append(doc)
            else:
                print(f"Пропускаю {name}: файл пустой.")
        else:
            print(
                f"Пропускаю {name}: формат '{ext}' не поддерживается для базы знаний "
                f"(поддерживаются: .json в формате question/answer, .txt, .md)."
            )
    return items


def main(force: bool = False):
    file_paths = _iter_watched_files()

    if not file_paths:
        raise RuntimeError(
            f"В {DATA_DIR} нет ни одного файла с базой знаний "
            f"(ожидается как минимум faqs.json)."
        )

    current_hash = _combined_hash(file_paths)
    saved_hash = _load_saved_hash()

    if not force and current_hash == saved_hash and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Файлы базы знаний не изменились с прошлой сборки — индекс актуален, пересборка не требуется.")
        return

    if not saved_hash and not os.path.exists(INDEX_PATH):
        print("Готового индекса нет — собираю с нуля.")
    else:
        print("Обнаружены изменения в data/ — пересобираю индекс.")

    items = _load_all_items(file_paths)

    if not items:
        raise RuntimeError("No data found to build index (data/ contains no usable Q&A items).")

    texts = [f"{item['question']}\n{item['answer']}" for item in items]

    print(f"Embedding {len(texts)} items...")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    meta = np.array(
        [
            {
                "question": item["question"],
                "answer": item["answer"],
            }
            for item in items
        ],
        dtype=object,
    )
    np.save(META_PATH, meta)
    _save_hash(current_hash)

    print(f"Index built and saved to {INDEX_PATH}")


if __name__ == "__main__":
    import sys
    force_rebuild = "--force" in sys.argv
    main(force=force_rebuild)
