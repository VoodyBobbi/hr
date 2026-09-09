import hashlib
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .kb_chunker import kb_files, load_and_chunk as load_kb_chunks
from .rag_index import load_faq_data, write_index

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "faqs.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
HASH_PATH = os.path.join(DATA_DIR, "data_hash.txt")

# Отдельный, второй индекс для базы знаний из data/kb/ — см. подробное
# объяснение в backend/assistant.py (комментарий у KB_INDEX_PATH) и
# backend/kb_chunker.py про то, почему раздельно от FAQ-индекса выше, а не
# в одном общем top-3.
KB_DIR = os.path.join(DATA_DIR, "kb")
KB_INDEX_PATH = os.path.join(DATA_DIR, "kb_faiss_index.bin")
KB_META_PATH = os.path.join(DATA_DIR, "kb_metadata.npy")
KB_HASH_PATH = os.path.join(DATA_DIR, "kb_data_hash.txt")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Версия способа построения векторов. Участвует в хеше (см. _combined_hash и
# build_kb_index), поэтому её изменение принудительно пересобирает ОБА
# индекса при следующем запуске.
#
# Зачем это нужно. Пересборка раньше зависела только от содержимого файлов в
# data/. Но индекс устаревает и тогда, когда меняется САМ СПОСОБ векторизации
# при неизменных исходных данных — например при включении нормализации или
# смене типа индекса с IndexFlatL2 на IndexFlatIP. В таком случае хеш файлов
# совпал бы со старым, пересборка пропускалась бы, и запросы искались бы по
# индексу, построенному по другим правилам — поиск молча перестал бы
# находить что-либо осмысленное.
#
# Поднимайте это значение при ЛЮБОМ изменении embed_texts, модели
# EMBEDDING_MODEL_NAME или флага нормализации.
EMBEDDING_VERSION = "v3-cosine-kb-folder"

# Файлы в data/, которые НЕ являются источником вопросов-ответов для FAQ-
# индекса и поэтому исключены из отслеживания FAQ-индекса ниже:
#   - faiss_index.bin, faqs_metadata.npy, data_hash.txt, faqs_hash.txt,
#     kb_faiss_index.bin, kb_metadata.npy, kb_data_hash.txt — сами
#     артефакты сборки (обоих индексов). Если их не исключить, пересборка
#     меняла бы их же, и при следующем запуске хеш "изменился" бы сам по
#     себе — вечная пересборка на каждом старте.
#   - папка kb/ сюда не попадает вообще: _iter_watched_files пропускает
#     каталоги. У базы знаний свой индекс и своя пересборка по своему хешу
#     (build_kb_index ниже) — независимая от FAQ-индекса, чтобы правка
#     одного тематического файла не пересобирала заново эмбеддинги
#     faqs.json, и наоборот.
#   - system_prompt.md — у него остался прежний отдельный механизм
#     отслеживания: assistant.py:_ensure_fresh() подхватывает его на лету
#     по mtime, БЕЗ пересборки векторного индекса (см. README, раздел
#     "Обновление базы знаний") — это неструктурированный текст инструкции
#     для GigaChat, не источник фактов для поиска.
_IGNORED_NAMES = {
    "faiss_index.bin",
    "faqs_metadata.npy",
    "data_hash.txt",
    "faqs_hash.txt",
    "kb_faiss_index.bin",
    "kb_metadata.npy",
    "kb_data_hash.txt",
    "system_prompt.md",
}

# Расширения, которые понимаются как "сырой текстовый документ с
# инструкциями" — весь файл целиком идёт в базу как один Q&A-элемент (см.
# _load_text_documents). faqs.json обрабатывается отдельно, по-другому (см.
# main) — у него есть структура question/answer, которую нет смысла терять.
_TEXT_EXTENSIONS = {".txt", ".md"}

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Векторы для FAISS-индекса.

    normalize_embeddings=True приводит длину каждого вектора к единице. У
    sentence-transformers значение по умолчанию — False, и раньше его никто
    не задавал ни здесь, ни в assistant.embed_text: векторы этой модели
    заметно длиннее единицы, поэтому квадрат расстояния даже между близкими
    по смыслу вопросами далеко превышал порог отсечки, и FAQ-поиск почти
    всегда возвращал пустой результат — весь search-first не срабатывал, и
    каждое сообщение уходило в GigaChat.

    Флаг ОБЯЗАН совпадать с assistant.embed_text: индекс и запрос должны
    строиться одинаково. При изменении поднимите EMBEDDING_VERSION выше."""
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def _build_faiss_from_items(items: List[dict], index_path: str, meta_path: str):
    """Общая механика построения FAISS-индекса из списка элементов —
    переиспользуется и для FAQ-индекса (main ниже), и для KB-индекса
    (build_kb_index ниже), чтобы не дублировать логику дважды.

    Текст для эмбеддинга берётся из поля "embed_text", если оно есть, иначе
    склеивается из question и answer. Разделение нужно потому, что искать и
    показывать надо РАЗНОЕ: чанк базы знаний ищется по заголовку, ключевым
    словам и тексту (см. kb_chunker.py), а показывается модели только
    заголовок и текст — ключевые слова это служебный поисковый мусор.
    Элементы faqs.json тоже могут иметь "keywords", он подмешивается здесь.

    Индекс — IndexFlatIP (скалярное произведение), а не IndexFlatL2. Векторы
    нормализованы, а для единичных векторов скалярное произведение равно
    косинусной близости: порог поиска становится читаемым числом от 0 до 1
    вместо квадрата расстояния в диапазоне 0..4 (см. rag_index.search_similar).
    Точность при этом та же — обе метрики на нормализованных векторах
    упорядочивают соседей одинаково."""
    texts = []
    for item in items:
        embed_text = item.get("embed_text")
        if not embed_text:
            keywords = item.get("keywords") or ""
            if isinstance(keywords, list):
                keywords = " ".join(str(k) for k in keywords)
            embed_text = f"{item['question']}. {keywords}. {item['answer']}"
        texts.append(embed_text.strip())

    print(f"Векторизую {len(texts)} элементов...")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    write_index(index, index_path)

    # В метаданные кладём только то, что реально нужно на этапе ответа:
    # заголовок и текст. embed_text и keywords сюда НЕ попадают — они бы
    # уехали в промпт GigaChat и потратили токены впустую.
    meta = np.array(
        [
            {
                "question": item["question"],
                "answer": item["answer"],
                "topic": item.get("topic", ""),
            }
            for item in items
        ],
        dtype=object,
    )
    np.save(meta_path, meta)

    print(f"Индекс сохранён: {index_path} ({len(items)} элементов, {dim} измерений)")


def _load_saved_hash(hash_path: str) -> str:
    if not os.path.exists(hash_path):
        return ""
    with open(hash_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _save_hash(hash_path: str, value: str):
    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(value)


def build_kb_index(force: bool = False):
    """Пересобирает ОТДЕЛЬНЫЙ индекс из папки data/kb/ — читает все
    тематические файлы (см. kb_chunker.py) и строит по ним свой FAISS-индекс,
    не смешивая с faqs.json.

    Индексы раздельные намеренно. У них разное назначение: попадание в FAQ
    отдаёт кандидату готовый ответ напрямую, минуя GigaChat, а чанк базы
    знаний лишь подмешивается в промпт как факт. Общий top-3 на два разных
    источника означал бы, что три хороших чанка базы знаний вытесняют
    точный FAQ-ответ, и наоборот.

    Пересобирается, только если содержимое папки изменилось (по хешу) или
    передан force=True."""
    files = kb_files(KB_DIR)
    if not files:
        print(f"В {KB_DIR} нет ни одного файла базы знаний — KB-индекс не собран.")
        return

    current_hash = _combined_hash(files)
    saved_hash = _load_saved_hash(KB_HASH_PATH)

    if not force and current_hash == saved_hash and os.path.exists(KB_INDEX_PATH):
        print("База знаний не изменилась — KB-индекс актуален, пересборка не требуется.")
        return

    if force:
        print("Принудительная пересборка KB-индекса.")
    else:
        print("Обнаружены изменения в data/kb/ — пересобираю KB-индекс.")

    chunks = load_kb_chunks(KB_DIR)
    if not chunks:
        raise RuntimeError(f"{KB_DIR} не дал ни одного чанка для индексации (пустые файлы?).")

    _build_faiss_from_items(chunks, KB_INDEX_PATH, KB_META_PATH)
    _save_hash(KB_HASH_PATH, current_hash)


def _iter_watched_files():
    """Все файлы в data/, которые участвуют в сборке FAQ-индекса — faqs.json
    плюс любые дополнительные файлы с вопросами-ответами, если их туда
    положили. Отсортировано по имени, чтобы порядок элементов в индексе был
    одинаковым от запуска к запуску, а не зависел от файловой системы.

    Папки пропускаются (os.path.isfile), поэтому data/kb/ сюда не попадает —
    у базы знаний свой отдельный индекс со своей пересборкой, см.
    build_kb_index."""
    if not os.path.isdir(DATA_DIR):
        return []
    paths = []
    for name in sorted(os.listdir(DATA_DIR)):
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
    (список путей тоже участвует в хеше, не только содержимое).

    EMBEDDING_VERSION тоже подмешивается в хеш — чтобы смена способа
    векторизации при неизменных данных тоже вызывала пересборку."""
    digest = hashlib.sha256()
    digest.update(EMBEDDING_VERSION.encode("utf-8"))
    for path in file_paths:
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


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
            print(f"Загружено {len(faq_items)} вопросов-ответов из {name}")
            items.extend(faq_items)
        elif ext == ".json":
            # Любой другой .json прямо в data/ (не в подпапке kb/) — тот же
            # формат [{"question": ..., "answer": ...}, ...], что и faqs.json.
            # Так HR может добавить, например, faqs_vahta.json отдельным
            # файлом, не редактируя существующий faqs.json.
            faq_items = load_faq_data(path)
            print(f"Загружено {len(faq_items)} вопросов-ответов из {name}")
            items.extend(faq_items)
        elif ext in _TEXT_EXTENSIONS:
            doc = _load_text_document(path)
            if doc["answer"]:
                print(f"Загружен текстовый документ {name}")
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
    saved_hash = _load_saved_hash(HASH_PATH)

    if not force and current_hash == saved_hash and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Файлы базы знаний не изменились с прошлой сборки — FAQ-индекс актуален, пересборка не требуется.")
    else:
        if not saved_hash and not os.path.exists(INDEX_PATH):
            print("Готового FAQ-индекса нет — собираю с нуля.")
        else:
            print("Обнаружены изменения в data/ — пересобираю FAQ-индекс.")

        items = _load_all_items(file_paths)

        if not items:
            raise RuntimeError("В data/ не найдено ни одного вопроса-ответа для сборки индекса.")

        _build_faiss_from_items(items, INDEX_PATH, META_PATH)
        _save_hash(HASH_PATH, current_hash)

    # KB-индекс — независимая пересборка со своим собственным хешем (см.
    # build_kb_index) — не завязана на хеш faqs.json выше, изменение одного
    # файла не должно триггерить пересборку другого.
    build_kb_index(force=force)


if __name__ == "__main__":
    import sys
    force_rebuild = "--force" in sys.argv
    main(force=force_rebuild)
