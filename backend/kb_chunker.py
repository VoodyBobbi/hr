"""
Разбивка data/knowledge_base.json на мелкие чанки для отдельного поиска в
FAISS (см. build_index.py — knowledge_base индексируется ОТДЕЛЬНЫМ поиском
от faqs.json, не смешивается в один общий top-3, см. rag_index.py).

Раньше весь knowledge_base.json целиком подавался в системный промпт при
каждом запросе (см. историю assistant.py — функция _build_system_prompt) —
это тратило токены на информацию, не относящуюся к конкретному вопросу
кандидата (например, полное описание вакансии альпиниста в промпте, когда
кандидат спрашивает про монтажника). Разбивка на чанки + отдельный поиск
позволяет подмешивать в промпт только те несколько фрагментов, которые
реально релевантны текущему вопросу.

Гранулярность — МЕЛКАЯ: каждое самое глубоко вложенное поле отдельным
чанком (не по верхнеуровневому разделу целиком) — точнее поиск, ценой
большего числа чанков. Структура knowledge_base.json неравномерная (где-то
2 уровня вложенности, где-то 5 — см. physical_tests.jump_height_30sec.5) —
поэтому обход рекурсивный, без фиксированной глубины: чанком становится
первое встреченное значение, которое НЕ является словарём (dict) — строка,
число, список. Списки НЕ разбиваются на отдельные элементы (например
selection_and_admission.forbidden_regions — один чанк из 9 регионов, не 9
чанков по одному региону) — элементы списка обычно логически единое целое,
разбивать их по отдельности потеряло бы смысл (одна строка из списка
регионов сама по себе бесполезна для поиска).
"""
import json

# Служебные поля верхнего уровня knowledge_base.json, не являющиеся
# содержательной информацией для поиска — исключены из обхода.
_IGNORED_TOP_LEVEL_KEYS = {"version"}


def _stringify_leaf(value) -> str:
    """Приводит лист (не-dict значение) к тексту для эмбеддинга/показа.
    Списки строк — через запятую, не через json.dumps (читаемее и для
    человека, и для модели эмбеддингов, чем сырой JSON-массив)."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def chunk_knowledge_base(kb: dict) -> list[dict]:
    """Рекурсивно обходит knowledge_base.json, возвращает список чанков в
    формате [{"question": "путь.до.поля", "answer": "текст"}, ...] — тот же
    формат {"question", "answer"}, что и элементы faqs.json (см.
    build_index.py/rag_index.py), чтобы переиспользовать существующую
    инфраструктуру эмбеддинга/поиска без дублирования.

    "question" — путь в структуре (например
    "vacancies.montazhnik.salary") — это НЕ вопрос в человеческом смысле, а
    заголовок чанка для контекста при показе/логировании; сам текстовый
    поиск (embed_texts в build_index.py) использует и question, и answer
    вместе, так что путь тоже участвует в сопоставлении с вопросом
    кандидата, просто не как основной сигнал.
    """
    chunks = []

    def walk(node, path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                if not path and key in _IGNORED_TOP_LEVEL_KEYS:
                    continue
                new_path = f"{path}.{key}" if path else key
                walk(value, new_path)
        else:
            text = _stringify_leaf(node)
            if text.strip():
                chunks.append({"question": path, "answer": text})
            # Пустая строка — не добавляем чанк вообще (не несёт пользы для
            # поиска), но это не ошибка данных, поэтому не логируем как проблему.

    walk(kb, "")
    return chunks


def load_and_chunk(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return chunk_knowledge_base(kb)
