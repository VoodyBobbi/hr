"""
Загрузка базы знаний из data/kb/*.json и превращение её в чанки для
векторного поиска.

## Зачем база разбита на файлы

Раньше вся база знаний лежала в одном knowledge_base.json как вложенное
дерево, а чанком становилось каждое листовое поле. Заголовком чанка при
этом был путь в дереве — "vacancies.montazhnik.salary". Для человека это
читаемо, но для поиска почти бесполезно: кандидат пишет "сколько платят
монтажнику", и точка соприкосновения с текстом "vacancies.montazhnik.salary"
минимальна.

Теперь база — это папка data/kb/ с тематическими файлами (company.json,
vacancies.json, medical.json и так далее). Два выигрыша:

1. Редактировать. HR правит medical.json, не открывая всё остальное и не
   рискуя сломать чужой раздел. Добавить новую тему — положить новый файл,
   в коде менять ничего не нужно.
2. Искать. У каждой записи есть человеческий заголовок (title) и список
   ключевых слов (keywords) — синонимы и формулировки, которыми кандидат
   реально задаёт вопрос. Всё это идёт в вектор вместе с текстом ответа,
   и попадание становится заметно точнее.

## Формат файла

    {
      "topic": "Медосмотр и здоровье",
      "entries": [
        {
          "id": "med.vision",
          "title": "Требования к зрению",
          "keywords": ["зрение", "очки", "линзы", "диоптрии"],
          "text": "Зрение должно быть в диапазоне от -2 до +2..."
        }
      ]
    }

Обязательны только "title" и "text". "id" нужен для логов и отладки,
"keywords" — необязательный, но сильно влияющий на качество поиска список.

## Что уходит в вектор, а что в промпт

Это РАЗНЫЕ строки, и в этом весь смысл:

- В вектор (embed_text) идёт title + keywords + text. Ключевые слова
  расширяют «зону попадания» записи, но сами по себе кандидату не нужны.
- В промпт GigaChat (assistant._build_system_prompt) идёт title + text.
  Ключевые слова туда НЕ попадают — это служебный поисковый мусор, который
  только тратил бы токены и сбивал модель.
"""
import glob
import json
import os


def _entry_to_chunk(entry: dict, topic: str, source_file: str) -> dict | None:
    """Одна запись базы знаний -> чанк для индекса.

    Возвращает None для записи без заголовка или без текста: такая запись
    бесполезна для поиска, но это не повод падать — HR мог оставить
    заготовку, чтобы дописать позже."""
    title = str(entry.get("title", "")).strip()
    text = str(entry.get("text", "")).strip()
    if not title or not text:
        return None

    keywords = entry.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords_text = " ".join(str(k).strip() for k in keywords if str(k).strip())

    return {
        # Ключи "question"/"answer" — тот же формат, что у элементов
        # faqs.json, чтобы переиспользовать общую механику эмбеддинга и
        # поиска (build_index._build_faiss_from_items) без дублирования.
        "question": title,
        "answer": text,
        # Отдельное поле для эмбеддинга: заголовок + ключевые слова + текст.
        # build_index берёт именно его, если оно есть.
        "embed_text": f"{title}. {keywords_text}. {text}".strip(),
        "topic": topic,
        "id": entry.get("id", ""),
        "source": source_file,
    }


def load_and_chunk(kb_dir: str) -> list[dict]:
    """Читает все *.json из папки базы знаний и возвращает список чанков.

    Файлы читаются в алфавитном порядке — чтобы состав и порядок чанков не
    зависели от порядка обхода файловой системы. Это важно: порядок влияет
    на номера в метаданных индекса, а значит на воспроизводимость сборки.

    Битый JSON пропускается с предупреждением, а не роняет сборку: одна
    опечатка в одном тематическом файле не должна оставлять бота вообще без
    базы знаний."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[kb] Пропускаю {name}: не удалось прочитать ({e}).")
            continue

        topic = str(doc.get("topic", "")).strip() or name
        entries = doc.get("entries")
        if not isinstance(entries, list):
            print(f"[kb] Пропускаю {name}: нет списка \"entries\".")
            continue

        added = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            chunk = _entry_to_chunk(entry, topic, name)
            if chunk:
                chunks.append(chunk)
                added += 1
        print(f"[kb] {name}: {added} записей (тема: {topic})")

    return chunks


def kb_files(kb_dir: str) -> list[str]:
    """Пути всех файлов базы знаний — нужны build_index.py для подсчёта
    хеша, по которому решается, пересобирать индекс или нет."""
    return sorted(glob.glob(os.path.join(kb_dir, "*.json")))
