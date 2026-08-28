import csv
import io
import json
import os
import threading
from datetime import datetime

from . import paths
from .crypto_utils import decrypt_bytes, encrypt_bytes
from .validators import FieldValidationError, validate_field

CANDIDATS_DIR = paths.CANDIDATS_DIR
CANDIDATES_PATH = os.path.join(CANDIDATS_DIR, "candidates.csv")
SESSIONS_PATH = os.path.join(CANDIDATS_DIR, "candidate_sessions.json")

_lock = threading.Lock()

FIELD_ORDER = [
    "ID кандидата",
    "Источник",
    "ДАТА заполнения",
    "Фамилия",
    "Имя",
    "Отчество",
    "Дата рождения",
    "Телефон",
    "Telegram",
    "Паспорт серия",
    "Паспорт номер",
    "Кем выдан",
    "Дата выдачи",
    "Код подразделения выдачи паспорта",
    "Адрес регистрации",
    "Адрес проживания",
    "СНИЛС",
    "ИНН",
    "Гражданство",
    "Город проживания",
    "Желаемая должность",
    "Опыт работы, напиши что считаешь нужным",
    "Последнее место работы",
    "Образование/удостоверения/напиши что есть",
    "Рост",
    "Вес",
    "Размер одежды",
    "Размер обуви",
    "Судимость",
    "Административный надзор",
    "Ограничения по здоровью",
    "Готовность к вахте, когда готов",
    "Факт ознакомления с ЗАКОНОМ",
    "Факт ознакомления с готовой карточкой",
]

_NORMALIZED_FIELDS = {f.strip().lower().replace("_", " "): f for f in FIELD_ORDER}

# Служебные поля, которые не являются "анкетными" данными кандидата —
# их не нужно требовать при проверке "анкета полностью заполнена".
_SERVICE_FIELDS = {
    "ID кандидата",
    "Источник",
    "ДАТА заполнения",
    "Факт ознакомления с ЗАКОНОМ",
    "Факт ознакомления с готовой карточкой",
}

# Ровно те 29 полей анкеты, которые модель обязана собрать перед CARD_CONFIRMED
# (см. системный промпт: "ЖЁСТКИЙ ИНВАРИАНТ" про 29 полей). Список полей
# сознательно не сокращается (решение: собирать все 29 полей в чат-боте).
REQUIRED_CANDIDATE_FIELDS = [f for f in FIELD_ORDER if f not in _SERVICE_FIELDS]


def normalize_field_name(name: str) -> str | None:
    """Пытается сопоставить произвольное имя поля (от модели) с реальным полем таблицы."""
    key = name.strip().lower().replace("_", " ")
    return _NORMALIZED_FIELDS.get(key)


def _load_sessions() -> dict:
    if not os.path.exists(SESSIONS_PATH) or os.path.getsize(SESSIONS_PATH) == 0:
        return {}
    try:
        with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[candidates] {SESSIONS_PATH} повреждён или пуст — начинаю с чистого списка сессий.")
        return {}


def _save_sessions(sessions: dict):
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def _load_table():
    """Читает candidates.csv с диска. Файл на диске хранится зашифрованным
    (Fernet) — здесь он расшифровывается в память и разбирается как CSV."""
    if not os.path.exists(CANDIDATES_PATH):
        return list(FIELD_ORDER), {}

    with open(CANDIDATES_PATH, "rb") as f:
        ciphertext = f.read()

    if not ciphertext:
        return list(FIELD_ORDER), {}

    plaintext = decrypt_bytes(ciphertext).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(plaintext)))

    if not rows:
        return list(FIELD_ORDER), {}

    header = rows[0]
    candidate_ids = header[1:]

    fields = [row[0] for row in rows[1:] if row]
    candidates = {cid: {} for cid in candidate_ids}

    for row in rows[1:]:
        if not row:
            continue
        field_name = row[0]
        for i, cid in enumerate(candidate_ids):
            value = row[i + 1] if i + 1 < len(row) else ""
            candidates[cid][field_name] = value

    return fields, candidates


def _save_table(fields: list, candidates: dict):
    """Сериализует таблицу в CSV в памяти, шифрует и пишет на диск —
    на диске никогда не оказывается незашифрованный CSV."""
    candidate_ids = sorted(candidates.keys(), key=lambda x: int(x))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Поле"] + candidate_ids)
    for field in fields:
        row = [field]
        for cid in candidate_ids:
            row.append(candidates[cid].get(field, ""))
        writer.writerow(row)

    plaintext = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    ciphertext = encrypt_bytes(plaintext)

    tmp_path = CANDIDATES_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(ciphertext)
    os.replace(tmp_path, CANDIDATES_PATH)  # атомарная замена — не оставит файл в битом состоянии при сбое


def get_or_create_candidate(source: str, external_id: str) -> str:
    with _lock:
        sessions = _load_sessions()
        key = f"{source}:{external_id}"

        if key in sessions:
            return sessions[key]

        fields, candidates = _load_table()
        existing_ids = [int(cid) for cid in candidates.keys()] if candidates else []
        new_id = str(max(existing_ids) + 1) if existing_ids else "1"

        candidates[new_id] = {field: "" for field in fields}
        candidates[new_id]["ID кандидата"] = new_id
        candidates[new_id]["Источник"] = source
        candidates[new_id]["ДАТА заполнения"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        _save_table(fields, candidates)

        sessions[key] = new_id
        _save_sessions(sessions)

        return new_id


def set_field(candidate_id: str, field_name: str, value: str) -> bool:
    """Сохраняет значение поля после валидации формата.

    Бросает FieldValidationError, если значение не проходит проверку формата
    (вызывающий код — assistant._process_markers — ловит её и мягко просит
    кандидата уточнить значение, ничего не сохраняя)."""
    real_field = normalize_field_name(field_name)
    if real_field is None:
        print(f"[candidates] Неизвестное поле от модели: '{field_name}' — игнорирую.")
        return False

    # Служебные поля (ID/источник/дата/отметки согласия) заполняются самим
    # кодом, а не значениями от модели/кандидата — валидация к ним не нужна.
    if real_field not in _SERVICE_FIELDS:
        value = validate_field(real_field, value)

    with _lock:
        fields, candidates = _load_table()
        if candidate_id not in candidates:
            return False
        candidates[candidate_id][real_field] = value
        _save_table(fields, candidates)
        return True


def mark_law_acknowledged(candidate_id: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_field(candidate_id, "Факт ознакомления с ЗАКОНОМ", f"Согласие получено {timestamp}")


def mark_card_confirmed(candidate_id: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_field(candidate_id, "Факт ознакомления с готовой карточкой",
              f"Кандидат подтвердил корректность анкеты {timestamp}")


def get_card(candidate_id: str) -> dict:
    with _lock:
        _, candidates = _load_table()
        return candidates.get(candidate_id, {})


def is_law_acknowledged(candidate_id: str) -> bool:
    """True, если кандидат уже дал согласие по 152-ФЗ (поле реально заполнено в таблице)."""
    card = get_card(candidate_id)
    return bool(card.get("Факт ознакомления с ЗАКОНОМ", "").strip())


def is_card_complete(candidate_id: str) -> bool:
    """
    Серверная проверка того самого 'ЖЁСТКОГО ИНВАРИАНТА' из системного промпта:
    все 29 полей анкеты должны быть реально заполнены, прежде чем можно
    считать анкету готовой к подтверждению (CARD_CONFIRMED).

    Модель может ошибиться и прислать CARD_CONFIRMED раньше времени — этот код
    не полагается на промпт, а проверяет факты по самой таблице кандидатов.
    """
    card = get_card(candidate_id)
    return all(card.get(field, "").strip() for field in REQUIRED_CANDIDATE_FIELDS)


def missing_fields(candidate_id: str) -> list:
    """Список полей анкеты, которые ещё не заполнены (полезно для логов/отладки)."""
    card = get_card(candidate_id)
    return [field for field in REQUIRED_CANDIDATE_FIELDS if not card.get(field, "").strip()]


def delete_candidate(candidate_id: str) -> dict:
    """Полное удаление данных кандидата по его запросу (право на удаление, 152-ФЗ).

    Удаляет: строку кандидата из candidates.csv, его запись(и) в
    candidate_sessions.json (маппинг source:external_id -> candidate_id) и
    файл(ы) истории переписки data*/conversations соответствующие найденным
    сессиям этого кандидата.

    Возвращает словарь с тем, что реально было удалено — полезно для лога/ответа HR.
    """
    with _lock:
        fields, candidates = _load_table()
        existed_in_table = candidate_id in candidates
        if existed_in_table:
            del candidates[candidate_id]
            _save_table(fields, candidates)

        sessions = _load_sessions()
        removed_sessions = [key for key, cid in sessions.items() if cid == candidate_id]
        for key in removed_sessions:
            del sessions[key]
        if removed_sessions:
            _save_sessions(sessions)

    removed_conversations = []
    for key in removed_sessions:
        source, _, external_id = key.partition(":")
        conv_path = paths.conversation_path(source, external_id)
        if os.path.exists(conv_path):
            os.remove(conv_path)
            removed_conversations.append(conv_path)

    return {
        "candidate_id": candidate_id,
        "removed_from_table": existed_in_table,
        "removed_sessions": removed_sessions,
        "removed_conversations": removed_conversations,
    }
