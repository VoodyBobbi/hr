import csv
import json
import os
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CANDIDATS_DIR = os.path.join(BASE_DIR, "candidats")
CANDIDATES_PATH = os.path.join(CANDIDATS_DIR, "candidates.csv")
SESSIONS_PATH = os.path.join(CANDIDATS_DIR, "candidate_sessions.json")

os.makedirs(CANDIDATS_DIR, exist_ok=True)

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
# (см. системный промпт: "ЖЁСТКИЙ ИНВАРИАНТ" про 29 полей).
REQUIRED_CANDIDATE_FIELDS = [f for f in FIELD_ORDER if f not in _SERVICE_FIELDS]


def normalize_field_name(name: str) -> str | None:
    """Пытается сопоставить произвольное имя поля (от модели) с реальным полем таблицы."""
    key = name.strip().lower().replace("_", " ")
    return _NORMALIZED_FIELDS.get(key)


def _load_sessions() -> dict:
    if not os.path.exists(SESSIONS_PATH):
        return {}
    with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_sessions(sessions: dict):
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def _load_table():
    if not os.path.exists(CANDIDATES_PATH):
        return list(FIELD_ORDER), {}

    with open(CANDIDATES_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

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
    candidate_ids = sorted(candidates.keys(), key=lambda x: int(x))

    with open(CANDIDATES_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Поле"] + candidate_ids)
        for field in fields:
            row = [field]
            for cid in candidate_ids:
                row.append(candidates[cid].get(field, ""))
            writer.writerow(row)


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
    real_field = normalize_field_name(field_name)
    if real_field is None:
        print(f"[candidates] Неизвестное поле от модели: '{field_name}' — игнорирую.")
        return False

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