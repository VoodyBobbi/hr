import csv
import io
import json
import os
import threading
from datetime import datetime

from . import logger
from . import paths
from .crypto_utils import decrypt_bytes, encrypt_bytes
from .filelock import cross_process_lock
from .validators import validate_field

CANDIDATS_DIR = paths.CANDIDATS_DIR
CANDIDATES_PATH = os.path.join(CANDIDATS_DIR, "candidates.csv")
SESSIONS_PATH = os.path.join(CANDIDATS_DIR, "candidate_sessions.json")

# threading.Lock защищает от одновременной записи РАЗНЫХ ПОТОКОВ одного
# процесса (Telegram-бот обслуживает кандидатов параллельно через
# asyncio.to_thread). Между ПРОЦЕССАМИ он бесполезен, а сайт и бот — это два
# отдельных процесса ОС (см. run_all.py), поэтому дополнительно берётся
# блокировка средствами ОС: cross_process_lock из backend/filelock.py.
# Версия набора полей анкеты. Поднимается при ЛЮБОМ изменении состава
# полей в anketa.ANKETA_STEPS: добавили поле, убрали, переименовали.
#
# Зачем. Карточка, начатая на прежнем наборе полей, после обновления
# оказывается несовместимой с новым: программа ждёт ответы на поля, о
# которых кандидат никогда не спрашивали, или наоборот. Раньше это
# проявлялось молча и странно — анкета сбивалась на середине, и понять
# причину со стороны было невозможно (так было при добавлении поля «Этап
# диалога»).
#
# Теперь версия записывается в карточку, а несовпадение видно явно:
# is_outdated ниже даёт коду возможность обработать такой случай осознанно.
ANKETA_VERSION = "2"

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
    "Этап диалога",
    "Версия анкеты",
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
    # На каком шаге диалога сейчас кандидат. Хранится ЯВНО, а не выводится
    # из текста предыдущего сообщения бота.
    #
    # Раньше шаг определялся сравнением last_bot_message с заготовленными
    # фразами: last_bot_message.startswith("Проверьте, пожалуйста, все
    # данные"). Это ломалось от любой правки текста — причём молча: бот
    # переставал понимать, на каком он шаге, и показывал карточку по кругу.
    # Именно так был устроен баг с невозможностью исправить поле.
    #
    # Значения см. в anketa.STAGE_*.
    "Этап диалога",
    # Версия набора полей, действовавшая при создании карточки.
    "Версия анкеты",
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


def find_candidate(source: str, external_id: str) -> str | None:
    """ID уже существующей карточки кандидата, либо None, если её ещё нет.

    В отличие от get_or_create_candidate НИЧЕГО не создаёт. Нужна потому, что
    карточка теперь заводится только в момент согласия по 152-ФЗ (см.
    assistant._handle_anketa_turn), а до этого бот должен спокойно вести
    обычный FAQ-диалог, не оставляя следов в таблице персональных данных."""
    with _lock:
        sessions = _load_sessions()
        return sessions.get(f"{source}:{external_id}")


def get_or_create_candidate(source: str, external_id: str) -> str:
    """Возвращает ID карточки, создавая её при необходимости.

    Вызывается ТОЛЬКО после явного согласия кандидата по 152-ФЗ — раньше
    вызывалась на каждое входящее сообщение, из-за чего таблица зарастала
    пустыми карточками случайных посетителей сайта."""
    with _lock, cross_process_lock(CANDIDATES_PATH):
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
        candidates[new_id]["Версия анкеты"] = ANKETA_VERSION
        candidates[new_id]["ДАТА заполнения"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        _save_table(fields, candidates)

        sessions[key] = new_id
        _save_sessions(sessions)

        return new_id


def set_field(candidate_id: str, field_name: str, value: str) -> bool:
    """Сохраняет значение поля после валидации формата.

    Бросает FieldValidationError, если значение не проходит проверку формата.
    На практике до этого не доходит: значения приходят из
    anketa.process_answer, которая вызывает validate_field раньше и сама
    показывает кандидату дружелюбный текст ошибки. Проверка здесь оставлена
    как последний рубеж на случай записи в обход анкеты (например из
    scripts/)."""
    real_field = normalize_field_name(field_name)
    if real_field is None:
        print(f"[candidates] Неизвестное поле от модели: '{field_name}' — игнорирую.")
        return False

    # Служебные поля (ID/источник/дата/отметки согласия) заполняются самим
    # кодом, а не значениями от модели/кандидата — валидация к ним не нужна.
    if real_field not in _SERVICE_FIELDS:
        value = validate_field(real_field, value)

    with _lock, cross_process_lock(CANDIDATES_PATH):
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


def delete_candidate(candidate_id: str) -> dict:
    """Полное удаление данных кандидата по его запросу (право на удаление, 152-ФЗ).

    Удаляет: строку кандидата из candidates.csv, его запись(и) в
    candidate_sessions.json (маппинг source:external_id -> candidate_id),
    файл(ы) истории переписки в history/ и строки в logs.csv,
    соответствующие найденным сессиям этого кандидата.

    Логи удаляются ОТДЕЛЬНО от истории переписки (logger.delete_logs_for_session,
    не просто os.remove как для conversations) намеренно: в отличие от
    истории (один файл на сессию), logs.csv — общий файл на ВСЕХ кандидатов
    сразу, поэтому удаление означает вырезание конкретных строк из общего
    файла, а не удаление файла целиком. До этого logs.csv не трогался
    вообще — при запросе на полное удаление данных в нём оставался тот же
    текст переписки (те же персональные данные), что и в уже удалённой
    истории — запрос выполнялся не полностью, что прямо противоречит праву
    на удаление по ст. 21 152-ФЗ.

    Возвращает словарь с тем, что реально было удалено — полезно для лога/ответа HR.
    """
    with _lock, cross_process_lock(CANDIDATES_PATH):
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

        # Удаление истории переписки и логов — ВНУТРИ той же блокировки.
        # Раньше блокировка снималась раньше, и в промежуток между снятием и
        # удалением файла кандидат успевал прислать сообщение — файл истории
        # создавался заново уже после «удаления». Данные формально удалены, а
        # фактически лежат на диске, что для 152-ФЗ недопустимо.
        removed_conversations = []
        removed_log_rows = 0
        for key in removed_sessions:
            source, _, external_id = key.partition(":")

            conv_path = paths.conversation_path(source, external_id)
            if os.path.exists(conv_path):
                os.remove(conv_path)
                removed_conversations.append(conv_path)

            removed_log_rows += logger.delete_logs_for_session(source, external_id)

    # След в журнале. Удаление по 152-ФЗ — то самое событие, по которому
    # потом придётся отчитываться: когда обратились и что именно стёрли.
    # Записывается ПОСЛЕ снятия блокировки, иначе получилась бы попытка
    # взять её повторно изнутри уже занятой секции.
    #
    # Текста кандидата здесь нет, только номер карточки и количества, —
    # поэтому запись открытая и читается панелью мониторинга без ключа.
    logger.log_event(
        logger.EVENT_DELETE,
        f"Удалены данные кандидата. Карточка: "
        f"{'была и удалена' if existed_in_table else 'не найдена'}. "
        f"Файлов переписки: {len(removed_conversations)}. "
        f"Строк в журнале: {removed_log_rows}.",
        external_id=str(candidate_id),
    )

    return {
        "candidate_id": candidate_id,
        "removed_from_table": existed_in_table,
        "removed_sessions": removed_sessions,
        "removed_conversations": removed_conversations,
        "removed_log_rows": removed_log_rows,
    }


def get_stage(candidate_id: str) -> str:
    """Текущий шаг диалога кандидата (anketa.STAGE_*). Пустая строка, если
    карточки ещё нет или шаг не выставлялся."""
    if not candidate_id:
        return ""
    return str(get_card(candidate_id).get("Этап диалога", "")).strip()


def set_stage(candidate_id: str, stage: str) -> None:
    """Запоминает шаг диалога. Вызывается каждый раз, когда бот переводит
    кандидата на следующий шаг, — именно на это значение опирается
    assistant._handle_anketa_turn при разборе следующего сообщения."""
    set_field(candidate_id, "Этап диалога", stage)


def is_outdated(candidate_id: str) -> bool:
    """True, если карточка заведена на ПРЕЖНЕМ наборе полей анкеты.

    Такую карточку нельзя продолжать заполнять: вопросы и сохранённые
    ответы разъехались. Вызывающий код (assistant._handle_anketa_turn)
    предлагает кандидату начать заново — это честнее, чем незаметно
    сбиться на середине, как было раньше.

    Пустая версия означает карточку, созданную до появления этого поля."""
    if not candidate_id:
        return False
    saved = str(get_card(candidate_id).get("Версия анкеты", "")).strip()
    return saved != ANKETA_VERSION
