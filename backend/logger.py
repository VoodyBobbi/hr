import csv
import os
import threading
from datetime import datetime, timedelta
from typing import Optional

from . import crypto_utils
from . import paths

# Логи хранят полный текст переписки кандидата — те же персональные данные,
# что и анкета, только неструктурированные (кандидат может упомянуть что
# угодно личное в свободном вопросе, в том числе раньше, чем дойдёт до
# соответствующего официального поля анкеты). У анкеты уже есть право на
# удаление (см. candidates.delete_candidate) — логи без срока хранения были
# бы несогласованным исключением из того же принципа. Решение: старые
# строки удаляются автоматически при каждом запуске сервера (см.
# purge_old_logs ниже, вызывается из run_all.py и из start.sh при старте;
# из app.py она НЕ вызывается, вопреки прежней редакции этого комментария —
# uvicorn можно запустить и напрямую, тогда очистка не произойдёт вовсе,
# см. README) — без отдельного cron/планировщика, чтобы работать
# при запуске одной командой.
#
# LOGS_RETENTION_DAYS в .env — сколько дней хранить строку лога с момента её
# записи. По умолчанию 90. Значение 0 означает "хранить логи вечно, не
# удалять ничего" — на случай, если это осознанно нужно для аудита.
#
# ШИФРОВАНИЕ: только колонки "Вопрос" и "Ответ" — это единственные два
# места в логе, где реально может оказаться личный текст кандидата (он
# волен написать что угодно в свободном сообщении). Остальные колонки
# ("Дата и время", "Источник", "Статус" и т.д.) — служебные метаданные, не
# персональные данные сами по себе, и оставлены открытым текстом намеренно:
# это позволяет быстро просматривать/фильтровать лог по дате или статусу
# (например, в Excel) без расшифровки каждой строки целиком.
#
# Каждое значение шифруется ОТДЕЛЬНО (Fernet поверх каждого поля по
# отдельности, не поверх всей строки и не поверх всего файла) — это
# сознательный выбор в пользу скорости записи и построчного доступа: чтобы
# дописать новую строку в конец файла, не нужно расшифровывать и
# перешифровывать весь файл целиком (как это устроено для candidates.csv,
# см. crypto_utils.py и candidates.py). Плата за это — расшифровка лога
# требует по одной операции на каждое зашифрованное значение, а не одну
# операцию на весь файл сразу; при разумном размере лога (тысячи, не
# миллионы строк) это не создаёт заметной задержки.
LOGS_DIR = paths.LOGS_DIR
LOG_PATH = os.path.join(LOGS_DIR, "logs.csv")
LOGS_RETENTION_DAYS = int(os.getenv("LOGS_RETENTION_DAYS", "90"))

_lock = threading.Lock()

FIELDNAMES = [
    "Дата и время",
    "Источник",
    "ID пользователя",
    "Вопрос",
    "Ответ",
    "Время ответа (мс)",
    "Статус",
    "Комментарий",
    "Токены на вход",
    "Токены на выход",
    "Всего токенов",
    "Кешировано токенов",
]

# Колонки, значения которых шифруются перед записью и расшифровываются при
# чтении. Список, а не захардкоженные имена внутри функций ниже — чтобы
# при необходимости расширить набор шифруемых полей это было видно в одном
# месте, а не разбросано по коду.
_ENCRYPTED_FIELDS = {"Вопрос", "Ответ"}

# Fernet-шифротекст — это ASCII-безопасный base64 сам по себе (см.
# cryptography.fernet), но в него не нужно "заворачивать" пустую строку —
# пустой "Вопрос"/"Ответ" (не должно случаться в норме, но защищаемся)
# остаётся пустой строкой без шифрования, чтобы не плодить лишние токены
# ради шифрования пустоты.
def _encrypt_field(value: str) -> str:
    if not value:
        return ""
    return crypto_utils.encrypt_bytes(value.encode("utf-8")).decode("ascii")


def _decrypt_field(value: str) -> str:
    if not value:
        return ""
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError:
        # Строка содержит НЕ-ASCII символы (например, русские буквы) — это
        # физически не может быть Fernet-токен (тот всегда чистый ASCII
        # base64), значит это открытый текст с прошлого формата, ДО
        # включения построчного шифрования. Возвращаем как есть, без
        # попытки "расшифровать" то, что не зашифровано.
        return value
    try:
        return crypto_utils.decrypt_bytes(raw).decode("utf-8")
    except crypto_utils.EncryptionKeyInvalid:
        # ASCII-строка похожа на Fernet-токен по алфавиту, но текущий
        # ENCRYPTION_KEY её не расшифровывает — либо записана другим
        # ключом, либо повреждена. Не роняем чтение всего лога из-за одной
        # нечитаемой строки — возвращаем как есть с пометкой, а не
        # выдуманное значение.
        return f"[не удалось расшифровать: {value}]"


def _ensure_header_up_to_date():
    """Если logs.csv уже существует с более коротким набором колонок (из
    версии до добавления учёта токенов) — переписывает шапку под текущий
    FIELDNAMES, заполняя старые строки пустыми значениями в новых колонках.

    Без этого шага log_interaction() ниже честно допишет новую 12-колоночную
    строку под старой 8-колоночной шапкой — колонки разъедутся при открытии
    файла (например, в Excel), т.к. количество значений в строке перестанет
    совпадать с количеством заголовков.

    Атомарная запись через tmp-файл + os.replace — тот же приём, что и в
    purge_old_logs() ниже, на случай если сайт и Telegram-бот запущены
    параллельно (run_all.py) и оба импортируют этот модуль почти одновременно."""
    if not os.path.exists(LOG_PATH):
        return

    with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
        try:
            existing_header = next(csv.reader(f))
        except StopIteration:
            return  # пустой файл — мигрировать нечего

    if existing_header == FIELDNAMES:
        return  # шапка уже актуальна

    with _lock:
        with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        tmp_path = LOG_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)  # csv.DictWriter сам подставит "" в новые колонки для старых строк
        os.replace(tmp_path, LOG_PATH)

    print(
        f"[logger] Шапка {LOG_PATH} обновлена под новый формат "
        f"(добавлены колонки по токенам; в старых строках эти поля пустые)."
    )


_ensure_header_up_to_date()


def log_interaction(source: str, external_id: str, query: str, response: str,
                     response_time_ms: int, status: str = "ok", comment: str = "",
                     prompt_tokens: Optional[int] = None,
                     completion_tokens: Optional[int] = None,
                     total_tokens: Optional[int] = None,
                     cached_tokens: Optional[int] = None):
    file_exists = os.path.exists(LOG_PATH)

    row = {
        "Дата и время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Источник": source,
        "ID пользователя": external_id,
        "Вопрос": _encrypt_field(query),
        "Ответ": _encrypt_field(response),
        "Время ответа (мс)": response_time_ms,
        "Статус": status,
        "Комментарий": comment,
        "Токены на вход": prompt_tokens if prompt_tokens is not None else "",
        "Токены на выход": completion_tokens if completion_tokens is not None else "",
        "Всего токенов": total_tokens if total_tokens is not None else "",
        "Кешировано токенов": cached_tokens if cached_tokens is not None else "",
    }

    with _lock:
        with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def purge_old_logs():
    """Удаляет из logs.csv строки старше LOGS_RETENTION_DAYS дней. Вызывается
    при старте: из run_all.py и из start.sh (Docker). При прямом запуске
    uvicorn backend.app:app очистка не выполняется, app.py её не вызывает.
    Это не бесконечный фоновый процесс, а простая проверка на каждом
    запуске — этого достаточно для локального сервера, который
    перезапускается не реже раза в LOGS_RETENTION_DAYS дней.

    Работает НЕ расшифровывая "Вопрос"/"Ответ" — сравнение идёт только по
    полю "Дата и время" (открытый текст), удаление/сохранение строки не
    требует знания ENCRYPTION_KEY вообще."""
    if LOGS_RETENTION_DAYS <= 0:
        return  # 0 — осознанное решение хранить логи вечно, ничего не делаем

    if not os.path.exists(LOG_PATH):
        return

    cutoff = datetime.now() - timedelta(days=LOGS_RETENTION_DAYS)

    def _is_old(row: dict) -> bool:
        try:
            row_date = datetime.strptime(row["Дата и время"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            # Строка повреждена/в неожиданном формате — не удаляем её молча
            # по ошибке, лучше оставить и разобраться руками.
            return False
        return row_date < cutoff

    removed_count = _remove_rows_matching(_is_old)
    if removed_count:
        print(
            f"[logger] Удалено {removed_count} строк(и) логов старше "
            f"{LOGS_RETENTION_DAYS} дней (LOGS_RETENTION_DAYS)."
        )


def delete_logs_for_session(source: str, external_id: str) -> int:
    """Удаляет из logs.csv все строки конкретного кандидата (по паре
    source+external_id — та же пара, что использует candidates.py для
    привязки сессии к кандидату, см. candidate_sessions.json).

    Вызывается из candidates.delete_candidate() — до этой функции
    удаление кандидата стирало анкету, привязку сессии и историю
    переписки, но НЕ логи, хотя в логах остаётся тот же текст переписки
    (те же персональные данные, что и в истории) — при запросе на полное
    удаление (ст. 21 152-ФЗ) это было неполным выполнением запроса.

    Возвращает количество реально удалённых строк.

    Работает по ОТКРЫТЫМ колонкам "Источник"/"ID пользователя" — не требует
    расшифровки "Вопрос"/"Ответ", т.е. не требует ENCRYPTION_KEY, как и
    purge_old_logs() выше."""
    return _remove_rows_matching(
        lambda row: row.get("Источник") == source and row.get("ID пользователя") == external_id
    )


def _remove_rows_matching(should_remove) -> int:
    """Общая механика для purge_old_logs()/delete_logs_for_session(): читает
    logs.csv, убирает строки, для которых should_remove(row) вернула True,
    атомарно перезаписывает файл (tmp + os.replace — тот же приём, что и в
    _ensure_header_up_to_date(), защищает от повреждения файла при сбое
    ровно в момент записи и от гонки, если сайт и Telegram-бот запущены
    параллельно и оба почти одновременно решат почистить логи)."""
    if not os.path.exists(LOG_PATH):
        return 0

    with _lock:
        with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        kept_rows = []
        removed_count = 0
        for row in rows:
            if should_remove(row):
                removed_count += 1
            else:
                kept_rows.append(row)

        if removed_count == 0:
            return 0

        tmp_path = LOG_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(tmp_path, LOG_PATH)

    return removed_count
