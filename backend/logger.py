import csv
import os
import threading
from datetime import datetime, timedelta
from typing import Optional

from . import crypto_utils
from . import paths
from .filelock import cross_process_lock

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

# threading.Lock защищает от параллельных ПОТОКОВ одного процесса. Сайт и
# Telegram-бот — разные процессы ОС (см. run_all.py), между ними он не
# работает, поэтому операции записи и удаления дополнительно берут
# блокировку средствами ОС (backend/filelock.py), как и candidates.csv.
_lock = threading.Lock()

FIELDNAMES = [
    "Дата и время",
    # Тип записи. Журнал ОДИН на всё: и разговоры с кандидатами, и
    # системные события. Раньше события уходили только в консоль и
    # исчезали вместе с ней — на сервере консоли под рукой нет, и о том,
    # что уведомление HR не дошло, узнать было бы неоткуда.
    #
    # Один файл вместо двух выбран намеренно: его можно целиком отдать
    # панели мониторинга, и вся картина будет в одном потоке — сообщения,
    # сбои, блокировки, удаления. Значения см. в EVENT_* ниже.
    "Событие",
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
# --- Типы записей в журнале ------------------------------------------------
#
# Латиницей и одним словом: значения читает панель мониторинга, и им лучше
# быть устойчивыми к правкам текста.
EVENT_DIALOG = "dialog"            # обычное сообщение кандидата
EVENT_START = "start"              # запуск проекта
EVENT_MODEL = "model"              # загрузка модели поиска
EVENT_INDEX = "index"              # пересборка поискового индекса
EVENT_GIGACHAT = "gigachat"        # сбой обращения к нейросети
EVENT_NOTIFY = "notify"            # уведомление HR-группе
EVENT_DELETE = "delete"            # удаление данных кандидата (152-ФЗ)
EVENT_BLOCK = "block"              # блокировка за спам
EVENT_MARKER = "marker"            # отклонена попытка навязать анкету

# Статусы — для раскраски в панели мониторинга.
STATUS_OK = "ok"
STATUS_WARN = "внимание"
STATUS_ERROR = "ошибка"

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

    with _lock, cross_process_lock(LOG_PATH):
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
        "Событие": EVENT_DIALOG,
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

    with _lock, cross_process_lock(LOG_PATH):
        with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def log_event(event: str, comment: str, status: str = STATUS_OK,
              source: str = "", external_id: str = "",
              details: str = "", duration_ms=None) -> None:
    """Пишет системное событие в ТОТ ЖЕ журнал, что и разговоры.

    Что куда кладётся, и почему именно так:

    - comment попадает в ОТКРЫТУЮ колонку «Комментарий». Значит панель
      мониторинга читает события без ключа шифрования, а переписка
      кандидатов при этом остаётся для неё закрытой. Из этого следует
      жёсткое правило: в comment НЕЛЬЗЯ класть текст, написанный человеком.
      Только факты о работе системы.
    - details — для текста, который всё-таки пришёл от кандидата (например
      сообщение, на котором сработала защита от навязанной анкеты). Он
      уходит в зашифрованную колонку «Вопрос», как и обычные вопросы.

    Ошибка записи в журнал НЕ должна ронять бота: если диск заполнен или
    файл занят, кандидат всё равно обязан получить ответ. Поэтому всё
    обёрнуто в try/except с выводом в консоль."""
    try:
        _ensure_header_up_to_date()
        file_exists = os.path.exists(LOG_PATH)

        row = {name: "" for name in FIELDNAMES}
        row.update({
            "Дата и время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Событие": event,
            "Источник": source,
            "ID пользователя": external_id,
            "Вопрос": _encrypt_field(details) if details else "",
            "Время ответа (мс)": duration_ms if duration_ms is not None else "",
            "Статус": status,
            "Комментарий": comment,
        })

        with _lock, cross_process_lock(LOG_PATH):
            with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
    except Exception as e:
        print(f"[logger] Не удалось записать событие {event!r}: {e}")


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

    with _lock, cross_process_lock(LOG_PATH):
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
