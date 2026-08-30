import csv
import os
import threading
from datetime import datetime, timedelta

from . import paths

# Логи хранят полный текст переписки кандидата — те же персональные данные,
# что и анкета, только неструктурированные (кандидат может упомянуть что
# угодно личное в свободном вопросе). У анкеты уже есть право на удаление
# (см. candidates.delete_candidate) — логи без срока хранения были бы
# несогласованным исключением из того же принципа. Решение: старые строки
# удаляются автоматически при каждом запуске сервера (см. purge_old_logs
# ниже, вызывается из app.py и telegram_bot.py при старте) — без отдельного
# cron/планировщика, чтобы работать при запуске одной командой (run_all.py).
#
# LOGS_RETENTION_DAYS в .env — сколько дней хранить строку лога с момента её
# записи. По умолчанию 90. Значение 0 означает "хранить логи вечно, не
# удалять ничего" — на случай, если это осознанно нужно для аудита.
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
]


def log_interaction(source: str, external_id: str, query: str, response: str,
                     response_time_ms: int, status: str = "ok", comment: str = ""):
    file_exists = os.path.exists(LOG_PATH)

    row = {
        "Дата и время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Источник": source,
        "ID пользователя": external_id,
        "Вопрос": query,
        "Ответ": response,
        "Время ответа (мс)": response_time_ms,
        "Статус": status,
        "Комментарий": comment,
    }

    with _lock:
        with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def purge_old_logs():
    """Удаляет из logs.csv строки старше LOGS_RETENTION_DAYS дней. Вызывается
    один раз при старте сервера (app.py и telegram_bot.py) — не бесконечный
    фоновый процесс, а простая проверка на каждом запуске, этого достаточно
    для локального сервера, который перезапускается не реже раза в
    LOGS_RETENTION_DAYS дней в обычной эксплуатации."""
    if LOGS_RETENTION_DAYS <= 0:
        return  # 0 — осознанное решение хранить логи вечно, ничего не делаем

    if not os.path.exists(LOG_PATH):
        return

    cutoff = datetime.now() - timedelta(days=LOGS_RETENTION_DAYS)

    with _lock:
        with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        kept_rows = []
        removed_count = 0
        for row in rows:
            try:
                row_date = datetime.strptime(row["Дата и время"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError):
                # Строка повреждена/в неожиданном формате — не удаляем её
                # молча по ошибке, лучше оставить и разобраться руками.
                kept_rows.append(row)
                continue
            if row_date >= cutoff:
                kept_rows.append(row)
            else:
                removed_count += 1

        if removed_count == 0:
            return

        tmp_path = LOG_PATH + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(tmp_path, LOG_PATH)

        print(
            f"[logger] Удалено {removed_count} строк(и) логов старше "
            f"{LOGS_RETENTION_DAYS} дней (LOGS_RETENTION_DAYS)."
        )
