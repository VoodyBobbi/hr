import csv
import os
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOGS_DIR, "logs.csv")

os.makedirs(LOGS_DIR, exist_ok=True)

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