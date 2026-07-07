"""
Конфигурация логирования событий.
"""

import os
from pathlib import Path

# Папка для логов событий
EVENTS_DIR = Path(__file__).parent.absolute()

# Путь к БД SQLite (события)
DATABASE_PATH = os.getenv("FAQ_EVENTS_DB", str(EVENTS_DIR / "events.db"))

# Ретеншен логов (в днях)
LOG_RETENTION_DAYS = int(os.getenv("FAQ_LOG_RETENTION_DAYS", "30"))

# Включить режим WAL
ENABLE_WAL = os.getenv("FAQ_ENABLE_WAL", "true").lower() == "true"

# Таймаут при занятой БД
DATABASE_TIMEOUT = int(os.getenv("FAQ_DB_TIMEOUT", "10"))

