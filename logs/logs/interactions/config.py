"""
Конфигурация логирования взаимодействий.
"""

import os
from pathlib import Path

# Папка для логов взаимодействий
INTERACTIONS_DIR = Path(__file__).parent.absolute()

# Путь к БД SQLite (взаимодействия)
DATABASE_PATH = os.getenv("FAQ_INTERACTIONS_DB", str(INTERACTIONS_DIR / "interactions.db"))

# Ретеншен логов (в днях)
LOG_RETENTION_DAYS = int(os.getenv("FAQ_LOG_RETENTION_DAYS", "30"))

# Включить режим WAL для устойчивости
ENABLE_WAL = os.getenv("FAQ_ENABLE_WAL", "true").lower() == "true"

# Таймаут при занятой БД (секунды)
DATABASE_TIMEOUT = int(os.getenv("FAQ_DB_TIMEOUT", "10"))

