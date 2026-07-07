"""
EventsLogger: логирование действий и событий в системе.
Таблица: events
Основные поля: identifier (кто), action (что сделал), status (результат), details (детали).
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import csv
import os

from .config import DATABASE_PATH, LOG_RETENTION_DAYS, ENABLE_WAL, DATABASE_TIMEOUT

logger = logging.getLogger(__name__)


class EventsLogger:
    """
    Логирует действия и события (кто сделал что и с каким результатом).
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Инициализирует логгер событий.

        Args:
            db_path: Путь к БД SQLite.
        """
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Получает соединение с БД."""
        conn = sqlite3.connect(self.db_path, timeout=DATABASE_TIMEOUT)
        if ENABLE_WAL:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Создаёт таблицу событий, если её нет."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    status TEXT,
                    result TEXT,
                    details TEXT,
                    error_message TEXT,
                    duration_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Индексы
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_identifier ON events(identifier)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_action_type ON events(action_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)
            """)

            conn.commit()
            conn.close()
            logger.info(f"Таблица events инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)
            raise

    def log_event(
        self,
        identifier: str,
        action_type: str,
        action_name: str,
        status: str = "completed",
        result: Optional[str] = None,
        details: Optional[Dict] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> int:
        """
        Логирует событие.

        Args:
            identifier: Идентификатор пользователя/системы (кто).
            action_type: Тип действия (query_process, vectorization, search, model_call, export и т.д.).
            action_name: Название действия (что сделал).
            status: Статус (completed, failed, pending, error и т.д.).
            result: Результат (успешно/неуспешно/с ошибкой).
            details: Дополнительные детали (как Dict, будет переведено в JSON).
            error_message: Сообщение об ошибке, если была.
            duration_ms: Длительность выполнения в мс.

        Returns:
            ID созданного события или 0 при ошибке.
        """
        try:
            timestamp = datetime.now().isoformat()
            details_json = json.dumps(details, ensure_ascii=False) if details else None

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO events (
                    timestamp, identifier, action_type, action_name, status,
                    result, details, error_message, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, identifier, action_type, action_name, status,
                result, details_json, error_message, duration_ms
            ))

            event_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return event_id

        except Exception as e:
            logger.error(f"Ошибка логирования события (identifier={identifier}): {e}", exc_info=True)
            return 0

    def get_stats(self) -> Dict:
        """Возвращает статистику по событиям."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as total FROM events")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(DISTINCT identifier) as unique_actors FROM events")
            unique_actors = cursor.fetchone()["unique_actors"]

            cursor.execute("SELECT COUNT(*) as completed FROM events WHERE status = 'completed'")
            completed = cursor.fetchone()["completed"]

            cursor.execute("SELECT COUNT(*) as failed FROM events WHERE status = 'failed' OR status = 'error'")
            failed = cursor.fetchone()["failed"]

            cursor.execute("SELECT AVG(duration_ms) as avg_duration FROM events WHERE duration_ms IS NOT NULL")
            avg_duration = cursor.fetchone()["avg_duration"] or 0

            conn.close()

            return {
                "total_events": total,
                "unique_actors": unique_actors,
                "completed": completed,
                "failed": failed,
                "avg_duration_ms": round(avg_duration, 2),
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
            return {}

    def get_user_events(self, identifier: str, limit: int = 20) -> List[Dict]:
        """Возвращает события пользователя/системного компонента."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM events
                WHERE identifier = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (identifier, limit))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Ошибка получения событий ({identifier}): {e}", exc_info=True)
            return []

    def get_failed_events(self, limit: int = 20) -> List[Dict]:
        """Возвращает ошибочные события."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM events
                WHERE status IN ('failed', 'error')
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Ошибка получения ошибочных событий: {e}", exc_info=True)
            return []

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Экспортирует все события в CSV."""
        try:
            if filepath is None:
                logs_dir = Path(self.db_path).parent
                logs_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = logs_dir / f"events_full_{timestamp}.csv"

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM events ORDER BY created_at DESC")
            rows = cursor.fetchall()

            if not rows:
                logger.warning("Нет событий для экспорта")
                conn.close()
                return filepath

            columns = [description[0] for description in cursor.description]

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            conn.close()
            logger.info(f"Экспорт событий выполнен: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}", exc_info=True)
            return ""

    def export_daily_csv(self, for_date: Optional[str] = None, filepath: Optional[str] = None) -> str:
        """Экспортирует события за день в CSV."""
        try:
            if for_date is None:
                for_date = datetime.now().strftime("%Y-%m-%d")

            if filepath is None:
                logs_dir = Path(self.db_path).parent
                logs_dir.mkdir(exist_ok=True)
                filepath = logs_dir / f"events_{for_date}.csv"

            conn = self._get_connection()
            cursor = conn.cursor()

            date_start = f"{for_date}T00:00:00"
            date_end = f"{for_date}T23:59:59"

            cursor.execute("""
                SELECT * FROM events
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (date_start, date_end))

            rows = cursor.fetchall()

            if not rows:
                logger.warning(f"Нет событий за {for_date}")
                conn.close()
                return filepath

            columns = [description[0] for description in cursor.description]

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            conn.close()
            logger.info(f"Дневной экспорт событий: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка дневного экспорта: {e}", exc_info=True)
            return ""

    def cleanup_old_logs(self, days: Optional[int] = None) -> int:
        """Удаляет старые события."""
        try:
            if days is None:
                days = LOG_RETENTION_DAYS

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_date,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Удалено {deleted_count} событий старше {days} дней")
            return deleted_count

        except Exception as e:
            logger.error(f"Ошибка очистки: {e}", exc_info=True)
            return 0

