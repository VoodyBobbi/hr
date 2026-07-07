"""
InteractionsLogger: логирование диалогов (вопрос-ответ) пользователей.
Таблица: interactions
Основные поля: identifier (кто), query (что спросил), response (ответ), результаты поиска.
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


class InteractionsLogger:
    """
    Логирует диалоги пользователей (вопрос-ответ) в таблицу interactions.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Инициализирует логгер.

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
        """Создаёт таблицу взаимодействий, если её нет."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'telegram',
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    from_cache INTEGER DEFAULT 0,
                    response_time_ms INTEGER,
                    is_valid INTEGER,
                    retrieved_count INTEGER,
                    answer_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Индексы
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_identifier ON interactions(identifier)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_source ON interactions(source)
            """)

            conn.commit()
            conn.close()
            logger.info(f"Таблица interactions инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации БД: {e}", exc_info=True)
            raise

    def log_interaction(
        self,
        identifier: str,
        query: str,
        response: str,
        source: str = "telegram",
        from_cache: bool = False,
        response_time_ms: Optional[int] = None,
        is_valid: Optional[bool] = None,
        retrieved_count: Optional[int] = None,
        answer_source: Optional[str] = None,
    ) -> int:
        """
        Логирует диалог.

        Args:
            identifier: Идентификатор пользователя (ник из Telegram или опознавательный признак).
            query: Вопрос.
            response: Ответ.
            source: Источник (telegram, console, api и т.д.).
            from_cache: Из кэша ли.
            response_time_ms: Время ответа в мс.
            is_valid: Валидный ли ответ.
            retrieved_count: Количество найденных похожих элементов.
            answer_source: Источник ответа (giga, cache, fallback и т.д.).

        Returns:
            ID созданной записи или 0 при ошибке.
        """
        try:
            timestamp = datetime.now().isoformat()
            from_cache_int = 1 if from_cache else 0
            is_valid_int = 1 if is_valid is True else (0 if is_valid is False else None)

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO interactions (
                    timestamp, identifier, source, query, response,
                    from_cache, response_time_ms, is_valid, retrieved_count, answer_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, identifier, source, query, response,
                from_cache_int, response_time_ms, is_valid_int, retrieved_count, answer_source
            ))

            interaction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return interaction_id

        except Exception as e:
            logger.error(f"Ошибка логирования диалога (identifier={identifier}): {e}", exc_info=True)
            return 0

    def get_stats(self) -> Dict:
        """Возвращает статистику по диалогам."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as total FROM interactions")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as from_cache FROM interactions WHERE from_cache = 1")
            from_cache_count = cursor.fetchone()["from_cache"]

            cursor.execute("SELECT COUNT(DISTINCT identifier) as unique_users FROM interactions")
            unique_users = cursor.fetchone()["unique_users"]

            cursor.execute("SELECT AVG(response_time_ms) as avg_time FROM interactions WHERE response_time_ms IS NOT NULL")
            avg_time = cursor.fetchone()["avg_time"] or 0

            cursor.execute("SELECT COUNT(*) as invalid FROM interactions WHERE is_valid = 0")
            invalid_count = cursor.fetchone()["invalid"]

            conn.close()

            return {
                "total_interactions": total,
                "from_cache": from_cache_count,
                "unique_users": unique_users,
                "avg_response_time_ms": round(avg_time, 2),
                "invalid_responses": invalid_count,
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
            return {}

    def get_history(self, identifier: str, limit: int = 10) -> List[Dict]:
        """Возвращает историю диалогов пользователя."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM interactions
                WHERE identifier = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (identifier, limit))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Ошибка получения истории ({identifier}): {e}", exc_info=True)
            return []

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Экспортирует все диалоги в CSV."""
        try:
            if filepath is None:
                logs_dir = Path(self.db_path).parent
                logs_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = logs_dir / f"interactions_full_{timestamp}.csv"

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM interactions ORDER BY created_at DESC")
            rows = cursor.fetchall()

            if not rows:
                logger.warning("Нет диалогов для экспорта")
                conn.close()
                return filepath

            columns = [description[0] for description in cursor.description]

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            conn.close()
            logger.info(f"Экспорт диалогов выполнен: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}", exc_info=True)
            return ""

    def export_daily_csv(self, for_date: Optional[str] = None, filepath: Optional[str] = None) -> str:
        """Экспортирует диалоги за день в CSV."""
        try:
            if for_date is None:
                for_date = datetime.now().strftime("%Y-%m-%d")

            if filepath is None:
                logs_dir = Path(self.db_path).parent
                logs_dir.mkdir(exist_ok=True)
                filepath = logs_dir / f"interactions_{for_date}.csv"

            conn = self._get_connection()
            cursor = conn.cursor()

            date_start = f"{for_date}T00:00:00"
            date_end = f"{for_date}T23:59:59"

            cursor.execute("""
                SELECT * FROM interactions
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (date_start, date_end))

            rows = cursor.fetchall()

            if not rows:
                logger.warning(f"Нет диалогов за {for_date}")
                conn.close()
                return filepath

            columns = [description[0] for description in cursor.description]

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            conn.close()
            logger.info(f"Дневной экспорт диалогов: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка дневного экспорта: {e}", exc_info=True)
            return ""

    def cleanup_old_logs(self, days: Optional[int] = None) -> int:
        """Удаляет старые логи."""
        try:
            if days is None:
                days = LOG_RETENTION_DAYS

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM interactions WHERE timestamp < ?", (cutoff_date,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Удалено {deleted_count} диалогов старше {days} дней")
            return deleted_count

        except Exception as e:
            logger.error(f"Ошибка очистки: {e}", exc_info=True)
            return 0

