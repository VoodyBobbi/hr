"""
Подсистема логирования взаимодействий (вопрос-ответ).
Логирует основные диалоги пользователей с FAQ-ботом.
"""

from .interactions.db_logger import InteractionsLogger

__all__ = ["InteractionsLogger"]

