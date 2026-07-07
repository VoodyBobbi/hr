#!/usr/bin/env python3
"""
Пример использования двухуровневой системы логирования.
1. InteractionsLogger - логирование диалогов (вопрос-ответ)
2. EventsLogger - логирование действий (кто что сделал)
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logs.logs.interactions import InteractionsLogger
from logs.logs.events import EventsLogger


def main():
    print("=" * 80)
    print("  ПРИМЕР: ДВУХУРОВНЕВОЕ ЛОГИРОВАНИЕ")
    print("=" * 80)

    # Инициализация обеих подсистем
    interactions_logger = InteractionsLogger()
    events_logger = EventsLogger()
    print("✅ Обе подсистемы инициализированы\n")

    # ========== СЦЕНАРИЙ 1: Пользователь задаёт вопрос ==========
    print("1️⃣  СЦЕНАРИЙ: Пользователь john_doe задаёт вопрос\n")

    # Событие: пользователь отправил сообщение
    events_logger.log_event(
        identifier="john_doe",
        action_type="user_action",
        action_name="Отправил вопрос в Telegram",
        status="completed",
        result="Вопрос получен",
    )
    print("   📝 Событие: Пользователь отправил вопрос")

    # Событие: система начала обработку
    events_logger.log_event(
        identifier="assistant_system",
        action_type="processing",
        action_name="Запустить обработку запроса",
        status="completed",
        result="Обработка начата",
    )
    print("   🔄 Событие: Система начала обработку")

    # Событие: векторизация запроса
    events_logger.log_event(
        identifier="assistant_system",
        action_type="vectorization",
        action_name="Преобразовать вопрос в вектор",
        status="completed",
        result="Успешно",
        duration_ms=45,
        details={"model": "paraphrase-multilingual-MiniLM-L12-v2", "dim": 384}
    )
    print("   ⚡ Событие: Вопрос преобразован в вектор (45 мс)")

    # Событие: поиск в FAISS индексе
    events_logger.log_event(
        identifier="assistant_system",
        action_type="search",
        action_name="Поиск похожих вопросов в FAISS",
        status="completed",
        result="Найдено 3 похожих вопроса",
        duration_ms=12,
        details={
            "index": "faiss_index.bin",
            "k": 3,
            "ids": [1, 2, 3],
            "distances": [0.12, 0.45, 0.67]
        }
    )
    print("   🔍 Событие: Найдено 3 похожих вопроса в БД")

    # Событие: вызов модели GigaChat
    events_logger.log_event(
        identifier="giga_model",
        action_type="model_call",
        action_name="Генерация ответа моделью GigaChat",
        status="completed",
        result="Ответ успешно получен",
        duration_ms=1250,
        details={"model": "GigaChat", "tokens": 150}
    )
    print("   🤖 Событие: GigaChat сгенерировал ответ (1250 мс)")

    # Событие: отправка ответа пользователю
    events_logger.log_event(
        identifier="telegram_bot",
        action_type="response_sent",
        action_name="Отправить ответ в Telegram",
        status="completed",
        result="Ответ отправлен",
        duration_ms=5,
        details={"chat_id": 123456, "message_id": 42}
    )
    print("   ✉️  Событие: Ответ отправлен пользователю\n")

    # Логирование диалога
    interaction_id = interactions_logger.log_interaction(
        identifier="john_doe",
        query="Какой график работы компании?",
        response="Мы работаем пн-пт с 9:00 до 18:00, сб-вс выходные.",
        source="telegram",
        from_cache=False,
        response_time_ms=1312,  # сумма всех компонентов
        is_valid=True,
        retrieved_count=3,
        answer_source="giga"
    )
    print(f"   💬 Диалог записан (ID: {interaction_id})")

    # ========== СЦЕНАРИЙ 2: Ошибка при обработке ==========
    print("\n2️⃣  СЦЕНАРИЙ: Ошибка при обращении к модели\n")

    # Событие: попытка обратиться к GigaChat
    events_logger.log_event(
        identifier="giga_model",
        action_type="model_call",
        action_name="Попытка обращения к GigaChat",
        status="error",
        result="Ошибка соединения",
        duration_ms=5000,
        error_message="Connection timeout after 5 seconds",
        details={"attempt": 1, "timeout": 5000}
    )
    print("   ⚠️  Событие: Ошибка соединения с GigaChat")

    # Событие: retry (повторная попытка)
    events_logger.log_event(
        identifier="assistant_system",
        action_type="retry",
        action_name="Повторная попытка обращения к модели",
        status="pending",
        result="Повторная попытка выполняется",
        details={"attempt": 2, "delay_ms": 1000}
    )
    print("   🔄 Событие: Система попыталась повторно (попытка 2)")

    # Событие: успех со второй попытки
    events_logger.log_event(
        identifier="giga_model",
        action_type="model_call",
        action_name="Генерация ответа (повторная попытка)",
        status="completed",
        result="Ответ получен",
        duration_ms=800,
        details={"attempt": 2}
    )
    print("   ✅ Событие: Со второй попытки ответ получен успешно\n")

    # Диалог с fallback-ом
    interaction_id2 = interactions_logger.log_interaction(
        identifier="alice_user",
        query="Есть ли удалённая работа?",
        response="Да, мы предоставляем возможность удалённой работы для некоторых позиций.",
        source="telegram",
        from_cache=False,
        response_time_ms=6800,
        is_valid=True,
        retrieved_count=2,
        answer_source="giga_retry"
    )
    print(f"   💬 Диалог с retry записан (ID: {interaction_id2})")

    # ========== СТАТИСТИКА ==========
    print("\n" + "=" * 80)
    print("  СТАТИСТИКА")
    print("=" * 80)

    interactions_stats = interactions_logger.get_stats()
    events_stats = events_logger.get_stats()

    print(f"\n📊 Диалоги (interactions):")
    for key, value in interactions_stats.items():
        print(f"   • {key}: {value}")

    print(f"\n📊 События (events):")
    for key, value in events_stats.items():
        print(f"   • {key}: {value}")

    # ========== ЭКСПОРТ ==========
    print("\n" + "=" * 80)
    print("  ЭКСПОРТ В CSV")
    print("=" * 80)

    interactions_csv = interactions_logger.export_to_csv()
    print(f"\n✅ Экспорт диалогов: {interactions_csv}")

    events_csv = events_logger.export_to_csv()
    print(f"✅ Экспорт событий: {events_csv}")

    # ========== ИСТОРИЯ ==========
    print("\n" + "=" * 80)
    print("  ИСТОРИЯ")
    print("=" * 80)

    print("\n📖 Диалоги пользователя john_doe:")
    john_interactions = interactions_logger.get_history("john_doe", limit=5)
    for i, interaction in enumerate(john_interactions, 1):
        print(f"   {i}. [{interaction['timestamp']}] {interaction['query'][:50]}...")

    print("\n📖 События пользователя/системы john_doe:")
    john_events = events_logger.get_user_events("john_doe", limit=5)
    for i, event in enumerate(john_events, 1):
        print(f"   {i}. [{event['timestamp']}] {event['action_name']} - {event['status']}")

    print("\n" + "=" * 80)
    print("  ✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()

