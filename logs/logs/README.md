# Система логирования FAQ-бота (двухуровневая)

Полная система логирования с разделением на два типа логов:
1. **Диалоги** (interactions) — кто спросил, что спросил, какой ответ
2. **События** (events) — кто сделал, что сделал, какой результат

## Структура папок

```
logs/
├── logs/
│   ├── interactions/     # Логирование диалогов (вопрос-ответ)
│   │   ├── config.py
│   │   ├── db_logger.py (InteractionsLogger)
│   │   ├── interactions.db
│   │   ├── interactions_*.csv (экспорты)
│   │   └── __init__.py
│   │
│   ├── events/           # Логирование событий (действий)
│   │   ├── config.py
│   │   ├── db_logger.py (EventsLogger)
│   │   ├── events.db
│   │   ├── events_*.csv (экспорты)
│   │   └── __init__.py
│   │
│   ├── example_usage.py
│   └── __init__.py
│
├── (остальные файлы логирования в корне logs/)
```

## Две таблицы логирования

### 1. Таблица `interactions` (диалоги)

Логирует каждый диалог пользователя с ботом.

| Поле | Тип | Описание |
|------|-----|---------|
| `id` | INTEGER | Автоинкремент |
| `timestamp` | TEXT | ISO-время запроса |
| **`identifier`** | TEXT | **Ник из Telegram или опознавательный признак (ОБЯЗАТЕЛЕН)** |
| `source` | TEXT | Источник (telegram, console, api) |
| `query` | TEXT | Вопрос пользователя |
| `response` | TEXT | Ответ бота |
| `from_cache` | INTEGER | 0/1 — из кэша ли |
| `response_time_ms` | INTEGER | Время ответа в мс |
| `is_valid` | INTEGER | 0/1/NULL — валидный ли ответ |
| `retrieved_count` | INTEGER | Сколько похожих найдено |
| `answer_source` | TEXT | Источник ответа (giga, cache, fallback) |
| `created_at` | TIMESTAMP | Время создания записи |

**Пример записи:**
```
identifier: "john_doe"
query: "Какой график работы?"
response: "Пн-пт с 9:00 до 18:00"
from_cache: 0
response_time_ms: 1312
is_valid: 1
retrieved_count: 3
answer_source: "giga"
```

### 2. Таблица `events` (события/действия)

Логирует каждое действие в системе (кто сделал, что, с каким результатом).

| Поле | Тип | Описание |
|------|-----|---------|
| `id` | INTEGER | Автоинкремент |
| `timestamp` | TEXT | ISO-время события |
| **`identifier`** | TEXT | **Кто сделал (username, гигачат, система и т.д.) (ОБЯЗАТЕЛЕН)** |
| `action_type` | TEXT | Тип действия (query_process, vectorization, search, model_call, error и т.д.) |
| `action_name` | TEXT | Описание действия (что сделал) |
| `status` | TEXT | Статус (completed, failed, pending, error) |
| `result` | TEXT | Результат действия |
| `details` | TEXT | JSON с дополнительными деталями |
| `error_message` | TEXT | Сообщение об ошибке, если была |
| `duration_ms` | INTEGER | Длительность в мс |
| `created_at` | TIMESTAMP | Время создания записи |

**Примеры событий:**
```
identifier: "john_doe" | action_type: "user_action" | action_name: "Отправил вопрос в Telegram"
identifier: "assistant_system" | action_type: "vectorization" | action_name: "Преобразовать в вектор"
identifier: "assistant_system" | action_type: "search" | action_name: "Поиск в FAISS"
identifier: "giga_model" | action_type: "model_call" | action_name: "Генерация ответа"
identifier: "telegram_bot" | action_type: "response_sent" | action_name: "Отправить ответ"
identifier: "giga_model" | action_type: "model_call" | action_name: "Обращение к модели" | status: "error"
```

## Использование

### Импорт

```python
from logs.logs.interactions import InteractionsLogger
from logs.logs.events import EventsLogger

interactions = InteractionsLogger()
events = EventsLogger()
```

### Логирование диалога

```python
interaction_id = interactions.log_interaction(
    identifier="john_doe",              # ник из Telegram
    query="Какие вакансии?",
    response="У нас есть вакансии для разработчиков",
    source="telegram",
    from_cache=False,
    response_time_ms=1312,
    is_valid=True,
    retrieved_count=3,
    answer_source="giga"
)
```

### Логирование события

```python
event_id = events.log_event(
    identifier="assistant_system",          # кто сделал
    action_type="vectorization",            # тип действия
    action_name="Преобразовать вопрос в вектор",  # что сделал
    status="completed",                     # статус
    result="Успешно",
    duration_ms=45,
    details={"model": "paraphrase-MiniLM", "dim": 384}
)
```

### Получение статистики

```python
# Статистика диалогов
stats = interactions.get_stats()
# {'total_interactions': 10, 'from_cache': 2, 'unique_users': 5, ...}

# Статистика событий
stats = events.get_stats()
# {'total_events': 50, 'unique_actors': 8, 'completed': 45, 'failed': 5, ...}
```

### История пользователя

```python
# Последние 10 диалогов john_doe
history = interactions.get_history("john_doe", limit=10)

# Последние 20 событий от john_doe
user_events = events.get_user_events("john_doe", limit=20)

# Все ошибочные события
failed = events.get_failed_events(limit=10)
```

### Экспорт в CSV

```python
# Экспорт всех диалогов
interactions.export_to_csv()           # → interactions_full_YYYYMMDD_HHMMSS.csv
interactions.export_daily_csv()        # → interactions_2025-07-07.csv

# Экспорт всех событий
events.export_to_csv()                 # → events_full_YYYYMMDD_HHMMSS.csv
events.export_daily_csv()              # → events_2025-07-07.csv
```

## Пример интеграции в бот

```python
from logs.logs.interactions import InteractionsLogger
from logs.logs.events import EventsLogger

interactions_logger = InteractionsLogger()
events_logger = EventsLogger()

async def handle_message(update, context):
    username = update.message.from_user.username
    query = update.message.text
    
    # События
    events_logger.log_event(
        identifier=username,
        action_type="user_action",
        action_name="Отправил вопрос",
        status="completed"
    )
    
    events_logger.log_event(
        identifier="assistant_system",
        action_type="vectorization",
        action_name="Преобразовать вопрос в вектор",
        status="completed",
        duration_ms=45
    )
    
    # Получить ответ...
    response = get_answer(query)
    
    events_logger.log_event(
        identifier="giga_model",
        action_type="model_call",
        action_name="Генерация ответа",
        status="completed",
        duration_ms=1200
    )
    
    # Логирование диалога
    interactions_logger.log_interaction(
        identifier=username,
        query=query,
        response=response,
        source="telegram",
        from_cache=False,
        response_time_ms=1245,
        is_valid=True,
        answer_source="giga"
    )
    
    await update.message.reply_text(response)
```

## CLI-команды (планируется)

Разработка CLI-интерфейса для просмотра логов:

```bash
python -m logs.logs.interactions stats
python -m logs.logs.interactions history --identifier john_doe
python -m logs.logs.interactions export

python -m logs.logs.events stats
python -m logs.logs.events history --identifier giga_model
python -m logs.logs.events failed-events
```

## Требования

- Python 3.7+
- SQLite 3 (встроенная в Python)

## Протестировано

Запустите пример:
```bash
python logs/logs/example_usage.py
```

Результат:
- 2 диалога записаны
- 8+ событий записано
- Статистика доступна
- CSV-экспорты созданы

## Преимущества двухуровневой системы

✅ **Четкое разделение:**
- Диалоги = основной результат (вопрос-ответ)
- События = детали обработки (кто что делал)

✅ **Полная история:**
- В `events` видна полная цепочка действий при обработке одного запроса
- Можно отследить каждый шаг: векторизация → поиск → модель → отправка

✅ **Идентификация:**
- Каждая запись имеет `identifier` (кто)
- Можно искать по пользователям, компонентам системы, моделям

✅ **Отладка:**
- При ошибке сразу видны все события, которые привели к ней
- Легко найти на каком этапе произошла проблема

✅ **Аналитика:**
- Отдельная статистика для диалогов и событий
- Можно анализировать производительность каждого компонента

## Файлы в папках

### logs/logs/interactions/
- `db_logger.py` — класс InteractionsLogger
- `config.py` — конфиг для interactions
- `interactions.db` — БД SQLite
- `interactions_*.csv` — экспорты

### logs/logs/events/
- `db_logger.py` — класс EventsLogger
- `config.py` — конфиг для events
- `events.db` — БД SQLite
- `events_*.csv` — экспорты

## Конфигурация через .env

```env
# Путь к БД диалогов
FAQ_INTERACTIONS_DB=logs/logs/interactions/interactions.db

# Путь к БД событий
FAQ_EVENTS_DB=logs/logs/events/events.db

# Общие параметры
FAQ_LOG_RETENTION_DAYS=30
FAQ_ENABLE_WAL=true
FAQ_DB_TIMEOUT=10
```

