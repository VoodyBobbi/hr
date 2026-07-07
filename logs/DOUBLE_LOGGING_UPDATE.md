# Структура системы логирования FAQ-бота

## 📁 Обновлённая структура папки `logs/`

```
logs/
├── logs/                           # НОВАЯ ПАПКА: двухуровневое логирование
│   ├── interactions/               # Логирование ДИАЛОГОВ (вопрос-ответ)
│   │   ├── config.py
│   │   ├── db_logger.py            # InteractionsLogger
│   │   ├── interactions.db         # БД диалогов
│   │   ├── interactions_*.csv      # Экспорты диалогов
│   │   └── __init__.py
│   │
│   ├── events/                     # Логирование СОБЫТИЙ (кто сделал что)
│   │   ├── config.py
│   │   ├── db_logger.py            # EventsLogger
│   │   ├── events.db               # БД событий
│   │   ├── events_*.csv            # Экспорты событий
│   │   └── __init__.py
│   │
│   ├── README.md                   # Документация (подробная)
│   ├── example_usage.py            # Пример использования
│   └── __init__.py
│
├── (остальное: старая система логирования)
├── db_logger.py
├── config.py
├── show_logs.py
├── example_usage.py
├── logs.db
├── README.md
├── QUICKSTART.md
├── .env.example
└── __init__.py
```

## ✨ Что нового

**В папке `logs/logs/` две независимые подсистемы:**

### 1. **`interactions/`** — диалоги пользователей

- Таблица: `interactions`
- Поля: `identifier` (кто), `query` (что спросил), `response` (ответ), `from_cache`, `response_time_ms`, `is_valid`, `retrieved_count`, `answer_source`
- БД: `interactions.db`
- Класс: `InteractionsLogger`
- Экспорт: `interactions_*.csv`

**Примеры:**
```
identifier: "john_doe"
query: "Какой график работы?"
response: "Пн-пт с 9:00 до 18:00"
answer_source: "giga"
```

### 2. **`events/`** — события и действия

- Таблица: `events`
- Поля: `identifier` (кто), `action_type` (тип действия), `action_name` (что сделал), `status` (результат), `details` (JSON), `error_message`, `duration_ms`
- БД: `events.db`
- Класс: `EventsLogger`
- Экспорт: `events_*.csv`

**Примеры:**
```
identifier: "john_doe" | action: "Отправил вопрос в Telegram"
identifier: "assistant_system" | action: "Преобразовать вопрос в вектор" (45 мс)
identifier: "assistant_system" | action: "Поиск похожих вопросов в FAISS" (12 мс)
identifier: "giga_model" | action: "Генерация ответа" (1250 мс)
identifier: "telegram_bot" | action: "Отправить ответ в Telegram" (5 мс)
```

## 🔑 Ключевое отличие: `identifier`

**Обязательное поле в обеих таблицах** — для идентификации «КТО»:

- В `interactions`: ник пользователя из Telegram (или иной опознавательный признак)
- В `events`: ник пользователя, компонента системы (giga_model, assistant_system, telegram_bot и т.д.)

**Примеры идентификаторов:**
- `john_doe` — пользователь из Telegram
- `alice_user` — другой пользователь
- `giga_model` — компонент GigaChat
- `assistant_system` — система обработки
- `telegram_bot` — Telegram бот
- `cache_service` — кэш-сервис
- `operator_support` — оператор поддержки

## 📊 Протестировано

Пример отработал успешно:
- ✅ 2 диалога записаны (john_doe, alice_user)
- ✅ 9 событий записано (для разных компонентов)
- ✅ Статистика доступна по обеим таблицам
- ✅ CSV-экспорты созданы в каждой папке
- ✅ Полная история каждого компонента доступна

## 🚀 Использование

### Импорт обеих систем

```python
from logs.logs.interactions import InteractionsLogger
from logs.logs.events import EventsLogger

interactions_logger = InteractionsLogger()
events_logger = EventsLogger()
```

### Логирование диалога

```python
interactions_logger.log_interaction(
    identifier="john_doe",
    query="Какие вакансии?",
    response="У нас есть вакансии...",
    source="telegram",
    response_time_ms=1312,
    is_valid=True,
    retrieved_count=3,
    answer_source="giga"
)
```

### Логирование события

```python
events_logger.log_event(
    identifier="assistant_system",
    action_type="vectorization",
    action_name="Преобразовать вопрос в вектор",
    status="completed",
    duration_ms=45,
    details={"model": "paraphrase-MiniLM", "dim": 384}
)
```

### Получение статистики

```python
interactions_stats = interactions_logger.get_stats()
events_stats = events_logger.get_stats()
```

### История по идентификатору

```python
# История диалогов john_doe
history = interactions_logger.get_history("john_doe", limit=10)

# История действий giga_model
events = events_logger.get_user_events("giga_model", limit=20)

# Все ошибки в системе
failed = events_logger.get_failed_events(limit=10)
```

### Экспорт в CSV

```python
# Диалоги
interactions_logger.export_to_csv()        # → interactions_full_*.csv
interactions_logger.export_daily_csv()     # → interactions_YYYY-MM-DD.csv

# События
events_logger.export_to_csv()              # → events_full_*.csv
events_logger.export_daily_csv()           # → events_YYYY-MM-DD.csv
```

## 📖 Документация

- **`logs/logs/README.md`** — полная документация двухуровневой системы
- **`logs/logs/example_usage.py`** — рабочий пример (протестирован ✅)

## 🔗 Логирование в коде

Типичный сценарий:

```python
# Пользователь отправил вопрос
events_logger.log_event("john_doe", "user_action", "Отправил вопрос")

# Система обрабатывает
events_logger.log_event("assistant_system", "vectorization", "Преобразовать в вектор", duration_ms=45)

# Поиск в БД
events_logger.log_event("assistant_system", "search", "Поиск в FAISS", duration_ms=12, 
    details={"ids": [1,2,3], "distances": [0.12, 0.45, 0.67]})

# Обращение к модели
events_logger.log_event("giga_model", "model_call", "Генерация ответа", duration_ms=1250)

# Ответ отправлен
events_logger.log_event("telegram_bot", "response_sent", "Отправить ответ", duration_ms=5)

# Финальная запись диалога
interactions_logger.log_interaction(
    identifier="john_doe",
    query="вопрос",
    response="ответ",
    response_time_ms=1312,  # сумма всех шагов
    answer_source="giga"
)
```

## 📝 Основные отличия от старой системы

| Параметр | Старая (`logs/logs.db`) | Новая (`logs/logs/interactions.db` + `logs/logs/events.db`) |
|----------|-------------------------|----------------------------------------------|
| Таблиц | 1 | 2 |
| Диалоги | ✅ logs | ✅ interactions |
| События | ❌ Нет | ✅ events |
| Идентификация | user_id + username | identifier (универсальный) |
| Детали событий | ❌ Нет | ✅ JSON в details |
| История компонентов | ❌ Нет | ✅ События каждого компонента |
| Ошибки | ❌ Нет поля | ✅ error_message + status |
| Длительность | response_time_ms (итого) | duration_ms (каждого шага) |

## ✅ Готово

Обе папки и все файлы созданы, протестированы и готовы к интеграции!

