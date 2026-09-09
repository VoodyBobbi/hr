"""
Балльная система ограничения по поведению — дополняет (не заменяет) простой
лимит slowapi в app.py (N запросов в минуту с одного IP).

Разница: slowapi ограничивает ЧИСЛО запросов, не заботясь о их содержании —
20 разных осмысленных вопросов за минуту проходят так же, как 20 одинаковых
спам-сообщений. Модуль ниже смотрит на ПАТТЕРН поведения одного собеседника
и начисляет баллы за подозрительные признаки:

- повтор одного и того же сообщения подряд: +2 балла
- слишком частые сообщения (быстрее REPEAT_WINDOW_SECONDS одно за другим): +2 балла
- аномально длинное сообщение (длиннее ANOMALOUS_LENGTH_CHARS символов): +1 балл

При достижении BLOCK_THRESHOLD баллов — блокировка на BLOCK_DURATION_MINUTES минут.
Баллы у каждого кандидата свои, не суммируются между разными людьми.

Хранение — в памяти процесса (обычный dict, не файл и не внешняя БД). Это
осознанное упрощение: uvicorn в этом проекте запускается БЕЗ --workers (один
процесс, см. run_all.py/start.sh) — единственный процесс сайта видит все
запросы, конфликта с несколькими воркерами нет. Если это изменится (сайт
начнёт запускаться с несколькими воркерами) — потребуется вынести хранение
в общее место (Redis, файл с блокировкой) — просто dict в памяти каждого
отдельного воркера видел бы только свою часть трафика.

Не хранится на диске и не переживает перезапуск сервера — это нормально
для защиты от спама-в-моменте, а не долгосрочного учёта: перезапуск сервера
и так сбрасывает подозрительную активность, это не персональные данные,
которые нужно беречь между перезапусками.
"""
import threading
import time

# --- Настраиваемые пороги -----------------------------------------------
#
# Значения ниже — начальные ориентиры, не результат замера реального
# трафика (у проекта его ещё не было) — при необходимости смените без
# затрагивания остальной логики модуля.
REPEAT_WINDOW_SECONDS = 5.0      # быстрее этого между сообщениями = "частит"
ANOMALOUS_LENGTH_CHARS = 1500    # длиннее этого = "аномальная длина"
BLOCK_THRESHOLD = 5              # баллов для блокировки
BLOCK_DURATION_MINUTES = 5

REPEAT_MESSAGE_POINTS = 2
HIGH_FREQUENCY_POINTS = 2
ANOMALOUS_LENGTH_POINTS = 1

_lock = threading.Lock()

# key -> {"last_message": str, "last_time": float, "points": int, "blocked_until": float}
_state: dict[str, dict] = {}


def _key(source: str, external_id: str) -> str:
    """Ключ счётчика.

    Что подставляется вторым аргументом, решает вызывающий код, и это разное
    для двух каналов: Telegram передаёт chat_id (подделать его пользователь
    не может), а сайт — IP-адрес. Сайт раньше передавал session_id, но сессию
    выдаёт сам сервер любому запросу без cookie: достаточно было не хранить
    cookie, чтобы на каждое сообщение получать чистый счётчик и обходить
    защиту целиком (см. backend/app.py)."""
    return f"{source}:{external_id}"


def is_blocked(source: str, external_id: str) -> tuple[bool, float]:
    """Возвращает (заблокирован_ли_сейчас, секунд_до_снятия_блокировки).
    Вторая часть кортежа осмысленна только при заблокирован_ли_сейчас=True."""
    key = _key(source, external_id)
    with _lock:
        entry = _state.get(key)
        if not entry:
            return False, 0.0
        blocked_until = entry.get("blocked_until", 0.0)
        now = time.time()
        if blocked_until > now:
            return True, blocked_until - now
        return False, 0.0


def record_message(source: str, external_id: str, message: str) -> None:
    """Начисляет баллы за текущее сообщение по трём признакам (повтор,
    частота, длина) и, если порог достигнут, устанавливает блокировку.
    Вызывается ПОСЛЕ is_blocked (см. app.py и telegram_bot.py: сначала
    проверка блокировки, потом начисление баллов за текущее сообщение) — то
    есть сообщение, которое только что довело счётчик до порога, само ещё
    проходит, а блокируется уже следующее. Это осознанный компромисс
    простоты, не критичный для цели модуля: задача — остановить
    продолжающийся спам, а не поймать самое первое сообщение атаки.

    Прежняя редакция этого докстринга описывала порядок наоборот; сам код
    при этом не менялся."""
    key = _key(source, external_id)
    now = time.time()

    with _lock:
        entry = _state.setdefault(key, {"last_message": "", "last_time": 0.0, "points": 0, "blocked_until": 0.0})

        points_added = 0

        if entry["last_message"] and message == entry["last_message"]:
            points_added += REPEAT_MESSAGE_POINTS

        if entry["last_time"] and (now - entry["last_time"]) < REPEAT_WINDOW_SECONDS:
            points_added += HIGH_FREQUENCY_POINTS

        if len(message) > ANOMALOUS_LENGTH_CHARS:
            points_added += ANOMALOUS_LENGTH_POINTS

        entry["points"] += points_added
        entry["last_message"] = message
        entry["last_time"] = now

        if entry["points"] >= BLOCK_THRESHOLD:
            entry["blocked_until"] = now + BLOCK_DURATION_MINUTES * 60
            entry["points"] = 0  # сбрасываем счётчик — блокировка сама по себе уже наказание


