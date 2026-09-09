"""
Межпроцессная блокировка файла candidates.csv.

Зачем это нужно. В candidates.py уже есть threading.Lock, но он защищает
только от одновременной записи РАЗНЫХ ПОТОКОВ одного процесса. А сайт
(backend/app.py) и Telegram-бот (backend/telegram_bot.py) запускаются как два
ОТДЕЛЬНЫХ процесса ОС (см. run_all.py и start.sh) — threading.Lock между ними
не работает вообще, каждый процесс держит свой собственный объект блокировки.

Чем это грозило. Запись поля анкеты — это цикл «прочитать весь файл ->
изменить одно значение -> записать весь файл обратно». Если кандидат из
Telegram и кандидат с сайта отвечают на вопрос анкеты в одну и ту же секунду,
оба процесса читают одно и то же состояние, и тот, кто записал вторым,
затирает чужое изменение. Сам файл при этом остаётся целым (os.replace
атомарен), поэтому ошибка тихая: поле просто молча не сохраняется, и бот
задаёт тот же вопрос ещё раз.

Как решено. Блокировка средствами операционной системы на отдельном файле
candidates.csv.lock: fcntl.flock на Linux/macOS, msvcrt.locking на Windows.
Такая блокировка снимается ядром автоматически при завершении процесса, даже
аварийном, поэтому зависшего «мёртвого» замка после падения не остаётся —
именно поэтому выбран этот способ, а не самодельный файл-флаг.

Если заблокировать не удалось (истёк таймаут, платформа без обоих модулей,
файловая система без поддержки блокировок — например сетевая шара), код
НЕ падает и НЕ ждёт вечно, а печатает предупреждение и продолжает работу без
блокировки. Это осознанный выбор: деградация до прежнего поведения хуже, чем
идеал, но намного лучше, чем повисший бот, который перестал отвечать всем
кандидатам сразу.
"""
import contextlib
import os
import time

try:  # Linux / macOS
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - Linux / macOS
    msvcrt = None


LOCK_TIMEOUT_SECONDS = 10.0
_RETRY_INTERVAL_SECONDS = 0.05


def _try_acquire(file_obj) -> bool:
    """Одна неблокирующая попытка захватить блокировку. True — получилось."""
    if fcntl is not None:
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    if msvcrt is not None:
        try:
            # Блокируем один байт с текущей позиции — этого достаточно,
            # содержимое lock-файла не используется, важен сам факт замка.
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    return False


def _release(file_obj) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        # Снятие не удалось — ядро всё равно снимет замок при закрытии файла
        # ниже (contextlib.closing в cross_process_lock), молчим.
        pass


@contextlib.contextmanager
def cross_process_lock(target_path: str, timeout: float = LOCK_TIMEOUT_SECONDS):
    """Контекстный менеджер: блокирует <target_path>.lock на время блока.

    Пример:
        with cross_process_lock(CANDIDATES_PATH):
            ...чтение, изменение и запись файла...
    """
    lock_path = target_path + ".lock"

    try:
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        # "a+b" не обрезает файл и создаёт его, если ещё нет.
        lock_file = open(lock_path, "a+b")
    except OSError as e:
        print(f"[filelock] Не удалось открыть файл блокировки {lock_path} ({e}) — работаю без неё.")
        yield False
        return

    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            lock_file.seek(0)
            if _try_acquire(lock_file):
                acquired = True
                break
            if time.monotonic() >= deadline:
                print(
                    f"[filelock] Не дождался блокировки {lock_path} за {timeout:g} с — "
                    f"продолжаю без неё. Если это повторяется, проверьте, не остался ли "
                    f"висеть старый процесс сайта или бота."
                )
                break
            time.sleep(_RETRY_INTERVAL_SECONDS)

        yield acquired
    finally:
        if acquired:
            _release(lock_file)
        try:
            lock_file.close()
        except OSError:
            pass
