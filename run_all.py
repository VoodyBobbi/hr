"""
Единая точка запуска всего проекта одной командой:

    python run_all.py

Что происходит по порядку, автоматически, без отдельных ручных шагов:
1. Проверка ключа шифрования (ENCRYPTION_KEY в .env) — без него сохранить
   анкету кандидата невозможно (см. backend/crypto_utils.py). Если не задан,
   выводится понятное сообщение и процесс не стартует — лучше остановиться
   сразу, чем упасть на первом сообщении кандидата (см. _check_encryption_key
   ниже). Сам ключ первым делом генерируется командой `python setup.py`
   (см. README, раздел «Установка и запуск») — run_all.py его не создаёт,
   только проверяет, что он уже на месте.
2. Пересборка векторного индекса (backend.build_index) — если ничего в
   data/ не менялось с прошлого запуска, пересборка пропускается (см.
   build_index.py, сравнение по хешу файлов); если что-то новое появилось
   или изменилось — пересобирается автоматически.
3. Очистка старых строк логов (backend.logger.purge_old_logs) — по сроку
   хранения LOGS_RETENTION_DAYS из .env.
4. Запуск сайта (uvicorn) и Telegram-бота как двух процессов.

Папки для персональных данных создаются автоматически при первом импорте
backend, вручную их заводить не нужно. По умолчанию все три лежат в корне
проекта: candidates/ (анкеты), logs/ (лог обращений), history/ (история
переписки). Прежняя промежуточная папка HR/ больше не используется, её
содержимое переносится автоматически — см. backend/paths.py.
"""
import os
import subprocess
import sys


def _check_encryption_key():
    """Останавливает запуск с понятным сообщением, если ENCRYPTION_KEY не
    задан — без него candidates.py упадёт при первом же сообщении кандидата
    (см. backend/crypto_utils.py), а не при старте сервера. Лучше сказать
    об этом сразу и явно, чем дать администратору решить, что всё работает,
    пока не пришёл первый кандидат."""
    from dotenv import load_dotenv
    load_dotenv()

    if os.environ.get("ENCRYPTION_KEY", "").strip():
        return

    print()
    print("=" * 70)
    print("ОШИБКА: ключ шифрования (ENCRYPTION_KEY) не найден в .env.")
    print()
    print("Без этого ключа программа не сможет сохранять анкеты кандидатов.")
    print("Сгенерируйте его командой:")
    print()
    print("    python start.py")
    print()
    print("(эта же команда поставит зависимости, установит сертификат")
    print("GigaChat и запустит проект — всё за один раз)")
    print("=" * 70)
    sys.exit(1)


def print_encryption_key_notice():
    """Показывает ключ шифрования анкет перед запуском серверов.

    Печатается ЗДЕСЬ, а не в самом конце main(), потому что после запуска
    серверов main() блокируется на p.wait() до остановки проекта — сообщение
    "в конце" кандидат увидел бы только при выключении, то есть никогда в
    обычной работе. Это последнее, что печатает сам проект: ниже пойдут уже
    логи uvicorn и Telegram-бота.

    Ключ показывается целиком намеренно: он нужен администратору, чтобы
    сохранить его в надёжном месте. Терять этот кусок вывода в скриншотах и
    пересылках не стоит — по нему расшифровывается вся таблица кандидатов.
    Если показ полного ключа нежелателен, замените строку с key ниже на
    маскированный вывод: key[:6] + "…" + key[-4:]."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("ENCRYPTION_KEY", "").strip()

    print()
    print("=" * 70)
    print("СОХРАНИТЕ Ключ шифрования анкет кандидатов.")
    print("ВНИМАНИЕ: потеряете ключ — не прочитаете уже собранные анкеты.")
    print()
    print(f"ENCRYPTION_KEY={key}")
    print()
    print("Он же записан в файле .env в корне проекта.")
    print("=" * 70)
    print()


def main():
    _check_encryption_key()

    print("Проверяю актуальность поискового индекса...")
    from backend import build_index
    build_index.main()

    print("Проверяю срок хранения логов...")
    from backend import logger
    logger.purge_old_logs()

    print_encryption_key_notice()

    processes = []

    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ])
    processes.append(server_process)

    bot_process = subprocess.Popen([
        sys.executable, "-m", "backend.telegram_bot",
    ])
    processes.append(bot_process)

    print("Запущено: сайт (http://localhost:8000) и Telegram-бот.")
    print("Нажмите Ctrl+C, чтобы остановить всё.")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("Остановка всех процессов...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
