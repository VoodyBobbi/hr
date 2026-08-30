"""
Единая точка запуска всего проекта одной командой:

    python run_all.py

Что происходит по порядку, автоматически, без отдельных ручных шагов:
1. Пересборка FAISS-индекса (backend.build_index) — если data/faqs.json не
   менялся с прошлого запуска, пересборка пропускается (см. build_index.py,
   сравнение по хешу файла); если менялся — пересобирается автоматически.
2. Очистка старых строк логов (backend.logger.purge_old_logs) — по сроку
   хранения LOGS_RETENTION_DAYS из .env.
3. Запуск сайта (uvicorn) и Telegram-бота как двух процессов.

Папки для персональных данных (candidats/conversations/logs в HR_DATA_DIR,
см. backend/paths.py) создаются автоматически при первом импорте backend —
отдельно создавать их вручную не нужно.
"""
import subprocess
import sys


def main():
    print("Проверяю актуальность поискового индекса...")
    from backend import build_index
    build_index.main()

    print("Проверяю срок хранения логов...")
    from backend import logger
    logger.purge_old_logs()

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
