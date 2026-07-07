import subprocess
import sys

def main():
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