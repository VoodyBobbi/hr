"""
Единая команда установки проекта:

    python setup.py

Делает по порядку:
1. Ставит зависимости из requirements.txt (pip install).
2. Если .env ещё нет, но есть .env.example — копирует .env.example в .env
   (см. _ensure_env_file), чтобы дальше было куда писать ключ.
3. Если в .env уже есть непустой ENCRYPTION_KEY — ничего не трогает (ключ
   уже сгенерирован раньше, перезаписывать его нельзя: старый
   candidates.csv, если он уже есть, зашифрован именно этим ключом и без
   него станет нечитаемым, см. backend/crypto_utils.py).
4. Если ENCRYPTION_KEY пустой или отсутствует — генерирует новый ключ,
   сам записывает строку "ENCRYPTION_KEY=..." в .env и печатает большими
   буквами понятное сообщение с этим же ключом.

Это НЕ заменяет python run_all.py — setup.py нужно запустить один раз при
установке (или повторно, если .env потерялся), run_all.py запускает сам
чат-бот и Telegram-бота и запускается каждый раз при старте.
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")
REQUIREMENTS_PATH = os.path.join(PROJECT_ROOT, "requirements.txt")


def _install_dependencies():
    print("Устанавливаю зависимости из requirements.txt (это может занять несколько минут)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH]
    )
    if result.returncode != 0:
        print()
        print("ОШИБКА: не удалось установить зависимости (pip завершился с ошибкой выше).")
        print("Ключ шифрования НЕ генерируется, пока зависимости не установлены — "
              "сначала устраните ошибку pip и запустите 'python setup.py' ещё раз.")
        sys.exit(1)
    print("Зависимости установлены.")


def _ensure_env_file():
    """Если .env ещё нет, но есть .env.example — создаёт .env как копию
    .env.example, чтобы было куда дальше дописать ENCRYPTION_KEY. Если нет
    ни .env, ни .env.example — создаёт пустой .env с нуля (на случай, если
    .env.example тоже случайно не попал в поставку)."""
    if os.path.exists(ENV_PATH):
        return

    if os.path.exists(ENV_EXAMPLE_PATH):
        with open(ENV_EXAMPLE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        print(".env не найден — создаю его из .env.example.")
    else:
        content = ""
        print(".env и .env.example не найдены — создаю пустой .env.")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def _read_env_lines() -> list:
    if not os.path.exists(ENV_PATH):
        return []
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _get_env_value(lines: list, key: str) -> str:
    """Значение переменной key из уже прочитанных строк .env — так же, как
    это делает python-dotenv: последняя строка вида 'KEY=значение' (без
    учёта закомментированных строк, начинающихся с '#')."""
    value = ""
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith(prefix):
            value = stripped[len(prefix):].strip()
    return value


def _write_encryption_key(lines: list, new_key: str) -> list:
    """Возвращает НОВЫЙ список строк .env с добавленной/обновлённой строкой
    ENCRYPTION_KEY=... . Если строка ENCRYPTION_KEY= (пустая или с любым
    значением) уже есть в файле — заменяется именно она, на том же месте,
    остальные строки и их порядок не трогаются. Если такой строки нет
    вообще — новая строка добавляется в конец файла."""
    prefix = "ENCRYPTION_KEY="
    new_line = f"ENCRYPTION_KEY={new_key}"
    result = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if not replaced and stripped.startswith(prefix) and not stripped.startswith("#"):
            result.append(new_line)
            replaced = True
        else:
            result.append(line)
    if not replaced:
        result.append(new_line)
    return result


def _ensure_encryption_key():
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)

    lines = _read_env_lines()
    existing_key = _get_env_value(lines, "ENCRYPTION_KEY")

    if existing_key:
        print()
        print("Ключ шифрования (ENCRYPTION_KEY) в .env уже задан — оставляю без изменений.")
        print("(если нужен новый ключ — учтите: старые данные кандидатов, зашифрованные")
        print("текущим ключом, станут нечитаемыми без него; меняйте вручную только если")
        print("точно понимаете, что делаете)")
        return

    from cryptography.fernet import Fernet
    new_key = Fernet.generate_key().decode("utf-8")

    new_lines = _write_encryption_key(lines, new_key)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print()
    print("=" * 70)
    print("ЗАВИСИМОСТИ УСТАНОВЛЕНЫ. ВОТ ВАШ КЛЮЧ ДЛЯ ШИФРОВАНИЯ ДАННЫХ:")
    print()
    print(f"    {new_key}")
    print()
    print("Он уже сохранён в файл .env (строка ENCRYPTION_KEY) — ничего")
    print("дополнительно вставлять не нужно, программа найдёт его сама.")
    print()
    print("ВАЖНО: сохраните этот ключ ещё и отдельно — например, в менеджере")
    print("паролей или в заметке. Если файл .env будет потерян или удалён,")
    print("а ключа не останется больше нигде — расшифровать уже сохранённые")
    print("анкеты кандидатов будет невозможно.")
    print("=" * 70)


def _check_other_required_settings():
    """GIGACHAT_CREDENTIALS и ALLOWED_ORIGINS тоже обязательны для запуска
    (см. backend/assistant.py, backend/app.py), но setup.py не может
    сгенерировать их сам — это внешние значения (доступ к GigaChat API,
    домен сайта), которые администратор должен вписать в .env вручную.
    Просто напоминаем, если они ещё не заполнены, не блокируя установку."""
    lines = _read_env_lines()
    gigachat = _get_env_value(lines, "GIGACHAT_CREDENTIALS")
    origins = _get_env_value(lines, "ALLOWED_ORIGINS")

    missing = []
    if not gigachat or gigachat == "ваш_ключ_авторизации_GigaChat":
        missing.append("GIGACHAT_CREDENTIALS (доступ к GigaChat API)")
    if not origins:
        missing.append("ALLOWED_ORIGINS (с каких сайтов разрешено обращаться к чату)")

    if missing:
        print()
        print("Перед запуском (python run_all.py) заполните в .env ещё:")
        for item in missing:
            print(f"  - {item}")


def main():
    _install_dependencies()
    _ensure_env_file()
    _ensure_encryption_key()
    _check_other_required_settings()

    print()
    print("Установка завершена. Для запуска программы используйте:")
    print()
    print("    python run_all.py")


if __name__ == "__main__":
    main()
