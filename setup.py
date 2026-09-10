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
# Оба имени: часть архиваторов и почтовых клиентов при распаковке теряет
# ведущую точку, и файл приезжает как env.example. Раньше setup.py его не
# находил и создавал ПУСТОЙ .env — без единой заготовки, из-за чего человек
# не понимал, что вообще нужно заполнять.
_ENV_EXAMPLE_NAMES = (".env.example", "env.example")
ENV_EXAMPLE_PATH = next(
    (os.path.join(PROJECT_ROOT, n) for n in _ENV_EXAMPLE_NAMES
     if os.path.exists(os.path.join(PROJECT_ROOT, n))),
    os.path.join(PROJECT_ROOT, ".env.example"),
)
REQUIREMENTS_PATH = os.path.join(PROJECT_ROOT, "requirements.txt")


def _install_gigachat_certificate():
    """GigaChat API использует сертификаты, выданные российским НУЦ
    Минцифры — этот корневой сертификат НЕ входит по умолчанию в системные
    доверенные хранилища (ни ОС, ни Python/certifi). Без него запрос к
    GigaChat падает с ошибкой "[SSL: CERTIFICATE_VERIFY_FAILED]... self-
    signed certificate in certificate chain" (см. официальную документацию:
    https://developers.sber.ru/docs/ru/gigachat/certificates) — это не
    ошибка конфигурации проекта, а следствие российской PKI-инфраструктуры,
    через которую работает GigaChat.

    Раньше это обходили передачей verify_ssl_certs=False в assistant.py —
    рабочий, но НЕБЕЗОПАСНЫЙ способ: он отключает проверку TLS-сертификата
    ЦЕЛИКОМ, а не только для GigaChat — при подмене сервера (MITM, например
    в незащищённой Wi-Fi сети) программа не отличит настоящий GigaChat от
    поддельного и молча отправит туда данные кандидатов.

    Правильный путь (используется здесь) — установить сам сертификат
    Минцифры в доверенное хранилище certifi ОДИН РАЗ, тогда обычная (по
    умолчанию включённая) проверка TLS начинает работать корректно, не
    ломая соединение — команда ниже взята из официальной документации SDK:
    https://pypi.org/project/gigachat/ (раздел "Установка корневого
    сертификата Минцифры России").

    Идемпотентно: если сертификат уже был добавлен раньше (повторный запуск
    setup.py) — не добавляет его повторно, определяя это по уникальной
    подстроке в самом сертификате."""
    try:
        import certifi
    except ImportError:
        # certifi ставится транзитивно вместе с httpx (зависимость
        # gigachat/fastapi) — если он всё ещё недоступен, значит
        # _install_dependencies() выше не отработал как ожидалось; не
        # валим всю установку из-за этого, просто предупреждаем.
        print()
        print("ВНИМАНИЕ: не удалось найти модуль certifi — сертификат Минцифры")
        print("не установлен автоматически. GigaChat может не заработать.")
        print("Установите вручную по инструкции:")
        print("https://developers.sber.ru/docs/ru/gigachat/certificates")
        return

    cert_bundle_path = certifi.where()
    # Уникальная подстрока именно этого сертификата (Common Name корневого
    # сертификата НУЦ Минцифры) — используется как маркер "уже установлен",
    # а не просто проверка "файл существует", потому что cacert.pem — общий
    # файл со множеством других сертификатов внутри.
    marker = "Russian Trusted Root CA"

    with open(cert_bundle_path, "r", encoding="utf-8", errors="ignore") as f:
        already_installed = marker in f.read()

    if already_installed:
        print("Сертификат НУЦ Минцифры для GigaChat уже установлен — пропускаю.")
        return

    print("Устанавливаю корневой сертификат НУЦ Минцифры (нужен для GigaChat)...")
    try:
        import urllib.request
        import ssl

        cert_url = "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer"
        # Официальная команда использует curl -k (без проверки TLS для ЭТОЙ
        # конкретной загрузки) — тот же принцип воспроизведён здесь через
        # ssl._create_unverified_context(): само же скачивание сертификата,
        # которым мы собираемся ПОЧИНИТЬ проверку сертификатов, не может
        # зависеть от уже работающей проверки сертификатов (это ещё не
        # установлено на этом шаге) — курица и яйцо. Риск здесь ограничен:
        # сама подмена ответа на этот конкретный запрос дала бы поддельный
        # сертификат, который добавился бы в доверенные, но URL жёстко
        # захардкожен на официальный государственный домен gu-st.ru, не
        # приходит откуда-либо ещё (не из .env, не из аргументов) и не
        # может быть подменён через конфигурацию проекта.
        unverified_context = ssl._create_unverified_context()
        with urllib.request.urlopen(cert_url, context=unverified_context, timeout=15) as response:
            cert_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"Не удалось скачать сертификат Минцифры автоматически: {e}")
        print("GigaChat может не заработать. Установите сертификат вручную:")
        print("https://developers.sber.ru/docs/ru/gigachat/certificates")
        return

    with open(cert_bundle_path, "a", encoding="utf-8") as f:
        f.write("\n" + cert_data)

    print("Сертификат НУЦ Минцифры установлен.")


def _install_dependencies():
    """Ставит зависимости молча (-q).

    Без -q pip печатает по строке "Requirement already satisfied" на каждый
    из полусотни пакетов при КАЖДОМ запуске — полезной информации ноль, а
    важные сообщения ниже тонут в этой простыне. Ошибки -q не скрывает: при
    сбое pip всё равно напечатает причину, и мы её увидим."""
    print("Проверяю зависимости...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_PATH]
    )
    if result.returncode != 0:
        print()
        print("ОШИБКА: не удалось установить зависимости (причина в выводе pip выше).")
        sys.exit(1)


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
        # Подробное предупреждение про потерю ключа печатается один раз в
        # конце запуска (run_all.print_encryption_key_notice) — дублировать
        # его здесь значит показывать одно и то же дважды за один старт.
        return

    from cryptography.fernet import Fernet
    new_key = Fernet.generate_key().decode("utf-8")

    new_lines = _write_encryption_key(lines, new_key)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print("Создан новый ключ шифрования, записан в .env.")


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
        print("Заполните в .env:")
        for item in missing:
            print(f"  - {item}")


def main(standalone: bool = True):
    """Установка: зависимости, сертификат GigaChat, .env, ключ шифрования.

    standalone=False передаёт start.py — он сразу продолжает запуском, и
    подсказка "теперь выполните python run_all.py" в этом случае не только
    лишняя, но и сбивает с толку: выглядит так, будто нужна вторая команда,
    хотя проект уже стартует следующей строкой."""
    _install_dependencies()
    _install_gigachat_certificate()
    _ensure_env_file()
    _ensure_encryption_key()
    _check_other_required_settings()

    if standalone:
        print()
        print("Установка завершена. Для запуска: python start.py")


if __name__ == "__main__":
    main()
