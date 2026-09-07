"""
Расположение файлов с персональными данными — ОТДЕЛЬНО от кода в git,
но физически внутри папки проекта.

Раньше анкеты, история переписки и логи лежали прямо в отслеживаемых git'ом
папках (`candidats/`, `data/conversations/`, `logs/`), из-за чего файл с
анкетами однажды попал в git и оказался в публичном GitHub-репозитории.

Защита от повторения — НЕ вынос данных за пределы папки проекта, а то, что
все папки с персональными данными добавлены в `.gitignore`: `git add .` их
не подхватывает, при `git push` они никогда не уходят на GitHub. Улетает
только код, сами данные остаются исключительно на диске пользователя.

По умолчанию (без HR_DATA_DIR):
- `candidates/` и `logs/` — на верхнем уровне папки проекта, рядом с
  `backend/`, `data/`, а не спрятаны во вложенной папке — чтобы HR сразу
  видел рабочую таблицу кандидатов и файл логов, не проваливаясь в
  подпапки. И то, и другое — то, во что HR регулярно заглядывает руками.
- `conversations/` (историю переписки — во внутреннюю логику бота никто
  вручную не заглядывает) остаётся в `HR/conversations/` — как раньше.

Расположение можно переопределить переменной окружения HR_DATA_DIR (в .env
или в переменных окружения ОС), если нужно хранить ВСЕ персональные данные
(включая candidates/ и logs/) в одном месте вне папки проекта — например,
на общем сетевом диске. Если HR_DATA_DIR задан, он имеет приоритет для
всех трёх папок, и верхнеуровневое расположение candidates/ и logs/,
описанное выше, не используется.

Папка с "мозгами" бота (data/faqs.json, data/system_prompt.md,
data/knowledge_base.json, поисковый индекс) — это часть кода/конфигурации,
не персональные данные, поэтому она никуда не переезжает и остаётся в
репозитории как раньше.
"""
import os
import shutil
import stat

from dotenv import load_dotenv

# load_dotenv() здесь, а не только в assistant.py/telegram_bot.py: paths.py
# может быть импортирован ДО того, как отработает load_dotenv() в
# вызывающем модуле (например, через "from . import candidates" раньше
# "load_dotenv()" по порядку строк) — тогда HR_DATA_DIR из .env ниже
# прочитался бы пустым. load_dotenv() безопасно вызывать повторно.
load_dotenv()

# Корень репозитория (папка, где лежит backend/, data/, requirements.txt и
# т.д.) — родитель папки backend/, где лежит этот файл.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Явно заданный HR_DATA_DIR имеет приоритет и переносит ВСЕ три папки в
# указанное место (см. докстринг выше). Без него — только conversations/
# идёт в HR/ внутри проекта; candidates/ и logs/ живут на верхнем уровне.
_hr_data_dir_override = os.environ.get("HR_DATA_DIR")
DATA_ROOT = os.path.abspath(_hr_data_dir_override or os.path.join(PROJECT_ROOT, "HR"))

if _hr_data_dir_override:
    CANDIDATS_DIR = os.path.join(DATA_ROOT, "candidates")
    LOGS_DIR = os.path.join(DATA_ROOT, "logs")
else:
    CANDIDATS_DIR = os.path.join(PROJECT_ROOT, "candidates")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

CONVERSATIONS_DIR = os.path.join(DATA_ROOT, "conversations")


def _ensure_private_dir(path: str) -> None:
    """Создаёт папку, если её нет, и по возможности ограничивает права доступа.

    chmod 700 реально ограничивает доступ на Linux/macOS (только владелец
    может читать содержимое). На Windows этот вызов, как правило, не даёт
    эффекта — там права нужно выставлять вручную через Свойства папки →
    Безопасность, ограничив доступ учётной записью администратора."""
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)  # rwx только для владельца
    except OSError:
        pass


def _migrate_legacy_dir(old_path: str, new_path: str, label: str) -> None:
    """Разовый перенос данных со старого расположения (candidates/ и logs/
    вложенными в HR/) на новое верхнеуровневое, при обновлении с версии, где
    HR_DATA_DIR не был задан. Переносит, только если старая папка есть, а
    новая — ещё нет (значит, перенос ещё не выполнялся, и новых данных,
    которые было бы страшно перезаписать, на новом месте ещё нет). Если и
    старая, и новая папки существуют — не трогает ничего: разрешать
    расхождение руками безопаснее, чем угадывать, что можно перезаписать.

    Обёрнуто в try/except: сайт и Telegram-бот запускаются как отдельные ОС
    процессы почти одновременно (run_all.py) и оба импортируют этот модуль
    при старте — если один уже перенёс папку, второй не должен упасть на
    ней, а просто пропустить перенос."""
    if os.path.isdir(old_path) and not os.path.exists(new_path):
        try:
            shutil.move(old_path, new_path)
            print(
                f"[paths] {label}: перенесено {old_path} -> {new_path} "
                f"(новое расположение по умолчанию, см. backend/paths.py)."
            )
        except OSError as e:
            print(
                f"[paths] {label}: не удалось перенести {old_path} -> {new_path} "
                f"({e}). Перенесите вручную, либо задайте HR_DATA_DIR={os.path.dirname(old_path)!r} "
                f"в .env, если хотите продолжать работать со старым расположением."
            )


if not _hr_data_dir_override:
    _migrate_legacy_dir(os.path.join(PROJECT_ROOT, "HR", "candidates"), CANDIDATS_DIR, "candidates")
    _migrate_legacy_dir(os.path.join(PROJECT_ROOT, "HR", "logs"), LOGS_DIR, "logs")


for _dir in (CANDIDATS_DIR, CONVERSATIONS_DIR, LOGS_DIR):
    _ensure_private_dir(_dir)


def conversation_path(source: str, external_id: str) -> str:
    """Путь к файлу истории переписки для (source, external_id).

    Общая для assistant.py (сохраняет/читает историю) и candidates.py
    (удаляет историю при delete_candidate) — вынесена сюда, а не в assistant.py,
    чтобы candidates.py не тянул за собой тяжёлые зависимости assistant.py
    (GigaChat, sentence-transformers) только ради построения пути."""
    safe_id = "".join(c if c.isalnum() else "_" for c in str(external_id))
    return os.path.join(CONVERSATIONS_DIR, f"{source}_{safe_id}.json")
