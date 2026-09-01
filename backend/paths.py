"""
Расположение файлов с персональными данными — ОТДЕЛЬНО от кода в git,
но физически внутри папки проекта.

Раньше анкеты, история переписки и логи лежали прямо в отслеживаемых git'ом
папках (`candidats/`, `data/conversations/`, `logs/`), из-за чего файл с
анкетами однажды попал в git и оказался в публичном GitHub-репозитории.

Защита от повторения — НЕ вынос данных за пределы папки проекта, а то, что
папка с персональными данными (по умолчанию `HR/candidates` рядом с
`backend/`, `data/` и т.д.) добавлена в `.gitignore`: `git add .` её не
подхватывает, при `git push` она никогда не уходит на GitHub. Улетает
только код, сама таблица кандидатов остаётся исключительно на диске
пользователя. Расположение можно переопределить переменной окружения
HR_DATA_DIR (в .env или в переменных окружения ОС), если нужно хранить
данные в другом месте — например, вне папки проекта, на общем сетевом диске.

Папка с "мозгами" бота (data/faqs.json, data/system_prompt.md,
data/knowledge_base.json, поисковый индекс) — это часть кода/конфигурации,
не персональные данные, поэтому она никуда не переезжает и остаётся в
репозитории как раньше.
"""
import os
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

# Явно заданный путь имеет приоритет. Если не задан — папка HR/candidates
# ВНУТРИ папки проекта (рядом с backend/, data/), но вне git (см. .gitignore).
DATA_ROOT = os.environ.get("HR_DATA_DIR") or os.path.join(PROJECT_ROOT, "HR")
DATA_ROOT = os.path.abspath(DATA_ROOT)

CANDIDATS_DIR = os.path.join(DATA_ROOT, "candidates")
CONVERSATIONS_DIR = os.path.join(DATA_ROOT, "conversations")
LOGS_DIR = os.path.join(DATA_ROOT, "logs")


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
