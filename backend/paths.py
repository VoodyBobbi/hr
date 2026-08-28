"""
Расположение файлов с персональными данными — ОТДЕЛЬНО от кода.

Раньше anketы, история переписки и логи лежали прямо внутри репозитория
(`candidats/`, `data/conversations/`, `logs/`), из-за чего файл с анкетами
однажды попал в git и оказался в публичном GitHub-репозитории.

Теперь все три папки живут вне кода — по умолчанию в домашней директории
администратора (`~/hr-data`), а не там, откуда запущен/склонирован репозиторий.
Расположение можно переопределить переменной окружения HR_DATA_DIR (в .env
или в переменных окружения ОС) — например, указать конкретную папку на диске
администратора.

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

# Явно заданный путь имеет приоритет. Если не задан — папка ВНЕ репозитория,
# в домашней директории пользователя, под именем hr-data.
DATA_ROOT = os.environ.get("HR_DATA_DIR") or os.path.join(os.path.expanduser("~"), "hr-data")
DATA_ROOT = os.path.abspath(DATA_ROOT)

CANDIDATS_DIR = os.path.join(DATA_ROOT, "candidats")
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
