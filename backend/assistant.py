import json
import os
import re
import threading
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
import gigachat.context
from gigachat import GigaChat
from gigachat.exceptions import (
    AuthenticationError,
    GigaChatException,
    RateLimitError,
    ResponseError,
)
from gigachat.models import Chat, Messages, MessagesRole
from sentence_transformers import SentenceTransformer

from .rag_index import load_index, search_similar
from . import anketa
from . import candidates
from . import crypto_utils
from . import logger
from . import notifications
from . import paths
from .validators import FieldValidationError

load_dotenv()

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    raise RuntimeError("GIGACHAT_CREDENTIALS is not set. Please set it in your .env file.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
# Отдельный, второй FAISS-индекс для knowledge_base.json (см.
# backend/kb_chunker.py, backend/build_index.py) — раздельный от FAQ-индекса
# выше по требованию "две раздельные поиска, не один общий top-3" (см.
# обсуждение пункта про сжатие промпта/базы): вопрос кандидата ищется И в
# faqs.json (готовые ответы, search-first — см. get_answer), И отдельно в
# knowledge_base.json (факты для контекста GigaChat, если FAQ не нашёл
# готового ответа) — это два разных назначения, смешивать их в один общий
# top-3 давало бы, например, что более релевантный FAQ-ответ может быть
# вытеснен менее релевантным KB-фактом просто по случайному порядку.
KB_INDEX_PATH = os.path.join(DATA_DIR, "kb_faiss_index.bin")
KB_META_PATH = os.path.join(DATA_DIR, "kb_metadata.npy")
AGENT_PROMPT_PATH = os.path.join(DATA_DIR, "system_prompt.md")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")

# История переписки — персональные данные кандидата, поэтому лежит ВНЕ
# репозитория (папка на диске администратора, см. backend/paths.py), в
# отличие от базы знаний бота выше (FAQ/промпт — это конфигурация, не ПД).
CONVERSATIONS_DIR = paths.CONVERSATIONS_DIR

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

WATCHED_FILES = [AGENT_PROMPT_PATH, KNOWLEDGE_BASE_PATH, INDEX_PATH, META_PATH, KB_INDEX_PATH, KB_META_PATH]

MAX_HISTORY_MESSAGES = 20

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

_state_lock = threading.RLock()
_state = {
    "mtimes": {},
    "agent_prompt": "",
    "knowledge_base": {},
    "index": None,
    "metadata": None,
}

_history_lock = threading.Lock()


def _get_mtimes() -> dict:
    return {
        path: os.path.getmtime(path) if os.path.exists(path) else None
        for path in WATCHED_FILES
    }


# _build_vacancy_summary/_detect_vacancy_key/_VACANCY_KEYWORDS — старая
# эвристическая логика выбора релевантной вакансии для промпта, полностью
# заменена отдельным KB-поиском по всей базе знаний (см.
# backend/kb_chunker.py, _build_system_prompt выше) — специальная
# обработка именно раздела vacancies больше не нужна: любой вопрос,
# включая про конкретную вакансию, находит релевантные факты через общий
# механизм поиска, не только про vacancies.


def _build_system_prompt(
    agent_prompt: str,
    knowledge_base: dict,
    kb_chunks: list | None = None,
) -> str:
    """Собирает системный промпт под конкретный вопрос кандидата.

    Раньше вся knowledge_base.json (кроме специально обработанного раздела
    vacancies) грузилась в промпт ЦЕЛИКОМ на каждом запросе — независимо от
    того, что реально спросил кандидат. Теперь: только базовая сводка о
    компании (название, вакансии — дёшево по токенам, нужно почти в любом
    диалоге) плюс kb_chunks — несколько фрагментов knowledge_base.json,
    найденных ОТДЕЛЬНЫМ поиском по вопросу кандидата (см. get_answer,
    backend/kb_chunker.py, backend/build_index.py — knowledge_base
    индексируется отдельно от faqs.json, с отдельным поиском, а не
    смешивается в один общий top-3). Не найдено ни одного релевантного
    чанка (kb_chunks пустой/None) — секция ниже просто не появляется в
    промпте, вместо пустого JSON-блока, как было раньше."""
    company = knowledge_base.get("company", {})
    vacancies = knowledge_base.get("vacancies", {})
    vacancy_titles = ", ".join(v.get("title", k) for k, v in vacancies.items())

    knowledge_summary = (
        f"\n\n## Компания\n"
        f"Название: {company.get('name', '')}\n"
        f"Описание: {company.get('description', '')}\n"
        f"Вакансии: {vacancy_titles}\n"
    )

    if kb_chunks:
        chunks_text = "\n".join(f"- {c['question']}: {c['answer']}" for c in kb_chunks)
        knowledge_summary += (
            f"\n## Найденные по вопросу кандидата факты из базы знаний\n"
            f"{chunks_text}\n"
            f"\nИспользуй эти факты как источник, не выдумывай детали сверх них. Если для "
            f"вопроса кандидата здесь не хватает данных — честно скажи, что уточнишь, "
            f"вместо того чтобы придумывать правдоподобный ответ."
        )

    return agent_prompt + knowledge_summary


def _reload_all_locked():
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        agent_prompt = f.read().strip()

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    new_index, new_metadata = load_index(INDEX_PATH, META_PATH)

    # KB-индекс, в отличие от FAQ-индекса выше, оборачиваем в try/except:
    # load_index() бросает RuntimeError, если файла нет на диске (см.
    # rag_index.py) — для FAQ-индекса это оправданно (без него бот
    # физически не может отвечать, пусть падает явно), но KB-индекс —
    # более новая функция (собирается build_index.py при следующей
    # пересборке, см. run_all.py/start.sh); на самой первой сборке проекта
    # ПОСЛЕ этой правки, до первого запуска build_index.py, файла ещё не
    # будет. Явный RuntimeError здесь уронил бы весь сервер только из-за
    # отсутствия опциональной оптимизации — вместо этого просто отключаем
    # KB-поиск (get_answer работает без него, как и раньше до этой правки).
    try:
        new_kb_index, new_kb_metadata = load_index(KB_INDEX_PATH, KB_META_PATH)
    except RuntimeError:
        new_kb_index, new_kb_metadata = None, None
        print(
            "[assistant] KB-индекс не найден (ожидается после следующего "
            "запуска `python -m backend.build_index`) — поиск по базе "
            "знаний временно отключён, отвечает только FAQ-поиск/GigaChat."
        )

    # Кешируем СЫРЫЕ agent_prompt/knowledge_base, а не готовую строку
    # системного промпта — сам промпт теперь собирается заново под каждого
    # конкретного кандидата в get_answer() (см. _build_system_prompt), в
    # зависимости от того, какая вакансия ему уже известна/интересна. Чтение
    # с диска и парсинг JSON — единственная тяжёлая часть, которую есть
    # смысл кешировать между запросами; сама сборка строки промпта из уже
    # распарсенного словаря — дёшево, её можно делать каждый раз.
    _state["agent_prompt"] = agent_prompt
    _state["knowledge_base"] = knowledge_base
    _state["index"] = new_index
    _state["metadata"] = new_metadata
    _state["kb_index"] = new_kb_index
    _state["kb_metadata"] = new_kb_metadata
    _state["mtimes"] = _get_mtimes()

    print("[assistant] База знаний и промпт (пере)загружены.")


def _ensure_fresh():
    with _state_lock:
        current_mtimes = _get_mtimes()
        if current_mtimes != _state["mtimes"]:
            _reload_all_locked()


with _state_lock:
    _reload_all_locked()


def embed_text(text: str) -> np.ndarray:
    vector = embedding_model.encode([text], convert_to_numpy=True)
    return vector.astype("float32")


def _encrypt_content(content: str) -> str:
    """Шифрует только текст сообщения ("content"), не всю запись целиком —
    "role" ("user"/"assistant") не персональные данные, оставлен открытым,
    чтобы файл истории оставался человекочитаемым по структуре диалога при
    беглом просмотре (кто говорил, сколько реплик), даже если сам текст
    реплик зашифрован."""
    if not content:
        return content
    return crypto_utils.encrypt_bytes(content.encode("utf-8")).decode("ascii")


def _decrypt_content(content: str) -> str:
    if not content:
        return content
    try:
        content.encode("ascii")
    except UnicodeEncodeError:
        # Не-ASCII символы — физически не Fernet-токен, значит открытый
        # текст с прошлого формата (до включения шифрования истории).
        return content
    try:
        return crypto_utils.decrypt_bytes(content.encode("ascii")).decode("utf-8")
    except crypto_utils.EncryptionKeyInvalid:
        # Другой ключ либо повреждённые данные — не роняем всю историю
        # диалога из-за одного нечитаемого сообщения.
        return f"[не удалось расшифровать: {content}]"


def _load_history(source: str, external_id: str) -> list:
    path = paths.conversation_path(source, external_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    for message in history:
        if "content" in message:
            message["content"] = _decrypt_content(message["content"])
    return history


def _save_history(source: str, external_id: str, history: list):
    path = paths.conversation_path(source, external_id)
    encrypted_history = [
        {**message, "content": _encrypt_content(message.get("content", ""))}
        for message in history
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encrypted_history, f, ensure_ascii=False, indent=2)


# --- Поиск служебных маркеров ---
#
# ВАЖНО: модель НЕ гарантирует, что маркер стоит на отдельной строке — часто
# лепит его в конец обычной фразы без переноса строки, например:
#   "Теперь, как ваше имя? ###SAVE_FIELD:Имя"
# Поэтому маркер ищется ГДЕ УГОДНО в тексте (не только в начале строки).
#
# Два паттерна, в порядке применения:
#
# 1) CLOSED — маркер с явным закрывающим ###. Значение ограничено [^\n#] —
#    не выходит за пределы строки и останавливается на первом же ###, поэтому
#    НЕ поглощает текст, который идёт после маркера в том же сообщении.
#    Без этого ограничения (просто .+? до конца строки/текста) наблюдался
#    реальный баг: "###SAVE_FIELD:Имя=Иван### Теперь, скажите..." — модель
#    закрыла маркер сразу после значения, но старая регулярка всё равно
#    дотягивалась до конца сообщения и записывала в поле "Иван### Теперь,
#    скажите, пожалуйста, вашу фамилию?" — а пользователю уходило пустое
#    сообщение (см. candidates.csv кандидата 2, поле "Имя").
#
# 2) OPEN — запасной вариант для маркера БЕЗ закрывающих ### (модель иногда
#    просто не дописывает закрытие в конце сообщения). Съедает до конца
#    СТРОКИ (\n или конец текста), но не дальше — не может поглотить
#    следующий абзац, если он есть.
#
# Оба риска (маркер не на отдельной строке, маркер без закрытия) актуальны
# и для единственного маркера, оставшегося после переделки анкеты ниже —
# паттерны и защита от них не переписывались, только сузился набор
# распознаваемых маркеров с четырёх до одного.
#
# Единственный маркер, оставшийся у GigaChat после переделки анкеты на
# детерминированную стейт-машину (см. backend/anketa.py и обсуждение пункта
# 2 в истории этого проекта: "зачем отправлять паспорт/СНИЛС/ИНН в
# GigaChat, если программа и так видит, заполнено поле или нет").
#
# START_ANKETA — чисто управляющий сигнал ("кандидат согласился начать
# анкету"), НЕ содержит никаких персональных данных вообще, в отличие от
# прежнего SAVE_FIELD:поле=значение, где как раз в промпт уходило само
# значение поля. GigaChat по-прежнему ведёт обычный диалог (FAQ,
# консультации по вакансиям, вопросы про условия) — это не убрано и не
# урезано — просто в какой-то момент естественно предлагает перейти к
# анкете и, когда кандидат соглашается, ставит этот маркер. Дальше —
# согласие 152-ФЗ, все 29 полей, подтверждение карточки, удаление —
# полностью на стороне _handle_anketa_turn/anketa.py, GigaChat к этому не
# возвращается, пока анкета не будет закрыта (см. get_answer).
CLOSED_MARKER_PATTERN = re.compile(r"#{2,}\s*(START_ANKETA)\s*#{2,}", re.IGNORECASE)
OPEN_MARKER_PATTERN = re.compile(r"#{2,}\s*(START_ANKETA)\s*#{0,}\s*(?=\n|$)", re.IGNORECASE)

# Модель иногда (по привычке из markdown) оборачивает маркер в тройные кавычки
# ```...```, хотя промпт этого не просит. Сам маркер вырезается по паттернам
# выше, а после этого от него может остаться пустая пара ```\n```. Эта
# регулярка вырезает ТОЛЬКО пустые/пробельные fence-пары — легитимные code
# block с реальным содержимым внутри не трогает.
EMPTY_FENCE_PATTERN = re.compile(r"```\w*[ \t]*\n?\s*```")


def _process_markers(raw_text: str) -> tuple[str, bool]:
    """
    Находит маркер START_ANKETA в тексте ответа GigaChat (в любом месте, не
    только на отдельной строке), убирает его из текста, который увидит
    кандидат.

    Возвращает (текст_без_маркера, анкета_запрошена):
    - анкета_запрошена — True, если в этом ответе появился START_ANKETA —
      используется в get_answer, чтобы на СЛЕДУЮЩЕМ ходу сразу передать
      диалог в анкетную ветку (_handle_anketa_turn), а не снова в GigaChat.
    """
    anketa_requested = False

    def handle_marker(marker_body: str) -> str:
        nonlocal anketa_requested
        if marker_body.strip().upper() == "START_ANKETA":
            anketa_requested = True
        return ""

    text = CLOSED_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), raw_text)
    text = OPEN_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), text)
    text = EMPTY_FENCE_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # схлопываем пустые строки, оставшиеся от вырезанного маркера

    clean_text = text.strip()

    if not clean_text:
        # Модель прислала ответ, состоящий ТОЛЬКО из маркера, без единого
        # слова для кандидата. Это баг промпта (модель обязана сопровождать
        # маркер текстом), не маскируем его красивой заглушкой, а громко
        # логируем, чтобы это было видно в консоли и не потерялось молча.
        print(
            f"[assistant] ВНИМАНИЕ: ответ модели стал пустым после зачистки "
            f"маркера. Сырой ответ модели: {raw_text!r}"
        )

    return clean_text, anketa_requested


def _handle_anketa_turn(candidate_id: str, source: str, external_id: str,
                         user_message: str, last_bot_message: str,
                         anketa_start_requested: bool = False):
    """Обрабатывает один ход диалога в рамках анкеты — согласие 152-ФЗ,
    сбор 29 полей, подтверждение готовой карточки, команду удаления.

    anketa_start_requested — True, если ИМЕННО в предыдущем ответе GigaChat
    поставила маркер START_ANKETA (см. _process_markers/get_answer) — то
    есть кандидат только что согласился начать анкету в обычном диалоге.
    Без этого флага функция НЕ инициирует показ текста согласия 152-ФЗ сама
    по себе — иначе ЛЮБОЕ первое сообщение любого нового кандидата (даже
    простой вопрос "какие у вас вакансии?") сразу превращалось бы в анкету,
    минуя обычный FAQ-диалог, для которого GigaChat как раз и нужна.

    Возвращает готовый текст ответа кандидату, если сообщение было
    обработано анкетной веткой (кандидат либо только что согласился начать
    анкету, либо уже находится в процессе её заполнения) — в этом случае
    GigaChat НЕ вызывается вообще для ЭТОГО хода, вызывающий код
    (get_answer) сразу отдаёт этот текст.

    Возвращает None, если анкетная ветка кандидата не касается — согласие
    ещё не запрошено (обычный диалог продолжается через GigaChat) либо уже
    дано и карточка уже подтверждена (тоже обычный диалог) — тогда
    get_answer продолжает свой обычный путь ниже.
    """
    # Ждём подтверждения удаления: ПРЕДЫДУЩИМ ответом бота было именно
    # предупреждение об удалении (DELETE_CONFIRMATION_PROMPT) — значит
    # сейчас кандидат либо подтверждает, либо нет. Проверяется в первую
    # очередь, до всего остального: команда удаления имеет приоритет над
    # обычным ходом анкеты на любом её шаге.
    if last_bot_message == anketa.DELETE_CONFIRMATION_PROMPT:
        if anketa.is_delete_confirmed(user_message):
            candidates.delete_candidate(candidate_id)
            return anketa.DELETE_DONE_MESSAGE
        # Не подтвердил явно — кандидат передумал удалять. Сама фраза
        # отказа ("не, погоди, продолжим") НЕ является ответом на текущий
        # вопрос анкеты — раньше здесь код "проваливался" в обычную
        # обработку с этой же фразой, что приводило к попытке сохранить её
        # как значение поля (например "Отчество: не, погоди, продолжим",
        # отклонённое валидатором с непонятной для кандидата ошибкой).
        # Вместо этого явно повторяем вопрос текущего поля — ничего не
        # было тронуто на шаге предупреждения, продолжаем с того же места.
        next_step = anketa.get_next_step(candidate_id)
        if next_step:
            return next_step[1]
        return anketa.format_card_for_confirmation(candidate_id)

    # Команда удаления может прозвучать на любом шаге анкеты (кандидат
    # передумал на середине заполнения) — проверяется до состояний ниже.
    # Условие ниже: согласие уже дано (иначе анкеты как таковой ещё нет,
    # удалять пока нечего — это отдельно решает "явный отказ" в состоянии 1
    # ниже) И карточка ещё не подтверждена (после подтверждения — это уже
    # состояние 5, обычный диалог, там команда удаления не перехватывается
    # этой веткой вообще).
    law_given = candidates.is_law_acknowledged(candidate_id)
    if law_given and not _card_already_confirmed(candidate_id):
        if anketa.is_delete_request(user_message):
            return anketa.DELETE_CONFIRMATION_PROMPT

    # Состояние 1: согласие 152-ФЗ ещё не получено.
    if not candidates.is_law_acknowledged(candidate_id):
        if last_bot_message == anketa.LAW_CONSENT_TEXT:
            # Это ответ на уже показанный текст согласия.
            if anketa.is_law_declined(user_message):
                # Явный отказ — анкету не начинаем, кандидат остаётся в
                # обычном диалоге (FAQ и консультации по-прежнему доступны,
                # см. get_answer ниже — там как раз обычный путь через
                # GigaChat, просто анкета не заводится).
                return (
                    "Хорошо, анкету заполнять не будем. Если у вас есть "
                    "вопросы о вакансиях — с радостью отвечу."
                )
            # Что угодно ещё — считаем согласием.
            candidates.mark_law_acknowledged(candidate_id)
            first_field, first_question = anketa.get_next_step(candidate_id)
            return first_question
        if anketa_start_requested:
            # GigaChat ИМЕННО СЕЙЧАС поставила START_ANKETA (см. docstring
            # выше) — кандидат только что согласился начать анкету в
            # обычном диалоге. Показываем текст согласия впервые.
            return anketa.LAW_CONSENT_TEXT
        # Согласия не было запрошено — это НЕ анкетная ветка, обычный
        # диалог продолжается через GigaChat (FAQ, вопросы про вакансии).
        return None

    # Состояние 5: карточка уже подтверждена ранее — анкета кандидата закрыта,
    # это НЕ анкетная ветка, дальше обычный диалог через GigaChat.
    if _card_already_confirmed(candidate_id):
        return None

    # Состояние 3: все 29 полей заполнены, ждём подтверждения карточки.
    if candidates.is_card_complete(candidate_id):
        if last_bot_message.startswith("Проверьте, пожалуйста, все данные"):
            if anketa.is_confirmation_yes(user_message):
                candidates.mark_card_confirmed(candidate_id)
                fresh_card = candidates.get_card(candidate_id)
                notifications.send_hr_notification(
                    notifications.format_new_candidate_notification(fresh_card)
                )
                return (
                    "Спасибо! Анкета принята, с вами свяжется наш менеджер. "
                    "Если появятся вопросы — я на связи."
                )
            # Не "да" — считаем, что кандидат хочет что-то поправить, но у
            # нас нет ИИ, чтобы понять, что именно он имеет в виду свободным
            # текстом (это и есть сама суть отказа от GigaChat внутри
            # анкеты) — просим прямо назвать поле.
            return (
                "Если нужно что-то исправить — назовите поле и новое "
                "значение, например: «Телефон 89991234567». Если всё "
                "верно — напишите «да»."
            )
        return anketa.format_card_for_confirmation(candidate_id)

    # Состояние 2: основной цикл сбора полей.
    #
    # Если мы дошли до этой точки — согласие уже получено (иначе вернулись
    # бы в состоянии 1 выше), карточка ещё не полностью заполнена (иначе
    # ушли бы в состояние 3 выше) — значит user_message по построению
    # ВСЕГДА является ответом на текущее незаполненное поле (next_step),
    # каким бы ни было предыдущее сообщение бота: сам вопрос, или текст
    # ошибки валидации ("не похоже на фамилию") после неудачной попытки —
    # в обоих случаях кандидат сейчас отвечает на один и тот же вопрос,
    # просто со второй/третьей попытки.
    #
    # ВАЖНОЕ ИСКЛЮЧЕНИЕ: сюда же попадает единственный случай, где
    # user_message НЕ является ответом на поле — момент, когда согласие
    # 152-ФЗ было получено только что, и бот ещё не успел показать первый
    # вопрос анкеты (это происходит в состоянии 1 выше явным return, минуя
    # этот код) — то есть на практике до этой строки такое сообщение никогда
    # не доходит, состояние 1 перехватывает его раньше.
    # Мета-вопрос "что я уже заполнил" — не значение поля, а вопрос о самом
    # процессе. Проверяется до попытки сохранить как значение (см. ниже) —
    # иначе такой вопрос ошибочно интерпретировался бы как некорректный
    # ответ на текущее поле (см. anketa.py:is_progress_question, там же
    # объяснение, почему это отдельная проверка нужна именно теперь, после
    # переделки анкеты на детерминированную стейт-машину).
    if anketa.is_progress_question(user_message):
        return anketa.format_progress_answer(candidate_id)

    next_step = anketa.get_next_step(candidate_id)
    if next_step:
        field, _ = next_step
        ok, message = anketa.process_answer(candidate_id, field, user_message)
        if ok and message is None:
            # Это было последнее из 29 полей — сразу показываем карточку.
            return anketa.format_card_for_confirmation(candidate_id)
        return message

    return None


def _card_already_confirmed(candidate_id: str) -> bool:
    card = candidates.get_card(candidate_id)
    return bool(card.get("Факт ознакомления с готовой карточкой", "").strip())


def _format_candidate_progress(card: dict) -> str:
    filled = {
        k: v for k, v in card.items()
        if v and k not in ("ID кандидата", "Источник", "ДАТА заполнения")
    }
    if not filled:
        return "Анкета этого кандидата ещё не начата."

    count = len(filled)
    return (
        f"Кандидат уже сообщил {count} пункт(ов) анкеты (используй это только для себя, "
        f"чтобы не спрашивать повторно; отвечай пользователю простыми словами, БЕЗ технических "
        f"названий полей и без выгрузки полного списка, если он явно не попросил это): "
        + ", ".join(filled.keys())
    )


def get_answer(user_message: str, source: str, external_id: str, top_k: int = 3):
    """
    source — "site" или "telegram".
    external_id — стабильный идентификатор диалога (session_id сайта или chat_id Telegram).
    """
    start_time = time.time()
    _ensure_fresh()

    candidate_id = candidates.get_or_create_candidate(source, external_id)

    with _history_lock:
        history = _load_history(source, external_id)

    # --- Анкетная ветка: полностью без GigaChat --------------------------
    #
    # Раньше вся анкета (включая распознавание согласия 152-ФЗ, самих
    # значений полей и команды удаления) шла через диалог с GigaChat —
    # модель ВИДЕЛА текст кандидата, включая паспорт/СНИЛС/ИНН, чтобы
    # понять, что сохранить (см. историю правок этого файла и обсуждение
    # пункта 2 в этом чате). Ветка ниже — детерминированная стейт-машина
    # (anketa.py): вопрос -> валидатор без ИИ -> следующий вопрос либо
    # дружелюбная ошибка. GigaChat здесь не вызывается ни разу.
    #
    # last_bot_message используется только для одного: чтобы понять, было
    # ли ПРЕДЫДУЩИМ сообщением бота предупреждение об удалении (см.
    # _handle_anketa_turn ниже) — это НЕ хранится как отдельное состояние в
    # candidates.csv (лишнее поле ради однооборотного состояния), достаточно
    # посмотреть последний ответ бота в уже загруженной истории диалога.
    last_bot_message = history[-1]["content"] if history and history[-1].get("role") == "assistant" else ""

    anketa_reply = _handle_anketa_turn(candidate_id, source, external_id, user_message, last_bot_message)
    if anketa_reply is not None:
        response_time_ms = int((time.time() - start_time) * 1000)
        history.append({"role": MessagesRole.USER, "content": user_message})
        history.append({"role": MessagesRole.ASSISTANT, "content": anketa_reply})
        with _history_lock:
            _save_history(source, external_id, history)
        logger.log_interaction(source, external_id, user_message, anketa_reply, response_time_ms, "ok")
        return anketa_reply, []

    card = candidates.get_card(candidate_id)
    progress_note = _format_candidate_progress(card)

    with _state_lock:
        current_index = _state["index"]
        current_metadata = _state["metadata"]
        current_kb_index = _state["kb_index"]
        current_kb_metadata = _state["kb_metadata"]
        current_agent_prompt = _state["agent_prompt"]
        current_knowledge_base = _state["knowledge_base"]

    query_vec = embed_text(user_message)
    similar_items = search_similar(current_index, current_metadata, query_vec, k=top_k)

    # --- Search-first: FAQ раньше GigaChat -------------------------------
    #
    # Раньше search_similar() использовалась только как ИСТОЧНИК КОНТЕКСТА
    # для GigaChat (её результат подмешивался в промпт, но сам ответ всегда
    # формировала модель, даже если нашёлся точный совпадающий вопрос из
    # faqs.json). Теперь: нашёлся релевантный вопрос (similar_items
    # непустой — search_similar уже фильтрует по порогу схожести
    # max_distance, см. rag_index.py) — отдаём готовый ответ ИЗ FAQ
    # НАПРЯМУЮ, без обращения к GigaChat вообще (промпт даже не строится —
    # см. ниже, построение промпта происходит ПОСЛЕ этой проверки, а не до,
    # чтобы не тратить время на KB-поиск и сборку промпта впустую, если
    # готовый ответ уже найден). Экономит реальные деньги (не тратится
    # вызов GigaChat на вопрос, ответ на который уже есть готовым текстом)
    # и время ответа (без сетевого запроса к GigaChat).
    #
    # Берём ПЕРВЫЙ (самый близкий) результат — top_k=3 по умолчанию
    # существует для контекста, который раньше подмешивался в промпт
    # GigaChat, а не для того, чтобы выбирать между несколькими готовыми
    # ответами: если первый результат прошёл порог схожести, он и есть
    # лучшее совпадение с вопросом кандидата.
    if similar_items:
        faq_answer = similar_items[0]["answer"]
        response_time_ms = int((time.time() - start_time) * 1000)
        history.append({"role": MessagesRole.USER, "content": user_message})
        history.append({"role": MessagesRole.ASSISTANT, "content": faq_answer})
        history = history[-MAX_HISTORY_MESSAGES:]
        with _history_lock:
            _save_history(source, external_id, history)
        logger.log_interaction(source, external_id, user_message, faq_answer, response_time_ms, "ok", "faq_direct")
        return faq_answer, similar_items

    # Не нашли релевантный вопрос в FAQ — обращаемся к GigaChat. Прежде чем
    # строить промпт, делаем ОТДЕЛЬНЫЙ поиск по knowledge_base.json (см.
    # backend/kb_chunker.py, backend/build_index.py) — раздельный от FAQ-
    # поиска выше индекс, чтобы найденные факты из базы знаний не
    # конкурировали за место в top_k с готовыми FAQ-ответами (которые сюда
    # в принципе не должны попадать вместе с KB-фактами — разное
    # назначение, см. комментарий у KB_INDEX_PATH). Если current_kb_index
    # ещё не собран (None — см. _reload_all_locked) — просто идём с пустым
    # списком чанков, промпт соберётся без раздела базы знаний, как было
    # изначально, до первой сборки KB-индекса.
    kb_chunks = []
    if current_kb_index is not None:
        kb_chunks = search_similar(current_kb_index, current_kb_metadata, query_vec, k=top_k)

    # ВАЖНО: явно добавляем user_message отдельным элементом уже внутри
    # messages ниже — если кандидат называет вакансию ПРЯМО В ЭТОМ
    # сообщении (например, самое первое сообщение диалога), в history его
    # ещё нет; для самого промпта это не нужно (kb_chunks уже найдены по
    # query_vec — векторному представлению именно user_message).
    current_system_prompt = _build_system_prompt(
        current_agent_prompt,
        current_knowledge_base,
        kb_chunks=kb_chunks,
    )

    messages = [Messages(role=MessagesRole.SYSTEM, content=current_system_prompt)]

    for turn in history:
        messages.append(Messages(role=turn["role"], content=turn["content"]))

    messages.append(
        Messages(
            role=MessagesRole.USER,
            content=(
                f"[Служебная информация, не для показа пользователю] {progress_note}\n\n"
                f"Вопрос пользователя: {user_message}"
            ),
        )
    )

    try:
        # X-Session-ID включает кеширование на стороне GigaChat: если контекст
        # (системный промпт + предыдущие сообщения) частично совпадает с
        # прошлым запросом под тем же идентификатором, GigaChat не пересчитывает
        # эту часть заново — платится и считается только то, что реально
        # изменилось (см. https://developers.sber.ru/docs/ru/gigachat/guides/keeping-context,
        # раздел "Кэширование запросов"; ответ API возвращает точное число
        # закешированных токенов в поле precached_prompt_tokens).
        #
        # Используем candidate_id, а не session_id/chat_id: candidate_id
        # одинаково устроен для обоих каналов (веб и Telegram) и стабилен на
        # весь диалог одного кандидата — ровно то, что нужно кешу, чтобы
        # находить совпадение между последовательными сообщениями одного и
        # того же разговора.
        #
        # Экономия не гарантирована на 100% каждый запрос — зависит от того,
        # сколько контекста реально совпало с предыдущим (system-часть внутри
        # одного диалога не меняется между сообщениями кандидата, значит по
        # документации должна кешироваться), и от TTL кеша на стороне
        # GigaChat, который в документации явно не указан числом.
        gigachat.context.session_id_cvar.set(candidate_id)

        # verify_ssl_certs=True (по умолчанию у самой библиотеки gigachat)
        # — раньше здесь стояло False, потому что без корневого сертификата
        # НУЦ Минцифры проверка TLS для GigaChat падает с ошибкой
        # "self-signed certificate in certificate chain" (см. setup.py,
        # функция _install_gigachat_certificate — там же подробное
        # объяснение, откуда эта проблема, и почему verify_ssl_certs=False
        # было небезопасным способом её обойти). Теперь сертификат
        # устанавливается один раз при установке проекта (python setup.py),
        # и обычная проверка сертификата работает корректно — если
        # ПОЯВИТСЯ ошибка сертификата снова после этой правки, вероятная
        # причина — setup.py не смог скачать сертификат автоматически
        # (например, недоступен gu-st.ru) и нужно установить его вручную,
        # см. https://developers.sber.ru/docs/ru/gigachat/certificates
        with GigaChat(credentials=GIGACHAT_CREDENTIALS) as giga:
            response = giga.chat(Chat(messages=messages))
        raw_answer = response.choices[0].message.content
        status = "ok"
        error_comment = ""

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        precached_tokens = getattr(usage, "precached_prompt_tokens", None) if usage else None
        if precached_tokens:
            print(f"[assistant] GigaChat: {precached_tokens} токенов взято из кеша (не тарифицировано).")
    except Exception as e:
        # ВАЖНО: кандидат НИКОГДА не должен увидеть, что что-то сломалось —
        # ни слов "ошибка", ни "API", ни "лимит", ни любого другого признака
        # технической неисправности (решение продукта). При любом сбое
        # GigaChat бот просто отвечает на основе того, что уже нашлось в FAQ
        # (search_similar чуть выше по функции), как обычный ответ по базе,
        # а не как сообщение о проблеме.
        #
        # Для администратора при этом причина различается точно (по типу
        # исключения библиотеки gigachat, а не по угадыванию текста ошибки),
        # чтобы в logs.csv было по-настоящему понятно, что случилось —
        # неверный ключ это, исчерпанный лимит запросов или что-то ещё.
        if isinstance(e, AuthenticationError):
            error_kind = "gigachat_auth"
            print(
                f"[assistant] GigaChat: ошибка авторизации — проверьте "
                f"GIGACHAT_CREDENTIALS в .env (ключ неверный или истёк). {e}"
            )
        elif isinstance(e, RateLimitError):
            error_kind = "gigachat_rate_limit"
            print(f"[assistant] GigaChat: исчерпан лимит запросов к API (429). {e}")
        elif isinstance(e, (ResponseError, GigaChatException)):
            error_kind = "gigachat_api"
            print(f"[assistant] GigaChat: ошибка на стороне API. {e}")
        else:
            error_kind = "gigachat_unknown"
            print(f"[assistant] GigaChat: непредвиденная ошибка соединения. {e}")

        if similar_items:
            # Есть подходящий ответ в FAQ — отдаём его как обычный ответ по
            # базе, без единого слова о том, что модель была недоступна.
            raw_answer = similar_items[0]["answer"]
        else:
            # В FAQ ничего релевантного не нашлось — честно, но БЕЗ
            # технических слов: как будто бот действительно не знает ответ,
            # а не "сломался".
            raw_answer = (
                "Пока не могу сформулировать точный ответ на этот вопрос. "
                "Уточните, пожалуйста, что именно вас интересует, либо "
                "свяжитесь с менеджером напрямую — контакты есть на сайте."
            )
        status = "error"
        error_comment = f"{error_kind}: {e}"
        prompt_tokens = completion_tokens = total_tokens = precached_tokens = None

    clean_answer, anketa_start_requested = _process_markers(raw_answer)

    if anketa_start_requested:
        # GigaChat только что поставила START_ANKETA — кандидат согласился
        # начать анкету ПРЯМО В ЭТОМ сообщении. Не отдаём кандидату голый
        # ответ GigaChat (что-то вроде "Отлично, давайте начнём!") без
        # продолжения — сразу же, в этом же ходу, показываем текст согласия
        # 152-ФЗ (см. _handle_anketa_turn, anketa_start_requested=True),
        # объединяя его с ответом GigaChat в одно сообщение кандидату.
        consent_text = _handle_anketa_turn(
            candidate_id, source, external_id, user_message, last_bot_message,
            anketa_start_requested=True,
        )
        if consent_text:
            clean_answer = f"{clean_answer}\n\n{consent_text}" if clean_answer else consent_text

    history.append({"role": MessagesRole.USER, "content": user_message})
    history.append({"role": MessagesRole.ASSISTANT, "content": clean_answer})
    history = history[-MAX_HISTORY_MESSAGES:]

    with _history_lock:
        _save_history(source, external_id, history)

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.log_interaction(
        source=source,
        external_id=external_id,
        query=user_message,
        response=clean_answer,
        response_time_ms=elapsed_ms,
        status=status,
        comment=error_comment,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=precached_tokens,
    )

    return clean_answer, similar_items