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
from . import candidates
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
AGENT_PROMPT_PATH = os.path.join(DATA_DIR, "system_prompt.md")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")

# История переписки — персональные данные кандидата, поэтому лежит ВНЕ
# репозитория (папка на диске администратора, см. backend/paths.py), в
# отличие от базы знаний бота выше (FAQ/промпт — это конфигурация, не ПД).
CONVERSATIONS_DIR = paths.CONVERSATIONS_DIR

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

WATCHED_FILES = [AGENT_PROMPT_PATH, KNOWLEDGE_BASE_PATH, INDEX_PATH, META_PATH]

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


def _build_vacancy_summary(vacancies: dict) -> str:
    """Краткая сводка всех вакансий (только название + зарплата) — показывается,
    пока не понятно, какая именно вакансия интересует кандидата. Экономит
    токены по сравнению с полными карточками всех трёх вакансий разом (детали
    каждой вакансии — duties, shift, schedule, height, equipment_and_safety и
    т.д. — в сводку не входят, только то, что нужно для первого выбора)."""
    lines = []
    for v in vacancies.values():
        lines.append(f"- {v.get('title', '')}: {v.get('salary', '')}")
    return "\n".join(lines)


_VACANCY_KEYWORDS = {
    "montazhnik": ["монтажник", "лес", "подмост"],
    "alpinist": ["альпинист", "альпинизм"],
    "izolyirovshik": ["изолировщик", "изоляц", "трубопровод"],
}


def _detect_vacancy_key(card: dict, recent_messages: list) -> str | None:
    """Определяет, о какой вакансии речь — по полю анкеты (если уже
    заполнено) или по ключевым словам в последних сообщениях диалога (если
    анкета ещё не дошла до этого поля, но кандидат уже назвал вакансию в
    обычном разговоре). Возвращает ключ из knowledge_base['vacancies'] или
    None, если вакансия ещё не понятна ни оттуда, ни оттуда."""
    declared = (card.get("Желаемая должность") or "").lower()
    for key, keywords in _VACANCY_KEYWORDS.items():
        if any(kw in declared for kw in keywords):
            return key

    # Поле анкеты ещё не заполнено (или не совпало) — смотрим в последних
    # сообщениях диалога, включая только что написанное. Идём с конца, самое
    # свежее упоминание вакансии важнее более раннего (кандидат мог сначала
    # спросить про одну вакансию, потом передумать и спросить про другую).
    for message in reversed(recent_messages):
        text = (message.get("content") or "").lower()
        for key, keywords in _VACANCY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return key
    return None


def _build_system_prompt(
    agent_prompt: str,
    knowledge_base: dict,
    candidate_card: dict | None = None,
    recent_messages: list | None = None,
) -> str:
    """Собирает системный промпт под конкретного кандидата: разделы
    knowledge_base НЕ привязанные к вакансии идут всегда целиком (ядро —
    company, general_conditions, selection_and_admission, документы,
    медотводы, правила самого ассистента и т.д. — то, что нужно почти в
    любом диалоге, независимо от вакансии). Раздел vacancies — самый
    большой по объёму и единственный явно разделённый на три
    самостоятельных, не связанных друг с другом блока — добавляется НЕ
    целиком: если понятно, какая вакансия интересует кандидата (см.
    _detect_vacancy_key), идёт только её блок; если ещё не понятно — идёт
    только краткая сводка (название + зарплата) всех трёх, а не полные
    карточки. Экономит основную часть токенов раздела vacancies на каждом
    запросе, при этом кандидат, уже назвавший вакансию, получает точный
    ответ по ней, а не усреднённый ответ по всем трём сразу."""
    candidate_card = candidate_card or {}
    recent_messages = recent_messages or []

    company = knowledge_base.get("company", {})
    vacancies = knowledge_base.get("vacancies", {})
    vacancy_titles = ", ".join(v.get("title", k) for k, v in vacancies.items())

    core = {k: v for k, v in knowledge_base.items() if k != "vacancies"}

    vacancy_key = _detect_vacancy_key(candidate_card, recent_messages)
    if vacancy_key and vacancy_key in vacancies:
        vacancy_section = (
            f"\nКандидат интересуется вакансией «{vacancies[vacancy_key].get('title', vacancy_key)}» "
            f"— вот полные условия именно по ней:\n"
            f"{json.dumps(vacancies[vacancy_key], ensure_ascii=False, indent=2)}\n"
            f"\nЕсли в разговоре станет понятно, что кандидата интересует ДРУГАЯ вакансия — "
            f"полные условия по ней появятся в следующем сообщении автоматически, не нужно "
            f"домысливать детали вакансий, которых нет в этом блоке."
        )
    else:
        vacancy_section = (
            f"\nЕщё не понятно, какая из трёх вакансий интересует кандидата — вот краткая сводка "
            f"(только название и зарплата, БЕЗ деталей по графику/обязанностям/требованиям):\n"
            f"{_build_vacancy_summary(vacancies)}\n"
            f"\nКак только кандидат назовёт вакансию (или это станет ясно из контекста), в "
            f"следующем сообщении появятся полные условия именно по ней — до этого не придумывай "
            f"детали графика, требований или обязанностей ни по одной из вакансий, опирайся "
            f"только на эту краткую сводку и уточняющие вопросы."
        )

    knowledge_summary = (
        f"\n\n## Компания\n"
        f"Название: {company.get('name', '')}\n"
        f"Описание: {company.get('description', '')}\n"
        f"Вакансии: {vacancy_titles}\n"
        f"\nБаза знаний компании, не привязанная к конкретной вакансии (JSON), используй как "
        f"источник фактов:\n"
        f"{json.dumps(core, ensure_ascii=False, indent=2)}\n"
        f"\n## Раздел по вакансии\n"
        f"{vacancy_section}"
    )
    return agent_prompt + knowledge_summary


def _reload_all_locked():
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        agent_prompt = f.read().strip()

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    new_index, new_metadata = load_index(INDEX_PATH, META_PATH)

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


def _load_history(source: str, external_id: str) -> list:
    path = paths.conversation_path(source, external_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_history(source: str, external_id: str, history: list):
    path = paths.conversation_path(source, external_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


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
CLOSED_MARKER_PATTERN = re.compile(
    r"#{2,}\s*(SAVE_FIELD\s*:\s*[^\n#]+?|LAW_ACK|CARD_CONFIRMED|DELETE_MY_DATA)\s*#{2,}",
    re.IGNORECASE,
)
OPEN_MARKER_PATTERN = re.compile(
    r"#{2,}\s*(SAVE_FIELD\s*:\s*[^\n#]+|LAW_ACK|CARD_CONFIRMED|DELETE_MY_DATA)\s*#{0,}\s*(?=\n|$)",
    re.IGNORECASE,
)

# Модель иногда (по привычке из markdown) оборачивает маркер в тройные кавычки
# ```...```, хотя промпт этого не просит. Сам маркер вырезается по паттернам
# выше, а после этого от него может остаться пустая пара ```\n```. Эта
# регулярка вырезает ТОЛЬКО пустые/пробельные fence-пары — легитимные code
# block с реальным содержимым внутри не трогает.
EMPTY_FENCE_PATTERN = re.compile(r"```\w*[ \t]*\n?\s*```")


def _process_markers(raw_text: str, candidate_id: str) -> tuple[str, bool, bool]:
    """
    Находит служебные маркеры в тексте (в любом месте, не только на отдельной
    строке), выполняет действия, убирает их из текста, который увидит кандидат.

    Возвращает (текст_без_маркеров, анкета_только_что_подтверждена, запрошено_удаление):
    - анкета_только_что_подтверждена — True, если ИМЕННО в этом ответе появился
      CARD_CONFIRMED (используется в get_answer для уведомления HR в Telegram).
    - запрошено_удаление — True, если кандидат подтвердил удаление своих данных
      (само удаление выполняется в get_answer ПОСЛЕ отправки этого ответа).

    ВАЖНО: модель (GigaChat) иногда нарушает собственные инструкции из системного
    промпта — например, присылает CARD_CONFIRMED раньше времени, когда часть из 29
    полей анкеты ещё пустая, пытается сохранить персональные данные до того,
    как кандидат дал согласие по 152-ФЗ (LAW_ACK), или путает синтаксис и пишет
    "SAVE_FIELD:LAW_ACK" вместо отдельного "LAW_ACK" (нормализуется ниже).
    Поэтому здесь добавлена серверная проверка "по факту" (на основе реальных
    данных в candidates.csv), а не только доверие тексту, который прислала модель.
    """
    law_acknowledged = candidates.is_law_acknowledged(candidate_id)
    validation_errors = []  # [(поле, сообщение), ...] — для короткой пометки кандидату в конце ответа
    card_just_confirmed = False  # True, если ИМЕННО в этом ответе анкета стала подтверждена — сигнал для уведомления HR (см. get_answer)
    delete_requested = False  # True, если кандидат подтвердил удаление — сам delete_candidate() вызывается ПОСЛЕ отправки ответа (см. get_answer), а не прямо здесь

    def handle_marker(marker_body: str) -> str:
        nonlocal law_acknowledged, card_just_confirmed, delete_requested
        body = marker_body.strip()

        if body.upper().startswith("SAVE_FIELD"):
            payload = body.split(":", 1)[1].strip() if ":" in body else ""

            # Защита от путаницы модели: "###SAVE_FIELD:LAW_ACK###" вместо
            # отдельного "###LAW_ACK###" — наблюдалось в реальных логах.
            if payload.upper() == "LAW_ACK":
                candidates.mark_law_acknowledged(candidate_id)
                law_acknowledged = True
                return ""

            if "=" not in payload:
                # Маркер без "=значение" (например модель проставила его ДО
                # ответа кандидата, просто пометив имя поля) — сохранять
                # нечего, просто убираем мусор из текста.
                return ""

            field_name, value = payload.split("=", 1)
            field_name = field_name.strip()
            real_field = candidates.normalize_field_name(field_name)

            # Жёсткий гейт по 152-ФЗ на уровне кода: пока нет согласия,
            # разрешено сохранять только "Желаемая должность".
            if real_field != "Желаемая должность" and not law_acknowledged:
                print(
                    f"[assistant] Заблокировано сохранение поля '{field_name}' "
                    f"для кандидата {candidate_id}: нет согласия по 152-ФЗ (LAW_ACK)."
                )
            else:
                try:
                    candidates.set_field(candidate_id, field_name, value.strip())
                except FieldValidationError as e:
                    print(
                        f"[assistant] Значение поля '{field_name}' для кандидата {candidate_id} "
                        f"не прошло проверку формата: {e.message} (получено: {value.strip()!r})"
                    )
                    # real_field гарантированно не None здесь: FieldValidationError
                    # бросается только для распознанных полей (см. candidates.set_field) —
                    # неизвестное имя поля отсекается раньше и без исключения.
                    validation_errors.append((real_field, e.message))

        elif body.upper() == "LAW_ACK":
            candidates.mark_law_acknowledged(candidate_id)
            law_acknowledged = True

        elif body.upper() == "CARD_CONFIRMED":
            if candidates.is_card_complete(candidate_id):
                candidates.mark_card_confirmed(candidate_id)
                card_just_confirmed = True
            else:
                missing = candidates.missing_fields(candidate_id)
                print(
                    f"[assistant] Заблокировано подтверждение анкеты кандидата {candidate_id}: "
                    f"не заполнены поля: {', '.join(missing)}."
                )

        elif body.upper() == "DELETE_MY_DATA":
            # Само удаление НЕ выполняется здесь (см. delete_requested в
            # get_answer) — оно должно произойти уже ПОСЛЕ того, как ответ
            # уйдёт кандидату и запишется история этого последнего обмена,
            # иначе get_answer попытается заново сохранить историю уже
            # удалённого кандидата на диск сразу после удаления.
            delete_requested = True

        return ""

    text = CLOSED_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), raw_text)
    text = OPEN_MARKER_PATTERN.sub(lambda m: handle_marker(m.group(1)), text)
    text = EMPTY_FENCE_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # схлопываем пустые строки, оставшиеся от вырезанных маркеров

    clean_text = text.strip()

    if not clean_text:
        # Модель прислала ответ, состоящий ТОЛЬКО из маркера(ов), без единого
        # слова для кандидата. Это баг промпта (см. системный промпт — модель
        # обязана сопровождать каждый маркер текстом), не маскируем его
        # красивой заглушкой, а громко логируем, чтобы это было видно в
        # logs.csv (колонка "Комментарий") и не потерялось молча, как раньше.
        print(
            f"[assistant] ВНИМАНИЕ: ответ модели кандидату {candidate_id} стал пустым "
            f"после зачистки маркеров. Сырой ответ модели: {raw_text!r}"
        )

    if validation_errors:
        # Значение не прошло проверку формата и НЕ было сохранено (см. handle_marker
        # выше) — кандидат должен об этом узнать, а не просто увидеть, что бот
        # промолчал про это поле и пошёл дальше.
        notes = " ".join(f"«{field}» — {message}" for field, message in validation_errors)
        clean_text = (clean_text + "\n\n" if clean_text else "") + f"Уточните, пожалуйста: {notes}"

    return clean_text, card_just_confirmed, delete_requested


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
    card = candidates.get_card(candidate_id)
    progress_note = _format_candidate_progress(card)

    with _history_lock:
        history = _load_history(source, external_id)

    with _state_lock:
        current_index = _state["index"]
        current_metadata = _state["metadata"]
        current_agent_prompt = _state["agent_prompt"]
        current_knowledge_base = _state["knowledge_base"]

    # Промпт собирается заново под ЭТОГО конкретного кандидата на каждый
    # запрос — не потому что это дёшево вообще (json.dumps на объект в
    # памяти — недорогая операция), а потому что раздел про вакансию должен
    # меняться вместе с тем, что уже известно об этом кандидате именно
    # сейчас (см. _build_system_prompt) — history содержит его прошлые
    # сообщения, card — уже сохранённые поля анкеты, если до них дошли.
    # ВАЖНО: явно добавляем user_message отдельным элементом — если кандидат
    # называет вакансию ПРЯМО В ЭТОМ сообщении (например, самое первое
    # сообщение диалога), в history его ещё нет, только в user_message.
    current_system_prompt = _build_system_prompt(
        current_agent_prompt,
        current_knowledge_base,
        candidate_card=card,
        recent_messages=history + [{"role": "user", "content": user_message}],
    )

    query_vec = embed_text(user_message)
    similar_items = search_similar(current_index, current_metadata, query_vec, k=top_k)

    faq_context = "\n\n".join(
        f"Вопрос: {item['question']}\nОтвет: {item['answer']}" for item in similar_items
    )

    messages = [Messages(role=MessagesRole.SYSTEM, content=current_system_prompt)]

    for turn in history:
        messages.append(Messages(role=turn["role"], content=turn["content"]))

    messages.append(
        Messages(
            role=MessagesRole.USER,
            content=(
                f"[Служебная информация, не для показа пользователю] {progress_note}\n\n"
                f"Похожие вопросы из FAQ:\n{faq_context}\n\n"
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

        with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as giga:
            response = giga.chat(Chat(messages=messages))
        raw_answer = response.choices[0].message.content
        status = "ok"
        error_comment = ""

        precached = getattr(response, "usage", None)
        precached_tokens = getattr(precached, "precached_prompt_tokens", None) if precached else None
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

    clean_answer, card_just_confirmed, delete_requested = _process_markers(raw_answer, candidate_id)

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
    )

    if card_just_confirmed:
        # Анкета именно СЕЙЧАС стала полностью заполнена и подтверждена —
        # уведомляем HR-группу в Telegram (если она настроена в .env). Берём
        # свежую карточку с диска, а не card из начала функции: та снята ДО
        # текущего ответа и ещё не содержит подтверждения.
        fresh_card = candidates.get_card(candidate_id)
        notifications.send_hr_notification(
            notifications.format_new_candidate_notification(fresh_card)
        )

    if delete_requested:
        # Удаление — ПОСЛЕ того, как этот ответ уже сохранён в историю и в
        # лог выше (см. _process_markers про порядок). Дальнейшие сообщения
        # от этого же source:external_id создадут НОВОГО кандидата (это
        # ожидаемо и правильно — старых данных больше не существует).
        result = candidates.delete_candidate(candidate_id)
        print(f"[assistant] Данные кандидата удалены по его запросу: {result}")

    return clean_answer, similar_items