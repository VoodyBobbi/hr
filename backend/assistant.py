import json
import os
import re
import threading
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sentence_transformers import SentenceTransformer

from .rag_index import load_index, search_similar
from . import candidates
from . import logger

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
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

WATCHED_FILES = [AGENT_PROMPT_PATH, KNOWLEDGE_BASE_PATH, INDEX_PATH, META_PATH]

MAX_HISTORY_MESSAGES = 20

os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

_state_lock = threading.RLock()
_state = {
    "mtimes": {},
    "system_prompt": "",
    "index": None,
    "metadata": None,
}

_history_lock = threading.Lock()


def _get_mtimes() -> dict:
    return {
        path: os.path.getmtime(path) if os.path.exists(path) else None
        for path in WATCHED_FILES
    }


def _build_system_prompt(agent_prompt: str, knowledge_base: dict) -> str:
    company = knowledge_base.get("company", {})
    vacancies = knowledge_base.get("vacancies", {})
    vacancy_titles = ", ".join(v.get("title", k) for k, v in vacancies.items())

    knowledge_summary = (
        f"\n\n## Компания\n"
        f"Название: {company.get('name', '')}\n"
        f"Описание: {company.get('description', '')}\n"
        f"Вакансии: {vacancy_titles}\n"
        f"\nПолная база знаний компании (JSON), используй как источник фактов:\n"
        f"{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}"
    )
    return agent_prompt + knowledge_summary


def _reload_all_locked():
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        agent_prompt = f.read().strip()

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    new_system_prompt = _build_system_prompt(agent_prompt, knowledge_base)
    new_index, new_metadata = load_index(INDEX_PATH, META_PATH)

    _state["system_prompt"] = new_system_prompt
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


def _conversation_path(source: str, external_id: str) -> str:
    safe_id = "".join(c if c.isalnum() else "_" for c in str(external_id))
    return os.path.join(CONVERSATIONS_DIR, f"{source}_{safe_id}.json")


def _load_history(source: str, external_id: str) -> list:
    path = _conversation_path(source, external_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_history(source: str, external_id: str, history: list):
    path = _conversation_path(source, external_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# Ищет строки вида: ##SAVE_FIELD:поле=значение##  или  ###SAVE_FIELD:поле=значение###
# (терпимо к 2 или 3+ решёткам, к пробелам, к регистру ключевых слов)
MARKER_LINE_PATTERN = re.compile(
    r"^\s*#{2,}\s*(SAVE_FIELD\s*:\s*.+?|LAW_ACK|CARD_CONFIRMED)\s*#{0,}\s*$",
    re.IGNORECASE,
)


def _process_markers(raw_text: str, candidate_id: str) -> str:
    """Построчно находит служебные маркеры, выполняет действия, убирает их из текста."""
    clean_lines = []

    for line in raw_text.split("\n"):
        match = MARKER_LINE_PATTERN.match(line)
        if not match:
            clean_lines.append(line)
            continue

        marker_body = match.group(1).strip()

        if marker_body.upper().startswith("SAVE_FIELD"):
            payload = marker_body.split(":", 1)[1] if ":" in marker_body else ""
            if "=" in payload:
                field_name, value = payload.split("=", 1)
                candidates.set_field(candidate_id, field_name.strip(), value.strip())
        elif marker_body.upper() == "LAW_ACK":
            candidates.mark_law_acknowledged(candidate_id)
        elif marker_body.upper() == "CARD_CONFIRMED":
            candidates.mark_card_confirmed(candidate_id)
        # строка-маркер не добавляется в clean_lines — пользователь её не увидит

    return "\n".join(clean_lines).strip()


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
        current_system_prompt = _state["system_prompt"]

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
        with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as giga:
            response = giga.chat(Chat(messages=messages))
        raw_answer = response.choices[0].message.content
        status = "ok"
        error_comment = ""
    except Exception as e:
        raw_answer = (
            "Сейчас нет доступа к серверу ассистента по техническим причинам. "
            "Пожалуйста, попробуйте написать чуть позже, либо свяжитесь с менеджером напрямую."
        )
        status = "error"
        error_comment = str(e)
        print(f"[assistant] Ошибка обращения к GigaChat: {e}")

    clean_answer = _process_markers(raw_answer, candidate_id)

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

    return clean_answer, similar_items