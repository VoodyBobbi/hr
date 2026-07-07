import json
import os

import numpy as np
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from sentence_transformers import SentenceTransformer

from .rag_index import load_index, search_similar

load_dotenv()

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
if not GIGACHAT_CREDENTIALS:
    raise RuntimeError("GIGACHAT_CREDENTIALS is not set. Please set it in your .env file.")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
AGENT_PROMPT_PATH = os.path.join(DATA_DIR, "system_prompt_hr_agent.md")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.json")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Файлы, за изменением которых следим (candidates.csv сюда НЕ входит намеренно)
WATCHED_FILES = [AGENT_PROMPT_PATH, KNOWLEDGE_BASE_PATH, INDEX_PATH, META_PATH]

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

_state = {
    "mtimes": {},
    "agent_prompt": "",
    "knowledge_base": {},
    "system_prompt": "",
    "index": None,
    "metadata": None,
}


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


def _reload_all():
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        _state["agent_prompt"] = f.read().strip()

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        _state["knowledge_base"] = json.load(f)

    _state["system_prompt"] = _build_system_prompt(
        _state["agent_prompt"], _state["knowledge_base"]
    )

    _state["index"], _state["metadata"] = load_index(INDEX_PATH, META_PATH)
    _state["mtimes"] = _get_mtimes()

    print("[assistant] База знаний и промпт (пере)загружены.")


def _ensure_fresh():
    current_mtimes = _get_mtimes()
    if current_mtimes != _state["mtimes"]:
        _reload_all()


# Первая загрузка при старте
_reload_all()


def embed_text(text: str) -> np.ndarray:
    vector = embedding_model.encode([text], convert_to_numpy=True)
    return vector.astype("float32")


def get_answer(user_message: str, top_k: int = 3):
    _ensure_fresh()

    query_vec = embed_text(user_message)
    similar_items = search_similar(_state["index"], _state["metadata"], query_vec, k=top_k)

    faq_context = "\n\n".join(
        f"Вопрос: {item['question']}\nОтвет: {item['answer']}" for item in similar_items
    )

    with GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as giga:
        response = giga.chat(
            Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=_state["system_prompt"]),
                    Messages(
                        role=MessagesRole.USER,
                        content=f"Похожие вопросы из FAQ:\n{faq_context}\n\nВопрос пользователя: {user_message}",
                    ),
                ]
            )
        )

    answer = response.choices[0].message.content
    return answer, similar_items