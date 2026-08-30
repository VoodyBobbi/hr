import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .assistant import get_answer

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# --- CORS: разрешаем обращаться к /chat из браузера только с сайтов,
# перечисленных в ALLOWED_ORIGINS (.env) — не с ЛЮБОГО сайта в интернете.
#
# ВАЖНО (чтобы не путать это с полноценной защитой от нагрузки): CORS —
# это правило для БРАУЗЕРА, а не для сервера. Он не даёт постороннему сайту
# незаметно встроить этот чат в свою страницу через JavaScript в браузере
# случайного посетителя. Он НЕ мешает прямому запросу из Postman, curl или
# любого скрипта — у них нет браузера, который проверяет CORS. От прямых
# скриптов и потока запросов защищает ограничение частоты запросов ниже
# (Limiter), а не эта настройка.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

if not ALLOWED_ORIGINS:
    print(
        "[app] ВНИМАНИЕ: ALLOWED_ORIGINS не задан в .env — CORS не разрешит "
        "обращаться к /chat ни с одного сайта в браузере (кроме прямых "
        "запросов не из браузера). Добавьте свой домен в .env, если чат "
        "должен работать на сайте."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# --- Ограничение частоты запросов к /chat: не больше CHAT_RATE_LIMIT_PER_MINUTE
# сообщений в минуту с одного IP-адреса. Без этого один пользователь или
# программа мог(ла) отправлять запросы без остановки, и это била бы либо по
# счёту за GigaChat, либо по стабильности сервера (см. README).
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "20"))

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # Намеренно НЕ используем стандартный обработчик slowapi как есть: он по
    # умолчанию отдаёт текст вроде "Rate limit exceeded: 20 per 1 minute" —
    # это техническая формулировка, а кандидат не должен видеть ничего, что
    # звучит как ошибка системы (см. README про то, что клиент не должен
    # видеть технические сообщения). Ответ ниже отдаётся в ТОМ ЖЕ формате
    # {answer, context, session_id}, что и обычный /chat, чтобы фронтенду не
    # нужно было отдельно обрабатывать код 429 — сообщение просто придёт
    # как обычный ответ бота.
    session_id = request.query_params.get("session_id") or str(uuid.uuid4())
    return JSONResponse(
        status_code=200,
        content={
            "answer": (
                "Вы отправляете сообщения слишком часто — пожалуйста, "
                "подождите немного и повторите."
            ),
            "context": [],
            "session_id": session_id,
        },
    )


class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    context: list
    session_id: str


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{CHAT_RATE_LIMIT_PER_MINUTE}/minute")
def chat(request: Request, body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())

    answer, similar_items = get_answer(
        body.message,
        source="site",
        external_id=session_id,
        top_k=body.top_k,
    )

    return ChatResponse(answer=answer, context=list(similar_items), session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
